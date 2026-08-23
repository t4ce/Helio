//! Helio-derived draw batching over SceneDB's canonical object query.

use helio_core::{DrawIndexedIndirectArgs, GpuDrawCall};
use helio_scenedb::{Entity, SceneMaterial, SceneObject};

use crate::handles::{entity_from_handle, MeshId};

use super::super::helpers::{
    object_groups, object_is_visible, object_material, object_mesh, object_movability,
};
use super::NO_OBJECT_PROJECTION_SLOT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RenderBucket {
    Opaque,
    Transparent,
    Forward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ObjectSortKey {
    class: u32,
    graph_hash: u64,
    bucket: RenderBucket,
    mesh_row: u32,
    material_row: u32,
}

#[derive(Debug, Clone, Copy)]
struct ObjectSortEntry {
    entity: Entity,
    key: ObjectSortKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaterialRangeKey {
    class: u32,
    graph_hash: u64,
    bucket: RenderBucket,
}

type MaterialRange = (u32, u64, u32, u32);

fn render_bucket(flags: u32) -> RenderBucket {
    if (flags & libhelio::FLAG_FORWARD_SHADING) != 0 {
        RenderBucket::Forward
    } else if (flags & libhelio::FLAG_TRANSPARENT_ONLY) != 0 {
        RenderBucket::Transparent
    } else {
        RenderBucket::Opaque
    }
}

fn build_material_ranges(
    keys: &[MaterialRangeKey],
) -> (Vec<MaterialRange>, Vec<MaterialRange>, Vec<MaterialRange>) {
    let mut opaque = Vec::new();
    let mut transparent = Vec::new();
    let mut forward = Vec::new();
    let mut index = 0;
    while index < keys.len() {
        let key = keys[index];
        let start = index as u32;
        while index < keys.len() && keys[index] == key {
            index += 1;
        }
        let range = (key.class, key.graph_hash, start, index as u32 - start);
        match key.bucket {
            RenderBucket::Opaque => opaque.push(range),
            RenderBucket::Transparent => transparent.push(range),
            RenderBucket::Forward => forward.push(range),
        }
    }
    (opaque, transparent, forward)
}

fn shadow_movable_partition_changed(
    old_sources: &[u32],
    new_sources: &[u32],
    old_draws: &[DrawIndexedIndirectArgs],
    new_draws: &[DrawIndexedIndirectArgs],
) -> bool {
    old_sources != new_sources
        || bytemuck::cast_slice::<DrawIndexedIndirectArgs, u8>(old_draws)
            != bytemuck::cast_slice::<DrawIndexedIndirectArgs, u8>(new_draws)
}

impl super::super::Scene {
    /// Rebuild compact draw order without cloning or repacking SceneDB rows.
    pub(in crate::scene) fn rebuild_instance_buffers(&mut self) {
        let mut entries: Vec<ObjectSortEntry> = self
            .authority
            .query::<SceneObject>()
            .map(|(entity, object)| {
                let material = self
                    .authority
                    .get::<SceneMaterial>(entity_from_handle(object_material(object)));
                let material_flags = material.map(|row| row.material.0.flags).unwrap_or(0);
                let (class, graph_hash) = material
                    .map(|row| (row.material.0.material_class, row.graph_hash))
                    .unwrap_or((0, 0));
                ObjectSortEntry {
                    entity,
                    key: ObjectSortKey {
                        class,
                        graph_hash,
                        bucket: render_bucket(material_flags),
                        mesh_row: object.render.mesh_row,
                        material_row: object.render.material_row,
                    },
                }
            })
            .collect();
        let object_count = entries.len();

        self.object_projection_slots
            .fill(NO_OBJECT_PROJECTION_SLOT);
        if let Some(max_index) = entries.iter().map(|entry| entry.entity.index()).max() {
            if self.object_projection_slots.len() <= max_index as usize {
                self.object_projection_slots
                    .resize(max_index as usize + 1, NO_OBJECT_PROJECTION_SLOT);
            }
        }

        if entries.is_empty() {
            self.gpu_scene.draw_calls.set_data(Vec::new());
            self.gpu_scene.draw_material_rows.clear();
            self.gpu_scene.draw_topology_generation =
                self.gpu_scene.draw_topology_generation.wrapping_add(1);
            self.gpu_scene.indirect.set_data(Vec::new());
            self.gpu_scene.visibility.set_data(Vec::new());
            self.gpu_scene.source_indices.set_data(Vec::new());
            self.gpu_scene.compacted_indices.set_data(Vec::new());
            self.gpu_scene.compacted_indices_2.set_data(Vec::new());
            self.gpu_scene.material_class_ranges.clear();
            self.gpu_scene.transparent_material_class_ranges.clear();
            self.gpu_scene.forward_material_class_ranges.clear();
            self.rebuild_shadow_partition_buffers();
            return;
        }

        entries.sort_by_key(|entry| entry.key);
        let mut draw_calls = Vec::<GpuDrawCall>::new();
        let mut draw_material_rows = Vec::<u32>::new();
        let mut indirect = Vec::<DrawIndexedIndirectArgs>::new();
        let mut visibility = Vec::<u32>::with_capacity(object_count);
        let mut source_indices = Vec::<u32>::with_capacity(object_count);
        let mut range_keys = Vec::<MaterialRangeKey>::new();
        let group_hidden = self.group_hidden();

        let mut cursor = 0;
        while cursor < entries.len() {
            let first_entry = entries[cursor];
            let first = self
                .authority
                .get::<SceneObject>(first_entry.entity)
                .expect("SceneDB query entry remained live during rebuild");
            let slice = self
                .mesh_pool()
                .get(object_mesh(first))
                .expect("live SceneObject must retain its GeometryArena/MeshPool edge")
                .slice;
            let group_start = source_indices.len() as u32;

            while cursor < entries.len() && entries[cursor].key == first_entry.key {
                let entry = entries[cursor];
                let object = self
                    .authority
                    .get::<SceneObject>(entry.entity)
                    .expect("SceneDB query entry remained live during rebuild");
                let projection_slot = source_indices.len() as u32;
                self.object_projection_slots[entry.entity.index() as usize] = projection_slot;
                source_indices.push(
                    self.authority
                        .gpu_row::<SceneObject>(entry.entity)
                        .expect("queried SceneObject must have a component-local GPU row"),
                );
                visibility.push(u32::from(object_is_visible(
                    object_groups(object),
                    group_hidden,
                )));
                cursor += 1;
            }

            let instance_count = source_indices.len() as u32 - group_start;
            draw_calls.push(GpuDrawCall {
                index_count: slice.index_count,
                first_index: slice.first_index,
                vertex_offset: slice.first_vertex as i32,
                first_instance: group_start,
                instance_count,
            });
            draw_material_rows.push(first_entry.key.material_row);
            indirect.push(DrawIndexedIndirectArgs {
                index_count: slice.index_count,
                instance_count,
                first_index: slice.first_index,
                base_vertex: slice.first_vertex as i32,
                first_instance: group_start,
            });
            range_keys.push(MaterialRangeKey {
                class: first_entry.key.class,
                graph_hash: first_entry.key.graph_hash,
                bucket: first_entry.key.bucket,
            });
        }

        let (opaque, transparent, forward) = build_material_ranges(&range_keys);
        self.gpu_scene.material_class_ranges = opaque;
        self.gpu_scene.transparent_material_class_ranges = transparent;
        self.gpu_scene.forward_material_class_ranges = forward;

        let compacted_capacity = vec![0u32; object_count];
        self.gpu_scene.draw_calls.set_data(draw_calls);
        self.gpu_scene.draw_material_rows = draw_material_rows;
        self.gpu_scene.draw_topology_generation =
            self.gpu_scene.draw_topology_generation.wrapping_add(1);
        self.gpu_scene.indirect.set_data(indirect);
        self.gpu_scene.visibility.set_data(visibility);
        self.gpu_scene.source_indices.set_data(source_indices);
        self.gpu_scene
            .compacted_indices
            .set_data(compacted_capacity.clone());
        self.gpu_scene
            .compacted_indices_2
            .set_data(compacted_capacity);
        self.rebuild_shadow_partition_buffers();
    }

    /// Batch shadow draws by mesh and publish compact slot -> entity-row maps.
    pub(in crate::scene) fn rebuild_shadow_partition_buffers(&mut self) {
        #[derive(Clone, Copy)]
        struct ShadowEntry {
            entity: Entity,
            mesh: MeshId,
        }

        let mut static_entries = Vec::<ShadowEntry>::new();
        let mut movable_entries = Vec::<ShadowEntry>::new();
        for (entity, object) in self.authority.query::<SceneObject>() {
            if (object.spatial.flags & libhelio::INSTANCE_FLAG_CASTS_SHADOW) == 0 {
                continue;
            }
            let entry = ShadowEntry {
                entity,
                mesh: object_mesh(object),
            };
            if object_movability(object).can_move() {
                movable_entries.push(entry);
            } else {
                static_entries.push(entry);
            }
        }

        let build = |entries: &mut Vec<ShadowEntry>| {
            entries.sort_by_key(|entry| (entry.mesh.slot(), entry.mesh.generation()));
            let mut sources = Vec::<u32>::with_capacity(entries.len());
            let mut draws = Vec::<DrawIndexedIndirectArgs>::new();
            let mut cursor = 0;
            while cursor < entries.len() {
                let mesh_id = entries[cursor].mesh;
                let slice = self
                    .mesh_pool()
                    .get(mesh_id)
                    .expect("live SceneObject must retain its GeometryArena/MeshPool edge")
                    .slice;
                let group_start = sources.len() as u32;
                while cursor < entries.len() && entries[cursor].mesh == mesh_id {
                    sources.push(
                        self.authority
                            .gpu_row::<SceneObject>(entries[cursor].entity)
                            .expect("queried SceneObject must have a component-local GPU row"),
                    );
                    cursor += 1;
                }
                draws.push(DrawIndexedIndirectArgs {
                    index_count: slice.index_count,
                    instance_count: sources.len() as u32 - group_start,
                    first_index: slice.first_index,
                    base_vertex: slice.first_vertex as i32,
                    first_instance: group_start,
                });
            }
            (sources, draws)
        };

        let (static_sources, static_indirect) = build(&mut static_entries);
        let (movable_sources, movable_indirect) = build(&mut movable_entries);
        let static_draw_count = static_indirect.len() as u32;
        let movable_draw_count = movable_indirect.len() as u32;

        if self.static_objects_dirty {
            self.gpu_scene.static_objects_generation += 1;
            self.static_objects_dirty = false;
        }
        self.gpu_scene.shadow_static_draw_count = static_draw_count;
        self.gpu_scene.shadow_movable_draw_count = movable_draw_count;
        if shadow_movable_partition_changed(
            self.gpu_scene.shadow_movable_source_indices.as_slice(),
            &movable_sources,
            self.gpu_scene.shadow_movable_indirect.as_slice(),
            &movable_indirect,
        ) {
            self.gpu_scene.shadow_movable_topology_generation = self
                .gpu_scene
                .shadow_movable_topology_generation
                .wrapping_add(1);
        }
        self.gpu_scene
            .shadow_static_source_indices
            .set_data(static_sources);
        self.gpu_scene
            .shadow_movable_source_indices
            .set_data(movable_sources);
        self.gpu_scene
            .shadow_static_indirect
            .set_data(static_indirect);
        self.gpu_scene
            .shadow_movable_indirect
            .set_data(movable_indirect);
    }
}

#[cfg(test)]
mod tests {
    use helio_core::DrawIndexedIndirectArgs;

    use super::{
        build_material_ranges, shadow_movable_partition_changed, MaterialRangeKey, RenderBucket,
    };

    #[test]
    fn material_ranges_do_not_merge_distinct_render_buckets() {
        let keys = [
            MaterialRangeKey {
                class: 7,
                graph_hash: 11,
                bucket: RenderBucket::Opaque,
            },
            MaterialRangeKey {
                class: 7,
                graph_hash: 11,
                bucket: RenderBucket::Transparent,
            },
            MaterialRangeKey {
                class: 7,
                graph_hash: 11,
                bucket: RenderBucket::Forward,
            },
        ];
        let (opaque, transparent, forward) = build_material_ranges(&keys);
        assert_eq!(opaque, vec![(7, 11, 0, 1)]);
        assert_eq!(transparent, vec![(7, 11, 1, 1)]);
        assert_eq!(forward, vec![(7, 11, 2, 1)]);
    }

    #[test]
    fn shadow_topology_includes_indirect_command_changes() {
        let sources = [4, 9];
        let old = [DrawIndexedIndirectArgs {
            index_count: 12,
            instance_count: 2,
            first_index: 3,
            base_vertex: 7,
            first_instance: 0,
        }];
        let mut changed = old;
        changed[0].first_index = 5;

        assert!(shadow_movable_partition_changed(
            &sources, &sources, &old, &changed
        ));
        assert!(!shadow_movable_partition_changed(
            &sources, &sources, &old, &old
        ));
    }
}
