//! Virtual geometry buffer rebuild and frame data packaging.
//!
//! Meshlet descriptors are stored once per referenced virtual mesh. A separate
//! object array supplies the instance index, conservative local bounds, measured
//! LOD errors, and the descriptor range for each LOD.

use std::collections::HashMap;

use libhelio::{
    GpuMeshletEntry, GpuVgObject, GpuVgWorkItem, VgFrameData,
    VG_CULL_MESHLETS_PER_WORK_ITEM, VG_LOD_LEVELS,
};

use crate::vg::VirtualMeshId;

use super::super::helpers::object_is_visible;
use super::super::types::VirtualMeshRecord;

fn append_unique_meshlets(
    referenced_meshes: impl IntoIterator<Item = VirtualMeshId>,
    mesh_records: &HashMap<VirtualMeshId, VirtualMeshRecord>,
    output: &mut Vec<GpuMeshletEntry>,
) -> HashMap<VirtualMeshId, u32> {
    let mut bases = HashMap::new();

    for mesh_id in referenced_meshes {
        if bases.contains_key(&mesh_id) {
            continue;
        }
        let Some(record) = mesh_records.get(&mesh_id) else {
            continue;
        };
        if record.meshlets.is_empty() || record.lod_count == 0 {
            continue;
        }

        let base = u32::try_from(output.len())
            .expect("virtual geometry exceeds the u32 descriptor address space");
        output.extend_from_slice(&record.meshlets);
        bases.insert(mesh_id, base);
    }

    bases
}

fn append_work_items(
    object_index: u32,
    max_meshlet_count: u32,
    output: &mut Vec<GpuVgWorkItem>,
) {
    for local_meshlet_base in
        (0..max_meshlet_count).step_by(VG_CULL_MESHLETS_PER_WORK_ITEM as usize)
    {
        output.push(GpuVgWorkItem {
            object_index,
            local_meshlet_base,
        });
    }
}

impl super::super::Scene {
    /// Returns the immutable mesh descriptors, object-level LOD metadata, and
    /// instance data consumed by the virtual-geometry pass.
    pub fn vg_frame_data(&self) -> Option<VgFrameData<'_>> {
        if self.vg_cpu_objects.is_empty() {
            return None;
        }
        debug_assert_eq!(self.vg_cpu_objects.len(), self.vg_cpu_instances.len());
        debug_assert_eq!(self.vg_cpu_instances.len(), self.vg_cpu_visibility.len());
        Some(VgFrameData {
            meshlets: bytemuck::cast_slice(&self.vg_cpu_meshlets),
            objects: bytemuck::cast_slice(&self.vg_cpu_objects),
            instances: bytemuck::cast_slice(&self.vg_cpu_instances),
            object_visibility: &self.vg_cpu_visibility,
            work_items: bytemuck::cast_slice(&self.vg_cpu_work_items),
            meshlet_count: u32::try_from(self.vg_cpu_meshlets.len())
                .expect("virtual geometry exceeds the u32 descriptor address space"),
            object_count: u32::try_from(self.vg_cpu_objects.len())
                .expect("virtual geometry exceeds the u32 object address space"),
            work_item_count: u32::try_from(self.vg_cpu_work_items.len())
                .expect("virtual geometry exceeds the u32 work-item address space"),
            max_draw_count: self.vg_max_draw_count,
            buffer_version: self.vg_buffer_version,
            instance_version: self.vg_instance_version,
            cull_signature_version: self.vg_cull_signature_version,
            instance_dirty_start: self
                .vg_published_instance_dirty_range
                .map_or(0, |(start, _)| u32::try_from(start).expect("VG dirty start exceeds u32")),
            instance_dirty_count: self
                .vg_published_instance_dirty_range
                .map_or(0, |(start, end)| {
                    u32::try_from(end - start).expect("VG dirty count exceeds u32")
                }),
        })
    }

    /// Rebuild the CPU mirrors used by the GPU-driven virtual-geometry pass.
    ///
    /// Each referenced virtual mesh contributes its descriptors exactly once,
    /// regardless of instance count. Each object then points at the shared
    /// per-LOD ranges. `vg_max_draw_count` is the exact worst case after one LOD
    /// is selected for every object, and therefore bounds every atomic append.
    pub(in crate::scene) fn rebuild_vg_buffers(&mut self) {
        let mut cpu_meshlets = std::mem::take(&mut self.vg_cpu_meshlets);
        let mut cpu_objects = std::mem::take(&mut self.vg_cpu_objects);
        let mut cpu_instances = std::mem::take(&mut self.vg_cpu_instances);
        let mut cpu_visibility = std::mem::take(&mut self.vg_cpu_visibility);
        let mut object_projection_slots =
            std::mem::take(&mut self.vg_object_projection_slots);
        let mut cpu_work_items = std::mem::take(&mut self.vg_cpu_work_items);
        cpu_meshlets.clear();
        cpu_objects.clear();
        cpu_instances.clear();
        cpu_visibility.clear();
        object_projection_slots.clear();
        cpu_work_items.clear();
        self.vg_instance_dirty_range = None;
        self.vg_published_instance_dirty_range = None;
        let mut max_draw_count = 0u32;

        let group_hidden = self.group_hidden();
        let storage = self.virtual_geometry();
        let dense_object_count = storage.objects.dense_len();
        cpu_objects.reserve(dense_object_count);
        cpu_instances.reserve(dense_object_count);
        cpu_visibility.reserve(dense_object_count);
        object_projection_slots.resize(dense_object_count, u32::MAX);

        let mesh_bases = append_unique_meshlets(
            (0..dense_object_count)
                .filter_map(|index| storage.objects.get_dense(index))
                .map(|object| object.virtual_mesh),
            &storage.meshes,
            &mut cpu_meshlets,
        );

        for dense_index in 0..dense_object_count {
            let Some(object) = storage.objects.get_dense(dense_index) else {
                continue;
            };
            let Some(mesh) = storage.meshes.get(&object.virtual_mesh) else {
                continue;
            };
            let Some(&mesh_base) = mesh_bases.get(&object.virtual_mesh) else {
                continue;
            };

            let instance_index = u32::try_from(cpu_instances.len())
                .expect("virtual geometry exceeds the u32 instance address space");
            let object_index = u32::try_from(cpu_objects.len())
                .expect("virtual geometry exceeds the u32 object address space");
            object_projection_slots[dense_index] = object_index;
            let mut lod_first_meshlets = [0; VG_LOD_LEVELS];
            for (level, first) in lod_first_meshlets.iter_mut().enumerate() {
                *first = mesh_base
                    .checked_add(mesh.lod_first_meshlets[level])
                    .expect("virtual geometry descriptor offset overflow");
            }

            cpu_instances.push(object.instance);
            cpu_visibility.push(u32::from(object_is_visible(object.groups, group_hidden)));
            cpu_objects.push(GpuVgObject {
                instance_index,
                lod_count: mesh.lod_count,
                max_meshlet_count: mesh.max_meshlet_count,
                reserved: 0,
                local_bounds: mesh.local_bounds,
                lod_errors: mesh.lod_errors,
                lod_first_meshlets,
                lod_meshlet_counts: mesh.lod_meshlet_counts,
            });
            append_work_items(
                object_index,
                mesh.max_meshlet_count,
                &mut cpu_work_items,
            );
            max_draw_count = max_draw_count
                .checked_add(mesh.max_meshlet_count)
                .expect("virtual geometry indirect draw capacity exceeds u32");
        }

        self.vg_cpu_meshlets = cpu_meshlets;
        self.vg_cpu_objects = cpu_objects;
        self.vg_cpu_instances = cpu_instances;
        self.vg_cpu_visibility = cpu_visibility;
        self.vg_object_projection_slots = object_projection_slots;
        self.vg_cpu_work_items = cpu_work_items;
        self.vg_max_draw_count = max_draw_count;

        self.vg_buffer_version = self.vg_buffer_version.wrapping_add(1);
        eprintln!(
            "[vg] rebuild: {} objects, {} unique meshlets, {} work spans, {} max draws",
            self.vg_cpu_objects.len(),
            self.vg_cpu_meshlets.len(),
            self.vg_cpu_work_items.len(),
            self.vg_max_draw_count,
        );
    }

    /// Refresh the compact VG visibility projection from canonical SceneDB
    /// group membership and the SceneVisibilityState singleton. Returns true
    /// only when at least one published object's effective visibility changed.
    ///
    /// A pending topology rebuild owns this work. Otherwise this is allocation
    /// free and advances only the pass-owned cull projection signature: meshlet,
    /// object, work-item, and instance buffers remain untouched.
    pub(in crate::scene) fn refresh_vg_group_visibility(
        &mut self,
        group_hidden: crate::groups::GroupMask,
    ) -> bool {
        if self.vg_objects_dirty {
            return false;
        }

        let instance_count = self.vg_cpu_instances.len();
        let mut visibility = std::mem::take(&mut self.vg_cpu_visibility);
        let projection_slots = std::mem::take(&mut self.vg_object_projection_slots);
        let mut changed = false;
        let mut projection_valid = visibility.len() == instance_count;
        {
            let storage = self.virtual_geometry();
            if !projection_valid || projection_slots.len() != storage.objects.dense_len() {
                projection_valid = false;
            } else {
                for dense_index in 0..storage.objects.dense_len() {
                    let projection_slot = projection_slots[dense_index];
                    if projection_slot == u32::MAX {
                        continue;
                    }
                    let Some(projected) = visibility.get_mut(projection_slot as usize) else {
                        projection_valid = false;
                        break;
                    };
                    let object = storage
                        .objects
                        .get_dense(dense_index)
                        .expect("dense VG projection references a missing canonical object");
                    let next = u32::from(object_is_visible(object.groups, group_hidden));
                    changed |= *projected != next;
                    *projected = next;
                }
            }
        }
        self.vg_cpu_visibility = visibility;
        self.vg_object_projection_slots = projection_slots;

        if !projection_valid {
            // This can only happen if a topology mutation failed to mark the
            // projection dirty. Recover conservatively rather than applying a
            // visibility value to the wrong object.
            self.vg_objects_dirty = true;
            return false;
        }
        if changed {
            self.vg_cull_signature_version = self.vg_cull_signature_version.wrapping_add(1);
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use libhelio::{GpuMeshletEntry, VG_LOD_LEVELS};

    use super::{append_unique_meshlets, append_work_items, VirtualMeshId, VirtualMeshRecord};

    fn meshlet(first_index: u32) -> GpuMeshletEntry {
        GpuMeshletEntry {
            center: [0.0; 3],
            radius: 1.0,
            cone_apex: [0.0; 3],
            cone_cutoff: 2.0,
            cone_axis: [0.0, 1.0, 0.0],
            lod_error: 0.0,
            first_index,
            index_count: 3,
            vertex_offset: 0,
            instance_index: 0,
        }
    }

    fn record(meshlets: Vec<GpuMeshletEntry>) -> VirtualMeshRecord {
        VirtualMeshRecord {
            mesh_ids: Vec::new(),
            meshlets,
            local_bounds: [0.0, 0.0, 0.0, 1.0],
            lod_count: 1,
            lod_errors: [0.0; VG_LOD_LEVELS],
            lod_first_meshlets: [0; VG_LOD_LEVELS],
            lod_meshlet_counts: [1; VG_LOD_LEVELS],
            max_meshlet_count: 1,
            ref_count: 0,
        }
    }

    #[test]
    fn repeated_instances_share_one_descriptor_copy() {
        let first = VirtualMeshId(3);
        let second = VirtualMeshId(7);
        let records = HashMap::from([
            (first, record(vec![meshlet(11), meshlet(12)])),
            (second, record(vec![meshlet(20)])),
        ]);
        let mut output = Vec::new();

        let bases = append_unique_meshlets(
            [first, first, second, first, second],
            &records,
            &mut output,
        );

        assert_eq!(output.len(), 3);
        assert_eq!(bases[&first], 0);
        assert_eq!(bases[&second], 2);
        assert_eq!(output.iter().map(|entry| entry.first_index).collect::<Vec<_>>(), [11, 12, 20]);
    }

    #[test]
    fn work_items_cover_exact_fixed_meshlet_spans() {
        let mut output = Vec::new();

        append_work_items(3, 0, &mut output);
        append_work_items(5, 1, &mut output);
        append_work_items(7, 64, &mut output);
        append_work_items(11, 65, &mut output);
        append_work_items(13, 130, &mut output);

        assert_eq!(
            output
                .iter()
                .map(|item| (item.object_index, item.local_meshlet_base))
                .collect::<Vec<_>>(),
            [
                (5, 0),
                (7, 0),
                (11, 0),
                (11, 64),
                (13, 0),
                (13, 64),
                (13, 128),
            ]
        );
    }
}
