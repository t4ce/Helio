//! SceneDB-authoritative object updates and read queries.

use glam::Mat4;
use helio_scenedb::{SceneMaterial, SceneObject};

use crate::handles::{
    bits_from_handle, entity_from_handle, handle_from_entity, MaterialId, ObjectId,
};

use super::super::errors::{invalid, Result};
use super::super::helpers::{
    normal_matrix, object_groups, object_material, object_mesh, object_movability,
};
use super::super::types::PickableObject;

fn pick_transform(
    local_model: [f32; 16],
    coordinate_space: Option<[f32; 16]>,
) -> Mat4 {
    let local = Mat4::from_cols_array(&local_model);
    coordinate_space
        .map(|space| Mat4::from_cols_array(&space) * local)
        .unwrap_or(local)
}

impl super::super::Scene {
    /// Update an object's authored transform through SceneDB's GPU-aware path.
    pub fn update_object_transform(&mut self, id: ObjectId, transform: Mat4) -> Result<()> {
        let entity = entity_from_handle(id);
        let movability = self
            .authority
            .get::<SceneObject>(entity)
            .map(object_movability)
            .ok_or_else(|| invalid("object"))?;
        if !movability.can_move() {
            log::warn!(
                "Attempted to update transform on Static object {:?}. Set movability to Movable to allow transform updates.",
                id
            );
            return Ok(());
        }
        let gpu_row = self
            .authority
            .gpu_row::<SceneObject>(entity)
            .ok_or_else(|| invalid("object"))?;

        let (current_sphere, current_flags) = self.authority
            .edit_gpu::<SceneObject, _>(entity, |record| {
                record.spatial.model = transform.to_cols_array();
                record.spatial.normal_mat = normal_matrix(transform);

                // Preserve the authored radius while following the transform's
                // translation, matching the pre-SceneDB object contract.
                let translation = transform.w_axis;
                record.spatial.sphere[0] = translation.x;
                record.spatial.sphere[1] = translation.y;
                record.spatial.sphere[2] = translation.z;
                (record.spatial.sphere, record.spatial.flags)
            })
            .ok_or_else(|| invalid("object"))?;
        self.gpu_scene
            .object_history
            .stage_current(
                gpu_row,
                transform.to_cols_array(),
                current_sphere,
                current_flags,
            );

        self.movable_objects_generation += 1;
        self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        Ok(())
    }

    /// Change the stable material reference and its compact GPU table row.
    pub fn update_object_material(&mut self, id: ObjectId, material: MaterialId) -> Result<()> {
        let entity = entity_from_handle(id);
        let old_material_id = self
            .authority
            .get::<SceneObject>(entity)
            .map(object_material)
            .ok_or_else(|| invalid("object"))?;

        let new_material_entity = entity_from_handle(material);
        let new_material_row = self
            .authority
            .gpu_row::<SceneMaterial>(new_material_entity)
            .ok_or_else(|| invalid("material"))?;
        self.authority
            .retain_material(new_material_entity)
            .map_err(super::super::errors::scene_asset)?;

        self.authority
            .edit_gpu::<SceneObject, _>(entity, |record| {
                record.material_handle_bits = bits_from_handle(material);
                record.render.material_row = new_material_row;
            })
            .expect("object liveness was validated immediately before edit");

        let _ = self
            .authority
            .release_material(entity_from_handle(old_material_id));
        self.objects_dirty = true;
        Ok(())
    }

    /// Update the authored bounding sphere. Culling derives any AABB it needs.
    pub fn update_object_bounds(&mut self, id: ObjectId, bounds: [f32; 4]) -> Result<()> {
        let entity = entity_from_handle(id);
        let gpu_row = self
            .authority
            .gpu_row::<SceneObject>(entity)
            .ok_or_else(|| invalid("object"))?;
        let (model, flags, movability) = self.authority
            .edit_gpu::<SceneObject, _>(entity, |record| {
                record.spatial.sphere = bounds;
                (
                    record.spatial.model,
                    record.spatial.flags,
                    object_movability(record),
                )
            })
            .ok_or_else(|| invalid("object"))?;
        self.gpu_scene
            .object_history
            .stage_current(gpu_row, model, bounds, flags);
        if movability.can_move() {
            self.movable_objects_generation = self.movable_objects_generation.wrapping_add(1);
            self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        }
        Ok(())
    }

    /// Associate non-movable objects with baked lightmap atlas rows.
    #[cfg(feature = "bake")]
    pub fn update_lightmap_indices(&mut self, regions: &[helio_bake::CachedAtlasRegion]) {
        use std::collections::HashMap;

        let region_map: HashMap<u32, u32> = regions
            .iter()
            .enumerate()
            .filter_map(|(index, region)| {
                (region.mesh_id[1] == 0).then_some((region.mesh_id[0] as u32, index as u32))
            })
            .collect();

        let mut static_count = 0usize;
        self.authority
            .edit_gpu_each::<SceneObject>(|_, _, record| {
                if object_movability(record).can_move() {
                    return false;
                }
                static_count += 1;
                let lightmap_index = region_map
                    .get(&object_mesh(record).slot())
                    .copied()
                    .unwrap_or(u32::MAX);
                if record.render.lightmap_index == lightmap_index {
                    return false;
                }
                record.render.lightmap_index = lightmap_index;
                true
            });

        log::info!(
            "[Scene] Updated lightmap indices for {} static objects ({} regions in atlas)",
            static_count,
            regions.len()
        );
    }

    pub fn get_object_transform(&self, id: ObjectId) -> Result<Mat4> {
        let record = self.object_record(id).ok_or_else(|| invalid("object"))?;
        Ok(Mat4::from_cols_array(&record.spatial.model))
    }

    pub fn get_object_bounds(&self, id: ObjectId) -> Result<[f32; 4]> {
        let record = self.object_record(id).ok_or_else(|| invalid("object"))?;
        Ok(record.spatial.sphere)
    }

    pub fn iter_objects_for_editor(
        &self,
    ) -> impl Iterator<Item = (ObjectId, Mat4, [f32; 4], u64)> + '_ {
        self.authority
            .query::<SceneObject>()
            .map(|(entity, record)| {
                (
                    handle_from_entity(entity),
                    Mat4::from_cols_array(&record.spatial.model),
                    record.spatial.sphere,
                    record.user_tag,
                )
            })
    }

    pub fn get_object_descriptor(
        &self,
        id: ObjectId,
    ) -> Result<crate::scene::types::ObjectDescriptor> {
        use crate::scene::types::ObjectDescriptor;

        let record = self.object_record(id).ok_or_else(|| invalid("object"))?;
        Ok(ObjectDescriptor {
            mesh: object_mesh(record),
            material: object_material(record),
            transform: Mat4::from_cols_array(&record.spatial.model),
            bounds: record.spatial.sphere,
            flags: record.spatial.flags,
            groups: object_groups(record),
            movability: Some(object_movability(record)),
            user_tag: record.user_tag,
        })
    }

    pub fn iter_pickable_objects(&self) -> impl Iterator<Item = PickableObject> + '_ {
        self.authority
            .query::<SceneObject>()
            .map(|(entity, record)| PickableObject {
                id: handle_from_entity(entity),
                mesh_id: object_mesh(record),
                transform: {
                    let space = libhelio::coordinate_space(record.spatial.flags);
                    pick_transform(
                        record.spatial.model,
                        (space != 0)
                            .then(|| self.gpu_scene.coordinate_space_history.slot(space)),
                    )
                },
                user_tag: record.user_tag,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::pick_transform;
    use glam::{Mat4, Vec3};

    #[test]
    fn pick_transform_composes_primary_coordinate_space_once() {
        let local = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let space = Mat4::from_translation(Vec3::new(10.0, -4.0, 8.0));

        let world = pick_transform(local.to_cols_array(), Some(space.to_cols_array()));

        assert_eq!(world.w_axis.truncate(), Vec3::new(11.0, -2.0, 11.0));
        assert_eq!(pick_transform(local.to_cols_array(), None), local);
    }
}
