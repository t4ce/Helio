//! Group transform operations for batch movement and transformation.
//!
//! Provides methods for applying transforms to all objects in a group simultaneously.

use glam::{Mat4, Vec3};
use helio_scenedb::SceneObject;

use crate::groups::GroupId;

use super::super::helpers::{normal_matrix, object_groups, object_movability};

impl super::super::Scene {
    /// Apply a transform delta to every object in a group.
    ///
    /// `delta` is pre-multiplied: `new_model = delta * old_model`. It is expressed
    /// in each object's authored coordinate-space frame: world space for ordinary
    /// objects (space 0), or sublevel-local space for sublevel members.
    ///
    /// # Parameters
    /// - `group`: Group to transform
    /// - `delta`: Transform delta (translation, rotation, scale, or combination)
    ///
    /// # Performance
    /// - CPU cost: O(N + V) over classic and virtual SceneDB records
    /// - GPU cost: O(M) bounded updates (where M = matching projected objects)
    ///   when each layout is stable
    /// - Memory: No allocations
    ///
    /// # Transform Application
    ///
    /// The delta is applied as: `new_transform = delta * old_transform`
    ///
    /// This means:
    /// - Translation moves objects along axes of their authored coordinate space
    /// - Rotation rotates objects around that coordinate space's origin
    /// - Scale scales objects from that coordinate space's origin
    ///
    /// To move an entire sublevel in world space, prefer
    /// [`move_sublevel`](crate::Scene::move_sublevel), which updates its shared
    /// coordinate-space placement in O(1). A world delta for one member can be
    /// conjugated into its local frame as `space.inverse() * delta * space`.
    ///
    /// # Bounds Updates
    ///
    /// Bounding sphere centers are transformed by the delta matrix. The radius
    /// is kept unchanged (this is an approximation - for accurate bounds after
    /// non-uniform scaling, manually update bounds with [`update_object_bounds`](crate::Scene::update_object_bounds)).
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    /// use glam::{Mat4, Vec3};
    ///
    /// let group_enemies = GroupId(0);
    ///
    /// // Move all enemies up by 5 units
    /// scene.move_group(group_enemies, Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0)));
    ///
    /// // Rotate all enemies 90 degrees around Y axis
    /// scene.move_group(group_enemies, Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2));
    ///
    /// // Scale all enemies to 2x size
    /// scene.move_group(group_enemies, Mat4::from_scale(Vec3::splat(2.0)));
    /// ```
    pub fn move_group(&mut self, group: GroupId, delta: Mat4) {
        let mut moved_movable = false;
        let mut moved_static = false;
        {
            let authority = &mut self.authority;
            let object_history = &mut self.gpu_scene.object_history;
            authority.edit_gpu_each::<SceneObject>(|_, gpu_row, object| {
                if !object_groups(object).contains(group) {
                    return false;
                }

                let can_move = object_movability(object).can_move();
                let new_transform = delta * Mat4::from_cols_array(&object.spatial.model);
                object.spatial.model = new_transform.to_cols_array();
                object.spatial.normal_mat = normal_matrix(new_transform);
                let old_center = Vec3::new(
                    object.spatial.sphere[0],
                    object.spatial.sphere[1],
                    object.spatial.sphere[2],
                );
                let new_center = delta.transform_point3(old_center);
                object.spatial.sphere[0] = new_center.x;
                object.spatial.sphere[1] = new_center.y;
                object.spatial.sphere[2] = new_center.z;
                object_history.stage_current(
                    gpu_row,
                    object.spatial.model,
                    object.spatial.sphere,
                    object.spatial.flags,
                );
                moved_movable |= can_move;
                moved_static |= !can_move;
                true
            });
        }

        // VirtualGeometryStorage is another canonical SceneDB subsystem. Keep
        // its authored transforms in the same group operation, while patching
        // only Helio's compact instance projection when topology is stable.
        // Taking the reusable vectors allows disjoint mutation without a
        // temporary per-object allocation.
        let topology_dirty = self.vg_objects_dirty;
        let mut cpu_instances = std::mem::take(&mut self.vg_cpu_instances);
        let projection_slots = std::mem::take(&mut self.vg_object_projection_slots);
        let mut vg_dirty_range: Option<(usize, usize)> = None;
        let mut projection_invalid = false;
        {
            let storage = self.virtual_geometry_mut();
            if !topology_dirty && projection_slots.len() != storage.objects.dense_len() {
                projection_invalid = true;
            }

            for dense_index in 0..storage.objects.dense_len() {
                let record = storage
                    .objects
                    .get_dense_mut(dense_index)
                    .expect("dense VG iteration returned a missing object");
                if !record.groups.contains(group) {
                    continue;
                }

                let old_center = Vec3::new(
                    record.instance.bounds[0],
                    record.instance.bounds[1],
                    record.instance.bounds[2],
                );
                let new_center = delta.transform_point3(old_center);
                let new_transform = delta * Mat4::from_cols_array(&record.instance.model);
                record.instance.model = new_transform.to_cols_array();
                record.instance.normal_mat = normal_matrix(new_transform);
                record.instance.bounds[0] = new_center.x;
                record.instance.bounds[1] = new_center.y;
                record.instance.bounds[2] = new_center.z;

                moved_movable |= record.movability.can_move();
                moved_static |= !record.movability.can_move();

                if topology_dirty || projection_invalid {
                    continue;
                }
                let projection_slot = projection_slots[dense_index];
                if projection_slot == u32::MAX {
                    continue;
                }
                let projection_slot = projection_slot as usize;
                let Some(projected) = cpu_instances.get_mut(projection_slot) else {
                    projection_invalid = true;
                    continue;
                };
                *projected = record.instance;
                vg_dirty_range = Some(match vg_dirty_range {
                    Some((start, end)) => {
                        (start.min(projection_slot), end.max(projection_slot + 1))
                    }
                    None => (projection_slot, projection_slot + 1),
                });
            }
        }
        self.vg_cpu_instances = cpu_instances;
        self.vg_object_projection_slots = projection_slots;
        if projection_invalid {
            self.vg_objects_dirty = true;
        } else if let Some((start, end)) = vg_dirty_range {
            self.vg_instance_dirty_range = Some(match self.vg_instance_dirty_range {
                Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
                None => (start, end),
            });
        }

        if moved_movable {
            self.movable_objects_generation = self.movable_objects_generation.wrapping_add(1);
            self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        }
        if moved_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }
    }

    /// Translate all objects in a group within their authored coordinate-space frames.
    ///
    /// Convenience wrapper around [`move_group`](Self::move_group) using a pure
    /// translation matrix.
    ///
    /// # Parameters
    /// - `group`: Group to translate
    /// - `delta`: Translation vector in each object's authored coordinate-space frame
    ///
    /// # Performance
    /// - CPU cost: O(N + V) over classic and virtual SceneDB records
    /// - GPU cost: O(M) bounded updates (where M = matching projected objects)
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    /// use glam::Vec3;
    ///
    /// let group_ui = GroupId(1);
    ///
    /// // Move all UI elements 10 units to the right
    /// scene.translate_group(group_ui, Vec3::new(10.0, 0.0, 0.0));
    ///
    /// // Move all UI elements back to origin
    /// let current_pos = Vec3::new(10.0, 0.0, 0.0);
    /// scene.translate_group(group_ui, -current_pos);
    /// ```
    pub fn translate_group(&mut self, group: GroupId, delta: Vec3) {
        self.move_group(group, Mat4::from_translation(delta));
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use glam::{Mat4, Vec3};
    use libhelio::Movability;

    use crate::groups::{GroupId, GroupMask};
    use crate::handles::entity_from_handle;
    use crate::mesh::{MeshUpload, PackedVertex};
    use crate::scene::{ObjectDescriptor, Scene};
    use crate::vg::{VirtualMeshUpload, VirtualObjectDescriptor};

    fn test_scene() -> Scene {
        let (device, queue) =
            crate::test_support::test_gpu().expect("no test GPU adapter found");
        Scene::new(device, queue)
    }

    #[test]
    fn group_transform_updates_virtual_authority_and_bounded_instance_projection() {
        let mut scene = test_scene();
        let vertex = |position| {
            PackedVertex::from_components(
                position,
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            )
        };
        let mesh = scene.insert_virtual_mesh(VirtualMeshUpload {
            vertices: vec![
                vertex([-1.0, -1.0, 0.0]),
                vertex([1.0, -1.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        });
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        let groups = GroupMask::from(GroupId::DYNAMIC);
        let classic_mesh = scene.insert_mesh(MeshUpload {
            vertices: vec![
                vertex([-1.0, -1.0, 0.0]),
                vertex([1.0, -1.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        });
        let classic = scene
            .insert_object(ObjectDescriptor {
                mesh: classic_mesh,
                material,
                transform: Mat4::IDENTITY,
                bounds: [2.0, 4.0, 6.0, 3.0],
                flags: 0,
                groups,
                movability: Some(Movability::Movable),
                user_tag: 0,
            })
            .expect("valid classic grouped object");
        let movable = scene
            .insert_virtual_object(VirtualObjectDescriptor {
                virtual_mesh: mesh,
                material_id: material,
                transform: Mat4::IDENTITY,
                bounds: [1.0, 2.0, 3.0, 4.0],
                flags: 0,
                groups,
                movability: Some(Movability::Movable),
            })
            .expect("valid movable VG object");
        let fixed = scene
            .insert_virtual_object(VirtualObjectDescriptor {
                virtual_mesh: mesh,
                material_id: material,
                transform: Mat4::IDENTITY,
                bounds: [-1.0, 0.0, 2.0, 1.0],
                flags: 0,
                groups,
                movability: Some(Movability::Static),
            })
            .expect("valid static VG object");
        scene.flush();

        scene.bake_invalidated = false;
        scene.static_objects_dirty = false;
        let topology_version = scene.vg_buffer_version;
        let instance_version = scene.vg_instance_version;
        let movable_generation = scene.movable_objects_generation;
        let delta = Vec3::new(5.0, -2.0, 1.0);

        scene.translate_group(GroupId::DYNAMIC, delta);

        let expected_model = Mat4::from_translation(delta).to_cols_array();
        let classic_record = scene
            .authority
            .get::<helio_scenedb::SceneObject>(entity_from_handle(classic))
            .expect("classic grouped object remains canonical");
        assert_eq!(classic_record.spatial.model, expected_model);
        assert_eq!(classic_record.spatial.sphere, [7.0, 2.0, 7.0, 3.0]);
        let movable_record = scene
            .virtual_geometry()
            .objects
            .get(movable)
            .expect("movable record remains canonical");
        let fixed_record = scene
            .virtual_geometry()
            .objects
            .get(fixed)
            .expect("static record remains canonical");
        assert_eq!(movable_record.instance.model, expected_model);
        assert_eq!(fixed_record.instance.model, expected_model);
        assert_eq!(movable_record.instance.bounds, [6.0, 0.0, 4.0, 4.0]);
        assert_eq!(fixed_record.instance.bounds, [4.0, -2.0, 3.0, 1.0]);
        assert_eq!(movable_record.instance.prev_model, Mat4::IDENTITY.to_cols_array());
        assert_eq!(fixed_record.instance.prev_model, Mat4::IDENTITY.to_cols_array());
        assert_eq!(scene.vg_instance_dirty_range, Some((0, 2)));
        assert_eq!(scene.movable_objects_generation, movable_generation + 1);
        assert!(scene.static_objects_dirty);
        assert!(scene.bake_invalidated);

        scene.flush();

        assert_eq!(scene.vg_buffer_version, topology_version);
        assert_eq!(scene.vg_instance_version, instance_version + 1);
        assert_eq!(scene.vg_published_instance_dirty_range, Some((0, 2)));
        assert_eq!(scene.vg_cpu_instances[0].model, expected_model);
        assert_eq!(scene.vg_cpu_instances[1].model, expected_model);
    }
}
