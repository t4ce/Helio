//! Object insertion operations.
//!
//! Provides the [`Scene::insert_object`](crate::Scene::insert_object) method for adding
//! renderable objects to the scene. Objects are automatically batched into instanced
//! draw calls on the next flush.

use crate::handles::{entity_from_handle, handle_from_entity, ObjectId};
use helio_scenedb::{SceneIndices, SceneMaterial};

use super::super::errors::{invalid, Result};
use super::super::helpers::{object_gpu_data, object_movability};
use super::super::types::ObjectDescriptor;
use super::NO_OBJECT_PROJECTION_SLOT;

impl super::super::Scene {
    /// Insert a renderable object into the scene.
    ///
    /// Creates a new object that references a mesh and material, with a world-space
    /// transform and optional group membership. Objects sharing the same mesh and
    /// material are automatically batched into instanced draw calls on the next flush.
    ///
    /// # Parameters
    /// - `desc`: Object descriptor containing:
    ///   - `mesh`: Mesh handle from [`insert_mesh`](crate::Scene::insert_mesh)
    ///   - `material`: Material handle from [`insert_material`](crate::Scene::insert_material)
    ///   - `transform`: World-space model matrix (column-major)
    ///   - `bounds`: Bounding sphere `[center.x, center.y, center.z, radius]`
    ///   - `flags`: Render flags (bit 0 = casts shadow, bit 1 = receives shadow)
    ///   - `groups`: Group membership mask for batch visibility control
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the mesh or material ID is invalid
    ///
    /// # Returns
    /// An [`ObjectId`] handle that can be used to update or remove the object.
    ///
    /// # Performance
    /// - CPU cost: amortized O(1) SceneDB entity/component insertion
    /// - GPU cost: Full optimized rebuild deferred to next `flush()` — includes
    ///   automatic instancing (objects with the same mesh+material share a draw call)
    ///
    /// # Example
    /// ```ignore
    /// use helio::{ObjectDescriptor, GroupMask};
    /// use glam::Mat4;
    ///
    /// let obj_id = scene.insert_object(ObjectDescriptor {
    ///     mesh: mesh_id,
    ///     material: material_id,
    ///     transform: Mat4::from_translation([0.0, 1.5, 0.0].into()),
    ///     bounds: [0.0, 1.5, 0.0, 1.6],  // Sphere at (0, 1.5, 0) with radius 1.6
    ///     flags: 0b11,                    // Casts and receives shadows
    ///     groups: GroupMask::NONE,        // Always visible
    /// })?;
    /// ```
    ///
    /// # Reference Counting
    ///
    /// Increments the reference count for the mesh and material. They cannot be removed
    /// while this object exists. Call [`remove_object`](crate::Scene::remove_object) to
    /// decrement reference counts.
    pub fn insert_object(&mut self, desc: ObjectDescriptor) -> Result<ObjectId> {
        self.mesh_pool_mut()
            .get(desc.mesh)
            .ok_or_else(|| invalid("mesh"))?;
        let material_entity = entity_from_handle(desc.material);
        let material_row = self
            .authority
            .gpu_row::<SceneMaterial>(material_entity)
            .ok_or_else(|| invalid("material"))?;
        self.authority
            .retain_material(material_entity)
            .map_err(super::super::errors::scene_asset)?;
        self.mesh_pool_mut()
            .get_mut(desc.mesh)
            .ok_or_else(|| invalid("mesh"))?
            .ref_count += 1;

        let record = object_gpu_data(desc.mesh, material_row, desc);
        let is_static = !object_movability(&record).can_move();
        let user_tag = record.user_tag;
        let initial_model = record.spatial.model;
        let initial_sphere = record.spatial.sphere;
        let initial_flags = record.spatial.flags;
        let entity = self.authority.insert(record);
        let gpu_row = self
            .authority
            .gpu_row::<helio_scenedb::SceneObject>(entity)
            .expect("newly inserted SceneObject must have a component-local GPU row");
        self.gpu_scene
            .object_history
            .insert(gpu_row, initial_model, initial_sphere, initial_flags);
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_object(user_tag, entity);
        let id = handle_from_entity(entity);

        let entity_index = entity.index() as usize;
        if self.object_projection_slots.len() <= entity_index {
            self.object_projection_slots
                .resize(entity_index + 1, NO_OBJECT_PROJECTION_SLOT);
        }
        self.object_projection_slots[entity_index] = NO_OBJECT_PROJECTION_SLOT;

        // Track static topology changes for shadow atlas caching
        if is_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }

        // Mark for full optimized rebuild on next flush — this automatically
        // batches objects with the same mesh+material into instanced draw calls.
        self.objects_dirty = true;

        Ok(id)
    }
}
