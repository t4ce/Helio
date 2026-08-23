//! Object removal operations.
//!
//! Provides the [`Scene::remove_object`](crate::Scene::remove_object) method for removing
//! renderable objects from the scene.

use helio_scenedb::{SceneIndices, SceneObject};

use crate::handles::{entity_from_handle, ObjectId};

use super::super::errors::{invalid, Result, SceneError};
use super::super::helpers::{object_material, object_mesh, object_movability};
use super::super::multi_mesh::SectionRelations;
use super::NO_OBJECT_PROJECTION_SLOT;

impl super::super::Scene {
    /// Remove an object from the scene.
    ///
    /// Despawns the canonical SceneDB entity and decrements mesh and material reference
    /// counts. GPU buffers are rebuilt on the next flush with automatic instancing.
    ///
    /// # Parameters
    /// - `id`: Object handle returned by [`insert_object`](crate::Scene::insert_object)
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    /// - [`SceneError::InvalidOperation`](super::super::SceneError::InvalidOperation) if the
    ///   object is one section of a live sectioned instance; remove that aggregate instead.
    ///
    /// # Returns
    /// `Ok(())` if the object was successfully removed.
    ///
    /// # Performance
    /// - CPU cost: amortized O(1) SceneDB despawn
    /// - GPU cost: Full optimized rebuild deferred to next `flush()`
    ///
    /// # Reference Counting
    ///
    /// Decrements reference counts for the mesh and material. If the reference count
    /// reaches zero, the mesh/material can be removed with [`remove_mesh`](crate::Scene::remove_mesh)
    /// or [`remove_material`](crate::Scene::remove_material).
    ///
    /// # Example
    /// ```ignore
    /// // Remove object
    /// scene.remove_object(obj_id)?;
    ///
    /// // Now mesh and material may be removable (if no other objects use them)
    /// if mesh_ref_count == 0 {
    ///     scene.remove_mesh(mesh_id)?;
    /// }
    /// if material_ref_count == 0 {
    ///     scene.remove_material(material_id)?;
    /// }
    /// ```
    pub fn remove_object(&mut self, id: ObjectId) -> Result<()> {
        // Capture handles and movability before removal.
        let entity = entity_from_handle(id);
        if self
            .authority
            .subsystem::<SectionRelations>()
            .and_then(|relations| relations.instance_for_object(entity))
            .is_some()
        {
            return Err(SceneError::InvalidOperation {
                reason: "object belongs to a sectioned instance; remove the sectioned instance",
            });
        }
        let (mesh_id, material_id, is_static, user_tag, gpu_row) = {
            let r = self
                .authority
                .get::<SceneObject>(entity)
                .ok_or_else(|| invalid("object"))?;
            (
                object_mesh(r),
                object_material(r),
                !object_movability(r).can_move(),
                r.user_tag,
                self.authority
                    .gpu_row::<SceneObject>(entity)
                    .ok_or_else(|| invalid("object"))?,
            )
        };

        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_object(user_tag, entity);
        if !self.authority.despawn(entity) {
            return Err(invalid("object"));
        }
        self.gpu_scene
            .object_history
            .remove(gpu_row);
        if let Some(slot) = self.object_projection_slots.get_mut(entity.index() as usize) {
            *slot = NO_OBJECT_PROJECTION_SLOT;
        }

        // Decrement ref counts
        let material_entity = entity_from_handle(material_id);
        let _ = self.authority.release_material(material_entity);
        if let Some(mesh) = self.mesh_pool_mut().get_mut(mesh_id) {
            mesh.ref_count = mesh.ref_count.saturating_sub(1);
        }

        // Mark for full optimized rebuild on next flush.
        self.objects_dirty = true;

        // Meshes and materials are independently authored assets. A zero
        // object-reference count makes them eligible for explicit removal,
        // but must not invalidate their handles: callers may temporarily
        // remove all instances and then place the same shared asset again.
        // `Scene::clear()` performs the deliberate zero-ref asset sweep.

        // After removal: mark static atlas dirty if a static object was removed
        if is_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }

        self.detach_actors_for_target(crate::scene::actor::SceneActorId::Object(id));

        Ok(())
    }
}
