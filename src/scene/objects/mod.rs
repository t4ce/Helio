//! Object management for the scene (insert, update, remove, rebuild).
//!
//! Objects are the primary renderable entities in Helio. Each object references
//! a mesh and material, has a world-space transform, and can be assigned to
//! visibility groups.
//!
//! # Automatic Instancing
//!
//! Objects sharing the same mesh and material are automatically batched into
//! instanced draw calls on every flush. No explicit optimization step is needed —
//! the renderer always sorts and groups objects by `(mesh_id, material_id)` when
//! rebuilding GPU buffers after topology changes.
//!
//! # Module Organization
//!
//! - [`insert`]: Object insertion
//! - [`update`]: Transform and material updates
//! - [`remove`]: Object removal
//! - [`rebuild`]: GPU buffer rebuild with automatic instancing

mod insert;
mod rebuild;
mod remove;
mod update;

use helio_scenedb::SceneObject;

use crate::handles::{entity_from_handle, ObjectId};

pub(in crate::scene) const NO_OBJECT_PROJECTION_SLOT: u32 = u32::MAX;

impl super::Scene {
    #[inline]
    pub(in crate::scene) fn object_record(&self, id: ObjectId) -> Option<&SceneObject> {
        self.authority.get(entity_from_handle(id))
    }

    #[inline]
    pub(in crate::scene) fn object_projection_slot(&self, id: ObjectId) -> Option<usize> {
        let entity = entity_from_handle(id);
        self.authority.get::<SceneObject>(entity)?;
        self.object_projection_slots
            .get(entity.index() as usize)
            .copied()
            .filter(|slot| *slot != NO_OBJECT_PROJECTION_SLOT)
            .map(|slot| slot as usize)
    }
}
