//! High-level access to SceneDB's ordered authored SDF authority.

use helio_scenedb::{
    SdfAuthority, SdfAuthorityError, SdfEdit, SdfEditId, SdfPickResult, TerrainConfig,
};

use super::{Result, Scene, SceneError};

impl Scene {
    /// Append one primitive to the canonical authored evaluation order.
    pub fn add_sdf_edit(&mut self, edit: SdfEdit) -> Result<SdfEditId> {
        self.sdf_authority_mut().add(edit).map_err(map_sdf_error)
    }

    /// Insert one primitive at an explicit canonical evaluation position.
    pub fn insert_sdf_edit(
        &mut self,
        order_index: usize,
        edit: SdfEdit,
    ) -> Result<SdfEditId> {
        self.sdf_authority_mut()
            .insert(order_index, edit)
            .map_err(map_sdf_error)
    }

    /// Replace an edit without changing its stable identity or order.
    pub fn update_sdf_edit(&mut self, id: SdfEditId, edit: SdfEdit) -> Result<()> {
        self.sdf_authority_mut()
            .set(id, edit)
            .map_err(map_sdf_error)
    }

    /// Remove an edit and invalidate its exact generation-bearing identity.
    pub fn remove_sdf_edit(&mut self, id: SdfEditId) -> Result<SdfEdit> {
        self.sdf_authority_mut().remove(id).map_err(map_sdf_error)
    }

    /// Move an existing edit to a new canonical evaluation position.
    pub fn move_sdf_edit(&mut self, id: SdfEditId, new_index: usize) -> Result<()> {
        self.sdf_authority_mut()
            .move_edit(id, new_index)
            .map_err(map_sdf_error)
    }

    pub fn clear_sdf_edits(&mut self) {
        self.sdf_authority_mut().clear();
    }

    pub fn sdf_edits(&self) -> &[SdfEdit] {
        self.sdf_authority().edits()
    }

    pub fn sdf_edit(&self, id: SdfEditId) -> Option<&SdfEdit> {
        self.sdf_authority().get(id)
    }

    pub fn sdf_edit_id_at(&self, order_index: usize) -> Option<SdfEditId> {
        self.sdf_authority().id_at(order_index)
    }

    /// Set or disable the one canonical procedural-terrain configuration.
    pub fn set_sdf_terrain(&mut self, terrain: Option<TerrainConfig>) -> Result<()> {
        self.sdf_authority_mut()
            .set_terrain(terrain)
            .map_err(map_sdf_error)
    }

    pub fn sdf_terrain(&self) -> Option<&TerrainConfig> {
        self.sdf_authority().terrain()
    }

    /// Read-only CPU query against the same canonical stream published to GPU.
    pub fn pick_sdf_surface(
        &self,
        ray_origin: glam::Vec3,
        ray_direction: glam::Vec3,
        max_distance: f32,
    ) -> Option<SdfPickResult> {
        self.sdf_authority()
            .pick_surface(ray_origin, ray_direction, max_distance)
    }

    pub(crate) fn sdf_authority(&self) -> &SdfAuthority {
        self.authority
            .subsystem::<SdfAuthority>()
            .expect("SDF authority is registered at Scene construction")
    }

    pub(crate) fn sdf_authority_mut(&mut self) -> &mut SdfAuthority {
        self.authority
            .subsystem_mut::<SdfAuthority>()
            .expect("SDF authority is registered at Scene construction")
    }
}

fn map_sdf_error(error: SdfAuthorityError) -> SceneError {
    match error {
        SdfAuthorityError::StaleEdit => SceneError::InvalidHandle {
            resource: "SDF edit",
        },
        SdfAuthorityError::CapacityExceeded => SceneError::InvalidOperation {
            reason: "SceneDB SDF edit residency exceeds the device storage limit",
        },
        SdfAuthorityError::InvalidEdit => SceneError::InvalidOperation {
            reason: "SDF edit parameters must be finite and positive; transforms must be finite affine similarity transforms (no non-uniform scale or shear)",
        },
        SdfAuthorityError::InvalidTerrain => SceneError::InvalidOperation {
            reason: "SDF terrain parameters are outside the finite supported domain",
        },
        SdfAuthorityError::InvalidIndex => SceneError::InvalidOperation {
            reason: "SDF authored order index is out of bounds",
        },
    }
}
