//! High-level access to SceneDB's canonical planetary frame authority.

use helio_scenedb::{
    PlanetFrameAuthority, PlanetFrameAuthorityError, PlanetFrameEntry, PlanetFrameId,
    PlanetFrameUniform, PlanetFrameUpdateOutcome, PlanetId,
};

use super::{Result, Scene, SceneError};

impl Scene {
    /// Register a new planet and return its stable generation-bearing identity.
    pub fn insert_planet_frame(&mut self, frame: PlanetFrameUniform) -> Result<PlanetFrameId> {
        self.planet_frame_authority_mut()
            .insert(frame)
            .map_err(map_planet_frame_error)
    }

    /// Insert or update a frame by canonical `PlanetId`.
    ///
    /// Existing planets retain both their SceneDB identity and direct GPU row.
    pub fn set_planet_frame(
        &mut self,
        frame: PlanetFrameUniform,
    ) -> Result<(PlanetFrameId, PlanetFrameUpdateOutcome)> {
        self.planet_frame_authority_mut()
            .upsert(frame)
            .map_err(map_planet_frame_error)
    }

    /// Update one existing stable identity. The planet id cannot be changed in
    /// place because it is the canonical key used by streamed page addresses.
    pub fn update_planet_frame(
        &mut self,
        id: PlanetFrameId,
        frame: PlanetFrameUniform,
    ) -> Result<PlanetFrameUpdateOutcome> {
        self.planet_frame_authority_mut()
            .set(id, frame)
            .map_err(map_planet_frame_error)
    }

    pub fn remove_planet_frame(&mut self, id: PlanetFrameId) -> Result<PlanetFrameUniform> {
        self.planet_frame_authority_mut()
            .remove(id)
            .map_err(map_planet_frame_error)
    }

    pub fn remove_planet(&mut self, planet: PlanetId) -> Result<Option<PlanetFrameUniform>> {
        self.planet_frame_authority_mut()
            .remove_planet(planet)
            .map_err(map_planet_frame_error)
    }

    pub fn clear_planet_frames(&mut self) {
        self.planet_frame_authority_mut().clear();
    }

    pub fn planet_frame(&self, id: PlanetFrameId) -> Option<&PlanetFrameUniform> {
        self.planet_frame_authority().get(id)
    }

    pub fn planet_frame_for(&self, planet: PlanetId) -> Option<&PlanetFrameUniform> {
        self.planet_frame_authority().frame_for_planet(planet)
    }

    pub fn planet_frame_id(&self, planet: PlanetId) -> Option<PlanetFrameId> {
        self.planet_frame_authority().id_for_planet(planet)
    }

    pub fn planet_frames(&self) -> &[PlanetFrameEntry] {
        self.planet_frame_authority().entries()
    }

    pub(crate) fn planet_frame_authority(&self) -> &PlanetFrameAuthority {
        self.authority
            .subsystem::<PlanetFrameAuthority>()
            .expect("planet-frame authority is registered at Scene construction")
    }

    pub(crate) fn planet_frame_authority_mut(&mut self) -> &mut PlanetFrameAuthority {
        self.authority
            .subsystem_mut::<PlanetFrameAuthority>()
            .expect("planet-frame authority is registered at Scene construction")
    }
}

fn map_planet_frame_error(error: PlanetFrameAuthorityError) -> SceneError {
    match error {
        PlanetFrameAuthorityError::StaleFrame => SceneError::InvalidHandle {
            resource: "planet frame",
        },
        PlanetFrameAuthorityError::DuplicatePlanet(_) => SceneError::InvalidOperation {
            reason: "planet already has a canonical frame",
        },
        PlanetFrameAuthorityError::PlanetIdentityMismatch { .. } => {
            SceneError::InvalidOperation {
                reason: "planet id cannot change without replacing its canonical identity",
            }
        }
        PlanetFrameAuthorityError::InvalidFrame => SceneError::InvalidOperation {
            reason: "planet frame does not match the canonical planetary coordinate contract",
        },
        PlanetFrameAuthorityError::CapacityExceeded => SceneError::InvalidOperation {
            reason: "SceneDB planet frames exceed the device storage-buffer limit",
        },
    }
}
