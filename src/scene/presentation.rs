//! SceneDB-owned authored inputs for renderer presentation systems.
//!
//! Billboards and Corona emitters retain their bulk-replace compatibility API,
//! so they do not have useful per-item entity identity. Their canonical lists
//! live in this registered SceneDB subsystem; Helio keeps only derived scratch,
//! simulation, and frame-resource buffers.

use bytemuck::{Pod, Zeroable};
use helio_scenedb::Subsystem;

use super::Scene;

/// One authored world-space billboard.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BillboardInstance {
    pub world_pos: [f32; 4],
    pub scale_flags: [f32; 4],
    pub color: [f32; 4],
}

#[derive(Default)]
pub(in crate::scene) struct ScenePresentationState {
    billboards: Vec<BillboardInstance>,
    billboard_generation: u64,
    corona_emitters: Vec<libhelio::GpuCoronaEmitter>,
    corona_generation: u64,
    /// Canonical lifetime boundary for Helio-owned Corona simulation state.
    corona_reset_epoch: u64,
}

impl ScenePresentationState {
    fn replace_billboards(&mut self, instances: &[BillboardInstance]) {
        if bytemuck::cast_slice::<BillboardInstance, u8>(&self.billboards)
            == bytemuck::cast_slice::<BillboardInstance, u8>(instances)
        {
            return;
        }
        self.billboards.clear();
        self.billboards.extend_from_slice(instances);
        self.billboard_generation = self.billboard_generation.wrapping_add(1);
    }

    fn replace_corona_emitters(&mut self, emitters: &[libhelio::GpuCoronaEmitter]) {
        if bytemuck::cast_slice::<libhelio::GpuCoronaEmitter, u8>(&self.corona_emitters)
            == bytemuck::cast_slice::<libhelio::GpuCoronaEmitter, u8>(emitters)
        {
            return;
        }
        self.corona_emitters.clear();
        self.corona_emitters.extend_from_slice(emitters);
        self.corona_generation = self.corona_generation.wrapping_add(1);
    }

    pub(super) fn clear(&mut self) {
        self.replace_billboards(&[]);
        self.replace_corona_emitters(&[]);
        // This is deliberately distinct from authored generation: clear then
        // re-add before a render must still retire the old GPU simulation.
        self.corona_reset_epoch = self.corona_reset_epoch.wrapping_add(1);
    }
}

impl Subsystem for ScenePresentationState {
    fn name(&self) -> &'static str {
        "helio.scene.presentation"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Scene {
    fn presentation(&self) -> &ScenePresentationState {
        self.authority
            .subsystem::<ScenePresentationState>()
            .expect("presentation subsystem is registered at Scene construction")
    }

    fn presentation_mut(&mut self) -> &mut ScenePresentationState {
        self.authority
            .subsystem_mut::<ScenePresentationState>()
            .expect("presentation subsystem is registered at Scene construction")
    }

    /// Replace the canonical authored billboard list.
    ///
    /// Identical bytes are a no-op, so static callers may safely resubmit a
    /// slice without forcing Helio to rebuild its derived billboard frame.
    pub fn set_billboard_instances(&mut self, instances: &[BillboardInstance]) {
        self.presentation_mut().replace_billboards(instances);
    }

    /// Replace the canonical authored Corona emitter definitions.
    ///
    /// Corona's evolving particle simulation remains render-pass state; this
    /// list is only the authored emitter input that seeds that simulation.
    /// Requests beyond [`libhelio::CORONA_MAX_EMITTERS`] are rejected without
    /// changing the prior list, matching the fixed shader table exactly.
    pub fn set_corona_emitters(
        &mut self,
        emitters: &[libhelio::GpuCoronaEmitter],
    ) -> crate::scene::Result<()> {
        let capacity = libhelio::CORONA_MAX_EMITTERS as usize;
        if emitters.len() > capacity {
            return Err(crate::scene::SceneError::CoronaEmitterCapacityExceeded {
                requested: emitters.len(),
                capacity,
            });
        }
        self.presentation_mut().replace_corona_emitters(emitters);
        Ok(())
    }

    pub(crate) fn authored_billboards(&self) -> &[BillboardInstance] {
        &self.presentation().billboards
    }

    pub(crate) fn corona_emitters(&self) -> &[libhelio::GpuCoronaEmitter] {
        &self.presentation().corona_emitters
    }

    pub(crate) fn presentation_generations(&self) -> (u64, u64) {
        let state = self.presentation();
        (state.billboard_generation, state.corona_generation)
    }

    pub(crate) fn corona_reset_epoch(&self) -> u64 {
        self.presentation().corona_reset_epoch
    }

    pub(in crate::scene) fn clear_presentation(&mut self) {
        self.presentation_mut().clear();
    }
}
