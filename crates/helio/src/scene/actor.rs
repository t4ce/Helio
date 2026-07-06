use crate::Scene;
use helio_core::SkyContext;

/// Trait for types that can be inserted into a [`Scene`].
///
/// Each implementor knows which scene method to call and what ID type to return,
/// eliminating the need for a unified enum and runtime `as_*()` downcasts.
///
/// # Example
///
/// ```ignore
/// let mesh_id: MeshId = scene.insert_actor(MeshUpload { vertices, indices });
/// ```
///
/// Implementations live alongside their types — not in this file.
pub trait IntoActor {
    type Id: Copy;
    fn insert(self, scene: &mut Scene) -> Self::Id;
}

/// Common behavior for custom scene actors with per-frame lifecycle.
pub trait SceneActorTrait {
    fn is_active(&self) -> bool { true }
    fn on_tick(&mut self, _scene: &mut Scene) {}
    fn sky_context(&self) -> Option<SkyContext> { None }
}
