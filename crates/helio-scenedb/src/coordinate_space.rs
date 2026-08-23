//! Canonical coordinate-space transforms shared by sublevels and portals.
//!
//! A single component type is deliberate: its component-local GPU rows are
//! the stable `space_id` values packed into object flags and portal views.
//! Keeping both features in this one domain prevents two allocators from ever
//! handing out the same shader index.

use std::sync::Arc;

use pulsar_scenedb::gpu::SceneGpuStore;
use pulsar_scenedb::page::Pod as SceneDbPod;
use pulsar_scenedb_derive::SceneStore;

pub const COORDINATE_SPACE_BUFFER_KEY: &str = "helio.scene.coordinate_spaces";

/// Shader-exact `mat4x4<f32>` coordinate-space row.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneCoordinateSpaceRow(pub [f32; 16]);

// SAFETY: transparent over a fully initialized scalar array, with no padding
// or invalid bit patterns.
unsafe impl SceneDbPod for SceneCoordinateSpaceRow {}

impl SceneCoordinateSpaceRow {
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);
}

/// Persistent authored coordinate-space transform.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneCoordinateSpace {
    #[gpu(buffer = "helio.scene.coordinate_spaces")]
    pub transform: SceneCoordinateSpaceRow,
}

impl SceneCoordinateSpace {
    pub const IDENTITY: Self = Self {
        transform: SceneCoordinateSpaceRow::IDENTITY,
    };
}

impl crate::storage::MutableGpuComponent for SceneCoordinateSpace {}

/// Register the shared coordinate-space partner before the World mirror is
/// attached. The shader ABI has a hard maximum of 32 rows, including row 0's
/// permanent world identity, so the Helio facade reserves that complete 2 KiB
/// table and rejects a 33rd live component before insertion. SceneDB's generic
/// World registration remains growable because `World::insert` has no capacity
/// error channel; the public scene API is the deliberate hard-cap boundary.
pub fn register_coordinate_space_buffer(
    store: &mut SceneGpuStore,
    device: &Arc<wgpu::Device>,
) {
    SceneCoordinateSpace::register_gpu_columns_growable(
        store,
        libhelio::MAX_COORDINATE_SPACES,
        device,
    );
}

const _: () = {
    assert!(std::mem::size_of::<SceneCoordinateSpaceRow>() == 64);
    assert!(std::mem::size_of::<SceneCoordinateSpace>() == 64);
};
