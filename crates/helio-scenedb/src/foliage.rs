//! Canonical foliage components and named GPU partners.
//!
//! SceneDB owns authored type, layer, and interactor records. Helio may derive compact
//! active-row projections for shaders whose records carry 8-bit type identities, but it
//! never repacks these canonical 96/32/32-byte tables.

use std::sync::Arc;

use helio_foliage_core::{GpuFoliageInteractor, GpuFoliageLayer, GpuFoliageType};
use pulsar_scenedb::gpu::SceneGpuStore;
use pulsar_scenedb::page::Pod as SceneDbPod;
use pulsar_scenedb::Entity;
use pulsar_scenedb_derive::SceneStore;

pub const FOLIAGE_TYPE_BUFFER_KEY: &str = "helio.scene.foliage.types";
pub const FOLIAGE_LAYER_BUFFER_KEY: &str = "helio.scene.foliage.layers";
pub const FOLIAGE_INTERACTOR_BUFFER_KEY: &str = "helio.scene.foliage.interactors";

macro_rules! foliage_row {
    ($name:ident, $inner:ty) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name(pub $inner);

        // SAFETY: each transparent wrapper has the exact byte layout of its bytemuck
        // Pod shader row and therefore has no invalid patterns or drop glue.
        unsafe impl SceneDbPod for $name {}

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        const _: () = {
            assert!(std::mem::size_of::<$name>() == std::mem::size_of::<$inner>());
            assert!(std::mem::align_of::<$name>() == std::mem::align_of::<$inner>());
        };
    };
}

foliage_row!(SceneFoliageTypeRow, GpuFoliageType);
foliage_row!(SceneFoliageLayerRow, GpuFoliageLayer);
foliage_row!(SceneFoliageInteractorRow, GpuFoliageInteractor);

/// One authored foliage type.
///
/// `material_entity_bits` is the generation-bearing authored relationship. The dense
/// material row cached inside `foliage.material_id` is only its GPU projection and must
/// never be used to retain/release or identify the material on CPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneFoliageType {
    pub material_entity_bits: u64,
    #[gpu(buffer = "helio.scene.foliage.types")]
    pub foliage: SceneFoliageTypeRow,
}

/// Fixed authored layer data plus its canonical 32-byte GPU row.
///
/// Variable-length type membership lives in [`SceneFoliageLayerTypes`] on the same
/// entity so archetype queries retain the relationship without forcing a second scene
/// authority or widening the shader ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneFoliageLayer {
    #[gpu(buffer = "helio.scene.foliage.layers")]
    pub foliage: SceneFoliageLayerRow,
    pub seed: u32,
    pub _pad: u32,
}

/// Generation-bearing authored type relationships for one foliage layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneFoliageLayerTypes {
    pub types: Vec<Entity>,
}

/// One moving body that displaces foliage.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneFoliageInteractor {
    #[gpu(buffer = "helio.scene.foliage.interactors")]
    pub interactor: SceneFoliageInteractorRow,
}

pub fn register_foliage_component_buffers(
    store: &mut SceneGpuStore,
    device: &Arc<wgpu::Device>,
) {
    const INITIAL_CAPACITY: u32 = 1;
    SceneFoliageType::register_gpu_columns_growable(store, INITIAL_CAPACITY, device);
    SceneFoliageLayer::register_gpu_columns_growable(store, INITIAL_CAPACITY, device);
    SceneFoliageInteractor::register_gpu_columns_growable(store, INITIAL_CAPACITY, device);
}

const _: () = {
    assert!(std::mem::size_of::<SceneFoliageType>() == 104);
    assert!(std::mem::offset_of!(SceneFoliageType, foliage) == 8);
    assert!(std::mem::size_of::<SceneFoliageLayer>() == 40);
    assert!(std::mem::size_of::<SceneFoliageInteractor>() == 32);
};

