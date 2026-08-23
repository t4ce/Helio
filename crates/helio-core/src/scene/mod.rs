//! Render-facing scene publications and derived GPU state.
//!
//! Persistent authored data belongs to SceneDB's `World` and typed subsystems.
//! Their `#[gpu]` partner buffers are published here without transferring
//! authority. Helio additionally owns compact draw projections, temporal
//! history, camera state, and pass-facing command buffers.
//!
//! # Design pattern
//!
//! The renderer-facing split is:
//!
//! 1. **SceneDB authority**: authored CPU values and component-local GPU partners
//! 2. **Helio projections**: compact/temporal buffers derived for rendering
//! 3. **Borrowed publication**: passes receive `&wgpu::Buffer` references
//! 4. **Epoch-aware growth**: physical buffers may grow and consumers rebind
//!
//! # Components
//!
//! - [`GpuScene`] - Derived renderer state plus canonical publications
//! - [`SceneResources`] - Borrowed resource references passed to passes
//!
//! # Performance
//!
//! - `flush()` uploads dirty ranges in Helio-owned managers.
//! - Clean managers issue no queue writes, though bounded checks still run.
//! - Growth and topology/projection rebuilds may allocate and do
//!   scene-dependent work.
//! - `SceneResources` borrows current allocations without cloning scene data.
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_core::GpuScene;
//! use std::sync::Arc;
//!
//! # fn example(device: wgpu::Device, queue: wgpu::Queue) {
//! let mut scene = GpuScene::new(
//!     Arc::new(device),
//!     Arc::new(queue),
//! );
//!
//! // Authored CRUD belongs to the higher-level SceneDB-backed Scene.
//! // This flush covers only Helio-owned derived buffers.
//! scene.flush();
//!
//! // Passes receive borrowed references.
//! let resources = scene.resources();
//! let light_buffer: &wgpu::Buffer = resources.lights;
//! // Execute the graph, then commit temporal history once for the rendered frame.
//! scene.advance_frame();
//! # }
//! ```

mod gpu_scene;
pub mod managers;
mod resources;

pub use crate::component::ComponentRegistry;
pub use gpu_scene::{
    GpuScene, WaterDropTarget, WaterSimulationTarget, WATER_SIM_SLOT_COUNT,
    WATER_SIM_SLOT_UNASSIGNED,
};
pub use managers::*;
pub use resources::SceneResources;
