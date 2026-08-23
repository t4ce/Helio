//! # Helio v3: GPU-Driven Rendering Core
//!
//! **Helio v3** is the render-graph executor and render-facing projection layer
//! used by the higher-level `helio` crate. It provides modular GPU-driven passes,
//! profiling, and borrowed access to published scene buffers.
//!
//! ## Design Philosophy
//!
//! Helio v3 is built on four core principles:
//!
//! 1. **Zero-Copy Access**: Passes receive borrowed references (`&wgpu::Buffer`) to GPU resources,
//!    never owned copies. This eliminates clones and ensures O(1) resource access.
//!
//! 2. **Explicit Scene Authority**: SceneDB owns persistent authored components
//!    and publishes component-local GPU partner rows. `GpuScene` owns only
//!    renderer-derived buffers, temporal state, and non-owning publications.
//!
//! 3. **Implicit Profiling**: CPU and GPU profiling happens automatically via `PassContext`.
//!    No manual instrumentation required - just implement `RenderPass` and profiling is injected.
//!
//! 4. **Trait-Based Modularity**: Render passes are separate crates implementing
//!    an object-safe trait and are composed dynamically by `RenderGraph`.
//!
//! ## Performance model
//!
//! The core avoids cloning complete authored scene tables into passes:
//!
//! - `SceneResources` is assembled from borrowed buffer references.
//! - Dirty Helio-owned projections upload changed ranges; clean managers issue no
//!   queue writes.
//! - GPU-driven passes avoid per-visible-instance CPU submission in their normal
//!   path.
//! - Buffer growth, bind-group/pipeline invalidation, topology rebuilds, and
//!   conservative backend fallbacks can allocate or perform scene-dependent work.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        helio-core (core)                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Traits                                                          │
//! │ └── RenderPass        : Object-safe render/compute pass         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Scene                                                           │
//! │ ├── GpuScene          : Derived renderer state + publications   │
//! │ ├── SceneResources    : Borrowed GPU resource references        │
//! │ └── Managers          : Growable derived/history buffers        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Graph                                                           │
//! │ ├── RenderGraph       : Graph executor with auto-profiling      │
//! │ └── ResourceBuilder   : Declare pass dependencies               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Profiling                                                       │
//! │ ├── Profiler          : Combined CPU/GPU profiler               │
//! │ ├── CpuProfiler       : Scoped CPU timing with RAII guards      │
//! │ └── GpuProfiler       : GPU timestamp queries                   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Context                                                         │
//! │ ├── PassContext       : Zero-copy context for execute()         │
//! │ └── PrepareContext    : Context for prepare() (uploads)         │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  Pass Crates (user-defined)                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ helio-gbuffer         : GBuffer geometry pass                   │
//! │ helio-shadow          : Cascaded shadow maps                    │
//! │ helio-deferred-light  : Deferred lighting pass                  │
//! │ helio-ssao            : Screen-space ambient occlusion          │
//! │ helio-bloom           : HDR bloom post-process                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start: Creating a Renderer
//!
//! ```rust,no_run
//! use helio_core::{RenderGraph, GpuScene, RenderPass, PassContext, Result};
//! use std::sync::Arc;
//!
//! // Define a simple pass
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str {
//!         "MyPass"
//!     }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::FrameResources<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // PassContext provides zero-copy access to scene resources
//!         let color_attachments = [Some(wgpu::RenderPassColorAttachment {
//!                 view: ctx.target,
//!                 resolve_target: None,
//!                 depth_slice: None,
//!                 ops: wgpu::Operations {
//!                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
//!                     store: wgpu::StoreOp::Store,
//!                 },
//!             })];
//!         let descriptor = wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"),
//!             color_attachments: &color_attachments,
//!             depth_stencil_attachment: None,
//!             timestamp_writes: None,
//!             occlusion_query_set: None,
//!             multiview_mask: None,
//!         };
//!         let mut pass = ctx.begin_render_pass(&descriptor);
//!
//!         pass.set_pipeline(&self.pipeline);
//!         // Access scene resources via ctx.scene
//!         // e.g., pass.set_bind_group(0, ctx.scene.lights.bind_group(), &[]);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//!
//! // Build the render graph
//! fn create_renderer(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
//!     let mut graph = RenderGraph::new(&device, &queue);
//!     let scene = GpuScene::new(device.clone(), queue.clone());
//!
//!     // Add passes (order matters)
//!     // graph.add_pass(Box::new(ShadowPass::new(&device)));
//!     // graph.add_pass(Box::new(GBufferPass::new(&device)));
//!     // graph.add_pass(Box::new(DeferredLightPass::new(&device)));
//!
//!     // Render loop
//!     // let target = surface.get_current_texture().unwrap();
//!     // let view = target.texture.create_view(&Default::default());
//!     // graph.execute(&scene, &view, &depth_view).unwrap();
//! }
//! ```
//!
//! ## How Passes Work
//!
//! A **render pass** is a single stage in the rendering pipeline. Passes implement the `RenderPass` trait:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, PrepareContext, Result};
//!
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//!     uniform_buffer: wgpu::Buffer,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str {
//!         "MyPass" // Used for profiling labels
//!     }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::FrameResources<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
//!         // Optional: Upload per-frame uniforms (called before execute)
//!         // ctx.queue.write_buffer(&self.uniform_buffer, 0, data);
//!         Ok(())
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Record GPU commands using ctx.encoder
//!         // Access scene resources via ctx.scene (zero-copy)
//!         Ok(())
//!     }
//! }
//! ```
//!
//! **Key Points:**
//! - `prepare()` runs on CPU before GPU submission (for uniform uploads)
//! - `execute()` records GPU commands into `ctx.encoder`
//! - `PassContext` provides zero-copy access to scene resources via `ctx.scene`
//! - Profiling is automatic (CPU scope + GPU timestamps)
//!
//! ## How scene publication works
//!
//! The higher-level `helio::Scene` mutates SceneDB's canonical components and
//! subsystems, then publishes their GPU partner buffers into `GpuScene`.
//! `GpuScene` does not provide an authored-object CRUD API; it retains
//! renderer-derived projections and non-owning canonical publications:
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
//! // Flush Helio-owned derived buffers. Canonical SceneDB partners are
//! // flushed and published by the high-level scene before graph execution.
//! scene.flush();
//!
//! // Assemble borrowed resource references for passes.
//! let resources = scene.resources();
//! let light_rows: &wgpu::Buffer = resources.lights;
//! # }
//! ```
//!
//! **Key Points:**
//! - SceneDB remains the persistent CPU authority for authored scene data.
//! - Canonical partner buffers use component-local rows, not `Entity::index()`.
//! - `GpuScene::flush()` uploads dirty Helio-derived ranges; it does not copy
//!   the canonical SceneDB tables.
//! - Passes receive `SceneResources<'_>` with borrowed buffer references.
//!
//! ## How Profiling Works
//!
//! Profiling is **automatic** and happens implicitly via `PassContext`:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str { "MyPass" }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::FrameResources<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // CPU profiling: Automatic scope created by RenderGraph
//!         // GPU profiling: Automatic timestamps via begin_render_pass
//!
//!         let descriptor = wgpu::RenderPassDescriptor {
//!             label: Some("MyPass"), // Used for GPU timestamp label
//!             // ...
//! #            color_attachments: &[],
//! #            depth_stencil_attachment: None,
//! #            timestamp_writes: None,
//! #            occlusion_query_set: None,
//! #            multiview_mask: None,
//!         };
//!         let mut pass = ctx.begin_render_pass(&descriptor);
//!
//!         // GPU timestamps automatically inserted at begin/end
//!         pass.set_pipeline(&self.pipeline);
//!         pass.draw(0..3, 0..1);
//!
//!         Ok(())
//!     }
//! }
//! ```
//!
//! **Key Points:**
//! - CPU profiling: `RenderGraph` creates scopes for each pass
//! - GPU profiling: `begin_render_pass` injects timestamp queries
//! - Zero instrumentation cost (compile-time feature flag `profiling`)
//! - Results available via `Profiler::export_timings()` for external telemetry
//!
//! ## Layering compared with the old monolithic renderer
//!
//! The current split keeps authored storage, derived render state, and pass
//! execution in separate layers:
//!
//! | v2 (Monolithic)                     | v3 (Modular)                          |
//! |-------------------------------------|---------------------------------------|
//! | `Renderer::render()`                | `RenderGraph::execute()`              |
//! | Renderer-owned scene collections    | SceneDB-backed high-level `Scene`     |
//! | One scene upload step               | SceneDB publication + derived flush   |
//! | Passes in `passes/` folder          | Separate crates (`helio-gbuffer`)     |
//! | Manual profiling calls              | Automatic via `PassContext`           |
//! | Shared mutable pass inputs          | Borrowed `SceneResources`             |
//! | Owned `wgpu::Buffer` in passes      | `&wgpu::Buffer` via `SceneResources`  |
//!
//! A low-level graph can still construct `GpuScene` directly, but authored
//! component insertion belongs to the high-level SceneDB-backed facade:
//!
//! ```rust,no_run
//! use helio_core::{RenderGraph, GpuScene};
//! use std::sync::Arc;
//!
//! # fn example(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
//! let mut graph = RenderGraph::new(&device, &queue);
//! let mut scene = GpuScene::new(device.clone(), queue.clone());
//!
//! scene.flush(); // Flush only Helio-owned derived state.
//! // graph.execute(&scene, &target, &depth);
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! - `profiling` (default): Enable automatic CPU/GPU profiling
//!   - Disable for maximum performance in shipping builds: `default-features = false`
//!
//! ## Performance Tips
//!
//! 1. Batch authored SceneDB edits before the high-level scene publication step.
//! 2. Reserve known component and projection capacities before large inserts.
//! 3. Cache bind groups and include published allocation epochs in cache keys.
//! 4. Expect topology changes and buffer growth to rebuild or allocate.
//! 5. Use the `profiling` feature to inspect CPU/GPU timings.
//!
//! ## Architecture Patterns
//!
//! ### Zero-Copy Resource Access
//!
//! Passes borrow shared scene publications. They can still own their pipelines,
//! bind groups, attachments, and pass-specific derived buffers:
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     bind_group_layout: wgpu::BindGroupLayout,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str { "MyPass" }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::FrameResources<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Borrow the current canonical light partner buffer.
//!         let light_buffer: &wgpu::Buffer = ctx.scene.lights;
//!
//!         // Resource creation can allocate; normally cache this by allocation epoch.
//!         // let bind_group = ctx.device.create_bind_group(...);
//!
//!         Ok(())
//!     }
//! }
//! ```
//!
//! ### Dirty Tracking Pattern
//!
//! SceneDB tracks changes to canonical `#[gpu]` partner rows. Helio separately
//! tracks ranges in its derived buffers. A clean Helio manager emits no queue
//! write, but `GpuScene::flush()` still performs bounded manager checks; buffer
//! growth and projection rebuilds are explicit non-steady-state work.
//!
//! ### Automatic Profiling Pattern
//!
//! Profiling is injected automatically by `RenderGraph`:
//!
//! ```text
//! RenderGraph::execute()
//! ├── CPU Scope: "ShadowPass" (automatic)
//! │   ├── prepare() (user code)
//! │   └── execute()
//! │       ├── GPU Timestamp: Start (automatic)
//! │       ├── GPU commands (user code)
//! │       └── GPU Timestamp: End (automatic)
//! ├── CPU Scope: "GBufferPass" (automatic)
//! │   └── ...
//! └── CPU Scope: "DeferredLightPass" (automatic)
//!     └── ...
//! ```
//!
//! ## See Also
//!
//! - [`RenderPass`] - Core trait for implementing render/compute passes
//! - [`GpuScene`] - Derived renderer state and canonical SceneDB publications
//! - [`RenderGraph`] - Graph executor with automatic profiling
//! - [`PassContext`] - Zero-copy context passed to `execute()`
//! - [`Profiler`] - Automatic CPU/GPU profiling system

/// Whether reflection features (screen-space, planar, and cubemap-capture
/// reflections) are available on this target.
///
/// Apple targets render them incorrectly, so the whole reflection group is
/// compiled out there: `SsrPass` and `PlanarReflectionPass` are left out of the
/// graph, and `DeferredLightPass` tells its shader to skip both composites and
/// the reflection-capture cube array. Everything else in deferred lighting —
/// direct light, ambient, RC GI — is untouched.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const REFLECTIONS_SUPPORTED: bool = false;
/// See the macOS/iOS variant above.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub const REFLECTIONS_SUPPORTED: bool = true;

pub mod acceleration;
pub mod actor;
pub mod component;
pub mod context;
pub mod entity;
pub mod error;
pub mod graph;
pub mod profiling;
pub mod scene;
pub mod shader;
pub mod traits;
pub mod upload;

// Re-export libhelio types for convenience
pub use libhelio::{
    DrawIndexedIndirectArgs, FrameResources, GBufferViews, GpuCameraUniforms, GpuDrawCall,
    GpuInstanceAabb, GpuInstanceData, GpuLight, GpuMaterial, GpuShadowMatrix,
};

pub use libhelio::sky::{SkyContext, SkyUniforms};
// Re-export managers
pub use crate::acceleration::{BlasManager, TlasInstanceInput, TlasManager};
pub use crate::scene::managers::*;
// Re-export core types
pub use actor::Actor;
pub use component::{Component, ComponentRegistry, ComponentSlot, ComponentVec};
pub use context::{PassContext, PrepareContext};
pub use entity::Entity;
pub use error::{Error, Result};
pub use graph::{DebugPassInfo, DebugResourceInfo, FrameDebugData, RenderGraph};
pub use profiling::{GpuTimingAvailability, Profiler, RenderPassTiming, RenderTimingSnapshot};
pub use scene::{
    GpuScene, SceneResources, WaterDropTarget, WaterSimulationTarget, WATER_SIM_SLOT_COUNT,
    WATER_SIM_SLOT_UNASSIGNED,
};
pub use traits::{AsAny, DebugViewDescriptor, MaybeSend, MaybeSync, RenderPass};
