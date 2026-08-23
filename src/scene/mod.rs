//! High-level scene management with automatic instancing.
//!
//! # Architecture
//!
//! Objects in the scene are automatically sorted by `(mesh_id, material_id)` and
//! grouped into instanced draw calls on every GPU buffer rebuild. No explicit
//! optimization step is required — the renderer always batches objects sharing
//! the same mesh and material.
//!
//! ## Steady-state CPU work
//!
//! - Transform updates are O(1) SceneDB component edits with deferred partner-row tracking
//! - GPU-driven paths do not iterate over visible scene objects on the CPU
//! - GPU frustum culling via indirect dispatch
//!
//! # Usage Example
//!
//! ```ignore
//! use helio::{Scene, ObjectDescriptor};
//! use glam::{Mat4, Vec3};
//!
//! // Create scene
//! let mut scene = Scene::new(device, queue);
//!
//! // Load resources
//! let mesh_id = scene.insert_mesh(mesh_upload);
//! let material_id = scene.insert_material(material);
//!
//! // Add objects — instancing is automatic
//! for transform in level_transforms {
//!     scene.insert_object(ObjectDescriptor {
//!         mesh: mesh_id,
//!         material: material_id,
//!         transform,
//!         bounds: [0.0, 1.0, 0.0, 1.0],
//!         flags: 0,
//!         groups: GroupMask::NONE,
//!     })?;
//! }
//!
//! // Render loop
//! loop {
//!     scene.update_camera(camera);
//!     scene.flush();
//!     renderer.render(&scene, target)?;
//! }
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Cost |
//! |-----------|------|
//! | `insert_object` / `remove_object` | Amortized O(1) authority mutation + deferred rebuild |
//! | `update_object_transform` | O(1) SceneDB edit + deferred partner/history update |
//! | GPU buffer rebuild | O(N log N) sort + O(N) upload |
//! | Render submission (CPU) | Independent of visible-instance count on GPU-driven paths; fallback-dependent |
//! | Draw calls (GPU) | D (one per unique mesh+material pair) |
//!
//! See the [GPU-Driven Pipeline](https://docs.farbeyondpulsar.com/helio/gpu-driven-pipeline)
//! documentation for complete architectural details.

mod actor;
mod camera;
mod core;
mod decals;
mod editor_debug;
#[cfg(test)]
mod environment_tests;
mod errors;
mod extension;
mod flush;
mod foliage;
mod foliage_authority;
mod groups;
mod helpers;
mod lifecycle;
mod multi_mesh;
mod objects;
mod portals;
mod planar_reflections;
mod planetary_voxel;
mod postprocess;
mod presentation;
mod resources;
mod sdf;
mod sprites;
mod stats;
mod sublevels;
mod types;
mod virtual_geometry;
mod voxel;
mod water;

pub use actor::{
    DecalActor, PostProcessVolumeActor, ReflectionCaptureActor, ReflectionCaptureDescriptor,
    SceneActor, SceneActorId, SceneActorTrait, WaterHitboxDescriptor, WaterHitboxActor,
    WaterVolumeDescriptor, WaterVolumeActor,
};
pub use camera::Camera;
pub use core::Scene;
pub use foliage::{
    FoliageInteractor, FoliageLayer, FoliageTypeDescriptor, GpuFoliageInteractor,
};
pub use errors::*;
pub use extension::{
    SceneComponentRegistrar, SceneCpuComponent, SceneDataBuffer, SceneDataComponent,
    SceneDataError, SceneDataMut, SceneDataSubsystem, SceneDataView, SceneDirtyTrackedComponent,
    SceneDirtyTrackedStorage, SceneExtensionEntity, SceneMixedComponent, SceneOnceComponent,
};
pub use portals::{portal_pose_facing, PortalDescriptor};
pub use planar_reflections::PlanarReflectorDescriptor;
pub use helio_scenedb::{
    BooleanOp, PlanetFrameEntry, PlanetFrameError, PlanetFrameId, PlanetFrameProjection,
    PlanetFrameUniform, PlanetFrameUpdateOutcome, PlanetId, PlanetPosition, PlanetRenderFrame,
    SdfEdit, SdfEditId, SdfPickResult, SdfShapeParams, SdfShapeType, TerrainConfig, TerrainStyle,
    SceneSpriteAtlasId, SceneSpriteId, SceneSpriteRow, SpriteAtlasSource, SpriteBufferSource,
    SpriteAuthorityError,
};
pub use presentation::BillboardInstance;
pub use sublevels::SublevelDescriptor;
pub use types::{ObjectDescriptor, PickableObject, VoxelVolumeDescriptor};
pub use voxel::VoxelMode;
