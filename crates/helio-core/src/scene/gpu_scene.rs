//! Renderer-owned GPU state and SceneDB buffer publications.
//!
//! Persistent authored scene data is owned by SceneDB. `GpuScene` borrows its
//! published partner allocations and owns only renderer/executor state:
//! compact draw projections, indirect commands, visibility/cull scratch,
//! temporal history, shadow assignments, and other pass outputs.

use crate::acceleration::{BlasManager, TlasManager};
use crate::scene::managers::GrowableBuffer;
use crate::scene::managers::{
    CoordinateSpaceHistory, GpuCameraBuffer, GpuCompactedIndices2Buffer,
    GpuCompactedIndicesBuffer, GpuDrawCallBuffer, GpuIndirectBuffer, GpuObjectHistoryBuffer,
    GpuShadowMatrixBuffer, GpuSourceIndicesBuffer, GpuVisibilityBuffer,
};
use crate::scene::SceneResources;
use std::sync::Arc;

/// Number of persistent heightfield residencies owned by the water simulation.
/// Authored water volumes beyond this limit remain valid SceneDB records but
/// receive [`WATER_SIM_SLOT_UNASSIGNED`] in the compact render projection.
pub const WATER_SIM_SLOT_COUNT: usize = 8;
pub const WATER_SIM_SLOT_UNASSIGNED: u32 = u32::MAX;

/// Derived identity for one pass-owned persistent water-simulation residency.
///
/// SceneDB remains the authority for the water volume itself. This token only
/// lets transient pass events target the matching heightfield without retaining
/// authored data or accidentally following a slot after it is reassigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaterSimulationTarget {
    sim_slot: u32,
    canonical_row: u32,
    residency_generation: u64,
}

impl WaterSimulationTarget {
    #[doc(hidden)]
    pub const fn from_parts(
        sim_slot: u32,
        canonical_row: u32,
        residency_generation: u64,
    ) -> Self {
        Self {
            sim_slot,
            canonical_row,
            residency_generation,
        }
    }

    pub const fn sim_slot(self) -> u32 {
        self.sim_slot
    }

    pub const fn canonical_row(self) -> u32 {
        self.canonical_row
    }

    pub const fn residency_generation(self) -> u64 {
        self.residency_generation
    }
}

/// Scene-validated world-space target for one transient water impulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterDropTarget {
    simulation: WaterSimulationTarget,
    world_center: [f32; 2],
}

impl WaterDropTarget {
    #[doc(hidden)]
    pub const fn from_parts(
        simulation: WaterSimulationTarget,
        world_center: [f32; 2],
    ) -> Self {
        Self {
            simulation,
            world_center,
        }
    }

    pub const fn simulation(self) -> WaterSimulationTarget {
        self.simulation
    }

    pub const fn world_center(self) -> [f32; 2] {
        self.world_center
    }
}

/// One published GPU allocation and the owner's allocation epoch/row span.
///
/// Cloning a `wgpu::Buffer` handle here does not transfer data authority:
/// SceneDB still owns canonical allocation, mutation, and residency.
pub struct PublishedSceneBuffer {
    buffer: Box<wgpu::Buffer>,
    epoch: u64,
    len: u32,
}

impl PublishedSceneBuffer {
    pub fn new(buffer: wgpu::Buffer, epoch: u64, len: u32) -> Self {
        Self {
            buffer: Box::new(buffer),
            epoch,
            len,
        }
    }

    /// Refresh the borrowed-allocation handle only when its owner reports a
    /// new epoch.  Boxing is load-bearing: pass bind-group caches fingerprint
    /// `&wgpu::Buffer` by address, which must change when the allocation does.
    pub fn publish(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        if self.epoch != epoch {
            self.buffer = Box::new(buffer);
            self.epoch = epoch;
        }
        self.len = len;
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref()
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn len(&self) -> u32 {
        self.len
    }
}

/// Non-owning publications of canonical scene buffers.
///
/// SceneDB owns allocation, residency, mutation, and epochs. `GpuScene`
/// retains cloned wgpu handles solely so the render graph can bind those
/// allocations. Empty entries bind a valid empty fallback buffer.
#[derive(Default)]
pub struct CanonicalSceneBuffers {
    pub object_spatial: Option<PublishedSceneBuffer>,
    pub object_render: Option<PublishedSceneBuffer>,
    pub lights: Option<PublishedSceneBuffer>,
    pub decals: Option<PublishedSceneBuffer>,
    pub water_volumes: Option<PublishedSceneBuffer>,
    pub water_hitboxes: Option<PublishedSceneBuffer>,
    pub post_process_volumes: Option<PublishedSceneBuffer>,
    pub reflection_captures: Option<PublishedSceneBuffer>,
    pub planar_reflectors: Option<PublishedSceneBuffer>,
    pub materials: Option<PublishedSceneBuffer>,
    pub material_textures: Option<PublishedSceneBuffer>,
    pub coordinate_spaces: Option<PublishedSceneBuffer>,
    pub voxel_volumes: Option<PublishedSceneBuffer>,
    pub voxel_bricks: Option<PublishedSceneBuffer>,
    pub voxel_data: Option<PublishedSceneBuffer>,
    pub voxel_palettes: Option<PublishedSceneBuffer>,
    pub sdf_edits: Option<PublishedSceneBuffer>,
    pub sdf_terrain: Option<PublishedSceneBuffer>,
    pub foliage_types: Option<PublishedSceneBuffer>,
    pub foliage_layers: Option<PublishedSceneBuffer>,
    pub foliage_interactors: Option<PublishedSceneBuffer>,
    pub planet_frames: Option<PublishedSceneBuffer>,
}

/// Frame-persistent renderer state plus non-owning canonical SceneDB publications.
///
/// Authored component CRUD belongs to the higher-level SceneDB-backed scene.
/// This type stores camera/temporal state and compact render projections, and
/// publishes SceneDB-owned partner buffers for passes. Derived managers upload
/// dirty ranges; projection rebuilds and buffer growth are explicitly allowed
/// to perform scene-dependent work or allocate.
pub struct GpuScene {
    /// GPU device (shared across scene).
    ///
    /// Used by managers to create buffers when capacity grows.
    pub device: Arc<wgpu::Device>,

    /// GPU queue (shared across scene).
    ///
    /// Used by `flush()` to upload data to GPU.
    pub queue: Arc<wgpu::Queue>,

    /// Current frame number (starts at 0).
    ///
    /// Incremented each frame, useful for time-based effects.
    pub frame_count: u64,

    /// Render target width in pixels.
    pub width: u32,

    /// Render target height in pixels.
    pub height: u32,

    /// Generation counter for movable objects - increments when any Movable object moves.
    /// Used by shadow caching to detect movement.
    pub movable_objects_generation: u64,

    /// Generation counter for movable lights - increments when any Movable light moves.
    /// Used by shadow caching to detect movement.
    pub movable_lights_generation: u64,

    /// Generation counter for camera - increments every time the camera is updated.
    /// Used by HiZ and light-cull passes to detect camera movement.
    pub camera_generation: u64,

    pub camera: GpuCameraBuffer,
    /// Read-only handles published by the persistent scene-data authority.
    pub canonical: CanonicalSceneBuffers,
    /// Valid storage binding used only before the first canonical publication.
    empty_scene_rows: Box<wgpu::Buffer>,
    /// Valid all-zero uniform binding used before a canonical singleton is published.
    empty_scene_uniform: Box<wgpu::Buffer>,
    /// Read-only CPU projection used by SdfPass to build its renderer-owned BVH.
    /// Repacked only when SceneDB's authored content generation changes.
    sdf_edit_bounds: Vec<[f32; 4]>,
    sdf_content_generation: u64,
    sdf_terrain_y_bounds: Option<[f32; 2]>,
    sdf_requires_canonical_scan: bool,
    /// Compact read projection of SceneDB's stable sparse planet-frame rows.
    /// Copied only when authored frame content changes; the direct GPU buffer
    /// remains published through `canonical.planet_frames`.
    planet_frames: Vec<helio_planet_voxel_core::PlanetFrameProjection>,
    planet_frame_authority_epoch: u64,
    planet_frame_content_generation: u64,
    /// Previous-frame object-local models, keyed by the stable component-local
    /// `SceneObject` GPU row.
    /// This is temporal renderer state, not a second persistent scene record.
    pub object_history: GpuObjectHistoryBuffer,
    pub draw_calls: GpuDrawCallBuffer,
    /// Canonical material row for each compact draw-call group. This is CPU
    /// render-topology metadata used by passes that build their own projected
    /// draw templates; persistent material identity remains in SceneDB.
    pub draw_material_rows: Vec<u32>,
    /// Changes whenever `draw_calls`/`draw_material_rows` are rebuilt.
    pub draw_topology_generation: u64,
    /// Compact realtime slot -> `[component-local light row, assigned shadow slice]`.
    pub light_projections: GrowableBuffer<[u32; 2]>,
    /// Compact active decal slot -> component-local SceneDB decal row.
    pub decal_indices: GrowableBuffer<u32>,
    /// Compact active water-volume slot -> `[canonical row, stable sim slot]`.
    /// The sim slot is renderer residency, never persistent scene authority.
    pub water_volume_projections: GrowableBuffer<[u32; 2]>,
    /// CPU-only reset epochs for the eight persistent simulation slots. A pass
    /// clears both ping-pong textures whenever its cached epoch differs.
    pub water_sim_slot_generations: [u64; WATER_SIM_SLOT_COUNT],
    /// Compact active water-hitbox slot -> canonical SceneDB component row.
    pub water_hitbox_indices: GrowableBuffer<u32>,
    /// Compact active post-process-volume slot -> canonical SceneDB component row.
    pub post_process_volume_indices: GrowableBuffer<u32>,
    /// Influence-sorted reflection slot -> `[canonical component row, cubemap layer]`.
    pub reflection_capture_projections: GrowableBuffer<[u32; 2]>,
    /// Compact active planar-reflector slot -> canonical SceneDB component row.
    pub planar_reflector_indices: GrowableBuffer<u32>,
    /// Compact active voxel-volume slot -> canonical SceneDB component row.
    pub voxel_volume_indices: GrowableBuffer<u32>,
    /// Stable Auto-mesh output slot -> canonical volume row and local brick.
    pub voxel_mesh_work: GrowableBuffer<helio_voxel_core::GpuVoxelMeshWork>,
    /// Changes when a coalesced Auto-mesh work batch is published.
    pub voxel_mesh_work_generation: u64,
    /// High-water occupied output slot used for indirect submission.
    pub voxel_mesh_draw_count: u32,
    /// Compact 8-bit blade type id -> canonical SceneDB foliage-type row.
    /// Fixed at 256 entries so cull can bind it as a guaranteed-small uniform rather
    /// than exceeding the 8-storage-buffer stage limit.
    pub foliage_type_indices: GrowableBuffer<u32>,
    /// Compact active layer -> canonical row plus flattened relation span and seed.
    pub foliage_layer_projections:
        GrowableBuffer<helio_foliage_core::GpuFoliageLayerProjection>,
    /// Flattened generation-validated layer-to-type relationships.
    pub foliage_layer_type_relations:
        GrowableBuffer<helio_foliage_core::GpuFoliageLayerTypeRelation>,
    /// Compact active interactor -> canonical SceneDB interactor row.
    pub foliage_interactor_indices: GrowableBuffer<u32>,
    pub shadow_matrices: GpuShadowMatrixBuffer,
    /// High-water span of faces assigned to live shadow casters. This may be
    /// zero even though `shadow_matrices` retains one element so its storage
    /// binding remains valid on APIs that reject zero-sized buffers.
    pub active_shadow_face_count: u32,
    pub indirect: GpuIndirectBuffer,
    pub visibility: GpuVisibilityBuffer,
    /// Compact draw-order slot -> canonical SceneDB row. This stays Helio-owned
    /// because sorting/batching is renderer-derived; the target rows and their
    /// instance/AABB fields remain solely SceneDB-owned.
    pub source_indices: GpuSourceIndicesBuffer,
    /// Static shadow draw slot -> component-local SceneObject row.
    pub shadow_static_source_indices: GrowableBuffer<u32>,
    /// Movable shadow draw slot -> component-local SceneObject row.
    pub shadow_movable_source_indices: GrowableBuffer<u32>,
    /// Per-instance original-slot indices surviving GPU frustum culling, packed
    /// per draw-call group. Written by IndirectDispatchPass, consumed by
    /// GBufferPass in place of a direct `instances[instance_index]` lookup.
    pub compacted_indices: GpuCompactedIndicesBuffer,
    /// Final surviving instance slots after frustum + Hi-Z occlusion culling.
    /// Written by OcclusionCullPass, consumed by GBufferPass/DepthPrepass.
    pub compacted_indices_2: GpuCompactedIndices2Buffer,

    /// Previous-frame projection and CPU staging for SceneDB-owned coordinate
    /// spaces. The current GPU allocation is published through `canonical`.
    pub coordinate_space_history: CoordinateSpaceHistory,

    // ── Shadow partition buffers (Unreal-style static/dynamic split) ──────────
    // NOTE: Both pass kinds use `instances` (the main transforms buffer) at binding 1.
    // We only partition the INDIRECT DRAW CALL buffers so that each atlas can be
    // rendered with a single `multi_draw_indexed_indirect` call. This means
    // `first_instance` in each indirect entry is the object's dense_index into
    // `instances`, keeping transform data in a single place that stays in sync
    // when `update_object_transform` writes to it.
    //
    // Obsolete approach (DO NOT restore): splitting instance data into two copies
    // (shadow_static_instances / shadow_movable_instances) caused dynamic shadows to
    // freeze because the copies were never updated on `update_object_transform`.
    /// Indirect draw commands for Static/Stationary objects (indexes into `instances`).
    pub shadow_static_indirect: GpuIndirectBuffer,
    /// Indirect draw commands for Movable objects (indexes into `instances`).
    pub shadow_movable_indirect: GpuIndirectBuffer,
    /// Number of draw calls in shadow_static_indirect.
    pub shadow_static_draw_count: u32,
    /// Number of draw calls in shadow_movable_indirect.
    pub shadow_movable_draw_count: u32,
    /// Changes whenever movable shadow source membership/order changes, even
    /// when object and draw counts remain identical.
    pub shadow_movable_topology_generation: u64,
    /// Increments when the static object set changes (add/remove of Static/Stationary objects).
    /// Used by ShadowPass to know when to re-render the static shadow atlas.
    pub static_objects_generation: u64,

    /// Number of movable lights in the lights buffer (at runtime, only movable lights are uploaded).
    /// Static/stationary lights are baked and excluded from real-time lighting calculations.
    pub movable_light_count: u32,

    /// Per-caster authored-input fingerprints. Each slot corresponds to one
    /// shadow caster (6 atlas faces). `Scene::flush()` hashes light state and,
    /// for directional lights, the complete unjittered camera frustum.
    /// Movable-object changes are detected separately by ShadowDirty on GPU.
    /// ShadowPass compares these values with its last-rendered snapshot.
    pub per_caster_dirty_gen: [u64; 42],


    /// Material class ranges for the GBuffer pass: [(class, graph_hash, start, count), ...]
    /// Each range is uniform in both material_class and graph_hash so a single
    /// PSO works for all indirect entries it covers.
    /// Built during `rebuild_instance_buffers_*`.
    pub material_class_ranges: Vec<(u32, u64, u32, u32)>,
    pub transparent_material_class_ranges: Vec<(u32, u64, u32, u32)>,
    /// Forward-shaded material class ranges (excluded from GBuffer pass).
    /// Drawn by the forward-lit pass instead.
    pub forward_material_class_ranges: Vec<(u32, u64, u32, u32)>,

    /// Renderer-derived material flags indexed by component-local material GPU
    /// row. Updated only on material mutations; virtual-geometry prepare uses
    /// this narrow cache instead of retaining a second full material table.
    pub material_flags: Vec<u32>,

    /// Render-facing clone of SceneDB's canonical graph WGSL assets, keyed by
    /// content hash. Material passes look up source by hash when building PSOs.
    pub graph_wgsl_snippets: std::collections::HashMap<u64, String>,
    /// Content epoch for `graph_wgsl_snippets`. Passes include this in their
    /// pipeline-cache validity so replacement under an existing hash cannot
    /// keep a pipeline compiled from stale source alive.
    pub graph_wgsl_epoch: u64,

    /// Custom template registrations that survive graph rebuilds.
    /// Stored as `Box<dyn Any>` — the GBufferPass downcasts it to
    /// `RadiantTemplateRegistry` at the start of every frame.
    pub template_registry: Option<Box<dyn std::any::Any + Send + Sync>>,

    /// Type-erased transparent template registry (`RadiantTemplateRegistry`).
    /// Separate from `template_registry` because transparent templates use a
    /// different base shader and bind group layout than gbuffer templates.
    pub transparent_template_registry: Option<Box<dyn std::any::Any + Send + Sync>>,

    /// Active portals' render data (clip transform + which coordinate space
    /// holds their content duplicate). Republished unconditionally each frame
    /// by `helio::Scene::flush()` from its private portal registry — portal
    /// counts are always small, so this is simpler than dirty-tracking it.
    /// Consumed by `helio-pass-portal-cull` / `helio-pass-portal-instances`.
    pub portal_views: GrowableBuffer<libhelio::GpuPortalView>,

    /// Every valid portal chain up to `libhelio::MAX_CHAIN_DEPTH` deep —
    /// see `SceneResources::portal_chains` for what this is and why it
    /// exists. Rebuilt only when the portal set changes.
    pub portal_chains: GrowableBuffer<libhelio::GpuPortalChain>,

    /// Bottom-Level Acceleration Structure manager (ray tracing).
    pub blas_manager: BlasManager,

    /// Top-Level Acceleration Structure manager (ray tracing, per-frame).
    pub tlas_manager: TlasManager,
}

impl GpuScene {
    fn publish_canonical(
        slot: &mut Option<PublishedSceneBuffer>,
        buffer: wgpu::Buffer,
        epoch: u64,
        len: u32,
    ) {
        match slot {
            Some(publication) => publication.publish(buffer, epoch, len),
            None => *slot = Some(PublishedSceneBuffer::new(buffer, epoch, len)),
        }
    }

    pub fn publish_object_spatial(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.object_spatial, buffer, epoch, len);
    }

    pub fn publish_object_render(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.object_render, buffer, epoch, len);
    }

    pub fn publish_lights(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.lights, buffer, epoch, len);
    }

    pub fn publish_decals(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.decals, buffer, epoch, len);
    }

    pub fn publish_water_volumes(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.water_volumes, buffer, epoch, len);
    }

    /// Mark a persistent water-simulation slot as newly assigned. The water
    /// pass observes this CPU epoch and clears all cascades before using the
    /// slot, so a newly inserted volume cannot inherit a prior occupant's
    /// heightfield.
    pub fn reset_water_sim_slot(&mut self, slot: u32) {
        let slot = usize::try_from(slot).expect("water sim slot must fit usize");
        let generation = self
            .water_sim_slot_generations
            .get_mut(slot)
            .expect("water sim slot is outside the fixed residency table");
        *generation = generation.wrapping_add(1);
    }

    pub fn publish_water_hitboxes(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.water_hitboxes, buffer, epoch, len);
    }

    pub fn publish_post_process_volumes(
        &mut self,
        buffer: wgpu::Buffer,
        epoch: u64,
        len: u32,
    ) {
        Self::publish_canonical(
            &mut self.canonical.post_process_volumes,
            buffer,
            epoch,
            len,
        );
    }

    pub fn publish_reflection_captures(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.reflection_captures, buffer, epoch, len);
    }

    pub fn publish_planar_reflectors(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.planar_reflectors, buffer, epoch, len);
    }

    pub fn publish_materials(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.materials, buffer, epoch, len);
    }

    pub fn publish_material_textures(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.material_textures, buffer, epoch, len);
    }

    pub fn publish_coordinate_spaces(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.coordinate_spaces, buffer, epoch, len);
    }

    pub fn publish_voxel_volumes(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.voxel_volumes, buffer, epoch, len);
    }

    /// Publish the three allocations owned by SceneDB's VoxelResidency
    /// subsystem. They grow as one allocation epoch and are never mutated by
    /// Helio's renderer state.
    pub fn publish_voxel_residency(
        &mut self,
        bricks: wgpu::Buffer,
        data: wgpu::Buffer,
        palettes: wgpu::Buffer,
        epoch: u64,
        capacity_bricks: u32,
        capacity_palette_rows: u32,
    ) {
        Self::publish_canonical(
            &mut self.canonical.voxel_bricks,
            bricks,
            epoch,
            capacity_bricks,
        );
        Self::publish_canonical(
            &mut self.canonical.voxel_data,
            data,
            epoch,
            capacity_bricks,
        );
        Self::publish_canonical(
            &mut self.canonical.voxel_palettes,
            palettes,
            epoch,
            capacity_palette_rows,
        );
    }

    pub fn publish_foliage_types(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.foliage_types, buffer, epoch, len);
    }

    pub fn publish_foliage_layers(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.foliage_layers, buffer, epoch, len);
    }

    pub fn publish_foliage_interactors(&mut self, buffer: wgpu::Buffer, epoch: u64, len: u32) {
        Self::publish_canonical(&mut self.canonical.foliage_interactors, buffer, epoch, len);
    }

    /// Publish SceneDB's sparse stable planet-frame rows directly and refresh
    /// the compact CPU projection only on authored mutations. Render passes use
    /// the latter to rebuild derived address tables without a GPU readback.
    pub fn publish_planet_frames(
        &mut self,
        buffer: wgpu::Buffer,
        authority_epoch: u64,
        allocation_epoch: u64,
        row_span: u32,
        content_generation: u64,
        frames: impl ExactSizeIterator<Item = helio_planet_voxel_core::PlanetFrameProjection>,
    ) {
        Self::publish_canonical(
            &mut self.canonical.planet_frames,
            buffer,
            allocation_epoch,
            row_span,
        );
        if self.planet_frame_authority_epoch != authority_epoch
            || self.planet_frame_content_generation != content_generation
        {
            self.planet_frames.clear();
            self.planet_frames.reserve(frames.len());
            self.planet_frames.extend(frames);
            self.planet_frame_authority_epoch = authority_epoch;
            self.planet_frame_content_generation = content_generation;
        }
    }

    pub fn planet_frame_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .planet_frames
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn planet_frame_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .planet_frames
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn planet_frame_row_span(&self) -> u32 {
        self.canonical
            .planet_frames
            .as_ref()
            .map(PublishedSceneBuffer::len)
            .unwrap_or(0)
    }

    pub fn planet_frames(&self) -> &[helio_planet_voxel_core::PlanetFrameProjection] {
        &self.planet_frames
    }

    pub const fn planet_frame_authority_epoch(&self) -> u64 {
        self.planet_frame_authority_epoch
    }

    pub const fn planet_frame_content_generation(&self) -> u64 {
        self.planet_frame_content_generation
    }

    /// Publish SceneDB's ordered authored SDF stream without copying its GPU
    /// rows. Bounds are the narrow CPU snapshot needed to build Helio's
    /// derived BVH and are copied only on authored mutations.
    #[allow(clippy::too_many_arguments)]
    pub fn publish_sdf_authority(
        &mut self,
        edits: wgpu::Buffer,
        edit_epoch: u64,
        edit_count: u32,
        terrain: wgpu::Buffer,
        terrain_epoch: u64,
        content_generation: u64,
        bounds: &[[f32; 4]],
        terrain_y_bounds: Option<[f32; 2]>,
        requires_canonical_scan: bool,
    ) {
        Self::publish_canonical(
            &mut self.canonical.sdf_edits,
            edits,
            edit_epoch,
            edit_count,
        );
        Self::publish_canonical(
            &mut self.canonical.sdf_terrain,
            terrain,
            terrain_epoch,
            1,
        );
        if self.sdf_content_generation != content_generation {
            self.sdf_edit_bounds.clear();
            self.sdf_edit_bounds.extend_from_slice(bounds);
            self.sdf_content_generation = content_generation;
            self.sdf_terrain_y_bounds = terrain_y_bounds;
            self.sdf_requires_canonical_scan = requires_canonical_scan;
        }
    }

    pub fn sdf_edit_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .sdf_edits
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn sdf_edit_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .sdf_edits
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn sdf_edit_count(&self) -> u32 {
        self.canonical
            .sdf_edits
            .as_ref()
            .map(PublishedSceneBuffer::len)
            .unwrap_or(0)
    }

    pub fn sdf_terrain_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .sdf_terrain
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_uniform.as_ref())
    }

    pub fn sdf_terrain_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .sdf_terrain
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn sdf_edit_bounds(&self) -> &[[f32; 4]] {
        &self.sdf_edit_bounds
    }

    pub fn sdf_content_generation(&self) -> u64 {
        self.sdf_content_generation
    }

    pub fn sdf_terrain_y_bounds(&self) -> Option<[f32; 2]> {
        self.sdf_terrain_y_bounds
    }

    pub fn sdf_requires_canonical_scan(&self) -> bool {
        self.sdf_requires_canonical_scan
    }

    pub fn coordinate_space_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .coordinate_spaces
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn coordinate_space_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .coordinate_spaces
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn material_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .materials
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn material_textures_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .material_textures
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn material_binding_key(
        &self,
        texture_epoch: u64,
    ) -> (u64, Option<u64>, Option<u64>) {
        (
            texture_epoch,
            self.canonical
                .materials
                .as_ref()
                .map(PublishedSceneBuffer::epoch),
            self.canonical
                .material_textures
                .as_ref()
                .map(PublishedSceneBuffer::epoch),
        )
    }

    pub fn object_spatial_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .object_spatial
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn object_render_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .object_render
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn light_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .lights
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn decal_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .decals
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn water_volume_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .water_volumes
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn water_volume_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .water_volumes
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn water_hitbox_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .water_hitboxes
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn water_hitbox_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .water_hitboxes
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn post_process_volume_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .post_process_volumes
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn reflection_capture_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .reflection_captures
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn planar_reflector_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .planar_reflectors
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn planar_reflector_buffer_epoch(&self) -> Option<u64> {
        self.canonical
            .planar_reflectors
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn voxel_volume_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .voxel_volumes
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn voxel_brick_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .voxel_bricks
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn voxel_data_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .voxel_data
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn voxel_palette_buffer(&self) -> &wgpu::Buffer {
        self.canonical
            .voxel_palettes
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref())
    }

    pub fn voxel_volume_epoch(&self) -> Option<u64> {
        self.canonical
            .voxel_volumes
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    pub fn voxel_residency_epoch(&self) -> Option<u64> {
        self.canonical
            .voxel_bricks
            .as_ref()
            .map(PublishedSceneBuffer::epoch)
    }

    /// Creates a new GPU scene.
    ///
    /// Initializes managers with default capacities (e.g., 1024 lights, 4096 meshes).
    /// Buffers are pre-allocated to avoid reallocation during gameplay.
    ///
    /// # Parameters
    ///
    /// - `device`: GPU device (wrapped in `Arc` for sharing)
    /// - `queue`: GPU queue (wrapped in `Arc` for sharing)
    ///
    /// # Performance
    ///
    /// - **O(1)**: Allocates buffers once at startup
    /// - **Pre-allocation**: Managers allocate initial capacity (e.g., 1024 lights)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_core::GpuScene;
    /// use std::sync::Arc;
    ///
    /// # fn example(device: wgpu::Device, queue: wgpu::Queue) {
    /// let scene = GpuScene::new(
    ///     Arc::new(device),
    ///     Arc::new(queue),
    /// );
    /// # }
    /// ```
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let camera = GpuCameraBuffer::new(&device);
        let object_history = GpuObjectHistoryBuffer::new(device.clone());
        let draw_calls = GpuDrawCallBuffer::new(device.clone());
        let light_projections = GrowableBuffer::new(
            device.clone(),
            256,
            wgpu::BufferUsages::STORAGE,
            "Light Projection Buffer",
        );
        let decal_indices = GrowableBuffer::new(
            device.clone(),
            1024,
            wgpu::BufferUsages::STORAGE,
            "Decal Active Row Buffer",
        );
        let water_volume_projections = GrowableBuffer::new(
            device.clone(),
            64,
            wgpu::BufferUsages::STORAGE,
            "Water Volume Projection Buffer",
        );
        let water_hitbox_indices = GrowableBuffer::new(
            device.clone(),
            256,
            wgpu::BufferUsages::STORAGE,
            "Water Hitbox Active Row Buffer",
        );
        let post_process_volume_indices = GrowableBuffer::new(
            device.clone(),
            64,
            wgpu::BufferUsages::STORAGE,
            "Post Process Volume Active Row Buffer",
        );
        let reflection_capture_projections = GrowableBuffer::new(
            device.clone(),
            64,
            wgpu::BufferUsages::STORAGE,
            "Reflection Capture Projection Buffer",
        );
        let planar_reflector_indices = GrowableBuffer::new(
            device.clone(),
            1,
            wgpu::BufferUsages::STORAGE,
            "Planar Reflector Active Row Buffer",
        );
        let voxel_volume_indices = GrowableBuffer::new(
            device.clone(),
            16,
            wgpu::BufferUsages::STORAGE,
            "Voxel Volume Active Row Buffer",
        );
        let voxel_mesh_work = GrowableBuffer::new(
            device.clone(),
            helio_voxel_core::MAX_VOLUMES as usize,
            wgpu::BufferUsages::STORAGE,
            "Voxel Mesh Work Projection",
        );
        let foliage_type_indices = GrowableBuffer::new(
            device.clone(),
            256,
            wgpu::BufferUsages::UNIFORM,
            "Foliage Type Row Projection",
        );
        let foliage_layer_projections = GrowableBuffer::new(
            device.clone(),
            64,
            wgpu::BufferUsages::STORAGE,
            "Foliage Layer Projection",
        );
        let foliage_layer_type_relations = GrowableBuffer::new(
            device.clone(),
            256,
            wgpu::BufferUsages::STORAGE,
            "Foliage Layer Type Relations",
        );
        let foliage_interactor_indices = GrowableBuffer::new(
            device.clone(),
            256,
            wgpu::BufferUsages::STORAGE,
            "Foliage Interactor Row Projection",
        );
        let empty_scene_rows = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empty Canonical Scene Rows"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let empty_scene_uniform = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empty Canonical Scene Uniform"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }));
        let shadow_matrices = GpuShadowMatrixBuffer::new(device.clone());
        let indirect = GpuIndirectBuffer::new(device.clone());
        let visibility = GpuVisibilityBuffer::new(device.clone());
        let source_indices = GpuSourceIndicesBuffer::new(device.clone());
        let shadow_static_source_indices = GrowableBuffer::new(
            device.clone(),
            4096,
            wgpu::BufferUsages::STORAGE,
            "Static Shadow Source Indices",
        );
        let shadow_movable_source_indices = GrowableBuffer::new(
            device.clone(),
            4096,
            wgpu::BufferUsages::STORAGE,
            "Movable Shadow Source Indices",
        );
        let compacted_indices = GpuCompactedIndicesBuffer::new(device.clone());
        let compacted_indices_2 = GpuCompactedIndices2Buffer::new(device.clone());
        let coordinate_space_history = CoordinateSpaceHistory::new(&device);
        let shadow_static_indirect = GpuIndirectBuffer::new(device.clone());
        let shadow_movable_indirect = GpuIndirectBuffer::new(device.clone());

        let portal_views = GrowableBuffer::new(
            device.clone(),
            8,
            wgpu::BufferUsages::STORAGE,
            "Portal Views Buffer",
        );
        let portal_chains = GrowableBuffer::new(
            device.clone(),
            32,
            wgpu::BufferUsages::STORAGE,
            "Portal Chains Buffer",
        );

        let device_for_rt = Arc::clone(&device);

        Self {
            device,
            queue,
            frame_count: 0,
            width: 0,
            height: 0,
            movable_objects_generation: 0,
            movable_lights_generation: 0,
            camera_generation: 0,
            static_objects_generation: 0,
            camera,
            canonical: CanonicalSceneBuffers::default(),
            empty_scene_rows,
            empty_scene_uniform,
            sdf_edit_bounds: Vec::new(),
            sdf_content_generation: 0,
            sdf_terrain_y_bounds: None,
            sdf_requires_canonical_scan: false,
            planet_frames: Vec::new(),
            planet_frame_authority_epoch: 0,
            planet_frame_content_generation: 0,
            object_history,
            draw_calls,
            draw_material_rows: Vec::new(),
            draw_topology_generation: 0,
            light_projections,
            decal_indices,
            water_volume_projections,
            water_sim_slot_generations: [0; WATER_SIM_SLOT_COUNT],
            water_hitbox_indices,
            post_process_volume_indices,
            reflection_capture_projections,
            planar_reflector_indices,
            voxel_volume_indices,
            voxel_mesh_work,
            voxel_mesh_work_generation: 0,
            voxel_mesh_draw_count: 0,
            foliage_type_indices,
            foliage_layer_projections,
            foliage_layer_type_relations,
            foliage_interactor_indices,
            shadow_matrices,
            active_shadow_face_count: 0,
            indirect,
            visibility,
            source_indices,
            shadow_static_source_indices,
            shadow_movable_source_indices,
            compacted_indices,
            compacted_indices_2,
            coordinate_space_history,
            shadow_static_indirect,
            shadow_movable_indirect,
            shadow_static_draw_count: 0,
            shadow_movable_draw_count: 0,
            shadow_movable_topology_generation: 0,
            movable_light_count: 0,
            per_caster_dirty_gen: [1u64; 42],
            material_class_ranges: Vec::new(),
            transparent_material_class_ranges: Vec::new(),
            forward_material_class_ranges: Vec::new(),
            material_flags: Vec::new(),
            graph_wgsl_snippets: std::collections::HashMap::new(),
            graph_wgsl_epoch: 0,
            template_registry: None,
            transparent_template_registry: None,
            portal_views,
            portal_chains,
            blas_manager: BlasManager::new(device_for_rt.clone()),
            tlas_manager: TlasManager::new(device_for_rt, 65536),
        }
    }

    /// Returns zero-copy references to GPU resources.
    ///
    /// Creates a `SceneResources` struct with borrowed references to all GPU buffers.
    /// Passes receive this struct via `PassContext::scene`.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns borrowed references (no clones)
    /// - **Zero-copy**: All fields are `&wgpu::Buffer` references
    ///
    /// # Lifetime
    ///
    /// The returned `SceneResources<'_>` borrows `self`, ensuring buffers are not
    /// freed while passes are using them.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_core::GpuScene;
    /// # use std::sync::Arc;
    /// # fn example(device: wgpu::Device, queue: wgpu::Queue) {
    /// # let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
    /// let resources = scene.resources();
    /// let light_rows: &wgpu::Buffer = resources.lights;
    /// let material_rows: &wgpu::Buffer = resources.material_buffer();
    /// # }
    /// ```
    pub fn resources(&self) -> SceneResources<'_> {
        let object_spatial = self
            .canonical
            .object_spatial
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref());
        let object_render = self
            .canonical
            .object_render
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref());
        let lights = self
            .canonical
            .lights
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref());
        let decals = self
            .canonical
            .decals
            .as_ref()
            .map(PublishedSceneBuffer::buffer)
            .unwrap_or(self.empty_scene_rows.as_ref());
        let water_volumes = self.water_volume_buffer();
        let water_hitboxes = self.water_hitbox_buffer();
        let post_process_volumes = self.post_process_volume_buffer();
        let reflection_captures = self.reflection_capture_buffer();
        let planar_reflectors = self.planar_reflector_buffer();
        let materials = self.material_buffer();
        let material_textures = self.material_textures_buffer();
        let material_buffer_epoch = self
            .canonical
            .materials
            .as_ref()
            .map(PublishedSceneBuffer::epoch);
        let material_textures_buffer_epoch = self
            .canonical
            .material_textures
            .as_ref()
            .map(PublishedSceneBuffer::epoch);
        let water_volume_projection_data = self.water_volume_projections.as_slice();
        let water_volume_count = water_volume_projection_data
            .len()
            .min(WATER_SIM_SLOT_COUNT);
        debug_assert!(water_volume_projection_data[..water_volume_count]
            .iter()
            .all(|projection| projection[1] < WATER_SIM_SLOT_COUNT as u32));
        debug_assert!(water_volume_projection_data[water_volume_count..]
            .iter()
            .all(|projection| projection[1] == WATER_SIM_SLOT_UNASSIGNED));

        SceneResources {
            camera: self.camera.buffer(),
            object_spatial,
            object_render,
            object_history: self.object_history.buffer(),
            draw_calls: self.draw_calls.buffer(),
            lights,
            light_projections: self.light_projections.buffer(),
            decals,
            decal_indices: self.decal_indices.buffer(),
            decal_count: self.decal_indices.len() as u32,
            water_volumes,
            water_volume_buffer_epoch: self.water_volume_buffer_epoch(),
            water_volume_projections: self.water_volume_projections.buffer(),
            water_volume_projection_epoch: self.water_volume_projections.buffer_version(),
            water_volume_projection_data,
            water_sim_slot_generations: &self.water_sim_slot_generations,
            water_volume_count: water_volume_count as u32,
            water_hitboxes,
            water_hitbox_buffer_epoch: self.water_hitbox_buffer_epoch(),
            water_hitbox_indices: self.water_hitbox_indices.buffer(),
            water_hitbox_projection_epoch: self.water_hitbox_indices.buffer_version(),
            water_hitbox_count: self.water_hitbox_indices.len() as u32,
            post_process_volumes,
            post_process_volume_indices: self.post_process_volume_indices.buffer(),
            post_process_volume_count: self.post_process_volume_indices.len() as u32,
            materials,
            material_textures,
            material_buffer_epoch,
            material_textures_buffer_epoch,
            shadow_matrices: self.shadow_matrices.buffer(),
            indirect: self.indirect.buffer(),
            visibility: self.visibility.buffer(),
            source_indices: self.source_indices.buffer(),
            shadow_static_source_indices: self.shadow_static_source_indices.buffer(),
            shadow_movable_source_indices: self.shadow_movable_source_indices.buffer(),
            compacted_indices: self.compacted_indices.buffer(),
            compacted_indices_2: self.compacted_indices_2.buffer(),
            coordinate_spaces: self.coordinate_space_buffer(),
            coordinate_spaces_prev: self.coordinate_space_history.prev_buffer(),
            // Draw topology is compact and Helio-owned. Canonical partner rows
            // use stable component-local addressing, so their allocation span
            // is not the number of live render instances.
            instance_count: self.source_indices.len() as u32,
            draw_count: self.draw_calls.len() as u32,
            light_count: self.movable_light_count,
            shadow_count: self.active_shadow_face_count,
            movable_objects_generation: self.movable_objects_generation,
            movable_lights_generation: self.movable_lights_generation,
            camera_generation: self.camera_generation,
            shadow_static_indirect: self.shadow_static_indirect.buffer(),
            shadow_movable_indirect: self.shadow_movable_indirect.buffer(),
            shadow_static_draw_count: self.shadow_static_draw_count,
            shadow_movable_draw_count: self.shadow_movable_draw_count,
            shadow_movable_topology_generation: self.shadow_movable_topology_generation,
            movable_light_count: self.movable_light_count,
            static_objects_generation: self.static_objects_generation,
            per_caster_dirty_gen: self.per_caster_dirty_gen,
            voxel_volumes: self.voxel_volume_buffer(),
            voxel_volume_epoch: self.voxel_volume_epoch(),
            voxel_volume_indices: self.voxel_volume_indices.buffer(),
            voxel_brick_pool: self.voxel_brick_buffer(),
            voxel_data_pool: self.voxel_data_buffer(),
            voxel_palette_pool: self.voxel_palette_buffer(),
            voxel_residency_epoch: self.voxel_residency_epoch(),
            voxel_volume_count: self.voxel_volume_indices.len() as u32,
            voxel_mesh_work: self.voxel_mesh_work.buffer(),
            voxel_mesh_work_epoch: self.voxel_mesh_work.buffer_version(),
            voxel_mesh_work_generation: self.voxel_mesh_work_generation,
            voxel_mesh_work_count: self.voxel_mesh_work.len() as u32,
            voxel_mesh_draw_count: self.voxel_mesh_draw_count,
            sdf_edits: self.sdf_edit_buffer(),
            sdf_edit_buffer_epoch: self.sdf_edit_buffer_epoch(),
            sdf_terrain: self.sdf_terrain_buffer(),
            sdf_terrain_buffer_epoch: self.sdf_terrain_buffer_epoch(),
            sdf_edit_count: self.sdf_edit_count(),
            sdf_content_generation: self.sdf_content_generation,
            sdf_edit_bounds: &self.sdf_edit_bounds,
            sdf_terrain_y_bounds: self.sdf_terrain_y_bounds,
            sdf_requires_canonical_scan: self.sdf_requires_canonical_scan,
            material_class_ranges: &self.material_class_ranges,
            transparent_material_class_ranges: &self.transparent_material_class_ranges,
            forward_material_class_ranges: &self.forward_material_class_ranges,
            graph_wgsl_snippets: &self.graph_wgsl_snippets,
            graph_wgsl_epoch: self.graph_wgsl_epoch,
            template_registry: &self.template_registry,
            transparent_template_registry: &self.transparent_template_registry,
            reflection_captures,
            reflection_capture_projections: self.reflection_capture_projections.buffer(),
            reflection_capture_count: self.reflection_capture_projections.len() as u32,
            planar_reflectors,
            planar_reflector_buffer_epoch: self.planar_reflector_buffer_epoch(),
            planar_reflector_indices: self.planar_reflector_indices.buffer(),
            planar_reflector_projection_epoch: self.planar_reflector_indices.buffer_version(),
            planar_reflector_count: self.planar_reflector_indices.len() as u32,
            portal_views: self.portal_views.buffer(),
            portal_view_count: self.portal_views.len() as u32,
            portal_chains: self.portal_chains.buffer(),
            portal_chain_count: self.portal_chains.len() as u32,
            rt_available: self.tlas_manager.is_rt_available(),
        }
    }

    /// Flushes dirty Helio-owned render projections and temporal buffers.
    ///
    /// Canonical SceneDB component buffers are flushed and published separately;
    /// this method does not copy their authored rows into another scene store.
    ///
    /// # Performance
    ///
    /// - Dirty managers upload their changed ranges.
    /// - Clean managers emit no queue writes, though bounded checks still run.
    /// - Capacity growth and projection rebuilds can allocate outside the clean
    ///   steady-state path.
    ///
    /// # Usage
    ///
    /// Call `flush()` as needed **before** `RenderGraph::execute()`, then call
    /// [`Self::advance_frame`] exactly once after the rendered frame:
    ///
    /// ```rust,no_run
    /// # use helio_core::{GpuScene, RenderGraph};
    /// # use std::sync::Arc;
    /// # fn example(
    /// #     device: Arc<wgpu::Device>,
    /// #     queue: Arc<wgpu::Queue>,
    /// #     view: wgpu::TextureView,
    /// #     depth_view: wgpu::TextureView,
    /// # ) {
    /// # let mut scene = GpuScene::new(device.clone(), queue.clone());
    /// # let mut graph = RenderGraph::new(&device, &queue);
    /// # let target = &view;
    /// # let depth = &depth_view;
    /// // Flush Helio-owned derived changes before graph execution.
    /// scene.flush();
    ///
    /// // Execute render graph (passes read GPU buffers)
    /// // graph.execute(&scene, target, depth);
    /// scene.advance_frame();
    /// # }
    /// ```
    ///
    pub fn flush(&mut self) {
        let queue: &wgpu::Queue = &self.queue;
        self.camera.flush(queue);
        self.object_history.flush(queue);
        self.draw_calls.flush(queue);
        self.light_projections.flush(queue);
        self.decal_indices.flush(queue);
        self.water_volume_projections.flush(queue);
        self.water_hitbox_indices.flush(queue);
        self.post_process_volume_indices.flush(queue);
        self.reflection_capture_projections.flush(queue);
        self.planar_reflector_indices.flush(queue);
        self.shadow_matrices.flush(queue);
        self.indirect.flush(queue);
        self.visibility.flush(queue);
        self.source_indices.flush(queue);
        self.shadow_static_source_indices.flush(queue);
        self.shadow_movable_source_indices.flush(queue);
        self.compacted_indices.flush(queue);
        self.compacted_indices_2.flush(queue);
        self.coordinate_space_history.flush(queue);
        self.shadow_static_indirect.flush(queue);
        self.shadow_movable_indirect.flush(queue);
        self.voxel_volume_indices.flush(queue);
        self.voxel_mesh_work.flush(queue);
        self.foliage_type_indices.flush(queue);
        self.foliage_layer_projections.flush(queue);
        self.foliage_layer_type_relations.flush(queue);
        self.foliage_interactor_indices.flush(queue);
        self.portal_views.flush(queue);
        self.portal_chains.flush(queue);

    }

    /// Commit the just-rendered current values as next frame's temporal
    /// history and advance the shader-visible frame counter.
    ///
    /// This is deliberately separate from [`Self::flush`]: callers may flush
    /// repeatedly before one render without collapsing current and previous.
    pub fn advance_frame(&mut self) {
        self.object_history.cycle_current();
        self.coordinate_space_history.cycle_current();
        self.frame_count = self.frame_count.wrapping_add(1);
    }
}
