//! Zero-copy scene resource references.
//!
//! `SceneResources` provides borrowed references to canonical SceneDB partner
//! buffers and Helio-owned render projections. It is passed to render passes via
//! `PassContext::scene` without cloning those buffers.
//!
//! # Design Pattern: Zero-Copy Access
//!
//! The render graph assembles a short-lived borrowed view:
//!
//! ```text
//! SceneDB World/subsystems ──publish──> canonical GPU partner buffers
//! Helio derived state      ──flush────> compact/history/pass buffers
//!                                      │
//!                                      └── SceneResources<'a>
//! ```
//!
//! # Lifetime
//!
//! The `'a` lifetime ensures that all borrowed references outlive the context. This prevents
//! dangling references and ensures safety without runtime overhead.
//!
//! # Performance
//!
//! - **O(1)**: Creating `SceneResources` is constant-time (no allocations)
//! - **Zero clones**: All fields are references (`&`)
//! - **Borrowed view**: Construction itself takes no scene-data lock
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
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
//!         // Borrow current canonical/derived buffers.
//!         let light_buffer: &wgpu::Buffer = ctx.scene.lights;
//!         let material_buffer: &wgpu::Buffer = ctx.scene.material_buffer();
//!
//!         // Use buffers in bind groups (no clones)
//!         // let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
//!         //     layout: &layout,
//!         //     entries: &[
//!         //         wgpu::BindGroupEntry {
//!         //             binding: 0,
//!         //             resource: light_buffer.as_entire_binding(),
//!         //         },
//!         //     ],
//!         //     label: Some("Scene Bind Group"),
//!         // });
//!
//!         Ok(())
//!     }
//! }
//! ```

/// Zero-copy references to GPU scene resources.
///
/// `SceneResources` provides borrowed references (`&`) to the GPU buffers
/// published for one graph execution. Some are canonical SceneDB component
/// partners; others are compact or temporal projections owned by Helio.
///
/// # Design
///
/// Component rows are addressed in their component-local SceneDB domain. They
/// are not indexed by `Entity::index()`; generation/liveness validation is a
/// separate explicitly indexed protocol. Renderer projections carry the
/// component row when they need to refer back to canonical data.
///
/// # Lifetime
///
/// The `'a` lifetime ties this struct to the `GpuScene` it was created from. This ensures
/// that buffers are not freed while passes are using them.
///
/// # Performance
///
/// - **O(1)**: Creating this struct is constant-time (no allocations)
/// - **Zero clones**: All fields are references
/// - **Borrowed view**: Construction itself takes no scene-data lock
///
/// # Example
///
/// ```rust,no_run
/// # use helio_core::{GpuScene, RenderPass, PassContext, Result};
/// # use std::sync::Arc;
/// # fn example(device: wgpu::Device, queue: wgpu::Queue) {
/// # let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
/// // Get zero-copy references
/// let resources = scene.resources();
///
/// let light_buffer: &wgpu::Buffer = resources.lights;
/// let material_buffer: &wgpu::Buffer = resources.material_buffer();
/// # }
/// ```
///
pub struct SceneResources<'a> {
    pub camera: &'a wgpu::Buffer,
    /// SceneDB-owned component-local authored spatial rows (144-byte stride).
    pub object_spatial: &'a wgpu::Buffer,
    /// SceneDB-owned component-local stable render-reference rows (16-byte stride).
    pub object_render: &'a wgpu::Buffer,
    /// Helio-owned previous-frame `{ model, sphere, flags }` rows keyed by the
    /// stable component-local `SceneObject` GPU row.
    pub object_history: &'a wgpu::Buffer,
    pub draw_calls: &'a wgpu::Buffer,
    /// SceneDB-owned component-local authored light rows.
    pub lights: &'a wgpu::Buffer,
    /// Compact realtime slot -> `{ light_row, shadow_index }`.
    pub light_projections: &'a wgpu::Buffer,
    /// SceneDB-owned component-local authored decal rows.
    pub decals: &'a wgpu::Buffer,
    /// Compact active decal slot -> component-local canonical decal row.
    pub decal_indices: &'a wgpu::Buffer,
    pub decal_count: u32,
    /// SceneDB-owned sparse authored water-volume rows.
    pub water_volumes: &'a wgpu::Buffer,
    /// Changes only when SceneDB reallocates the canonical water row buffer.
    pub water_volume_buffer_epoch: Option<u64>,
    /// Compact active water-volume slot -> component-local SceneDB row.
    pub water_volume_projections: &'a wgpu::Buffer,
    /// Helio projection-buffer allocation epoch (content edits do not change it).
    pub water_volume_projection_epoch: u64,
    /// CPU mirror of `water_volume_projections`, used only to select stable
    /// persistent simulation layers without a GPU readback.
    pub water_volume_projection_data: &'a [[u32; 2]],
    pub water_sim_slot_generations: &'a [u64; super::WATER_SIM_SLOT_COUNT],
    pub water_volume_count: u32,
    /// SceneDB-owned sparse authored water-hitbox rows.
    pub water_hitboxes: &'a wgpu::Buffer,
    pub water_hitbox_buffer_epoch: Option<u64>,
    /// Compact active water-hitbox slot -> component-local SceneDB row.
    pub water_hitbox_indices: &'a wgpu::Buffer,
    pub water_hitbox_projection_epoch: u64,
    pub water_hitbox_count: u32,
    /// SceneDB-owned sparse authored post-process-volume rows.
    pub post_process_volumes: &'a wgpu::Buffer,
    /// Compact active post-process-volume slot -> component-local SceneDB row.
    pub post_process_volume_indices: &'a wgpu::Buffer,
    pub post_process_volume_count: u32,
    /// SceneDB-owned 96-byte material shader rows, addressed by the stable
    /// component-local row resolved for `SceneMaterial`.
    pub(crate) materials: &'a wgpu::Buffer,
    /// SceneDB-owned 352-byte texture-selection shader rows, addressed by
    /// the same component-local material row.
    pub(crate) material_textures: &'a wgpu::Buffer,
    /// Allocation epochs, changed only when the corresponding SceneDB
    /// growable buffer is rebound. Material bind-group caches must include
    /// both values as well as the texture residency epoch.
    pub(crate) material_buffer_epoch: Option<u64>,
    pub(crate) material_textures_buffer_epoch: Option<u64>,
    pub shadow_matrices: &'a wgpu::Buffer,
    pub indirect: &'a wgpu::Buffer,
    pub visibility: &'a wgpu::Buffer,
    /// Helio-derived compact draw slot -> component-local SceneObject row.
    /// This indirection lets draw batches stay compact without repacking the
    /// persistent object columns owned by SceneDB.
    pub source_indices: &'a wgpu::Buffer,
    /// Static-shadow draw slot -> component-local SceneObject row.
    pub shadow_static_source_indices: &'a wgpu::Buffer,
    /// Movable-shadow draw slot -> component-local SceneObject row.
    pub shadow_movable_source_indices: &'a wgpu::Buffer,
    /// Per-draw-call-group compacted original instance slots surviving GPU
    /// frustum culling (see `IndirectDispatchPass`). Passes drawing through
    /// `indirect`/`draw_calls` should index canonical object rows through this
    /// buffer rather than using `instance_index` directly.
    pub compacted_indices: &'a wgpu::Buffer,
    /// Final surviving instance slots after frustum + Hi-Z occlusion culling.
    /// Consumers drawing through `indirect`/`draw_calls` should use this one,
    /// not `compacted_indices` (which is frustum-only, an intermediate stage).
    pub compacted_indices_2: &'a wgpu::Buffer,
    /// SceneDB-owned coordinate-space partner transforms (current frame).
    /// Component-local row 0 is the permanent identity. Shaders index this
    /// with the id packed into SceneObjectSpatial flags bits 8-15
    /// (`libhelio::coordinate_space`) to place sublevel/portal content.
    pub coordinate_spaces: &'a wgpu::Buffer,
    /// Coordinate-space transforms as of the previous frame — same indexing as
    /// `coordinate_spaces`, used to compute correct per-space motion vectors.
    pub coordinate_spaces_prev: &'a wgpu::Buffer,
    pub instance_count: u32,
    pub draw_count: u32,
    pub light_count: u32,
    pub shadow_count: u32,
    /// Generation counter for movable objects (increments when any Movable object moves)
    pub movable_objects_generation: u64,
    /// Generation counter for movable lights (increments when any Movable light moves)
    pub movable_lights_generation: u64,
    /// Generation counter for camera (increments when camera view/projection changes)
    pub camera_generation: u64,

    // ── Shadow partition buffers (Unreal-style static/dynamic split) ──────────
    // Both passes use canonical object rows — only the indirect call lists differ.
    /// Indirect draw commands for Static/Stationary objects.
    pub shadow_static_indirect: &'a wgpu::Buffer,
    /// Indirect draw commands for Movable objects.
    pub shadow_movable_indirect: &'a wgpu::Buffer,
    /// Number of draw calls in shadow_static_indirect.
    pub shadow_static_draw_count: u32,
    /// Number of draw calls in shadow_movable_indirect.
    pub shadow_movable_draw_count: u32,
    /// Changes whenever movable shadow membership/order/batching changes,
    /// including equal-count remove+insert replacement.
    pub shadow_movable_topology_generation: u64,
    /// Increments when static object topology changes; triggers static atlas re-render.
    pub static_objects_generation: u64,
    /// Number of movable lights in the lights buffer (static/stationary excluded from runtime).
    pub movable_light_count: u32,
    /// Per-caster authored-input fingerprints (one per caster slot, 42 max).
    /// ShadowPass compares these with its last-rendered values; object movement
    /// is tracked independently by the GPU dirty pass.
    pub per_caster_dirty_gen: [u64; 42],

    /// SceneDB-owned sparse authored voxel-volume rows.
    pub voxel_volumes: &'a wgpu::Buffer,
    pub voxel_volume_epoch: Option<u64>,
    /// Compact active voxel-volume slot -> component-local SceneDB row.
    pub voxel_volume_indices: &'a wgpu::Buffer,
    pub voxel_brick_pool: &'a wgpu::Buffer,
    pub voxel_data_pool: &'a wgpu::Buffer,
    pub voxel_palette_pool: &'a wgpu::Buffer,
    pub voxel_residency_epoch: Option<u64>,
    pub voxel_volume_count: u32,
    /// Helio-derived stable output-slot rows consumed by VoxelMeshPass.
    pub voxel_mesh_work: &'a wgpu::Buffer,
    pub voxel_mesh_work_epoch: u64,
    pub voxel_mesh_work_generation: u64,
    pub voxel_mesh_work_count: u32,
    pub voxel_mesh_draw_count: u32,

    /// SceneDB-owned ordered authored SDF rows and singleton terrain uniform.
    pub sdf_edits: &'a wgpu::Buffer,
    pub sdf_edit_buffer_epoch: Option<u64>,
    pub sdf_terrain: &'a wgpu::Buffer,
    pub sdf_terrain_buffer_epoch: Option<u64>,
    pub sdf_edit_count: u32,
    pub sdf_content_generation: u64,
    /// Narrow read-only snapshot used only for Helio's derived BVH rebuild.
    pub sdf_edit_bounds: &'a [[f32; 4]],
    pub sdf_terrain_y_bounds: Option<[f32; 2]>,
    /// Ordered streams containing intersection operands bypass brick-local
    /// edit lists and scan the canonical SceneDB buffer in authored order.
    pub sdf_requires_canonical_scan: bool,

    /// Material class ranges for the GBuffer pass: [(class, graph_hash, start, count), ...]
    /// Each range is uniform in both material_class and graph_hash so a single
    /// PSO works for all indirect entries it covers.
    /// Built during scene flush.
    pub material_class_ranges: &'a [(u32, u64, u32, u32)],
    pub transparent_material_class_ranges: &'a [(u32, u64, u32, u32)],
    /// Forward-shaded material class ranges (excluded from GBuffer pass).
    pub forward_material_class_ranges: &'a [(u32, u64, u32, u32)],

    /// Compiled graph WGSL snippets keyed by hash. Populated during flush.
    pub graph_wgsl_snippets: &'a std::collections::HashMap<u64, String>,
    /// Content epoch for the graph-source registry. This changes for insert,
    /// removal, and byte-different replacement under an existing hash.
    pub graph_wgsl_epoch: u64,

    /// Custom template registrations that survive graph rebuilds.
    /// GBufferPass downcasts to `RadiantTemplateRegistry` before each frame.
    pub template_registry: &'a Option<Box<dyn std::any::Any + Send + Sync>>,

    /// Separate template registry for transparent materials (water, glass, etc.).
    /// TransparentPass reads this instead of `template_registry` to avoid picking
    /// up gbuffer templates with incompatible bind group layouts.
    pub transparent_template_registry: &'a Option<Box<dyn std::any::Any + Send + Sync>>,

    /// SceneDB-owned sparse authored reflection-capture rows.
    pub reflection_captures: &'a wgpu::Buffer,
    /// Influence-sorted active slot -> `{ component_row, cubemap_layer }`.
    pub reflection_capture_projections: &'a wgpu::Buffer,
    /// Number of active reflection-capture projections.
    pub reflection_capture_count: u32,

    /// SceneDB-owned sparse authored planar-reflector rows.
    pub planar_reflectors: &'a wgpu::Buffer,
    /// SceneDB allocation epoch for the canonical row buffer.
    pub planar_reflector_buffer_epoch: Option<u64>,
    /// Compact active slot -> component-local canonical row.
    pub planar_reflector_indices: &'a wgpu::Buffer,
    /// Helio projection-buffer allocation epoch.
    pub planar_reflector_projection_epoch: u64,
    pub planar_reflector_count: u32,

    /// Active portals' render data (`libhelio::GpuPortalView`). Consumed by
    /// `helio-pass-portal-cull` / `helio-pass-portal-instances`.
    pub portal_views: &'a wgpu::Buffer,
    /// Number of active portals in `portal_views`.
    pub portal_view_count: u32,

    /// Every valid portal *chain* (`libhelio::GpuPortalChain`) up to
    /// `libhelio::MAX_CHAIN_DEPTH` deep — every sequence of portal indices,
    /// including repeats, that represents "look through this portal, then
    /// through this one, then...". This is what makes portals recursively
    /// reflect each other automatically: content is mapped through the
    /// *composed* transform of a whole chain, not just one portal, and each
    /// stage is independently clip-tested against its own portal's opening.
    /// Rebuilt whenever the portal set changes (add/remove/pose update), not
    /// every frame — see `helio::Scene::add_portal` and neighbors.
    pub portal_chains: &'a wgpu::Buffer,
    /// Number of valid chains in `portal_chains`.
    pub portal_chain_count: u32,

    /// Whether hardware ray tracing (TLAS + ray queries) is available.
    pub rt_available: bool,
}

impl SceneResources<'_> {
    pub fn material_buffer(&self) -> &wgpu::Buffer {
        self.materials
    }

    pub fn material_textures_buffer(&self) -> &wgpu::Buffer {
        self.material_textures
    }

    /// Complete cache key for the material bind group. `texture_epoch`
    /// covers view/sampler array membership; the other two terms cover
    /// independent SceneDB buffer growth.
    pub fn material_binding_key(
        &self,
        texture_epoch: u64,
    ) -> (u64, Option<u64>, Option<u64>) {
        (
            texture_epoch,
            self.material_buffer_epoch,
            self.material_textures_buffer_epoch,
        )
    }
}
