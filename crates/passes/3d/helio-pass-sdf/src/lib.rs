pub mod edit_list;
pub mod gpu_bvh;
pub mod noise;
pub mod primitives;
pub mod rendering;
pub mod terrain;
pub mod uniforms;

pub use edit_list::{BooleanOp, GpuSdfEdit, SdfEdit, SdfEditId};
pub use primitives::{SdfShapeParams, SdfShapeType};
pub use terrain::{GpuTerrainParams, TerrainConfig, TerrainStyle};
pub use uniforms::SdfGridParams;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

pub(crate) const INITIAL_BVH_CAPACITY: usize = 2048;
pub(crate) const MAX_BRICKS_PER_LEVEL: u32 = 4096;
/// Classify consumes the complete default WebGPU compute-stage storage budget.
pub const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE: u32 = 8;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Compatibility alias; CPU picking now lives on SceneDB's authored authority
/// (`Scene::pick_sdf_surface` in the high-level Helio API).
pub use helio_scenedb::SdfPickResult as PickResult;

/// Checked construction failures for the fixed-stride SDF clipmap layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfPassConfigError {
    GridNotBrickAligned,
    BrickCapacityExceeded { requested: u32, maximum: u32 },
    InvalidVolumeBounds,
    DeviceStorageLimit,
}

impl std::fmt::Display for SdfPassConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GridNotBrickAligned => formatter.write_str(
                "SDF grid dimension must be a non-zero multiple of the 8-voxel brick size",
            ),
            Self::BrickCapacityExceeded { requested, maximum } => write!(
                formatter,
                "SDF grid requires {requested} bricks per level; the checked maximum is {maximum}",
            ),
            Self::InvalidVolumeBounds => {
                formatter.write_str("SDF volume bounds must be finite and strictly increasing")
            }
            Self::DeviceStorageLimit => formatter.write_str(
                "SDF clipmap allocation exceeds the device storage-buffer binding limit",
            ),
        }
    }
}

impl std::error::Error for SdfPassConfigError {}

// ═══════════════════════════════════════════════════════════════════════════════
// SdfPass — fully GPU-native SDF render pass
// ═══════════════════════════════════════════════════════════════════════════════

pub struct SdfPass {
    pub(crate) scroll_pipeline: wgpu::ComputePipeline,
    pub(crate) classify_pipeline: wgpu::ComputePipeline,
    pub(crate) eval_pipeline: wgpu::ComputePipeline,
    pub(crate) march_pipeline: wgpu::RenderPipeline,
    pub(crate) march_bgl: wgpu::BindGroupLayout,
    pub(crate) bvh_nodes_buffer: wgpu::Buffer,
    pub(crate) scroll_state_buffer: wgpu::Buffer,
    pub(crate) dirty_flags_buffer: wgpu::Buffer,
    pub(crate) clip_config_buffer: wgpu::Buffer,
    pub(crate) per_brick_hashes_buffer: wgpu::Buffer,
    pub(crate) per_brick_edit_lists_buffer: wgpu::Buffer,
    pub(crate) all_brick_indices_buffer: wgpu::Buffer,
    pub(crate) dirty_bricks_buffer: wgpu::Buffer,
    pub(crate) eval_indirect_buffer: wgpu::Buffer,
    pub(crate) eval_indirect_template_buffer: wgpu::Buffer,
    pub(crate) atlas_buffer: wgpu::Buffer,
    pub(crate) atlas_level_byte_stride: u64,
    pub(crate) atlas_words_per_level: u32,
    pub(crate) level_params_buffers: Vec<wgpu::Buffer>,
    pub(crate) scroll_bg: Option<wgpu::BindGroup>,
    pub(crate) scroll_bg_camera_key: usize,
    pub(crate) classify_bg: wgpu::BindGroup,
    pub(crate) eval_bgs: Vec<wgpu::BindGroup>,
    pub(crate) march_bg: Option<wgpu::BindGroup>,
    pub(crate) march_bg_camera_key: usize,
    pub(crate) last_gen: u64,
    pub(crate) bound_edit_epoch: Option<u64>,
    pub(crate) bound_terrain_epoch: Option<u64>,
    pub(crate) edit_generation: u32,
    pub(crate) bindings_dirty: bool,
    pub(crate) debug_mode: bool,
    pub(crate) enabled: bool,
    pub(crate) preserve_framebuffer: bool,
    pub(crate) level_count: u32,
    pub(crate) bricks_per_level: u32,
    pub(crate) brick_grid_dim: u32,
    pub(crate) brick_size: u32,
    pub(crate) grid_dim: u32,
    pub(crate) base_voxel_size: f32,
    pub(crate) volume_min: [f32; 3],
    pub(crate) volume_max: [f32; 3],
    pub(crate) cached_snap_origins: [[i32; 3]; 8],
    pub(crate) gpu_passes_clean: bool,
    pub(crate) authority_bound: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

impl SdfPass {
    pub fn toggle_debug(&mut self) {
        self.debug_mode = !self.debug_mode;
        self.last_gen = u64::MAX;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_preserve_framebuffer(&mut self, preserve: bool) {
        self.preserve_framebuffer = preserve;
    }

    pub fn preserve_framebuffer(&self) -> bool {
        self.preserve_framebuffer
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    pub(crate) fn voxel_size_for_level(&self, level: u32) -> f32 {
        self.base_voxel_size * (1u32 << level) as f32
    }

    pub(crate) fn build_clip_config(
        &self,
        edit_count: u32,
        bvh_node_count: u32,
        terrain_y_bounds: Option<[f32; 2]>,
        canonical_order_scan: bool,
    ) -> crate::rendering::GpuClipConfig {
        let mut voxel_sizes_lo = [0.0f32; 4];
        let mut voxel_sizes_hi = [0.0f32; 4];
        for i in 0..4usize {
            voxel_sizes_lo[i] = self.voxel_size_for_level(i as u32);
            voxel_sizes_hi[i] = self.voxel_size_for_level((4 + i) as u32);
        }
        let (terrain_enabled, terrain_y_min, terrain_y_max) = match terrain_y_bounds {
            Some([minimum, maximum]) => (1u32, minimum, maximum),
            None => (0u32, -1e10, 1e10),
        };
        crate::rendering::GpuClipConfig {
            level_count: self.level_count,
            grid_dim: self.grid_dim,
            brick_size: self.brick_size,
            brick_grid_dim: self.brick_grid_dim,
            bricks_per_level: self.bricks_per_level,
            atlas_bricks_per_axis: self.brick_grid_dim,
            base_voxel_size: self.base_voxel_size,
            edit_count,
            bvh_node_count,
            terrain_enabled,
            terrain_y_min,
            terrain_y_max,
            atlas_words_per_level: self.atlas_words_per_level,
            canonical_order_scan: u32::from(canonical_order_scan),
            _pad2: 0,
            _pad3: 0,
            voxel_sizes_lo,
            voxel_sizes_hi,
        }
    }

    pub(crate) fn build_level_params(&self, level: u32, edit_count: u32) -> SdfGridParams {
        let vs = self.voxel_size_for_level(level);
        let max_march_dist = self.grid_dim as f32 * vs * 2.0;
        SdfGridParams {
            volume_min: self.volume_min,
            _pad0: 0.0,
            volume_max: self.volume_max,
            _pad1: 0.0,
            grid_dim: self.grid_dim,
            edit_count,
            voxel_size: vs,
            max_march_dist,
            brick_size: self.brick_size,
            brick_grid_dim: self.brick_grid_dim,
            level_idx: level,
            atlas_bricks_per_axis: self.brick_grid_dim,
            grid_origin: [0.0; 3],
            debug_flags: if self.debug_mode { 1 } else { 0 },
            bricks_per_level: self.bricks_per_level,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        }
    }
}
