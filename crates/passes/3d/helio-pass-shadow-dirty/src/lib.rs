//! GPU-driven per-face shadow dirty detection.
//!
//! Runs as a compute pass immediately after `ShadowMatrixPass`. For each movable
//! shadow-caster object, it compares the object's current world-space position
//! with the stored previous-frame position.  If the object moved, it sphere-tests
//! the object's bounding sphere against every active shadow-face frustum (planes
//! extracted from the VP matrix via Gribb-Hartmann).  Any intersecting face is
//! marked dirty in a GPU buffer that `ShadowPass` reads directly via
//! `multi_draw_indexed_indirect_count` — no CPU readback and no O(N·M) CPU loop.
//!
//! # Architecture
//!
//! ```text
//! ShadowMatrixPass  ─writes─►  shadow_mats (VP per face)
//!                   ─writes─►  light_dirty (per-caster matrix changes)
//!        ↓
//! ShadowDirtyPass   ─reads──►  spatial rows, source rows, history, shadow_mats
//!                   ─writes─►  face_dirty[256]     (0/1, is this face dirty?)
//!        ↓
//! ShadowPass        ─reads──►  face_dirty (as clear-draw indirect count)
//! ShadowCullPass    ─writes─►  per-face compacted commands and counts
//! ```
//!
//! # Granularity
//!
//! The dirty check is **per shadow face**, not per caster.  A spinning object on the
//! +X side of a point light does NOT re-render the -X, ±Y, ±Z cube faces.
//!
//! # Topology changes
//!
//! When movable membership or draw batching changes between frames,
//! the pass sets `force_dirty_all = 1` in its uniform buffer, causing the shader to
//! dirty every active face. Temporal history is advanced separately at the rendered
//! frame boundary by `GpuScene`.
//! Subsequent frames return to normal per-object dirty detection.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum shadow atlas faces.  Must match `MAX_FACES` in the WGSL shader and
/// `MAX_SHADOW_FACES` in `helio-pass-shadow`.
const MAX_SHADOW_FACES: usize = 256;

const WORKGROUP_SIZE: u32 = 64;

// ── Uniforms ──────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowDirtyUniforms {
    movable_object_count: u32,
    face_count: u32,
    /// 1 when movable object/draw topology changes — forces all faces dirty.
    force_dirty_all: u32,
    _pad: u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowDirtyPass {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,

    /// Uniform buffer holding per-frame parameters.
    uniform_buf: wgpu::Buffer,

    /// Per-face dirty flag: 0 = clean, 1 = dirty (atomic u32 array, 256 entries).
    /// Shared with `ShadowPass` — published via `Arc` so the shadow pass can bind it.
    pub face_dirty_buf: Arc<wgpu::Buffer>,

    /// Per-caster flags written by ShadowMatrixPass when a light matrix changes.
    light_dirty_buf: Arc<wgpu::Buffer>,

    /// Bind group (lazy; rebuilt whenever the `instances` or `shadow_mats` buffer
    /// pointer changes due to `GrowableBuffer` reallocation).
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize, usize, usize, usize, usize)>,

    /// Movable source/draw topology seen last frame.
    last_topology: (u64, u32, u32),
    movable_object_count: u32,
}

impl ShadowDirtyPass {
    /// Allocate all GPU resources.  Pass the shared buffers to `ShadowPass::new()`.
    pub fn new(device: &wgpu::Device, light_dirty_buf: Arc<wgpu::Buffer>) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowDirty Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_dirty.wgsl").into()),
        });

        // ── Bind Group Layout ─────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowDirty BGL"),
            entries: &[
                // 0: canonical object spatial rows
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: movable shadow slot -> component-local SceneObject row
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Helio-owned previous object spatial history
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: shadow_mats (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: face_dirty (read-write storage, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: current coordinate-space transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: per-caster light dirty flags from ShadowMatrixPass
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: previous-frame coordinate-space transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShadowDirty PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ShadowDirty Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Buffers ───────────────────────────────────────────────────────────

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/Uniforms"),
            size: std::mem::size_of::<ShadowDirtyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // face_dirty: one atomic<u32> per shadow face. Cleared by the command
        // encoder before the compute dispatch, which provides ordering across
        // every workgroup (a shader workgroup barrier cannot do that).
        let face_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowDirty/FaceDirty"),
            size: (MAX_SHADOW_FACES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Self {
            pipeline,
            bgl,
            uniform_buf,
            face_dirty_buf,
            light_dirty_buf,
            bind_group: None,
            bind_group_key: None,
            last_topology: (u64::MAX, u32::MAX, u32::MAX),
            movable_object_count: 0,
        }
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowDirtyPass {
    fn name(&self) -> &'static str {
        "ShadowDirty"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let movable_draw_count = ctx.scene.shadow_movable_draw_count;
        let movable_object_count = ctx.scene.shadow_movable_source_indices.len() as u32;
        let face_count = (ctx.scene.shadow_matrices.len() as u32).min(MAX_SHADOW_FACES as u32);

        // Detect object or mesh-batch topology changes.
        let topology = (
            ctx.scene.shadow_movable_topology_generation,
            movable_object_count,
            movable_draw_count,
        );
        let force_dirty_all = if topology != self.last_topology {
            self.last_topology = topology;
            1u32
        } else {
            0u32
        };
        self.movable_object_count = movable_object_count;

        let u = ShadowDirtyUniforms {
            movable_object_count,
            face_count,
            force_dirty_all,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let movable_object_count = self.movable_object_count;
        let face_count = ctx.scene.shadow_count;

        if face_count == 0 {
            return Ok(());
        }

        // ── Lazy bind group rebuild on GrowableBuffer reallocation ─────────────
        let spatial_ptr = ctx.scene.object_spatial as *const _ as usize;
        let source_ptr = ctx.scene.shadow_movable_source_indices as *const _ as usize;
        let history_ptr = ctx.scene.object_history as *const _ as usize;
        let sm_ptr = ctx.scene.shadow_matrices as *const _ as usize;
        let ld_ptr = &*self.light_dirty_buf as *const _ as usize;
        let spaces_ptr = ctx.scene.coordinate_spaces as *const _ as usize;
        let spaces_prev_ptr = ctx.scene.coordinate_spaces_prev as *const _ as usize;
        let key = (
            spatial_ptr,
            source_ptr,
            history_ptr,
            sm_ptr,
            ld_ptr,
            spaces_ptr,
            spaces_prev_ptr,
        );

        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ShadowDirty BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.object_spatial.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.shadow_movable_source_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.object_history.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.shadow_matrices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.face_dirty_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.scene.coordinate_spaces.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.light_dirty_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: ctx.scene.coordinate_spaces_prev.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let bg = self.bind_group.as_ref().unwrap();

        // Reset the complete output arrays before dispatch. Doing this as
        // encoder commands avoids the cross-workgroup race that occurs when
        // invocation zero clears storage while other workgroups write it.
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        encoder.clear_buffer(&self.face_dirty_buf, 0, None);

        // Dispatch enough threads to cover all movable draw calls.
        // Dispatch at least one thread so topology changes with an empty
        // movable set still pass through the force-dirty path.
        let thread_count = movable_object_count.max(1);
        let workgroups = thread_count.div_ceil(WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ShadowDirty"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}
