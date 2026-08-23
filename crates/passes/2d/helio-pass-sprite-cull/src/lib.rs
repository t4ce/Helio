//! GPU-side culling + depth sort for SceneDB's growable, handle-addressed
//! sprite component. The normal constructor consumes a [`SpriteBufferSource`]
//! published by `SpriteBatchPass`; this keeps the pass crate decoupled from
//! the batch implementation while still following SceneDB value, presence,
//! and optional runtime-projection allocation epochs.
//!
//! Runs as its own [`RenderPass`], added to the graph *before* the pass that
//! renders the culled/sorted result — this is what lets that pass's
//! `execute()` never know the visible instance count on the CPU at all.
//! Every frame, entirely on the GPU:
//!
//! 1. `cs_cull` (`shaders/sprite_cull.wgsl`) — one thread per addressable
//!    component-local row. Present + in-view rows are atomically compacted into `indices_a`/`keys_a`,
//!    and the surviving count is atomically accumulated directly into
//!    [`SpriteCullPass::indirect_buf`]'s `instance_count` field.
//! 2. `cs_prepare` (`shaders/sprite_sort.wgsl`) — a single thread. Turns that
//!    GPU-written visible count into `num_blocks` and an indirect dispatch
//!    arg buffer for step 3, so the sort's dispatch size tracks the actual
//!    per-frame visible count instead of the pool's worst-case capacity.
//! 3. 32 single-bit passes of a GPU LSD radix sort (`shaders/sprite_sort.wgsl`)
//!    over the compacted list — see that shader's module doc comment for the
//!    three-kernel (histogram / scan / scatter) design, and for why it's 32
//!    one-bit passes rather than 4 eight-bit-digit passes (a real stability
//!    bug in an earlier 8-bit version, caught by
//!    `tests/gpu_sort_validation.rs`).
//!
//! The CPU touches none of this per frame beyond ~97 tiny dispatches. The
//! cull launch follows SceneDB's current component-local row span and uses a
//! shader-side grid-stride loop if that span exceeds one device dispatch; the
//! 32 sort passes' `cs_histogram`/
//! `cs_scatter` dispatches are *indirect* (`dispatch_workgroups_indirect`),
//! sized by `cs_prepare` from the cull pass's GPU-written visible count —
//! not from `max_visible` (the pool-sized worst case). Without that,
//! zooming a camera in to where only a few thousand sprites are visible
//! would still dispatch enough workgroups to cover `max_visible` sprites on
//! every one of the 32 sort passes, every frame — the fixed ceiling has to
//! size the *buffers* (a real GPU allocation can't grow mid-frame), but the
//! *dispatch* should track what's actually there. There is no per-instance
//! CPU work in `prepare()`/`execute()` regardless of row-span size either way.
//!
//! The instance byte layout is the shader-exact 80-byte
//! `helio_scenedb::SceneSpriteRow` partner protocol shared by all three sprite
//! passes.
//!
//! SceneDB value/presence reallocations are followed by allocation epoch, so
//! reserving before graph construction is only a performance hint. The one
//! explicit capacity is `max_visible`: it sizes sort/draw scratch and excess
//! visible rows are transactionally counted in overflow telemetry.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result};
use helio_scenedb::SpriteBufferSource;
use std::fmt;
use std::sync::Arc;

const WG_SIZE: u32 = 256;
/// `DrawIndexedIndirectArgs` is 5 × u32 = 20 bytes: index_count,
/// instance_count, first_index, base_vertex, first_instance.
const INDIRECT_ARGS_SIZE: u64 = 24;
const INDIRECT_INSTANCE_COUNT_OFFSET: u64 = 4;
pub const INDIRECT_OVERFLOW_COUNT_OFFSET: u64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteCullError {
    ZeroMaxVisible,
    DeviceCapacityExceeded { requested: u32, maximum: u32 },
}

impl fmt::Display for SpriteCullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaxVisible => write!(f, "SpriteCullPass max_visible must be non-zero"),
            Self::DeviceCapacityExceeded { requested, maximum } => write!(
                f,
                "SpriteCullPass max_visible {requested} exceeds this device's sort domain {maximum}"
            ),
        }
    }
}

impl std::error::Error for SpriteCullError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    view_min: [f32; 2],
    view_max: [f32; 2],
    slot_count: u32,
    max_visible: u32,
    runtime_capacity: u32,
    dispatched_threads: u32,
}

const SORT_BITS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortUniforms {
    bit: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-side compute pass: culls a growable SceneDB sprite component against a view rect
/// and depth-sorts the survivors, entirely on the GPU. See the module doc
/// comment for the full design.
pub struct SpriteCullPass {
    slot_capacity: u32,
    max_visible: u32,
    max_cull_workgroups: u32,
    cull_pipeline: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    cull_uniform_buf: wgpu::Buffer,
    cull_bind_group: Option<wgpu::BindGroup>,
    buffer_source: SpriteBufferSource,
    bound_instances_epoch: u64,
    bound_presence_epoch: u64,
    bound_runtime_epoch: u64,
    runtime_capacity: u32,
    fallback_runtime_buf: wgpu::Buffer,
    cull_keys_buf: wgpu::Buffer,
    view_min: [f32; 2],
    view_max: [f32; 2],
    view_dirty: bool,

    prepare_pipeline: wgpu::ComputePipeline,
    prepare_bind_group: wgpu::BindGroup,

    hist_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    /// `FrameUniform{count,num_blocks}`, GPU-written once per frame by
    /// `cs_prepare` and read by every one of this frame's `SORT_BITS` sort
    /// passes (bound as both storage, for `cs_prepare`'s write, and uniform,
    /// for the histogram/scan/scatter kernels' reads).
    _frame_uniform_buf: wgpu::Buffer,
    /// `[num_blocks, 1, 1]`, GPU-written once per frame by `cs_prepare` — the
    /// indirect dispatch args `cs_histogram`/`cs_scatter` dispatch against,
    /// so their dispatch size tracks the real per-frame visible count.
    dispatch_args_buf: wgpu::Buffer,

    /// One bind group per bit pass — `bit` is baked in at construction, and
    /// every buffer reference is fixed for the pass's lifetime, so nothing
    /// here needs rebuilding per frame.
    hist_bind_groups: [wgpu::BindGroup; SORT_BITS],
    scatter_bind_groups: [wgpu::BindGroup; SORT_BITS],
    scan_bind_group: wgpu::BindGroup,

    /// Final draw-order buffer after `SORT_BITS` (even) bit passes — always
    /// lands back in the "a" ping-pong buffer; see `radix_sort_indices`'s
    /// CPU-side sibling in `helio-pass-sprite-batch` for the same parity
    /// argument.
    pub draw_order_buf: Arc<wgpu::Buffer>,
    /// `DrawIndexedIndirectArgs`, `instance_count` GPU-written by `cs_cull`.
    /// Bind this to `SpriteBatchPass::use_gpu_culling`.
    pub indirect_buf: Arc<wgpu::Buffer>,
}

impl SpriteCullPass {
    fn cull_dispatch_workgroups(&self) -> u32 {
        self.slot_capacity
            .div_ceil(WG_SIZE)
            .min(self.max_cull_workgroups)
    }

    fn cull_dispatched_threads(&self) -> u32 {
        self.cull_dispatch_workgroups().saturating_mul(WG_SIZE)
    }

    fn refresh_cull_binding(&mut self, device: &wgpu::Device) -> bool {
        let source = self.buffer_source.snapshot();
        let runtime_epoch = source.runtime.as_ref().map_or(0, |runtime| runtime.epoch);
        let runtime_capacity = source.runtime.as_ref().map_or(0, |runtime| runtime.row_capacity);
        let uniform_changed = self.slot_capacity != source.row_span
            || self.runtime_capacity != runtime_capacity;
        if self.cull_bind_group.is_none()
            || self.bound_instances_epoch != source.instances_epoch
            || self.bound_presence_epoch != source.presence_epoch
            || self.bound_runtime_epoch != runtime_epoch
        {
            let runtime_buffer = source
                .runtime
                .as_ref()
                .map_or(&self.fallback_runtime_buf, |runtime| &runtime.buffer);
            self.cull_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Cull BG"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.cull_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: source.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: source.presence.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.draw_order_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.cull_keys_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.indirect_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: runtime_buffer.as_entire_binding(),
                    },
                ],
            }));
            self.bound_instances_epoch = source.instances_epoch;
            self.bound_presence_epoch = source.presence_epoch;
            self.bound_runtime_epoch = runtime_epoch;
        }
        self.slot_capacity = source.row_span;
        self.runtime_capacity = runtime_capacity;
        uniform_changed
    }

    /// `instances_buf`/`alive_buf` must be
    /// [`crate::SpriteBatchPass::instances_buffer`]/
    /// [`crate::SpriteBatchPass::alive_buffer`] on the pass this culls for,
    /// sized for at least `slot_capacity` component rows. `max_visible` bounds how
    /// many sprites can be visible at once — size it for the worst case your
    /// scene can put on screen simultaneously, not the total row span.
    /// This compatibility constructor cannot observe later buffer replacement;
    /// SceneDB integrations should use [`Self::from_source`].
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances_buf: Arc<wgpu::Buffer>,
        alive_buf: Arc<wgpu::Buffer>,
        slot_capacity: u32,
        max_visible: u32,
    ) -> Self {
        Self::try_new(
            device,
            queue,
            instances_buf,
            alive_buf,
            slot_capacity,
            max_visible,
        )
        .expect("SpriteCullPass::new capacity is unsupported by this device")
    }

    pub fn try_new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances_buf: Arc<wgpu::Buffer>,
        alive_buf: Arc<wgpu::Buffer>,
        slot_capacity: u32,
        max_visible: u32,
    ) -> std::result::Result<Self, SpriteCullError> {
        let buffer_source = SpriteBufferSource::new(
            instances_buf.as_ref().clone(),
            0,
            alive_buf.as_ref().clone(),
            0,
            slot_capacity,
        );
        Self::try_from_source(device, queue, buffer_source, max_visible)
    }

    /// Construct against SceneDB's allocation-epoch-aware sprite
    /// publication. Value/presence/runtime buffer reallocations are rebound
    /// before dispatch; `max_visible` remains the explicit sort/draw domain.
    pub fn from_source(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_source: SpriteBufferSource,
        max_visible: u32,
    ) -> Self {
        Self::try_from_source(device, queue, buffer_source, max_visible)
            .expect("SpriteCullPass::from_source capacity is unsupported by this device")
    }

    pub fn try_from_source(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_source: SpriteBufferSource,
        max_visible: u32,
    ) -> std::result::Result<Self, SpriteCullError> {
        if max_visible == 0 {
            return Err(SpriteCullError::ZeroMaxVisible);
        }
        let maximum = maximum_visible_for_device(device);
        if max_visible > maximum {
            return Err(SpriteCullError::DeviceCapacityExceeded {
                requested: max_visible,
                maximum,
            });
        }
        let source = buffer_source.snapshot();
        let slot_capacity = source.row_span;
        let max_blocks = max_visible.div_ceil(WG_SIZE).max(1);

        // ── Cull ────────────────────────────────────────────────────────────
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite_cull.wgsl").into()),
        });
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Cull BGL"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
            ],
        });
        let cull_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Cull PL"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Cull Pipeline"),
            layout: Some(&cull_pl),
            module: &cull_shader,
            entry_point: Some("cs_cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cull_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Cull Uniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let keys_a = create_u32_buffer(device, "Sprite Sort Keys A", max_visible);
        let keys_b = create_u32_buffer(device, "Sprite Sort Keys B", max_visible);
        let indices_a = Arc::new(create_u32_buffer(device, "Sprite Sort Indices A", max_visible));
        let indices_b = create_u32_buffer(device, "Sprite Sort Indices B", max_visible);

        let indirect_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Indirect Draw Args"),
            size: INDIRECT_ARGS_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        // Static fields, written once: index_count=6 (one quad), instance_count=0
        // (reset every frame before culling), first_index=0, base_vertex=0,
        // first_instance=0.
        queue.write_buffer(
            &indirect_buf,
            0,
            bytemuck::cast_slice(&[6u32, 0, 0, 0, 0, 0]),
        );

        let fallback_runtime_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Cull Runtime Fallback"),
            size: 24,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let runtime_buffer = source
            .runtime
            .as_ref()
            .map_or(&fallback_runtime_buf, |runtime| &runtime.buffer);

        let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Cull BG"),
            layout: &cull_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cull_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source.instances.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: source.presence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: indices_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: keys_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: indirect_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: runtime_buffer.as_entire_binding() },
            ],
        });

        // ── Prepare (drives indirect dispatch sizing for the sort passes) ────
        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Sort Frame Uniform"),
            // FrameUniform{count,num_blocks} is 8 bytes in WGSL; pad the
            // allocation for backend safety margin, mirroring the old
            // CountUniform buffer this replaces.
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatch_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Sort Dispatch Args"),
            size: 12, // [num_blocks, 1, 1] as u32s
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // y/z workgroup counts are always 1 — only `cs_prepare` ever updates
        // [0] again, once per frame.
        queue.write_buffer(&dispatch_args_buf, 0, bytemuck::cast_slice(&[max_blocks, 1u32, 1u32]));

        let prepare_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Prepare BGL"),
            entries: &[storage_entry(0, true), storage_entry(1, false), storage_entry(2, false)],
        });
        let prepare_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Prepare PL"),
            bind_group_layouts: &[Some(&prepare_bgl)],
            immediate_size: 0,
        });

        // ── Sort ────────────────────────────────────────────────────────────
        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite_sort.wgsl").into()),
        });

        let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Prepare Pipeline"),
            layout: Some(&prepare_pl),
            module: &sort_shader,
            entry_point: Some("cs_prepare"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Sort Prepare BG"),
            layout: &prepare_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: indirect_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: frame_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dispatch_args_buf.as_entire_binding() },
            ],
        });

        let hist_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Histogram BGL"),
            entries: &[uniform_entry(0), uniform_entry(1), storage_entry(2, true), storage_entry(3, false)],
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Scan BGL"),
            entries: &[uniform_entry(0), storage_entry(1, false)],
        });
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Scatter BGL"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
            ],
        });

        let hist_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Histogram PL"),
            bind_group_layouts: &[Some(&hist_bgl)],
            immediate_size: 0,
        });
        let scan_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Scan PL"),
            bind_group_layouts: &[Some(&scan_bgl)],
            immediate_size: 0,
        });
        let scatter_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Scatter PL"),
            bind_group_layouts: &[Some(&scatter_bgl)],
            immediate_size: 0,
        });

        let hist_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Histogram Pipeline"),
            layout: Some(&hist_pl),
            module: &sort_shader,
            entry_point: Some("cs_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Scan Pipeline"),
            layout: Some(&scan_pl),
            module: &sort_shader,
            entry_point: Some("cs_scan"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Scatter Pipeline"),
            layout: Some(&scatter_pl),
            module: &sort_shader,
            entry_point: Some("cs_scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 2 buckets (0/1) per block per bit pass, not 256 — see the module
        // doc comment on why this is a 1-bit-per-pass sort. Sized for the
        // worst case (`max_blocks`); actual per-frame usage is bounded by
        // `frame_uniform_buf.num_blocks` at dispatch time, not by reallocating
        // this buffer.
        let block_hist_buf = create_u32_buffer(device, "Sprite Sort Block Histogram", max_blocks * 2);

        // `bit` is compile-time-fixed given the pass index, so its uniform
        // buffer is written once here and never touched again. `count`/
        // `num_blocks` (GPU-computed, per-frame) live in `frame_uniform_buf`
        // instead, refreshed once per frame by `cs_prepare`, shared by all
        // `SORT_BITS` passes.
        let pass_uniforms: [wgpu::Buffer; SORT_BITS] = std::array::from_fn(|i| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sprite Sort Pass Uniforms"),
                size: std::mem::size_of::<SortUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let u = SortUniforms { bit: i as u32, _pad0: 0, _pad1: 0, _pad2: 0 };
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
            buf
        });

        // Ping-pong role per pass, mirroring the CPU radix sort's parity:
        // pass 0 reads A writes B, pass 1 reads B writes A, and so on — so
        // after `SORT_BITS` (even) passes the sorted result is back in A.
        let hist_bind_groups: [wgpu::BindGroup; SORT_BITS] = std::array::from_fn(|i| {
            let src_keys = if i % 2 == 0 { &keys_a } else { &keys_b };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Sort Histogram BG"),
                layout: &hist_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pass_uniforms[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: frame_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: block_hist_buf.as_entire_binding() },
                ],
            })
        });
        let scan_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Sort Scan BG"),
            layout: &scan_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: block_hist_buf.as_entire_binding() },
            ],
        });
        let scatter_bind_groups: [wgpu::BindGroup; SORT_BITS] = std::array::from_fn(|i| {
            let (src_keys, src_indices, dst_keys, dst_indices) = if i % 2 == 0 {
                (&keys_a, &*indices_a, &keys_b, &indices_b)
            } else {
                (&keys_b, &indices_b, &keys_a, &*indices_a)
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Sort Scatter BG"),
                layout: &scatter_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pass_uniforms[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: frame_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: src_indices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: dst_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: dst_indices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: block_hist_buf.as_entire_binding() },
                ],
            })
        });

        Ok(Self {
            slot_capacity,
            max_visible,
            max_cull_workgroups: device
                .limits()
                .max_compute_workgroups_per_dimension
                .max(1),
            cull_pipeline,
            cull_bgl,
            cull_uniform_buf,
            cull_bind_group: Some(cull_bind_group),
            buffer_source,
            bound_instances_epoch: source.instances_epoch,
            bound_presence_epoch: source.presence_epoch,
            bound_runtime_epoch: source.runtime.as_ref().map_or(0, |runtime| runtime.epoch),
            runtime_capacity: source.runtime.as_ref().map_or(0, |runtime| runtime.row_capacity),
            fallback_runtime_buf,
            cull_keys_buf: keys_a.clone(),
            view_min: [0.0, 0.0],
            view_max: [0.0, 0.0],
            view_dirty: true,
            prepare_pipeline,
            prepare_bind_group,
            hist_pipeline,
            scan_pipeline,
            scatter_pipeline,
            _frame_uniform_buf: frame_uniform_buf,
            dispatch_args_buf,
            hist_bind_groups,
            scatter_bind_groups,
            scan_bind_group,
            draw_order_buf: indices_a,
            indirect_buf,
        })
    }

    /// Sets the world-space view rect sprites are culled against — must
    /// match the paired `SpriteBatchPass::set_camera`'s effective view, or
    /// sprites will be culled against the wrong bounds. Unlike the batch
    /// pass, this rect isn't derived from the render target size
    /// automatically (this pass has no render target to read a size from) —
    /// callers using `half_extent: None` on the batch pass must resize this
    /// to match themselves.
    pub fn set_view_rect(&mut self, center: [f32; 2], half_extent: [f32; 2]) {
        self.view_min = [center[0] - half_extent[0], center[1] - half_extent[1]];
        self.view_max = [center[0] + half_extent[0], center[1] + half_extent[1]];
        self.view_dirty = true;
    }
}

fn create_u32_buffer(device: &wgpu::Device, label: &str, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (count.max(1) as u64) * 4,
        // COPY_SRC costs nothing at runtime and is what lets
        // `draw_order_buf` (== `indices_a`) be read back at all, including
        // by `tests/gpu_sort_validation.rs`.
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn maximum_visible_for_device(device: &wgpu::Device) -> u32 {
    let limits = device.limits();
    let storage_u32s = (limits.max_buffer_size / 4)
        .min(u64::from(limits.max_storage_buffer_binding_size) / 4)
        .min(u64::from(u32::MAX));
    let histogram_blocks = storage_u32s / 2;
    let dispatch_blocks = u64::from(limits.max_compute_workgroups_per_dimension);
    let block_limited_rows = histogram_blocks
        .min(dispatch_blocks)
        .saturating_mul(u64::from(WG_SIZE));
    storage_u32s.min(block_limited_rows) as u32
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl RenderPass for SpriteCullPass {
    fn name(&self) -> &'static str {
        "SpriteCull"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None // compute-only pass
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let source_changed = self.refresh_cull_binding(ctx.device);
        if self.view_dirty || source_changed {
            self.view_dirty = false;
            let u = CullUniforms {
                view_min: self.view_min,
                view_max: self.view_max,
                slot_count: self.slot_capacity,
                max_visible: self.max_visible,
                runtime_capacity: self.runtime_capacity,
                dispatched_threads: self.cull_dispatched_threads(),
            };
            ctx.write_buffer(&self.cull_uniform_buf, 0, bytemuck::bytes_of(&u));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        self.refresh_cull_binding(ctx.device);
        self.record(unsafe { &mut *ctx.encoder_ptr });
        Ok(())
    }
}

impl SpriteCullPass {
    /// Records the cull + prepare + 32-pass sort dispatch sequence. Shared
    /// by the `RenderPass::execute()` trait impl and
    /// [`SpriteCullPass::run_once_for_testing`] — the graph-integrated path
    /// and the standalone test path must record identically, or a test pass
    /// proves nothing about the real one.
    fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        // Reset the atomic visible-count / instance_count field to zero
        // before culling. The other four `DrawIndexedIndirectArgs` fields
        // are static and were written once at construction.
        encoder.clear_buffer(&self.indirect_buf, INDIRECT_INSTANCE_COUNT_OFFSET, Some(4));
        encoder.clear_buffer(&self.indirect_buf, INDIRECT_OVERFLOW_COUNT_OFFSET, Some(4));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpriteCull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(
                0,
                self.cull_bind_group
                    .as_ref()
                    .expect("sprite cull binding is initialized"),
                &[],
            );
            let workgroups = self.cull_dispatch_workgroups();
            if workgroups != 0 {
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // Turns the GPU-computed visible count into `frame_uniform_buf`
        // (read by every sort kernel below) and `dispatch_args_buf` (the
        // indirect dispatch size for histogram/scatter) — this is what lets
        // those dispatches shrink when fewer sprites are visible instead of
        // always covering `max_visible`.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpriteSort Prepare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare_pipeline);
            pass.set_bind_group(0, &self.prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        for i in 0..SORT_BITS {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpriteSort Histogram"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.hist_pipeline);
                pass.set_bind_group(0, &self.hist_bind_groups[i], &[]);
                pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpriteSort Scan"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_pipeline);
                pass.set_bind_group(0, &self.scan_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpriteSort Scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &self.scatter_bind_groups[i], &[]);
                pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);
            }
        }
    }

    /// Runs cull + sort once, outside a `RenderGraph`, blocking until the
    /// GPU is done. Not for the hot path — this exists for integration tests
    /// that need to read back [`SpriteCullPass::draw_order_buf`]/
    /// [`SpriteCullPass::indirect_buf`] afterward without standing up a full
    /// graph + `PassContext`.
    pub fn run_once_for_testing(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let source_changed = self.refresh_cull_binding(device);
        if self.view_dirty || source_changed {
            self.view_dirty = false;
            let u = CullUniforms {
                view_min: self.view_min,
                view_max: self.view_max,
                slot_count: self.slot_capacity,
                max_visible: self.max_visible,
                runtime_capacity: self.runtime_capacity,
                dispatched_threads: self.cull_dispatched_threads(),
            };
            queue.write_buffer(&self.cull_uniform_buf, 0, bytemuck::bytes_of(&u));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SpriteCullPass::run_once_for_testing"),
        });
        self.record(&mut encoder);
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
}
