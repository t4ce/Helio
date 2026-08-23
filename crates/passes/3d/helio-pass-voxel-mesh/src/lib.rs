//! GPU-driven voxel meshlet rendering pass.
//!
//! Manages surface extraction (Marching Cubes compute shader) and indirect
//! multi-draw rendering of per-brick meshlets. Canonical volume, brick, voxel,
//! and palette data is bound directly from SceneDB; this pass owns only its
//! derived mesh output buffers.

mod marching_cubes;

use bytemuck::{Pod, Zeroable};
use helio_core::{
    graph::{ResourceBuilder, ResourceSize},
    PassContext, PrepareContext, RenderPass, Result as HelioResult,
};
use helio_voxel_core::{
    GpuVoxelMeshVertex, MAX_SURFACE_INDICES_PER_BRICK, MAX_SURFACE_VERTS_PER_BRICK,
};
use libhelio::DrawIndexedIndirectArgs;

use marching_cubes::PACKED_TRI_TABLE;
// ── Constants ─────────────────────────────────────────────────────────────────

// Kept modest because vertex_buf/index_buf scale with
// max_bricks * MAX_SURFACE_VERTS_PER_BRICK — at the 2048-vert budget needed to
// avoid truncating textured terrain (see constants.rs), the original 8192
// would allocate hundreds of MB for buffers this example never uses more than
// 512 bricks of.
pub const VOXEL_MESH_MAX_BRICKS: u32 = 1024;
#[cfg(test)]
const EXTRACT_STORAGE_BINDINGS: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttachmentMode {
    Standalone,
    Composited,
}

impl AttachmentMode {
    fn color_load(self) -> wgpu::LoadOp<wgpu::Color> {
        match self {
            Self::Standalone => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            Self::Composited => wgpu::LoadOp::Load,
        }
    }

    fn depth_load(self) -> wgpu::LoadOp<f32> {
        match self {
            Self::Standalone => wgpu::LoadOp::Clear(1.0),
            Self::Composited => wgpu::LoadOp::Load,
        }
    }
}

// ── GPU types ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExtractParams {
    generation: u32,
    bootstrap: u32,
    work_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshletParams {
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn needs_render_pass(attachment_mode: AttachmentMode, draw_count: u32) -> bool {
    attachment_mode == AttachmentMode::Standalone || draw_count > 0
}

// ── Pass ──────────────────────────────────────────────────────────────────────

pub struct VoxelMeshPass {
    // Pipelines
    extract_pipeline: wgpu::ComputePipeline,
    extract_bgl: wgpu::BindGroupLayout,
    extract_bind_group: Option<wgpu::BindGroup>,
    extract_bind_group_key: Option<(usize, u64, usize, u64, usize, u64, usize, u64)>,
    extract_params_buf: wgpu::Buffer,
    processed_work_generation: Option<u64>,
    pending_work_count: u32,
    draw_count: u32,

    render_pipeline: wgpu::RenderPipeline,
    render_bgl: wgpu::BindGroupLayout,
    render_bind_group: Option<wgpu::BindGroup>,
    render_bind_group_key: Option<(usize, usize, usize, usize, u64, usize, u64)>,
    meshlet_params_buf: wgpu::Buffer,

    // GPU buffers
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    indirect_buf: wgpu::Buffer,
    packed_tri_table_buf: wgpu::Buffer,
    surface_format: wgpu::TextureFormat,
    attachment_mode: AttachmentMode,
}

impl VoxelMeshPass {
    /// Creates a standalone pass that clears color and depth before drawing.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_with_attachment_mode(
            device,
            queue,
            surface_format,
            AttachmentMode::Standalone,
        )
    }

    /// Creates a pass that loads existing color and depth for composition.
    pub fn new_composited(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_with_attachment_mode(
            device,
            queue,
            surface_format,
            AttachmentMode::Composited,
        )
    }

    fn new_with_attachment_mode(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        attachment_mode: AttachmentMode,
    ) -> Self {
        let max_bricks = VOXEL_MESH_MAX_BRICKS as u64;
        let max_verts = MAX_SURFACE_VERTS_PER_BRICK as u64;
        let max_indices = MAX_SURFACE_INDICES_PER_BRICK as u64;

        // ── Buffers ──────────────────────────────────────────────────────────
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Vertices"),
            size: max_bricks
                * max_verts
                * std::mem::size_of::<GpuVoxelMeshVertex>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Indices"),
            size: max_bricks * max_indices * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect_buf = {
            let indirect_size = max_bricks * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
            let zeros = vec![0u8; indirect_size as usize];
            // Keep the always-present default graph off WebGPU's fragile
            // mappedAtCreation path; queue writes preserve initialization order.
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelMesh Indirect"),
                size: indirect_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &zeros);
            buffer
        };
        let extract_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Extract Params"),
            size: std::mem::size_of::<ExtractParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let packed_tri_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Packed Triangle Table"),
            size: std::mem::size_of_val(&PACKED_TRI_TABLE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &packed_tri_table_buf,
            0,
            bytemuck::cast_slice(&PACKED_TRI_TABLE),
        );

        // ── Shaders ──────────────────────────────────────────────────────────
        let extract_src = include_str!("../shaders/voxel_surface_extract.wgsl");
        let meshlet_src = include_str!("../shaders/voxel_meshlet.wgsl");

        let extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelSurfaceExtract"),
            source: wgpu::ShaderSource::Wgsl(extract_src.into()),
        });
        let meshlet_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelMeshlet"),
            source: wgpu::ShaderSource::Wgsl(meshlet_src.into()),
        });

        // ── Extract (compute) bind group layout ──────────────────────────────
        let extract_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelMesh Extract BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Extract pipeline ─────────────────────────────────────────────────
        let extract_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VoxelMesh Extract PL"),
            bind_group_layouts: &[Some(&extract_bgl)],
            immediate_size: 0,
        });
        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelMesh Extract"),
            layout: Some(&extract_pl),
            module: &extract_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Render bind group layout ─────────────────────────────────────────
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelMesh Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let meshlet_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Meshlet Params"),
            size: std::mem::size_of::<MeshletParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Render pipeline ──────────────────────────────────────────────────
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VoxelMesh Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelMesh Render"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &meshlet_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuVoxelMeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 1,
                            },
                        ],
                    })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &meshlet_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Marching-cubes triangle winding from TRI_TABLE isn't
                // guaranteed to come out consistently front-facing for every
                // one of the 256 cases against this crate's edge_vertex/
                // local_pos convention (unlike a hand-authored mesh). Backface
                // culling here would silently drop roughly half the surface —
                // exactly the patchy, gap-riddled look this pass had with
                // Face::Back culling on. Disable culling instead of chasing
                // per-case winding correctness.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            extract_pipeline,
            extract_bgl,
            extract_bind_group: None,
            extract_bind_group_key: None,
            extract_params_buf,
            processed_work_generation: None,
            pending_work_count: 0,
            draw_count: 0,
            render_pipeline,
            render_bgl,
            meshlet_params_buf,
            render_bind_group: None,
            render_bind_group_key: None,
            vertex_buf,
            index_buf,
            indirect_buf,
            packed_tri_table_buf,
            surface_format,
            attachment_mode,
        }
    }
}

// ── RenderPass trait ──────────────────────────────────────────────────────────

impl RenderPass for VoxelMeshPass {
    fn name(&self) -> &'static str {
        "VoxelMesh"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    // The constructor selects whether this pass initializes `pre_aa` and depth
    // or composites over attachments produced by earlier graph passes.
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw("pre_aa", self.surface_format, ResourceSize::MatchSurface);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.draw_count = ctx.scene.voxel_mesh_draw_count;
        let generation = ctx.scene.voxel_mesh_work_generation;
        if self.processed_work_generation != Some(generation) {
            self.pending_work_count = ctx.scene.voxel_mesh_work.len() as u32;
            let params = ExtractParams {
                generation: generation as u32,
                bootstrap: u32::from(self.processed_work_generation.is_none()),
                work_count: self.pending_work_count,
                _pad: 0,
            };
            ctx.write_buffer(
                &self.extract_params_buf,
                0,
                bytemuck::bytes_of(&params),
            );
        }
        if !needs_render_pass(self.attachment_mode, self.draw_count) {
            return Ok(());
        }
        let params = MeshletParams {
            light_count: ctx.scene.movable_light_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.meshlet_params_buf, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── Step 1: Compute — consume SceneDB-backed coalesced work rows ─────
        if self.pending_work_count > 0 {
            let residency_epoch = ctx.scene.voxel_residency_epoch.unwrap_or(0);
            let volume_epoch = ctx.scene.voxel_volume_epoch.unwrap_or(0);
            let extract_key = (
                ctx.scene.voxel_brick_pool as *const _ as usize,
                residency_epoch,
                ctx.scene.voxel_data_pool as *const _ as usize,
                residency_epoch,
                ctx.scene.voxel_mesh_work as *const _ as usize,
                ctx.scene.voxel_mesh_work_epoch,
                ctx.scene.voxel_volumes as *const _ as usize,
                volume_epoch,
            );
            if self.extract_bind_group_key != Some(extract_key) {
                self.extract_bind_group = Some(ctx.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("VoxelMesh Extract BG"),
                        layout: &self.extract_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: ctx.scene.voxel_brick_pool.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: ctx.scene.voxel_data_pool.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.vertex_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.index_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.indirect_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: ctx.scene.voxel_mesh_work.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.packed_tri_table_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: ctx.scene.voxel_volumes.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: self.extract_params_buf.as_entire_binding(),
                            },
                        ],
                    },
                ));
                self.extract_bind_group_key = Some(extract_key);
            }
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VoxelMesh Extract"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.extract_pipeline);
            cpass.set_bind_group(0, self.extract_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups(self.pending_work_count, 1, 1);
        }
        self.processed_work_generation = Some(ctx.scene.voxel_mesh_work_generation);
        self.pending_work_count = 0;

        // A composited pass with no occupied bricks intentionally has no active
        // render pass. Inactive work rows still reach compute above so stale
        // indirect arguments are cleared before output slots are recycled.
        if !needs_render_pass(self.attachment_mode, self.draw_count) {
            return Ok(());
        }

        // ── Step 2: Render — draw the resident brick range indirectly ───────
        // Rebuild the bind group when the camera or lights buffer pointer changes
        // (the lights buffer can be reallocated by GrowableBuffer as it grows).
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let lights_ptr = ctx.scene.lights as *const _ as usize;
        let light_projections_ptr = ctx.scene.light_projections as *const _ as usize;
        let volume_ptr = ctx.scene.voxel_volumes as *const _ as usize;
        let volume_epoch = ctx.scene.voxel_volume_epoch.unwrap_or(0);
        let palette_ptr = ctx.scene.voxel_palette_pool as *const _ as usize;
        let residency_epoch = ctx.scene.voxel_residency_epoch.unwrap_or(0);
        let render_key = (
            camera_ptr,
            lights_ptr,
            light_projections_ptr,
            volume_ptr,
            volume_epoch,
            palette_ptr,
            residency_epoch,
        );
        if self.render_bind_group_key != Some(render_key) {
            self.render_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VoxelMesh Render BG"),
                layout: &self.render_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.meshlet_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.light_projections.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.voxel_volumes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.scene.voxel_palette_pool.as_entire_binding(),
                    },
                ],
            }));
            self.render_bind_group_key = Some(render_key);
        }

        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, self.render_bind_group.as_ref().unwrap(), &[]);
        rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);

        // Slots below draw_count may still contain zero-count entries. The CPU
        // bound only removes the unused capacity after the final active slot.
        #[cfg(not(target_arch = "wasm32"))]
        rp.multi_draw_indexed_indirect(&self.indirect_buf, 0, self.draw_count);
        #[cfg(target_arch = "wasm32")]
        for i in 0..self.draw_count {
            let off = i as u64 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
            rp.draw_indexed_indirect(&self.indirect_buf, off);
        }

        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if !needs_render_pass(self.attachment_mode, self.draw_count) {
            return None;
        }

        let pre_aa_view = resources.pre_aa.read("VoxelMesh")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: pre_aa_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: self.attachment_mode.color_load(),
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("VoxelMesh"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: self.attachment_mode.depth_load(),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{needs_render_pass, AttachmentMode, EXTRACT_STORAGE_BINDINGS};

    #[test]
    fn standalone_mode_initializes_color_and_depth() {
        assert!(matches!(
            AttachmentMode::Standalone.color_load(),
            wgpu::LoadOp::Clear(color) if color == wgpu::Color::TRANSPARENT
        ));
        assert!(matches!(
            AttachmentMode::Standalone.depth_load(),
            wgpu::LoadOp::Clear(1.0)
        ));
    }

    #[test]
    fn composited_mode_preserves_color_and_depth() {
        assert!(matches!(
            AttachmentMode::Composited.color_load(),
            wgpu::LoadOp::Load
        ));
        assert!(matches!(
            AttachmentMode::Composited.depth_load(),
            wgpu::LoadOp::Load
        ));
    }

    #[test]
    fn extraction_stays_within_the_portable_storage_binding_budget() {
        assert_eq!(EXTRACT_STORAGE_BINDINGS, 8);
        assert_eq!(std::mem::size_of::<helio_voxel_core::GpuVoxelMeshVertex>(), 32);
    }

    #[test]
    fn empty_composited_pass_is_skipped() {
        assert!(!needs_render_pass(AttachmentMode::Composited, 0));
        assert!(needs_render_pass(AttachmentMode::Composited, 1));
    }

    #[test]
    fn empty_standalone_pass_still_initializes_attachments() {
        assert!(needs_render_pass(AttachmentMode::Standalone, 0));
    }
}
