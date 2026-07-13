//! GPU-driven voxel meshlet rendering pass.
//!
//! Manages surface extraction (Marching Cubes compute shader) and indirect
//! multi-draw rendering of per-brick meshlets.  CPU only touches a small
//! dirty-brick list each frame.

use bytemuck::{Pod, Zeroable};
use helio_core::{
    PassContext, PrepareContext, RenderPass, Result as HelioResult,
};
use helio_voxel_core::{
    GpuBrickMeshlet, GpuBrickMeta, MAX_SURFACE_INDICES_PER_BRICK, MAX_SURFACE_VERTS_PER_BRICK,
};
use libhelio::DrawIndexedIndirectArgs;
use wgpu::util::DeviceExt;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const VOXEL_MESH_MAX_BRICKS: u32 = 8192;
pub const VOXEL_MESH_MAX_DIRTY: u32 = 4096;
pub const VOXEL_MESH_BRICK_VOXEL_WORDS: u64 = 128; // 512 voxels / 4 per u32

// ── GPU types ─────────────────────────────────────────────────────────────────

/// Per-brick dirty entry uploaded to the GPU each frame.
/// WGSL layout: brick_slot(u32) + volume_id(u32) + _pad(u32×2) + origin_size(vec4<f32>).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DirtyBrick {
    brick_slot: u32,
    volume_id: u32,
    _pad: [u32; 2],
    origin_size: [f32; 4], // xyz = world origin, w = voxel size
}

// ── Pass ──────────────────────────────────────────────────────────────────────

pub struct VoxelMeshPass {
    // Pipelines
    extract_pipeline: wgpu::ComputePipeline,
    extract_bgl: wgpu::BindGroupLayout,
    extract_bind_group: wgpu::BindGroup,

    render_pipeline: wgpu::RenderPipeline,
    render_bgl: wgpu::BindGroupLayout,
    render_bind_group: Option<wgpu::BindGroup>,
    render_bind_group_key: Option<usize>,

    // GPU buffers
    brick_meta_buf: wgpu::Buffer,
    voxel_data_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    descriptor_buf: wgpu::Buffer,
    indirect_buf: wgpu::Buffer,
    dirty_brick_buf: wgpu::Buffer,

    // Owned color attachment
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,

    // CPU-side dirty list (uploaded each frame, cleared after compute dispatch)
    dirty_bricks: Vec<DirtyBrick>,

    normal_buf: wgpu::Buffer,
    surface_format: wgpu::TextureFormat,
}

impl VoxelMeshPass {
    /// Creates the pass, allocating all GPU buffers and compiling both pipelines.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let max_bricks = VOXEL_MESH_MAX_BRICKS as u64;
        let max_verts = MAX_SURFACE_VERTS_PER_BRICK as u64;
        let max_indices = MAX_SURFACE_INDICES_PER_BRICK as u64;
        let max_dirty = VOXEL_MESH_MAX_DIRTY as u64;

        // ── Buffers ──────────────────────────────────────────────────────────
        let brick_meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh BrickMeta"),
            size: max_bricks * std::mem::size_of::<GpuBrickMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let voxel_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh VoxelData"),
            size: max_bricks * VOXEL_MESH_BRICK_VOXEL_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Vertices"),
            size: max_bricks * max_verts * 16,
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
        let descriptor_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Descriptors"),
            size: max_bricks * std::mem::size_of::<GpuBrickMeshlet>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect_buf = {
            let indirect_size =
                max_bricks * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
            let zeros = vec![0u8; indirect_size as usize];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VoxelMesh Indirect"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            })
        };
        let dirty_brick_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh DirtyBricks"),
            size: max_dirty * std::mem::size_of::<DirtyBrick>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let normal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelMesh Normals"),
            size: max_bricks * max_verts * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Color attachment (initial 1×1; resized in on_resize) ────────────
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VoxelMesh Color"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&Default::default());

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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let extract_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VoxelMesh Extract BG"),
            layout: &extract_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: brick_meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: voxel_data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vertex_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: descriptor_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: indirect_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dirty_brick_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: normal_buf.as_entire_binding() },
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
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 1,
                        }],
                    },
                ],
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
            extract_bind_group,
            render_pipeline,
            render_bgl,
            render_bind_group: None,
            render_bind_group_key: None,
            brick_meta_buf,
            voxel_data_buf,
            vertex_buf,
            index_buf,
            descriptor_buf,
            indirect_buf,
            dirty_brick_buf,
            color_tex,
            color_view,
            dirty_bricks: Vec::new(),
            normal_buf,
            surface_format,
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    pub fn brick_meta_buf(&self) -> &wgpu::Buffer {
        &self.brick_meta_buf
    }

    pub fn voxel_data_buf(&self) -> &wgpu::Buffer {
        &self.voxel_data_buf
    }

    /// Mark a brick for re-extraction on the next frame.
    pub fn mark_dirty(
        &mut self,
        brick_slot: u32,
        volume_id: u32,
        origin: [f32; 3],
        voxel_size: f32,
    ) {
        if self.dirty_bricks.len() < VOXEL_MESH_MAX_DIRTY as usize {
            self.dirty_bricks.push(DirtyBrick {
                brick_slot,
                volume_id,
                _pad: [0u32; 2],
                origin_size: [origin[0], origin[1], origin[2], voxel_size],
            });
        } else {
            log::warn!("VoxelMeshPass: dirty brick list overflow (dropping slot {brick_slot})");
        }
    }

    /// Zero out the indirect draw for a brick slot so it stops being rendered.
    /// Call this when a brick is deallocated.
    pub fn clear_brick_slot(
        &self,
        queue: &wgpu::Queue,
        brick_slot: u32,
    ) {
        const ZERO: DrawIndexedIndirectArgs = DrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let off = brick_slot as u64 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
        queue.write_buffer(&self.indirect_buf, off, bytemuck::bytes_of(&ZERO));
    }
}

// ── RenderPass trait ──────────────────────────────────────────────────────────

impl RenderPass for VoxelMeshPass {
    fn name(&self) -> &'static str {
        "VoxelMesh"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if !self.dirty_bricks.is_empty() {
            let bytes = bytemuck::cast_slice(&self.dirty_bricks);
            ctx.write_buffer(&self.dirty_brick_buf, 0, bytes);
            log::debug!("VoxelMeshPass: {} dirty bricks", self.dirty_bricks.len());
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let dirty_count = self.dirty_bricks.len() as u32;

        // ── Step 1: Compute — surface extraction on all dirty bricks ─────────
        if dirty_count > 0 {
            let mut cpass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VoxelMesh Extract"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.extract_pipeline);
            cpass.set_bind_group(0, &self.extract_bind_group, &[]);
            cpass.dispatch_workgroups(dirty_count, 1, 1);
        }

        // ── Step 2: Render — draw all bricks via indirect multi-draw ─────────
        // Rebuild camera bind group when the camera buffer pointer changes.
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        if self.render_bind_group_key != Some(camera_ptr) {
            self.render_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VoxelMesh Render BG"),
                layout: &self.render_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.scene.camera.as_entire_binding(),
                }],
            }));
            self.render_bind_group_key = Some(camera_ptr);
        }

        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, self.render_bind_group.as_ref().unwrap(), &[]);
        rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rp.set_vertex_buffer(1, self.normal_buf.slice(..));
        rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);

        // Draw all bricks — entries with index/instance count == 0 are no-ops.
        #[cfg(not(target_arch = "wasm32"))]
        rp.multi_draw_indexed_indirect(&self.indirect_buf, 0, VOXEL_MESH_MAX_BRICKS);
        #[cfg(target_arch = "wasm32")]
        for i in 0..VOXEL_MESH_MAX_BRICKS {
            let off = i as u64 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
            rp.draw_indexed_indirect(&self.indirect_buf, off);
        }

        // Flush the CPU-side dirty list after the GPU has consumed it.
        self.dirty_bricks.clear();

        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("VoxelMesh"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VoxelMesh Color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.color_view = self.color_tex.create_view(&Default::default());
    }
}
