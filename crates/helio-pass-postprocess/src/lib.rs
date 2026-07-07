//! GPU-native post-processing pipeline for Helio.
//!
//! Single pass with multiple sub-stages chained together:
//!
//!   1. `cs_exposure`     — luminance histogram → avg luminance (compute)
//!   2. `cs_bloom_clear`  — clear bloom mip chain
//!   3. `cs_bloom_down`   — extract brights + downsample (compute)
//!   4. `cs_bloom_up`     — upsample + accumulate (compute)
//!   5. `fs_uber`         — tonemap, color grade, vignette, CA, grain (render)
//!
//! The renderer writes blended `GpuPostProcessUniforms` to a buffer that is
//! exposed via FrameResources.postprocess_uniforms. The pass reads it in execute().

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const BLOOM_MIPS: u32 = 5;
const WG_BLOOM: u32 = 8;
const WG_EXPOSURE_X: u32 = 16;
const WG_EXPOSURE_Y: u32 = 16;

pub struct PostProcessPass {
    // Owned staging buffer for initial luminance value
    avg_luminance_buf: wgpu::Buffer,

    // Pipelines
    exposure_pipeline: wgpu::ComputePipeline,
    bloom_clear_pipeline: wgpu::ComputePipeline,
    bloom_down_pipeline: wgpu::ComputePipeline,
    bloom_up_pipeline: wgpu::ComputePipeline,
    uber_pipeline: wgpu::RenderPipeline,

    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize, usize)>,

    bloom_textures: Vec<wgpu::Texture>,
    bloom_views: Vec<wgpu::TextureView>,

    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    width: u32,
    height: u32,
    #[allow(dead_code)]
    format: wgpu::TextureFormat,

    first_frame: bool,
}

impl PostProcessPass {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/postprocess.wgsl").into()),
        });

        let avg_luminance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Avg Luminance"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let (bloom_textures, bloom_views) = Self::create_bloom_mips(device, width, height);

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess BGL"),
            entries: &[
                // b0: GpuPostProcessUniforms (renderer-owned)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b1: CameraUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b2: HDR input (pre_aa)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // b3: Depth input
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // b4: Linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // b5: Point sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // b6: Bloom mips (texture binding array)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: std::num::NonZeroU32::new(BLOOM_MIPS),
                },
                // b7: Avg luminance storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        };
        let pl = device.create_pipeline_layout(&pl_desc);

        let mk_compute = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let exposure_pipeline = mk_compute("PostProcess Exposure", "cs_exposure");
        let bloom_clear_pipeline = mk_compute("PostProcess Bloom Clear", "cs_bloom_clear");
        let bloom_down_pipeline = mk_compute("PostProcess Bloom Down", "cs_bloom_down");
        let bloom_up_pipeline = mk_compute("PostProcess Bloom Up", "cs_bloom_up");
        let uber_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Uber Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_uber"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            avg_luminance_buf,
            exposure_pipeline,
            bloom_clear_pipeline,
            bloom_down_pipeline,
            bloom_up_pipeline,
            uber_pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            bloom_textures,
            bloom_views,
            linear_sampler,
            point_sampler,
            width,
            height,
            format,
            first_frame: true,
        }
    }

    fn create_bloom_mips(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>) {
        let mut textures = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut views = Vec::with_capacity(BLOOM_MIPS as usize);

        for i in 0..BLOOM_MIPS {
            let mw = (width >> (i + 1)).max(1);
            let mh = (height >> (i + 1)).max(1);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Mip {}", i)),
                size: wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            textures.push(tex);
            views.push(view);
        }

        (textures, views)
    }

    fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        postprocess_buf: &wgpu::Buffer,
        pre_aa_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
    ) {
        let bloom_refs: Vec<&wgpu::TextureView> = self.bloom_views.iter().collect();

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: postprocess_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(pre_aa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureViewArray(&bloom_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.avg_luminance_buf.as_entire_binding(),
                },
            ],
        }));
    }

    fn mip_dims(&self, mip: u32) -> (u32, u32) {
        ((self.width >> (mip + 1)).max(1), (self.height >> (mip + 1)).max(1))
    }
}

impl RenderPass for PostProcessPass {
    fn name(&self) -> &'static str { "PostProcess" }

    fn reads(&self) -> &'static [&'static str] { &["pre_aa"] }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (textures, views) = Self::create_bloom_mips(device, width, height);
        self.bloom_textures = textures;
        self.bloom_views = views;
        self.bind_group = None;
        self.bind_group_key = None;
        self.first_frame = true;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if self.first_frame {
            self.first_frame = false;
            let initial: f32 = 0.18;
            ctx.queue.write_buffer(&self.avg_luminance_buf, 0, bytemuck::bytes_of(&initial));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let pre_aa_view = ctx.resources.pre_aa.read("PostProcess").ok_or_else(|| {
            helio_core::Error::InvalidPassConfig(
                "PostProcess requires frame.pre_aa (from DeferredLightPass)".to_string(),
            )
        })?;

        let postprocess_buf = ctx.resources.postprocess_uniforms.read("PostProcess").ok_or_else(|| {
            helio_core::Error::InvalidPassConfig(
                "PostProcess requires frame.postprocess_uniforms (from Renderer)".to_string(),
            )
        })?;

        let camera_buf = ctx.scene.camera;

        // Lazy bind group
        let bg_key = (
            pre_aa_view as *const _ as usize,
            ctx.depth as *const _ as usize,
            camera_buf as *const _ as usize,
            postprocess_buf as *const _ as usize,
        );
        if self.bind_group_key != Some(bg_key) {
            self.rebuild_bind_group(ctx.device, postprocess_buf, pre_aa_view, ctx.depth, camera_buf);
            self.bind_group_key = Some(bg_key);
        }

        let bg = self.bind_group.as_ref().unwrap();
        let ce = ctx.compute_encoder_ptr;

        // 1. Auto-exposure histogram
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Exposure"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.exposure_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let gx = (self.width / (4 * WG_EXPOSURE_X)).max(1);
            let gy = (self.height / (4 * WG_EXPOSURE_Y)).max(1);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // 2. Bloom: clear → downscale → upscale
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Bloom Clear"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_clear_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let (mw, mh) = self.mip_dims(0);
            let gx = (mw + WG_BLOOM - 1) / WG_BLOOM;
            let gy = (mh + WG_BLOOM - 1) / WG_BLOOM;
            cpass.dispatch_workgroups(gx, gy, BLOOM_MIPS);
        }

        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Bloom Down"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_down_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let (mw, mh) = self.mip_dims(0);
            let gx = (mw + WG_BLOOM - 1) / WG_BLOOM;
            let gy = (mh + WG_BLOOM - 1) / WG_BLOOM;
            cpass.dispatch_workgroups(gx, gy, BLOOM_MIPS);
        }

        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Bloom Up"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_up_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let (mw, mh) = self.mip_dims(0);
            let gx = (mw + WG_BLOOM - 1) / WG_BLOOM;
            let gy = (mh + WG_BLOOM - 1) / WG_BLOOM;
            cpass.dispatch_workgroups(gx, gy, BLOOM_MIPS - 1);
        }

        // 3. Uber pass (tonemap + color grade + vignette + CA + grain + bloom composite)
        {
            let color = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PostProcess Uber"),
                color_attachments: &color,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.uber_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
