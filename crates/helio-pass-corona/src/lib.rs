//! Corona — fully GPU-native particle system.
//!
//! Per-frame GPU pipeline:
//!   1. Simulate  — physics + aging
//!   2. Emit      — ring-buffer spawn per emitter
//!   3. Compact   — scatter alive indices into per-emitter compact_buf sub-ranges
//!   4. Build     — write one DrawArgs per emitter from alive counts
//!   5. Render    — one draw_indirect per emitter, reads compact_buf[ii] → particles[idx]

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_MAX_PARTICLES: u32 = libhelio::CORONA_MAX_PARTICLES;
const MAX_EMITTERS: u32 = 64;
const WG: u32 = 256;

// ── CPU uniform mirror ───────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoronaUniforms {
    delta_time:      f32,
    total_particles: u32,
    emitter_count:   u32,
    frame_count:     u32,
}

// ── Pass ─────────────────────────────────────────────────────────────────────

pub struct CoronaPass {
    // ── Compute pipelines ────────────────────────────────────────────────────
    simulate_pipeline:      wgpu::ComputePipeline,
    emit_pipeline:          wgpu::ComputePipeline,
    compact_reset_pipeline: wgpu::ComputePipeline,
    compact_pipeline:       wgpu::ComputePipeline,
    build_multi_pipeline:   wgpu::ComputePipeline,

    // ── Render pipeline ──────────────────────────────────────────────────────
    render_pipeline: wgpu::RenderPipeline,

    // ── Bind group layouts / pipeline layouts ────────────────────────────────
    compute_bgl: wgpu::BindGroupLayout,
    render_bgl:  wgpu::BindGroupLayout,

    // ── GPU buffers ──────────────────────────────────────────────────────────
    uniform_buf:        wgpu::Buffer,
    particle_buf:       wgpu::Buffer,
    emitter_buf:        wgpu::Buffer,
    /// compact_buf[emitter.particle_offset .. +alive_count] = alive particle indices
    compact_buf:        wgpu::Buffer,
    /// per-emitter atomic alive counter, reset by cs_compact_reset each frame
    emitter_alive_buf:  wgpu::Buffer,
    /// written by cs_build_multi (STORAGE), copied to draw_args_buf (INDIRECT)
    draw_args_staging:  wgpu::Buffer,
    /// only flag: INDIRECT | COPY_DST — never bound in any bind group
    draw_args_buf:      wgpu::Buffer,

    // ── Particle texture ─────────────────────────────────────────────────────
    particle_tex:     wgpu::Texture,
    particle_view:    wgpu::TextureView,
    particle_sampler: wgpu::Sampler,

    // ── Bind groups (rebuilt when buffer pointers change) ────────────────────
    compute_bg:     Option<wgpu::BindGroup>,
    compute_bg_key: Option<usize>,   // particle_buf ptr
    render_bg:      Option<wgpu::BindGroup>,
    render_bg_key:  Option<usize>,   // camera_buf ptr

    // ── State ────────────────────────────────────────────────────────────────
    uploaded_generation: u64,
    max_particles:       u32,
    emitter_count:       u32,

    // CPU-side emitter list: preserves spawn_cursor and assigned particle_offset
    // across frames so particles travel rather than resetting every update.
    cpu_emitters: Vec<libhelio::GpuCoronaEmitter>,
}

impl CoronaPass {
    pub fn new(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Corona Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/corona.wgsl").into()),
        });

        // ── Buffers ──────────────────────────────────────────────────────────

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Uniforms"),
            size:  std::mem::size_of::<CoronaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_size = std::mem::size_of::<libhelio::GpuCoronaParticle>() as u64;
        let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Particles"),
            size:  DEFAULT_MAX_PARTICLES as u64 * particle_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let emitter_size = std::mem::size_of::<libhelio::GpuCoronaEmitter>() as u64;
        let emitter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitters"),
            size:  MAX_EMITTERS as u64 * emitter_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compact_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Compact"),
            size:  DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let emitter_alive_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitter Alive"),
            size:  MAX_EMITTERS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let draw_args_size = MAX_EMITTERS as u64
            * std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        let draw_args_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona DrawArgs Staging"),
            size:  draw_args_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let draw_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona DrawArgs"),
            size:  draw_args_size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Particle texture (soft circle) ───────────────────────────────────

        let tex_size = 32u32;
        let tex_data = Self::make_soft_circle_tex(tex_size);
        let particle_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Corona Particle Tex"),
            size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &particle_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &tex_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tex_size * 4),
                rows_per_image: Some(tex_size),
            },
            wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
        );
        let particle_view = particle_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let particle_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Corona Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ── Compute bind group layout (b0–b5) ────────────────────────────────

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Corona Compute BGL"),
            entries: &[
                Self::uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                Self::storage_entry(1, wgpu::ShaderStages::COMPUTE, false),
                Self::storage_entry(2, wgpu::ShaderStages::COMPUTE, false),
                Self::storage_entry(3, wgpu::ShaderStages::COMPUTE, false),
                Self::storage_entry(4, wgpu::ShaderStages::COMPUTE, false),
                Self::storage_entry(5, wgpu::ShaderStages::COMPUTE, false),
            ],
        });

        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona Compute PL"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        // ── Render bind group layout (b0–b8, superset of compute) ────────────
        // b0-b5 are present so the pipeline layout satisfies all WGSL globals;
        // vs/fs only use b1 (particles), b3 (compact), b6 (camera), b7-b8 (tex).

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Corona Render BGL"),
            entries: &[
                Self::uniform_entry(0, wgpu::ShaderStages::VERTEX),
                Self::storage_entry(1, wgpu::ShaderStages::VERTEX, false),
                Self::storage_entry(2, wgpu::ShaderStages::COMPUTE, false),
                Self::storage_entry(3, wgpu::ShaderStages::VERTEX, false),
                Self::storage_entry(4, wgpu::ShaderStages::COMPUTE, false),
                Self::storage_entry(5, wgpu::ShaderStages::COMPUTE, false),
                Self::uniform_entry(6, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        // ── Compute pipelines ────────────────────────────────────────────────

        let mk = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Corona {entry}")),
                layout: Some(&compute_pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let simulate_pipeline      = mk("cs_simulate");
        let emit_pipeline          = mk("cs_emit");
        let compact_reset_pipeline = mk("cs_compact_reset");
        let compact_pipeline       = mk("cs_compact");
        let build_multi_pipeline   = mk("cs_build_multi");

        // ── Render pipeline ──────────────────────────────────────────────────

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Corona Render Pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                compilation_options: Default::default(),
                buffers:             &[],
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend:  Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:               wgpu::TextureFormat::Depth32Float,
                depth_write_enabled:  Some(false),
                depth_compare:        Some(wgpu::CompareFunction::LessEqual),
                stencil:              wgpu::StencilState::default(),
                bias:                 wgpu::DepthBiasState::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ── Initial bind groups ──────────────────────────────────────────────

        let camera_ptr = camera_buf as *const _ as usize;
        let part_ptr   = &particle_buf as *const _ as usize;

        let compute_bg = Some(Self::build_compute_bg(
            device, &compute_bgl,
            &uniform_buf, &particle_buf, &emitter_buf,
            &compact_buf, &emitter_alive_buf, &draw_args_staging,
        ));
        let render_bg = Some(Self::build_render_bg(
            device, &render_bgl,
            &uniform_buf, &particle_buf, &emitter_buf,
            &compact_buf, &emitter_alive_buf, &draw_args_staging,
            camera_buf, &particle_view, &particle_sampler,
        ));

        Self {
            simulate_pipeline,
            emit_pipeline,
            compact_reset_pipeline,
            compact_pipeline,
            build_multi_pipeline,
            render_pipeline,
            compute_bgl,
            render_bgl,
            uniform_buf,
            particle_buf,
            emitter_buf,
            compact_buf,
            emitter_alive_buf,
            draw_args_staging,
            draw_args_buf,
            particle_tex,
            particle_view,
            particle_sampler,
            compute_bg,
            compute_bg_key: Some(part_ptr),
            render_bg,
            render_bg_key: Some(camera_ptr),
            uploaded_generation: u64::MAX,
            max_particles: DEFAULT_MAX_PARTICLES,
            emitter_count: 0,
            cpu_emitters: Vec::new(),
        }
    }

    // ── BGL entry helpers ────────────────────────────────────────────────────

    fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    // ── Bind group builders ──────────────────────────────────────────────────

    fn build_compute_bg(
        device: &wgpu::Device, bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer, particles: &wgpu::Buffer, emitters: &wgpu::Buffer,
        compact: &wgpu::Buffer, alive: &wgpu::Buffer, draw_staging: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corona Compute BG"), layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: emitters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: compact.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: alive.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: draw_staging.as_entire_binding() },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_render_bg(
        device: &wgpu::Device, bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer, particles: &wgpu::Buffer, emitters: &wgpu::Buffer,
        compact: &wgpu::Buffer, alive: &wgpu::Buffer, draw_staging: &wgpu::Buffer,
        camera: &wgpu::Buffer, tex_view: &wgpu::TextureView, sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corona Render BG"), layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: emitters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: compact.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: alive.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: draw_staging.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(tex_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    // ── Texture helper ───────────────────────────────────────────────────────

    fn make_soft_circle_tex(size: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((size * size * 4) as usize);
        let half = size as f32 * 0.5;
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f32 + 0.5 - half) / half;
                let dy = (y as f32 + 0.5 - half) / half;
                let d  = (dx * dx + dy * dy).sqrt();
                let a  = (1.0 - d).clamp(0.0, 1.0);
                let s  = a * a * (3.0 - 2.0 * a);
                data.extend_from_slice(&[255, 255, 255, (s * 255.0) as u8]);
            }
        }
        data
    }
}

// ── RenderPass impl ──────────────────────────────────────────────────────────

impl RenderPass for CoronaPass {
    fn name(&self) -> &'static str { "Corona" }

    fn reads(&self) -> &'static [helio_v3::graph::ResourceSlot] {
        &[
            helio_v3::graph::ResourceSlot::PreAa,
            helio_v3::graph::ResourceSlot::FullResDepth,
            helio_v3::graph::ResourceSlot::CoronaEmitters,
        ]
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if let Some(data) = ctx.frame_resources.corona_emitters.get() {
            if data.generation != self.uploaded_generation {
                let em_size = std::mem::size_of::<libhelio::GpuCoronaEmitter>();
                let count   = (data.count as usize).min(MAX_EMITTERS as usize);
                let src: &[libhelio::GpuCoronaEmitter] =
                    bytemuck::cast_slice(&data.emitters[..count * em_size]);

                // Grow/shrink CPU list; new slots start zero (spawn_cursor=0).
                self.cpu_emitters.resize(count, libhelio::GpuCoronaEmitter::zeroed());

                // Assign non-overlapping particle ranges and preserve spawn cursors
                // so particles travel across frames rather than resetting each update.
                let mut upload = self.cpu_emitters[..count].to_vec();
                let dt = ctx.delta_time;
                let mut offset = 0u32;
                for (i, src_em) in src.iter().enumerate() {
                    let this_cursor = self.cpu_emitters[i].spawn_cursor;
                    let emit_count  = (src_em.emit_params[0] * dt) as u32;
                    let range       = src_em.particle_count.max(1);

                    upload[i] = *src_em;
                    upload[i].particle_offset = offset;
                    upload[i].spawn_cursor    = this_cursor;

                    // Advance CPU cursor to mirror what GPU will do in cs_emit.
                    self.cpu_emitters[i].spawn_cursor = (this_cursor + emit_count) % range;

                    offset = offset.saturating_add(src_em.particle_count);
                }

                let bytes: &[u8] = bytemuck::cast_slice(&upload);
                let cap = (MAX_EMITTERS as usize * em_size).min(bytes.len());
                ctx.write_buffer(&self.emitter_buf, 0, &bytes[..cap]);

                self.emitter_count       = count as u32;
                self.uploaded_generation = data.generation;
                self.max_particles       = data.max_particles.clamp(1024, DEFAULT_MAX_PARTICLES);
            }
        } else {
            self.emitter_count = 0;
        }

        let uniforms = CoronaUniforms {
            delta_time:      ctx.delta_time,
            total_particles: self.max_particles,
            emitter_count:   self.emitter_count,
            frame_count:     ctx.frame_num as u32,
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.emitter_count == 0 { return Ok(()); }

        // ── Bind group rebuild when camera or particle buffer pointers change ─

        let part_ptr   = &self.particle_buf as *const _ as usize;
        let camera_ptr = ctx.scene.camera   as *const _ as usize;

        if self.compute_bg_key != Some(part_ptr) {
            self.compute_bg = Some(Self::build_compute_bg(
                ctx.device, &self.compute_bgl,
                &self.uniform_buf, &self.particle_buf, &self.emitter_buf,
                &self.compact_buf, &self.emitter_alive_buf, &self.draw_args_staging,
            ));
            self.compute_bg_key = Some(part_ptr);
        }
        if self.render_bg_key != Some(camera_ptr) {
            self.render_bg = Some(Self::build_render_bg(
                ctx.device, &self.render_bgl,
                &self.uniform_buf, &self.particle_buf, &self.emitter_buf,
                &self.compact_buf, &self.emitter_alive_buf, &self.draw_args_staging,
                ctx.scene.camera, &self.particle_view, &self.particle_sampler,
            ));
            self.render_bg_key = Some(camera_ptr);
        }

        let cbg = self.compute_bg.as_ref().unwrap();
        let wg  = (self.max_particles + WG - 1) / WG;
        let ec  = self.emitter_count;

        // ── Pass 1: Simulate ─────────────────────────────────────────────────
        {
            let mut p = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Simulate"), timestamp_writes: None,
            });
            p.set_pipeline(&self.simulate_pipeline);
            p.set_bind_group(0, cbg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 2: Emit ─────────────────────────────────────────────────────
        {
            let mut p = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Emit"), timestamp_writes: None,
            });
            p.set_pipeline(&self.emit_pipeline);
            p.set_bind_group(0, cbg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // ── Pass 3: Compact reset (one workgroup per emitter) ────────────────
        {
            let mut p = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona CompactReset"), timestamp_writes: None,
            });
            p.set_pipeline(&self.compact_reset_pipeline);
            p.set_bind_group(0, cbg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // ── Pass 4: Compact — scatter alive indices into compact_buf ─────────
        {
            let mut p = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Compact"), timestamp_writes: None,
            });
            p.set_pipeline(&self.compact_pipeline);
            p.set_bind_group(0, cbg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 5: Build per-emitter draw args ──────────────────────────────
        {
            let mut p = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona BuildMulti"), timestamp_writes: None,
            });
            p.set_pipeline(&self.build_multi_pipeline);
            p.set_bind_group(0, cbg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // Copy draw args from STORAGE staging → INDIRECT draw_args_buf.
        let args_size = ec as u64
            * std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        ctx.encoder.copy_buffer_to_buffer(
            &self.draw_args_staging, 0,
            &self.draw_args_buf, 0,
            args_size,
        );

        // ── Render pass ──────────────────────────────────────────────────────

        let target_view = ctx.resources.pre_aa.get().unwrap_or(ctx.target);

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Corona Render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });

        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, self.render_bg.as_ref().unwrap(), &[]);

        // One draw_indirect per emitter — each draws only its alive particles.
        let stride = std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        for i in 0..ec {
            pass.draw_indirect(&self.draw_args_buf, i as u64 * stride);
        }

        Ok(())
    }
}
