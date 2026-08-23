use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlanarDispatch {
    reflector_count: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<PlanarDispatch>() == 16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlanarSceneBindingKey {
    canonical_epoch: Option<u64>,
    projection_epoch: u64,
}

pub struct PlanarReflectionPass {
    pipeline: wgpu::ComputePipeline,
    bgl_1: wgpu::BindGroupLayout,
    bgl_2: wgpu::BindGroupLayout,
    bg_0: wgpu::BindGroup,
    bg_1: Option<wgpu::BindGroup>,
    bg_1_key: Option<(usize, usize, usize, usize)>,
    bg_2: Option<wgpu::BindGroup>,
    bg_2_key: Option<PlanarSceneBindingKey>,
    linear_sampler: wgpu::Sampler,
    dispatch_buf: wgpu::Buffer,
    prepared_reflector_count: u32,
    width: u32,
    height: u32,
}

impl PlanarReflectionPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        _surface_format: wgpu::TextureFormat,
    ) -> Self {
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Planar Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let shader = helio_core::shader::module(
            device,
            "Planar Trace Shader",
            include_str!("../shaders/planar_trace.wgsl"),
        );

        let dispatch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planar Dispatch"),
            size: std::mem::size_of::<PlanarDispatch>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planar BGL0"),
            entries: &[
                buffer_camera_entry(0),
                buffer_uniform_entry(1),
            ],
        });

        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planar BGL1"),
            entries: &[
                texture_float_entry(0),
                texture_depth_entry(1),
                texture_float_entry(2),
                sampler_entry(3),
                storage_entry(4),
            ],
        });

        let bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planar SceneDB BGL2"),
            entries: &[buffer_read_only_entry(0), buffer_read_only_entry(1)],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planar PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1), Some(&bgl_2)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Planar Trace Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planar BG0"),
            layout: &bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dispatch_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bgl_1,
            bgl_2,
            bg_0,
            bg_1: None,
            bg_1_key: None,
            bg_2: None,
            bg_2_key: None,
            linear_sampler,
            dispatch_buf,
            prepared_reflector_count: u32::MAX,
            width: 0,
            height: 0,
        }
    }
}

impl RenderPass for PlanarReflectionPass {
    fn name(&self) -> &'static str {
        "PlanarReflection"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "depth", "pre_aa"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["planar_reflection"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "planar_reflection",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::MatchSurface,
        );
        builder.with_extra_usage(
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.bg_1 = None;
        self.bg_1_key = None;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let reflector_count = ctx.scene.planar_reflector_indices.len() as u32;
        if self.prepared_reflector_count != reflector_count {
            let dispatch = PlanarDispatch {
                reflector_count,
                _pad: [0; 3],
            };
            ctx.write_buffer(&self.dispatch_buf, 0, bytemuck::bytes_of(&dispatch));
            self.prepared_reflector_count = reflector_count;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gbuffer = match ctx.resources.gbuffer.read("PlanarReflection") {
            Some(g) => g,
            None => return Ok(()),
        };
        let depth_view = ctx.depth;
        let pre_aa_view = match ctx.resources.pre_aa.get() {
            Some(v) => v,
            None => return Ok(()),
        };
        let planar_tex = match ctx.resource_pool.get_view("planar_reflection") {
            Some(v) => v,
            None => return Ok(()),
        };

        let key = (
            gbuffer.normal as *const _ as usize,
            depth_view as *const _ as usize,
            pre_aa_view as *const _ as usize,
            planar_tex as *const _ as usize,
        );

        if self.bg_1_key != Some(key) {
            self.bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Planar BG1"),
                layout: &self.bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.normal),
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    texture_view_entry(2, pre_aa_view),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(planar_tex),
                    },
                ],
            }));
            self.bg_1_key = Some(key);
        }

        let scene_key = PlanarSceneBindingKey {
            canonical_epoch: ctx.scene.planar_reflector_buffer_epoch,
            projection_epoch: ctx.scene.planar_reflector_projection_epoch,
        };
        if self.bg_2_key != Some(scene_key) {
            self.bg_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Planar SceneDB BG2"),
                layout: &self.bgl_2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.planar_reflectors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.planar_reflector_indices.as_entire_binding(),
                    },
                ],
            }));
            self.bg_2_key = Some(scene_key);
        }

        let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
        let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planar Reflection Trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg_0, &[]);
        pass.set_bind_group(1, self.bg_1.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bg_2.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);

        Ok(())
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {
        // Published by the graph automatically via the resource pool name.
    }
}

#[cfg(test)]
mod tests {
    use super::PlanarSceneBindingKey;

    #[test]
    fn either_owner_allocation_epoch_invalidates_the_complete_scene_bind_key() {
        let initial = PlanarSceneBindingKey {
            canonical_epoch: Some(3),
            projection_epoch: 5,
        };
        assert_ne!(
            initial,
            PlanarSceneBindingKey {
                canonical_epoch: Some(4),
                ..initial
            }
        );
        assert_ne!(
            initial,
            PlanarSceneBindingKey {
                projection_epoch: 6,
                ..initial
            }
        );
        assert_ne!(
            initial,
            PlanarSceneBindingKey {
                canonical_epoch: None,
                ..initial
            }
        );
    }
}

fn buffer_camera_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn buffer_read_only_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_float_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn texture_view_entry<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}
