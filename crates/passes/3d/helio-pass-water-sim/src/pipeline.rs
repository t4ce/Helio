use wgpu::util::DeviceExt;
use crate::simulation::{DeltaUniform, DropUniform, HitboxCountUniform, NormalUniform};
use crate::{
    make_static_box_mesh, make_top_grid, make_caustics_grid, vec4_vbl, WaterSimPass, BLIT_WGSL,
    CASCADE_COUNT, SIM_SIZE, MAX_SIM_VOLUMES,
};

impl WaterSimPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        _camera_buf: &wgpu::Buffer,
        internal_width: u32,
        internal_height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let internal_width = internal_width.max(1);
        let internal_height = internal_height.max(1);
        let vert = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim VS"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fullscreen.vert.wgsl").into(),
            ),
        });
        let drop_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Drop FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/drop.frag.wgsl").into()),
        });
        let update_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Update FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/update.frag.wgsl").into()),
        });
        let normal_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Normal FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/normal.frag.wgsl").into()),
        });
        let hitbox_frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WaterSim Hitbox FS"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hitbox.frag.wgsl").into()),
        });

        let sim_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
            ],
        });

        // Updates and targeted world-space drops read authored data directly
        // from canonical SceneDB water rows. Keeping this separate from
        // `sim_bgl` avoids charging the normal pipeline an unused storage binding.
        // One storage buffer remains below every WebGPU baseline limit.
        let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim Canonical Update BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
            ],
        });

        let hitbox_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WaterSim Hitbox BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let sim_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim PL"),
            bind_group_layouts: &[Some(&sim_bgl)],
            immediate_size: 0,
        });
        let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim Canonical Update PL"),
            bind_group_layouts: &[Some(&update_bgl)],
            immediate_size: 0,
        });
        let hitbox_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WaterSim Hitbox PL"),
            bind_group_layouts: &[Some(&hitbox_bgl)],
            immediate_size: 0,
        });

        let make_sim_pipeline =
            |label, layout: &wgpu::PipelineLayout, frag: &wgpu::ShaderModule| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &vert,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: frag,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
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
                })
            };

        let drop_pipeline = make_sim_pipeline("WaterSim Targeted Drop", &update_pl, &drop_frag);
        let update_pipeline =
            make_sim_pipeline("WaterSim Canonical Update", &update_pl, &update_frag);
        let normal_pipeline = make_sim_pipeline("WaterSim Normal", &sim_pl, &normal_frag);
        let hitbox_pipeline = make_sim_pipeline("WaterSim Hitbox", &hitbox_pl, &hitbox_frag);

        let total_layers = CASCADE_COUNT as u32 * MAX_SIM_VOLUMES;

        let make_array_tex = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: SIM_SIZE,
                    height: SIM_SIZE,
                    depth_or_array_layers: total_layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let sim_tex_a = make_array_tex("WaterSim Tex A");
        let sim_tex_b = make_array_tex("WaterSim Tex B");

        let mut sim_layer_views_a = Vec::with_capacity(total_layers as usize);
        let mut sim_layer_views_b = Vec::with_capacity(total_layers as usize);
        for layer in 0..total_layers {
            let va = sim_tex_a.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("WaterSim Layer A {}", layer)),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let vb = sim_tex_b.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("WaterSim Layer B {}", layer)),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            sim_layer_views_a.push(va);
            sim_layer_views_b.push(vb);
        }

        let sim_array_view_a = sim_tex_a.create_view(&wgpu::TextureViewDescriptor {
            label: Some("WaterSim Array View A"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(total_layers),
            ..Default::default()
        });
        let sim_array_view_b = sim_tex_b.create_view(&wgpu::TextureViewDescriptor {
            label: Some("WaterSim Array View B"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(total_layers),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterSim Internal Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let output_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WaterSim Output Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Depth Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let make_ubuf = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let drop_buf = make_ubuf("WaterSim Drop Uniform", std::mem::size_of::<DropUniform>());
        let mut update_bufs_vec: Vec<wgpu::Buffer> = Vec::with_capacity(total_layers as usize);
        for _sim_slot in 0..MAX_SIM_VOLUMES {
            for _cascade in 0..CASCADE_COUNT {
                update_bufs_vec.push(make_ubuf(
                    "WaterSim Canonical Update Uniform",
                    std::mem::size_of::<DeltaUniform>(),
                ));
            }
        }
        let normal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("WaterSim Static Normal Uniform"),
            contents: bytemuck::bytes_of(&NormalUniform {
                delta: [1.0 / SIM_SIZE as f32, 1.0 / SIM_SIZE as f32],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let hitbox_count_buf =
            make_ubuf("WaterSim Hitbox Count", std::mem::size_of::<HitboxCountUniform>());
        let volume_count_buf =
            make_ubuf("WaterSim Volume Count", std::mem::size_of::<HitboxCountUniform>());

        let caustics_render_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("WaterCaustics Render BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });


        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // hiz_min — R32Float with a mip chain, read via textureLoad, so
                // it is unfilterable and needs no sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

            ],
        });

        let caustics_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Caustics Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let pre_aa_fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water PreAA Fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pre_aa_fallback_view =
            pre_aa_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let gbuffer_fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water GBuffer Fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let gbuffer_fallback_view =
            gbuffer_fallback_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Blit BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let blit_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Blit PL"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Blit Pipeline"),
            layout: Some(&blit_pl_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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

        let tint_scratch_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Tint Scratch"),
            size: wgpu::Extent3d {
                width: internal_width.max(1),
                height: internal_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tint_scratch_view =
            tint_scratch_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let underwater_tint_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Water Underwater Tint BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Scene depth — the underwater medium is integrated per
                    // pixel against the distance the view ray travels.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // Heightfield — the containment test runs against the
                    // displaced surface, not the flat plane.
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Caustics, for submerged geometry seen directly rather
                    // than through the surface.
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let underwater_tint_shader = helio_core::shader::module(
            device,
            "Water Underwater Tint Shader",
            include_str!("../shaders/underwater_fog.wgsl"),
        );
        let underwater_tint_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Underwater Tint PL"),
            bind_group_layouts: &[Some(&underwater_tint_bgl)],
            immediate_size: 0,
        });
        let underwater_tint_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Water Underwater Tint Pipeline"),
                layout: Some(&underwater_tint_pl),
                vertex: wgpu::VertexState {
                    module: &underwater_tint_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &underwater_tint_shader,
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
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let caustics_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Caustics PL"),
            bind_group_layouts: &[Some(&caustics_render_bgl)],
            immediate_size: 0,
        });
        let render_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        let caustics_shader = helio_core::shader::module(
            device,
            "Water Caustics Shader",
            include_str!("../shaders/caustics.wgsl"),
        );
        // One module, two fragment entry points: `fs_above` and `fs_under` share
        // the vertex stage and every helper, so the surface can only be shaded
        // one way. They used to be separate files with the whole coordinate and
        // SSR apparatus duplicated between them.
        let surface_shader = helio_core::shader::module(
            device,
            "Water Surface Shader",
            include_str!("../shaders/surface.wgsl"),
        );

        let vbl = vec4_vbl();

        let caustics_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Caustics Pipeline"),
            layout: Some(&caustics_pl_layout),
            vertex: wgpu::VertexState {
                module: &caustics_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vbl.clone())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &caustics_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let surface_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Water Surface Pipeline"),
                layout: Some(&render_pl_layout),
                vertex: wgpu::VertexState {
                    module: &surface_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(vbl.clone())],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &surface_shader,
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
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let (static_box_vbuf, static_box_ibuf, static_box_index_count) = make_static_box_mesh(device);
        let (top_vbuf, top_ibuf, top_index_count) = make_top_grid(device);
        let (caustics_vbuf, caustics_ibuf, caustics_index_count) = make_caustics_grid(device);

        let n_updates = total_layers as usize;

        Self {
            sim_bgl,
            update_bgl,
            hitbox_bgl,
            drop_pipeline,
            update_pipeline,
            normal_pipeline,
            hitbox_pipeline,
            sim_tex_a,
            sim_tex_b,
            sim_array_view_a,
            sim_array_view_b,
            sim_layer_views_a,
            sim_layer_views_b,
            front_per_layer: vec![true; n_updates],
            sim_slot_generations: [u64::MAX; helio_core::WATER_SIM_SLOT_COUNT],
            sampler,
            output_sampler,
            depth_sampler,
            drop_buf,
            update_bufs: update_bufs_vec,
            normal_buf,
            hitbox_count_buf,
            volume_count_buf,
            pending_drops: std::collections::VecDeque::new(),
            staged_drop: None,
            static_box_vbuf,
            static_box_ibuf,
            static_box_index_count,
            top_vbuf,
            top_ibuf,
            top_index_count,
            caustics_vbuf,
            caustics_ibuf,
            caustics_index_count,
            caustics_sampler,
            caustics_render_bgl,
            render_bgl,
            caustics_pipeline,
            surface_pipeline,
            _pre_aa_fallback_tex: pre_aa_fallback_tex,
            pre_aa_fallback_view,
            _gbuffer_fallback_tex: gbuffer_fallback_tex,
            gbuffer_fallback_view,
            internal_width,
            internal_height,
            surface_format,
            viewport_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Water Viewport"),
                contents: bytemuck::cast_slice(&[
                    internal_width as f32,
                    internal_height as f32,
                    1.0 / internal_width as f32,
                    1.0 / internal_height as f32,
                ]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            blit_bgl,
            blit_pipeline,
            blit_bg: None,
            blit_bg_key: None,
            water_output_view: None,
            caustics_bg_key: None,
            caustics_bg: None,
            render_bg: None,
            render_bg_key: None,
            normal_bgs: (0..n_updates).map(|_| [None, None]).collect(),
            normal_bg_keys: vec![[None, None]; n_updates],
            hitbox_bgs: (0..n_updates).map(|_| [None, None]).collect(),
            hitbox_bg_keys: vec![[None, None]; n_updates],
            drop_bgs: (0..n_updates).map(|_| [None, None]).collect(),
            drop_bg_keys: vec![[None, None]; n_updates],
            update_bgs: (0..n_updates).map(|_| [None, None]).collect(),
            update_bg_keys: vec![[None, None]; n_updates],
            underwater_tint_bg: None,
            underwater_tint_bg_key: None,
            tint_blit_bg: None,
            tint_blit_bg_key: None,
            _tint_scratch_tex: tint_scratch_tex,
            tint_scratch_view,
            underwater_tint_bgl,
            underwater_tint_pipeline,
            sim_time: 0.0,
        }
    }
}
