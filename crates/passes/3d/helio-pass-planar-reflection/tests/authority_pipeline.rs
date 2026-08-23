use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DispatchUniform {
    reflector_count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReflectorRow {
    position_tolerance: [f32; 4],
    normal_cos_threshold: [f32; 4],
    tangent_priority: [f32; 4],
    half_extents_reserved: [f32; 4],
}

async fn context() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapter = None;
    for force_fallback_adapter in [false, true] {
        if let Ok(candidate) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            adapter = Some(candidate);
            break;
        }
    }
    let adapter = adapter?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Planar Reflection Default-Limits Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .ok()?;
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("planar reflection pipeline validation error: {error:?}");
    }));
    Some((device, queue))
}

fn storage_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn read_output(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bg0: &wgpu::BindGroup,
    bg1: &wgpu::BindGroup,
    bg2: &wgpu::BindGroup,
    output: &wgpu::Texture,
) -> [u8; 8] {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planar Reflection Readback"),
        size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planar Reflection Contract Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planar Reflection Contract Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg0, &[]);
        pass.set_bind_group(1, bg1, &[]);
        pass.set_bind_group(2, bg2, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("map planar-reflection output")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll planar-reflection output");
    let mapped = slice
        .get_mapped_range()
        .expect("read planar-reflection output");
    let result = mapped[..8].try_into().unwrap();
    drop(mapped);
    readback.unmap();
    result
}

#[test]
fn real_pipeline_rejects_a_displaced_parallel_plane_and_compiles_at_baseline_limits() {
    let Some((device, queue)) = pollster::block_on(context()) else {
        eprintln!("skipping planar-reflection GPU test: no adapter");
        return;
    };

    // Construct the production pass as well as the direct contract harness:
    // this validates its complete three-group layout at downlevel defaults.
    let production_camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planar Production Camera"),
        size: (std::mem::size_of::<libhelio::GpuCameraUniforms>() * 2) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let _production = helio_pass_planar_reflection::PlanarReflectionPass::new(
        &device,
        &production_camera,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let shader = helio_core::shader::module(
        &device,
        "Planar Reflection Contract Shader",
        include_str!("../shaders/planar_trace.wgsl"),
    );
    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Planar Contract BGL0"),
        entries: &[
            storage_layout_entry(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Planar Contract BGL1"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Planar Contract BGL2"),
        entries: &[storage_layout_entry(0), storage_layout_entry(1)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Planar Contract Layout"),
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Planar Contract Pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let trace_projection = [
        0.1, 0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.001, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let camera = libhelio::GpuCameraUniforms {
        view: identity,
        proj: trace_projection,
        view_proj: identity,
        inv_view_proj: identity,
        position_near: [0.0, 1.0, 1.0, 0.1],
        forward_far: [0.0, 0.0, -1.0, 100.0],
        jitter_frame: [0.0; 4],
        prev_view_proj: identity,
    };
    let cameras = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Planar Contract Cameras"),
        contents: bytemuck::cast_slice(&[camera, camera]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Planar Contract Dispatch"),
        contents: bytemuck::bytes_of(&DispatchUniform {
            reflector_count: 2,
            _pad: [0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Planar Contract BG0"),
        layout: &bgl0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cameras.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dispatch.as_entire_binding(),
            },
        ],
    });

    let sampled = |label: &'static str, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    };
    let normal = sampled("Planar Contract Normal", wgpu::TextureFormat::Rgba16Float);
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Planar Contract Color"),
        size: wgpu::Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let depth = sampled("Planar Contract Depth", wgpu::TextureFormat::Depth32Float);
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Planar Contract Output"),
        size: normal.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    queue.write_texture(
        normal.as_image_copy(),
        bytemuck::cast_slice(&[0x0000_u16, 0x3c00, 0x0000, 0x0000]),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        normal.size(),
    );
    queue.write_texture(
        color.as_image_copy(),
        // Red at the left texel, blue at the right. Tilted reflector normals
        // trace to opposite sides, making winner selection observable.
        bytemuck::cast_slice(&[
            0x3c00_u16, 0x0000, 0x0000, 0x3c00,
            0x0000, 0x0000, 0x3c00, 0x3c00,
        ]),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(1),
        },
        color.size(),
    );
    let depth_clear_view = depth.create_view(&Default::default());
    let mut depth_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planar Contract Depth Clear"),
    });
    {
        let _pass = depth_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Planar Contract Depth Pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_clear_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.5),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    queue.submit([depth_encoder.finish()]);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Planar Contract Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let normal_view = normal.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let color_view = color.create_view(&Default::default());
    let output_view = output.create_view(&Default::default());
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Planar Contract BG1"),
        layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });
    let row = |plane_y: f32, normal: [f32; 3], priority: f32| ReflectorRow {
        // Identity inverse VP reconstructs this pixel at (0, 0, 0.5).
        position_tolerance: [0.0, plane_y, 0.5, 0.05],
        normal_cos_threshold: [normal[0], normal[1], normal[2], 0.8],
        tangent_priority: [normal[1], -normal[0], 0.0, priority],
        half_extents_reserved: [10.0, 10.0, 0.0, 0.0],
    };
    let run = |rows: [ReflectorRow; 2], active_rows: [u32; 2]| {
        let reflectors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planar Contract Canonical Rows"),
            contents: bytemuck::cast_slice(&rows),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let projection = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planar Contract Projection"),
            contents: bytemuck::cast_slice(&active_rows),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planar Contract BG2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reflectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: projection.as_entire_binding(),
                },
            ],
        });
        read_output(&device, &queue, &pipeline, &bg0, &bg1, &bg2, &output)
    };

    let displaced = run(
        [
            row(5.0, [0.0, 1.0, 0.0], 0.0),
            row(-5.0, [0.0, 1.0, 0.0], 0.0),
        ],
        [0, 1],
    );
    assert_eq!(
        displaced, [0; 8],
        "a parallel surface outside the authored plane tolerance must not match"
    );
    let coincident = run(
        [
            row(0.0, [0.0, 1.0, 0.0], 0.0),
            row(5.0, [0.0, 1.0, 0.0], 0.0),
        ],
        [0, 1],
    );
    assert!(
        coincident.iter().any(|byte| *byte != 0),
        "a coincident, aligned surface must produce a reflection sample"
    );

    const TILT_Y: f32 = 0.866_025_4;
    let positive_tilt = row(0.0, [0.5, TILT_Y, 0.0], 0.0);
    let negative_tilt = row(0.0, [-0.5, TILT_Y, 0.0], 0.0);
    let nonmatch = row(5.0, [0.0, 1.0, 0.0], 0.0);
    let row_zero_reference = run([positive_tilt, nonmatch], [0, 1]);
    let row_one_reference = run([nonmatch, negative_tilt], [0, 1]);
    assert_ne!(row_zero_reference, row_one_reference);

    let equal_priority_reversed_projection = run(
        [positive_tilt, negative_tilt],
        [1, 0],
    );
    assert_eq!(
        equal_priority_reversed_projection, row_zero_reference,
        "equal priorities must select the lower canonical row, not active-list order"
    );

    let higher_priority_row_one = run(
        [
            positive_tilt,
            ReflectorRow {
                tangent_priority: [
                    negative_tilt.tangent_priority[0],
                    negative_tilt.tangent_priority[1],
                    negative_tilt.tangent_priority[2],
                    1.0,
                ],
                ..negative_tilt
            },
        ],
        [0, 1],
    );
    assert_eq!(
        higher_priority_row_one, row_one_reference,
        "priority must win before canonical-row tie-breaking"
    );
}
