//! The hitbox pass must map simulation UV through the selected water volume's
//! authored world bounds. A hardcoded [-1,1] pool silently misses this case.

use std::sync::Arc;

use wgpu::util::DeviceExt;

const SIZE: u32 = 64;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;
    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let shift = mantissa.leading_zeros() - 21;
        let e = 127 - 15 - shift;
        let m = (mantissa << (shift + 1)) & 0x03ff;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mantissa << 13))
}

async fn displaced_center_signals() -> Option<(f32, f32)> {
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
            label: Some("Water Hitbox Bounds Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("water hitbox GPU validation error: {error:?}");
    }));

    let source = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("Water Hitbox Zero Source"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &vec![0; (SIZE * SIZE * 8) as usize],
    );
    let source_view = source.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        min_filter: wgpu::FilterMode::Linear,
        mag_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let hitbox = libhelio::GpuWaterHitbox {
        old_min: [-16.0, 0.0, -6.0, 0.0],
        old_max: [-14.0, 3.0, -4.0, 0.0],
        new_min: [100.0, 0.0, 100.0, 0.0],
        new_max: [101.0, 1.0, 101.0, 0.0],
        params: [0.25, 1.0, 0.0, 0.0],
    };
    let mut volume = libhelio::GpuWaterVolume::default_lake();
    volume.bounds_min = [-20.0, -2.0, -10.0, 0.0];
    volume.bounds_max = [-10.0, 5.0, 0.0, 2.0];

    let count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Hitbox Test Count"),
        contents: bytemuck::cast_slice(&[1u32, 0, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let hitboxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Hitbox Test Rows"),
        contents: bytemuck::bytes_of(&hitbox),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let hitbox_rows = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Hitbox Test Active Rows"),
        contents: bytemuck::bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let volumes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Hitbox Test Volumes"),
        contents: bytemuck::bytes_of(&volume),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let projections = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Hitbox Test Projections"),
        contents: bytemuck::cast_slice(&[0u32, 0]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let entries = [
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
    ];
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Hitbox Bounds Test BGL"),
        entries: &entries,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Hitbox Bounds Test BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: hitboxes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: hitbox_rows.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: volumes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: projections.as_entire_binding() },
        ],
    });
    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Hitbox Bounds Test VS"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.vert.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Hitbox Bounds Test FS"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hitbox.frag.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Hitbox Bounds Test PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Hitbox Bounds Test Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vertex,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Hitbox Bounds Test Target"),
        size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let row_bytes = SIZE * 8;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Water Hitbox Bounds Test Readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water Hitbox Bounds Test Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map callback").expect("map succeeds");
    let data = slice.get_mapped_range().expect("mapped range");
    let halves: &[u16] = bytemuck::cast_slice(&data);
    // fract([-15, -5] / 30) = [0.5, 5/6]. The volume-bounds midpoint is
    // [0.5, 0.5], which was the old (incorrect) mapping.
    let tiled = (((SIZE * 5 / 6) * SIZE + SIZE / 2) * 4) as usize;
    let legacy_bounds_midpoint = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    Some((
        f16_to_f32(halves[tiled]),
        f16_to_f32(halves[legacy_bounds_midpoint]),
    ))
}

#[test]
fn negative_world_hitbox_matches_surface_cascade_tiling() {
    let Some((signal, legacy_signal)) = pollster::block_on(displaced_center_signals()) else {
        eprintln!("skipping water hitbox bounds test: no GPU adapter available");
        return;
    };
    assert!(
        signal > 0.05,
        "hitbox at a negative translated world point produced no tiled displacement: {signal}"
    );
    assert!(
        legacy_signal.abs() < 0.001,
        "the removed volume-bounds mapping still produced a false midpoint match: {legacy_signal}"
    );
}
