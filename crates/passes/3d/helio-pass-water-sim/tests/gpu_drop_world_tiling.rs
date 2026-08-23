//! A targeted drop is authored in world XZ, gated by canonical volume bounds,
//! and mapped through cascade-0's periodic 30 metre tile. This regression uses
//! negative coordinates so a bounds-normalized implementation cannot pass by
//! accident.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DropUniform {
    world_center: [f32; 2],
    radius: f32,
    strength: f32,
    volume_row: u32,
    _pad: [u32; 3],
}

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

async fn render_drop() -> Option<(f32, f32, f32)> {
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
            label: Some("Water Drop World Tiling Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("water drop world-tiling GPU validation error: {error:?}");
    }));

    let source = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("Water Drop Zero Source"),
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
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
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        ..Default::default()
    });

    let mut volume = libhelio::GpuWaterVolume::default_lake();
    volume.bounds_min = [-20.0, -2.0, -10.0, 0.0];
    volume.bounds_max = [-10.0, 5.0, 0.0, 2.0];
    let volumes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Drop Canonical Volume"),
        contents: bytemuck::bytes_of(&volume),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let inside_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Drop Inside Uniform"),
        contents: bytemuck::bytes_of(&DropUniform {
            world_center: [-15.0, -5.0],
            radius: 1.5,
            strength: 0.8,
            volume_row: 0,
            _pad: [0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let outside_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Drop Outside Uniform"),
        contents: bytemuck::bytes_of(&DropUniform {
            world_center: [45.0, -5.0],
            radius: 1.5,
            strength: 0.8,
            volume_row: 0,
            _pad: [0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Drop World Tiling BGL"),
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
    let make_bg = |label, uniform: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: volumes.as_entire_binding() },
            ],
        })
    };
    let inside_bg = make_bg("Water Drop Inside BG", &inside_uniform);
    let outside_bg = make_bg("Water Drop Outside BG", &outside_uniform);

    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Drop World Tiling VS"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.vert.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Drop World Tiling FS"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/drop.frag.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Drop World Tiling Layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Drop World Tiling Pipeline"),
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

    let make_target = |label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let inside_target = make_target("Water Drop Inside Target");
    let outside_target = make_target("Water Drop Outside Target");
    let inside_view = inside_target.create_view(&Default::default());
    let outside_view = outside_target.create_view(&Default::default());
    let bytes_per_image = (SIZE * SIZE * 8) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Water Drop World Tiling Readback"),
        size: bytes_per_image * 2,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    for (view, bind_group) in [(&inside_view, &inside_bg), (&outside_view, &outside_bg)] {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water Drop World Tiling Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
    for (index, texture) in [&inside_target, &outside_target].into_iter().enumerate() {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: index as u64 * bytes_per_image,
                    bytes_per_row: Some(SIZE * 8),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
    }
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

    // fract([-15, -5] / 30) = [0.5, 5/6]. Bounds normalization would put the
    // same point at [0.5, 0.5], so sample both to exclude the legacy mapping.
    let tiled = (((SIZE * 5 / 6) * SIZE + SIZE / 2) * 4) as usize;
    let legacy = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    let outside_base = (bytes_per_image / 2) as usize;
    let mut outside_max = 0.0f32;
    for pixel in 0..(SIZE * SIZE) as usize {
        outside_max = outside_max.max(f16_to_f32(halves[outside_base + pixel * 4]).abs());
    }
    Some((
        f16_to_f32(halves[tiled]),
        f16_to_f32(halves[legacy]),
        outside_max,
    ))
}

#[test]
fn targeted_drop_matches_negative_world_tiling_and_canonical_bounds() {
    let Some((signal, legacy_signal, outside_max)) = pollster::block_on(render_drop()) else {
        eprintln!("skipping water drop world-tiling test: no GPU adapter available");
        return;
    };
    assert!(signal > 0.05, "negative world-space drop missed cascade-0 tile: {signal}");
    assert!(
        legacy_signal.abs() < 0.001,
        "removed bounds-normalized drop mapping still produced a midpoint signal: {legacy_signal}"
    );
    assert!(
        outside_max < 0.001,
        "shader accepted a forged target outside the canonical volume: {outside_max}"
    );
}
