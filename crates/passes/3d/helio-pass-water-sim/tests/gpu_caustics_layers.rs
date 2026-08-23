//! Stable water simulation slots are also stable caustics array layers.
//!
//! This exercises the production caustics shader with two displaced volumes,
//! samples the result through their compact projection rows, then models a
//! removal/promotion/reuse transition. The vacated layer must be black before
//! a new volume is allowed to occupy it.

use std::sync::Arc;

use wgpu::util::DeviceExt;

const SIZE: u32 = 64;
const DETAIL: u32 = 32;
const SIM_SLOTS: u32 = 8;
const SIM_LAYERS: u32 = SIM_SLOTS * 3;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mantissa >> 13) as u16)
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

fn synthetic_heightfield(wavelength: f32, amplitude: f32) -> Vec<u16> {
    let n = SIZE as usize;
    let mut height = vec![0.0f32; n * n];
    let k = std::f32::consts::TAU / wavelength;
    for y in 0..n {
        for x in 0..n {
            height[y * n + x] = amplitude
                * ((x as f32 * k).sin()
                    + 0.65 * ((y as f32 * 0.83 + x as f32 * 0.31) * k).sin());
        }
    }

    let delta = 1.0 / SIZE as f32;
    let mut texels = vec![0u16; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let h = height[y * n + x];
            let h_right = height[y * n + (x + 1) % n];
            let h_up = height[((y + 1) % n) * n + x];
            let nx = -(h_right - h) * delta;
            let ny = delta * delta;
            let nz = -(h_up - h) * delta;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-9);
            let i = (y * n + x) * 4;
            texels[i] = f32_to_f16(h);
            texels[i + 1] = f32_to_f16(0.0);
            texels[i + 2] = f32_to_f16(nx / len);
            texels[i + 3] = f32_to_f16(nz / len);
        }
    }
    texels
}

fn volume(bounds_min: [f32; 3], bounds_max: [f32; 3], intensity: f32) -> libhelio::GpuWaterVolume {
    let mut volume = libhelio::GpuWaterVolume::default_lake();
    volume.bounds_min = [bounds_min[0], bounds_min[1], bounds_min[2], 0.0];
    volume.bounds_max = [bounds_max[0], bounds_max[1], bounds_max[2], bounds_max[1] - 0.5];
    volume.wave_params[0] = 0.4;
    volume.caustics_params = [1.0, intensity, 8.0, 0.0];
    volume.sim_params[0] = 1.333;
    volume.sim_params[1] = intensity;
    volume.sun_direction = [0.4082483, 0.8164966, 0.4082483, 0.0];
    volume
}

fn grid_mesh() -> (Vec<[f32; 4]>, Vec<u32>) {
    let n = DETAIL + 1;
    let mut vertices = Vec::with_capacity((n * n) as usize);
    let mut indices = Vec::with_capacity((DETAIL * DETAIL * 6) as usize);
    for y in 0..n {
        for x in 0..n {
            vertices.push([
                x as f32 / DETAIL as f32 * 2.0 - 1.0,
                y as f32 / DETAIL as f32 * 2.0 - 1.0,
                0.0,
                0.0,
            ]);
        }
    }
    for y in 0..DETAIL {
        for x in 0..DETAIL {
            let tl = y * n + x;
            let tr = tl + 1;
            let bl = (y + 1) * n + x;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    (vertices, indices)
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    mean_abs: f32,
    max_abs: f32,
}

fn stats_for_rows(data: &[u8], first_row: u32, row_count: u32) -> Stats {
    let halves: &[u16] = bytemuck::cast_slice(data);
    let mut sum = 0.0f64;
    let mut max_abs = 0.0f32;
    for y in first_row..first_row + row_count {
        for x in 0..SIZE {
            let value = f16_to_f32(halves[((y * SIZE + x) * 4) as usize]).abs();
            sum += value as f64;
            max_abs = max_abs.max(value);
        }
    }
    Stats {
        mean_abs: (sum / (SIZE * row_count) as f64) as f32,
        max_abs,
    }
}

fn map_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map callback").expect("map succeeds");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let data = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    data
}

async fn layered_caustics_lifecycle() -> Option<(Stats, Stats, Stats, Stats, Stats)> {
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
            label: Some("Water Layered Caustics Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("water layered caustics GPU validation error: {error:?}");
    }));

    let sim_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Layered Caustics Sim"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: SIM_LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let write_sim_layer = |slot: u32, texels: &[u16]| {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &sim_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: slot * 3 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 8),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
    };
    write_sim_layer(2, &synthetic_heightfield(11.0, 0.55));
    write_sim_layer(6, &synthetic_heightfield(17.0, 0.38));
    let sim_view = sim_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(SIM_LAYERS),
        ..Default::default()
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        min_filter: wgpu::FilterMode::Linear,
        mag_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        ..Default::default()
    });
    let volumes = [
        volume([-27.0, -3.0, -16.0], [-13.0, 3.0, -2.0], 0.7),
        volume([38.0, -4.0, 21.0], [54.0, 4.0, 37.0], 1.8),
        volume([-72.0, -2.0, 43.0], [-58.0, 5.0, 57.0], 1.15),
    ];
    let volumes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Layered Caustics Volumes"),
        contents: bytemuck::cast_slice(&volumes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let initial_projections = [[0u32, 2u32], [1u32, 6u32]];
    let projections_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Layered Caustics Projections"),
        contents: bytemuck::cast_slice(&initial_projections),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let producer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Layered Caustics Producer BGL"),
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
    let producer_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Layered Caustics Producer BG"),
        layout: &producer_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: volumes_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sim_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: projections_buffer.as_entire_binding() },
        ],
    });
    let producer_shader = helio_core::shader::module(
        &device,
        "Water Layered Caustics Producer Shader",
        include_str!("../shaders/caustics.wgsl"),
    );
    let producer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Layered Caustics Producer Layout"),
        bind_group_layouts: &[Some(&producer_bgl)],
        immediate_size: 0,
    });
    let producer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Layered Caustics Producer Pipeline"),
        layout: Some(&producer_layout),
        vertex: wgpu::VertexState {
            module: &producer_shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                }],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &producer_shader,
            entry_point: Some("fs_main"),
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
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let (vertices, indices) = grid_mesh();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Layered Caustics Vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Layered Caustics Indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let caustics_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Layered Caustics Array"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: SIM_SLOTS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let caustics_view = caustics_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(SIM_SLOTS),
        ..Default::default()
    });
    let layer_views: Vec<_> = (0..SIM_SLOTS)
        .map(|layer| {
            caustics_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();

    let consumer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Layered Caustics Consumer Shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Projection { entity_row: u32, sim_slot: u32 }
@group(0) @binding(0) var caustics: texture_2d_array<f32>;
@group(0) @binding(1) var caustics_sampler: sampler;
@group(0) @binding(2) var<storage, read> projections: array<Projection>;

struct Out { @builtin(position) position: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> Out {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: Out;
    out.position = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
    let projection_index = min(u32(p.y) / 64u, 1u);
    let local_y = fract(p.y / 64.0);
    let uv = vec2f(p.x / 64.0, local_y);
    let layer = projections[projection_index].sim_slot;
    return textureSampleLevel(caustics, caustics_sampler, uv, layer, 0.0);
}
"#
            .into(),
        ),
    });
    let consumer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Layered Caustics Consumer BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let consumer_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Layered Caustics Consumer BG"),
        layout: &consumer_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&caustics_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: projections_buffer.as_entire_binding() },
        ],
    });
    let consumer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Layered Caustics Consumer Layout"),
        bind_group_layouts: &[Some(&consumer_bgl)],
        immediate_size: 0,
    });
    let consumer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Layered Caustics Consumer Pipeline"),
        layout: Some(&consumer_layout),
        vertex: wgpu::VertexState {
            module: &consumer_shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &consumer_shader,
            entry_point: Some("fs"),
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
    let sampled_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Layered Caustics Sampled Output"),
        size: wgpu::Extent3d { width: SIZE, height: SIZE * 2, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let sampled_view = sampled_texture.create_view(&Default::default());
    let sampled_bytes = (SIZE * 8 * SIZE * 2) as u64;
    let make_readback = |label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: sampled_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };

    let encode_producer = |encoder: &mut wgpu::CommandEncoder, instance: u32, slot: u32| {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water Layered Caustics Produce"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &layer_views[slot as usize],
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
        pass.set_pipeline(&producer_pipeline);
        pass.set_bind_group(0, &producer_bg, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..indices.len() as u32, 0, instance..instance + 1);
    };
    let encode_consumer = |encoder: &mut wgpu::CommandEncoder, readback: &wgpu::Buffer| {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Water Layered Caustics Sample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &sampled_view,
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
            pass.set_pipeline(&consumer_pipeline);
            pass.set_bind_group(0, &consumer_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            sampled_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 8),
                    rows_per_image: Some(SIZE * 2),
                },
            },
            wgpu::Extent3d { width: SIZE, height: SIZE * 2, depth_or_array_layers: 1 },
        );
    };

    let initial_readback = make_readback("Water Layered Caustics Initial Readback");
    let mut encoder = device.create_command_encoder(&Default::default());
    encode_producer(&mut encoder, 0, 2);
    encode_producer(&mut encoder, 1, 6);
    encode_consumer(&mut encoder, &initial_readback);
    queue.submit([encoder.finish()]);
    let initial_data = map_buffer(&device, &initial_readback);
    let initial_first = stats_for_rows(&initial_data, 0, SIZE);
    let initial_second = stats_for_rows(&initial_data, SIZE, SIZE);

    // Remove row 0: row 1 is promoted to compact projection 0 but retains
    // stable slot 6. Slot 2 is cleared at the same residency-generation edge.
    queue.write_buffer(
        &projections_buffer,
        0,
        bytemuck::cast_slice(&[[1u32, 6u32], [1u32, 6u32]]),
    );
    let cleared_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Water Layered Caustics Cleared Layer Readback"),
        size: (SIZE * 8 * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water Layered Caustics Clear Removed Slot"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &layer_views[2],
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
        drop(pass);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &caustics_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &cleared_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 8),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let cleared_data = map_buffer(&device, &cleared_readback);
    let cleared = stats_for_rows(&cleared_data, 0, SIZE);

    // Reuse slot 2 for a third, separately displaced canonical volume. The
    // promoted volume remains compact index 0 / stable slot 6; the new volume
    // is compact index 1 / reused stable slot 2.
    write_sim_layer(2, &synthetic_heightfield(7.0, 0.46));
    queue.write_buffer(
        &projections_buffer,
        0,
        bytemuck::cast_slice(&[[1u32, 6u32], [2u32, 2u32]]),
    );
    let reused_readback = make_readback("Water Layered Caustics Reused Readback");
    let mut encoder = device.create_command_encoder(&Default::default());
    encode_producer(&mut encoder, 1, 2);
    encode_consumer(&mut encoder, &reused_readback);
    queue.submit([encoder.finish()]);
    let reused_data = map_buffer(&device, &reused_readback);
    let promoted = stats_for_rows(&reused_data, 0, SIZE);
    let reused = stats_for_rows(&reused_data, SIZE, SIZE);

    Some((initial_first, initial_second, cleared, promoted, reused))
}

#[test]
fn displaced_volumes_use_distinct_stable_layers_and_reuse_starts_clear() {
    let Some((first, second, cleared, promoted, reused)) =
        pollster::block_on(layered_caustics_lifecycle())
    else {
        eprintln!("skipping layered water caustics test: no GPU adapter available");
        return;
    };

    assert!(first.mean_abs > 1e-4, "first displaced volume produced no caustics: {first:?}");
    assert!(second.mean_abs > 1e-4, "second displaced volume produced no caustics: {second:?}");
    assert!(
        (first.mean_abs - second.mean_abs).abs() > 1e-4,
        "distinct volume inputs collapsed to one sampled layer: first={first:?}, second={second:?}"
    );
    assert!(
        cleared.max_abs < 1e-5,
        "removed stable slot retained its previous occupant's caustics: {cleared:?}"
    );
    assert!(
        promoted.mean_abs > 1e-4,
        "compact-row promotion lost the surviving stable layer: {promoted:?}"
    );
    assert!(reused.mean_abs > 1e-4, "reused stable layer was not populated: {reused:?}");
    assert!(
        (promoted.mean_abs - reused.mean_abs).abs() > 1e-4,
        "promoted and reused projections sampled the same layer: promoted={promoted:?}, reused={reused:?}"
    );
}
