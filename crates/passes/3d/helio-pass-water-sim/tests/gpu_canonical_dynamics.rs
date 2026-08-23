//! Per-volume water dynamics must come from the canonical SceneDB partner
//! buffer. Editing a row in place must affect the next update without mutating
//! pass state or rebuilding its bind group.

use std::sync::Arc;

use wgpu::util::DeviceExt;

const SIZE: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateUniform {
    delta: [f32; 2],
    time: f32,
    time_step: f32,
    cascade_patch_size: f32,
    volume_row: u32,
    _pad: [u32; 2],
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

fn render_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::Texture,
    target_view: &wgpu::TextureView,
) -> Vec<[f32; 2]> {
    let row_bytes = SIZE * 8;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Water Canonical Dynamics Readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Water Canonical Dynamics Encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water Canonical Dynamics Draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
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
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
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
    let values = halves
        .chunks_exact(4)
        .map(|pixel| [f16_to_f32(pixel[0]), f16_to_f32(pixel[1])])
        .collect();
    drop(data);
    readback.unmap();
    values
}

async fn canonical_dynamics_outputs() -> Option<(f32, f32)> {
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
            label: Some("Water Canonical Dynamics Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("water canonical dynamics GPU validation error: {error:?}");
    }));

    // Alternating height plus uniform velocity exercises both authored spring
    // and damping, while the second row additionally injects authored wind.
    let mut source_halves = vec![0u16; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let pixel = ((y * SIZE + x) * 4) as usize;
            source_halves[pixel] = if (x + y) % 2 == 0 { 0x3800 } else { 0xb800 };
            source_halves[pixel + 1] = 0x3400;
        }
    }
    let source = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("Water Canonical Dynamics Source"),
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
        bytemuck::cast_slice(&source_halves),
    );
    let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Canonical Dynamics Target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        min_filter: wgpu::FilterMode::Linear,
        mag_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let mut calm = libhelio::GpuWaterVolume::default_lake();
    calm.wave_params[2] = 0.75;
    calm.sim_dynamics = [0.2, 0.2, 0.25, 0.0];
    calm.wind_params = [0.0, 0.0, 0.0, 0.0];
    let mut energetic = calm;
    energetic.wave_params[2] = 3.0;
    energetic.sim_dynamics = [1.8, 0.95, 1.75, 0.0];
    energetic.wind_params = [0.6, 0.8, 5.0, 0.0];
    let volumes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Water Canonical Dynamics Rows"),
        contents: bytemuck::cast_slice(&[calm, energetic]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Canonical Dynamics BGL"),
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
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Canonical Dynamics PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Canonical Dynamics VS"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fullscreen.vert.wgsl").into()),
    });
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Canonical Dynamics FS"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/update.frag.wgsl").into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Canonical Dynamics Pipeline"),
        layout: Some(&pipeline_layout),
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

    let uniforms = [0u32, 1].map(|volume_row| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Canonical Dynamics Stable-Slot Uniform"),
            contents: bytemuck::bytes_of(&UpdateUniform {
                delta: [1.0 / SIZE as f32, 1.0 / SIZE as f32],
                time: 1.25,
                time_step: 1.0 / 60.0,
                cascade_patch_size: 30.0,
                volume_row,
                _pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    });
    let bind_groups = uniforms.each_ref().map(|uniform| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Canonical Dynamics Stable-Slot BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: volumes.as_entire_binding(),
                },
            ],
        })
    });

    let calm_output = render_and_read(
        &device,
        &queue,
        &pipeline,
        &bind_groups[0],
        &target,
        &target_view,
    );
    let energetic_output = render_and_read(
        &device,
        &queue,
        &pipeline,
        &bind_groups[1],
        &target,
        &target_view,
    );
    let initial_difference = calm_output
        .iter()
        .zip(&energetic_output)
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum::<f32>()
        / calm_output.len() as f32;

    // Edit the canonical row in place. `bind_groups[1]` and its stable-slot
    // uniform are deliberately reused without any pass-side parameter change.
    queue.write_buffer(
        &volumes,
        std::mem::size_of::<libhelio::GpuWaterVolume>() as u64,
        bytemuck::bytes_of(&calm),
    );
    let edited_output = render_and_read(
        &device,
        &queue,
        &pipeline,
        &bind_groups[1],
        &target,
        &target_view,
    );
    let post_edit_difference = calm_output
        .iter()
        .zip(&edited_output)
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum::<f32>()
        / calm_output.len() as f32;

    Some((initial_difference, post_edit_difference))
}

#[test]
fn canonical_rows_drive_independent_dynamics_and_hot_edits() {
    let Some((initial_difference, post_edit_difference)) =
        pollster::block_on(canonical_dynamics_outputs())
    else {
        eprintln!("skipping water canonical dynamics test: no GPU adapter available");
        return;
    };

    assert!(
        initial_difference > 0.05,
        "two differently authored water volumes did not diverge: {initial_difference}"
    );
    assert!(
        post_edit_difference < 0.001,
        "an in-place canonical edit did not affect the reused binding: {post_edit_difference}"
    );
}
