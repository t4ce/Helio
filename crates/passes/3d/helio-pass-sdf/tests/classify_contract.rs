use std::sync::Arc;

use helio_pass_sdf::{
    gpu_bvh::build_flat_bvh, REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipConfig {
    level_count: u32,
    grid_dim: u32,
    brick_size: u32,
    brick_grid_dim: u32,
    bricks_per_level: u32,
    atlas_bricks_per_axis: u32,
    base_voxel_size: f32,
    edit_count: u32,
    bvh_node_count: u32,
    terrain_enabled: u32,
    terrain_y_min: f32,
    terrain_y_max: f32,
    atlas_words_per_level: u32,
    canonical_order_scan: u32,
    _pad2: u32,
    _pad3: u32,
    voxel_sizes_lo: [f32; 4],
    voxel_sizes_hi: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScrollState {
    snap_origins: [[i32; 4]; 8],
    edit_gen: u32,
    prev_edit_gen: u32,
    _pad0: u32,
    _pad1: u32,
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
    if adapter.limits().max_storage_buffers_per_shader_stage
        < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE
    {
        return None;
    }
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage =
        REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("SDF Classify Contract Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .expect("adapter must provide the default eight-storage tier");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("SDF classify validation error: {error:?}");
    }));
    Some((device, queue))
}

fn storage_init(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF Classify Readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SDF Classify Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("map SDF result"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll SDF readback");
    let bytes = slice
        .get_mapped_range()
        .expect("mapped SDF result")
        .to_vec();
    staging.unmap();
    bytes
}

fn classify(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bounds: &[[f32; 4]],
    edit_generation: u32,
    canonical_order_scan: bool,
    world_brick: [i32; 3],
) -> (u32, Vec<u32>, u32, [i32; 4]) {
    let bvh = build_flat_bvh(bounds);
    let clip = ClipConfig {
        level_count: 1,
        grid_dim: 8,
        brick_size: 8,
        brick_grid_dim: 1,
        bricks_per_level: 1,
        atlas_bricks_per_axis: 1,
        base_voxel_size: 1.0,
        edit_count: bounds.len() as u32,
        bvh_node_count: bvh.len() as u32,
        terrain_enabled: 0,
        terrain_y_min: -1.0e10,
        terrain_y_max: 1.0e10,
        atlas_words_per_level: 1,
        canonical_order_scan: u32::from(canonical_order_scan),
        _pad2: 0,
        _pad3: 0,
        voxel_sizes_lo: [1.0, 2.0, 4.0, 8.0],
        voxel_sizes_hi: [16.0, 32.0, 64.0, 128.0],
    };
    let mut snap_origins = [[0; 4]; 8];
    snap_origins[0] = [world_brick[0], world_brick[1], world_brick[2], 0];
    let scroll = ScrollState {
        snap_origins,
        edit_gen: edit_generation,
        prev_edit_gen: 0,
        _pad0: 0,
        _pad1: 0,
    };
    let clip_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SDF Test Clip Config"),
        contents: bytemuck::bytes_of(&clip),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let scroll_buffer = storage_init(device, "SDF Test Scroll", bytemuck::bytes_of(&scroll));
    let dirty_flags = storage_init(device, "SDF Test Dirty Flags", bytemuck::cast_slice(&[1u32]));
    let bvh_buffer = storage_init(device, "SDF Test BVH", bytemuck::cast_slice(&bvh));
    let hashes = storage_init(device, "SDF Test Hashes", bytemuck::cast_slice(&[0u32]));
    let edit_lists = storage_init(
        device,
        "SDF Test Edit Lists",
        bytemuck::cast_slice(&[0u32; 65]),
    );
    let all_indices = storage_init(device, "SDF Test All Indices", bytemuck::cast_slice(&[0u32]));
    let dirty_bricks = storage_init(
        device,
        "SDF Test Dirty Bricks",
        bytemuck::cast_slice(&[[0i32; 4]; 4096]),
    );
    let indirect = storage_init(
        device,
        "SDF Test Indirect",
        bytemuck::cast_slice(&[0u32, 1, 1]),
    );
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SDF Test Classify Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: clip_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scroll_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dirty_flags.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bvh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: hashes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: edit_lists.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: all_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: dirty_bricks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: indirect.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SDF Test Classify Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Test Classify Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let list_bytes = readback(device, queue, &edit_lists, 65 * 4);
    let list: Vec<u32> = list_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let hash_bytes = readback(device, queue, &hashes, 4);
    let hash = u32::from_le_bytes(hash_bytes.try_into().unwrap());
    let coordinate_bytes = readback(device, queue, &dirty_bricks, 16);
    let mut coordinate = [0i32; 4];
    for (value, bytes) in coordinate.iter_mut().zip(coordinate_bytes.chunks_exact(4)) {
        *value = i32::from_le_bytes(bytes.try_into().unwrap());
    }
    (list[0], list[1..].to_vec(), hash, coordinate)
}

#[test]
fn classifier_restores_authored_order_hashes_content_and_marks_overflow() {
    let Some((device, queue)) = pollster::block_on(context()) else {
        eprintln!("skipping SDF classify contract: no eight-storage adapter available");
        return;
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SDF Classify Contract Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../shaders/sdf_classify.wgsl").into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SDF Classify Contract Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("cs_classify"),
        compilation_options: Default::default(),
        cache: None,
    });

    let authored = [3.0, 1.0, 4.0, 0.0, 2.0].map(|x| [x, 0.0, 0.0, 100_000.0]);
    let (count, indices, first_hash, coordinate) = classify(
        &device,
        &queue,
        &pipeline,
        &authored,
        7,
        false,
        [700, -900, 2048],
    );
    assert_eq!(count, authored.len() as u32);
    assert_eq!(&indices[..authored.len()], &[0, 1, 2, 3, 4]);
    assert_eq!(
        coordinate,
        [700, -900, 2048, 0],
        "dirty coordinates must not wrap at the former signed ten-bit ceiling",
    );
    let (_, _, second_hash, _) = classify(
        &device,
        &queue,
        &pipeline,
        &authored,
        8,
        false,
        [0; 3],
    );
    assert_ne!(
        first_hash, second_hash,
        "content generation must invalidate unchanged index membership",
    );

    let overlapping = vec![[0.0, 0.0, 0.0, 100.0]; 65];
    let (descriptor, _, _, _) = classify(
        &device,
        &queue,
        &pipeline,
        &overlapping,
        9,
        false,
        [0; 3],
    );
    assert_eq!(
        descriptor,
        u32::MAX,
        "the 65th overlap must select canonical ordered scanning, never truncate",
    );

    let (descriptor, _, _, _) = classify(
        &device,
        &queue,
        &pipeline,
        &authored,
        10,
        true,
        [0; 3],
    );
    assert_eq!(
        descriptor,
        u32::MAX,
        "intersection-bearing streams must preserve non-local boolean semantics",
    );
}
