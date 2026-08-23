use std::sync::{mpsc, Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

// Column-major clip transform `clip.x = world.x + 10`. The visible x range is
// therefore [-11, -9]. This deliberately catches treating WGSL matrix columns
// as Gribb-Hartmann rows, which an identity-only test cannot distinguish.
const SHIFTED_VP: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    10.0, 0.0, 0.0, 1.0,
];

// x' = x + 2y. Its largest column length is sqrt(5), but its spectral
// expansion is 1 + sqrt(2), so max-column radius scaling is not conservative.
const SHEARED_SPACE: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    2.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneObjectSpatial {
    model: [f32; 16],
    normal: [f32; 12],
    sphere: [f32; 4],
    flags: u32,
    pad: [u32; 3],
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn read_buffer<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let size = (count * std::mem::size_of::<T>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Cull Test Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Shadow Cull Test Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll shadow cull readback");
    receiver
        .recv()
        .expect("shadow cull map callback")
        .expect("map shadow cull output");
    let mapped = slice
        .get_mapped_range()
        .expect("read mapped shadow cull output");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}

#[test]
fn gpu_cull_keeps_a_batch_when_a_nonrepresentative_member_is_visible() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("skipping shadow batch cull regression: no GPU adapter available");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Shadow Batch Cull Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("adapter must create a shadow cull test device");
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("shadow cull GPU validation error: {error:?}");
        }));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Batch Cull Test Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_cull.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow Batch Cull Test Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let spatial = [
            // The first member of draw 0 is outside the shifted frustum.
            SceneObjectSpatial {
                model: IDENTITY,
                normal: [0.0; 12],
                sphere: [100.0, 0.0, 0.5, 0.1],
                flags: 0,
                pad: [0; 3],
            },
            // Its second member is inside, so the complete batch must survive.
            SceneObjectSpatial {
                model: IDENTITY,
                normal: [0.0; 12],
                sphere: [-10.0, 0.0, 0.5, 0.1],
                flags: 0,
                pad: [0; 3],
            },
            // Draw 1 is entirely outside and must be rejected.
            SceneObjectSpatial {
                model: IDENTITY,
                normal: [0.0; 12],
                sphere: [100.0, 0.0, 0.5, 0.1],
                flags: 0,
                pad: [0; 3],
            },
            // ALWAYS_VISIBLE must override an intentionally empty bound.
            SceneObjectSpatial {
                model: IDENTITY,
                normal: [0.0; 12],
                sphere: [100.0, 0.0, 0.5, 0.0],
                flags: 4,
                pad: [0; 3],
            },
            // The sheared sphere intersects the right plane at x=-9. Max-column
            // scaling rejects it; a conservative affine bound keeps it.
            SceneObjectSpatial {
                model: IDENTITY,
                normal: [0.0; 12],
                sphere: [-8.765, 0.0, 0.5, 0.1],
                flags: 1 << 8,
                pad: [0; 3],
            },
        ];
        let draws = [
            DrawIndexedIndirect {
                index_count: 36,
                instance_count: 2,
                first_index: 7,
                base_vertex: -3,
                first_instance: 0,
            },
            DrawIndexedIndirect {
                index_count: 24,
                instance_count: 1,
                first_index: 19,
                base_vertex: 5,
                first_instance: 2,
            },
            DrawIndexedIndirect {
                index_count: 12,
                instance_count: 1,
                first_index: 43,
                base_vertex: 9,
                first_instance: 3,
            },
            DrawIndexedIndirect {
                index_count: 18,
                instance_count: 1,
                first_index: 61,
                base_vertex: 13,
                first_instance: 4,
            },
        ];
        let uniforms = [draws.len() as u32, draws.len() as u32, 0, 0];
        let mut face_dirty = [0u32; 256];
        face_dirty[0] = 1;

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Cull Test Uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shadow_matrices = storage_buffer(
            &device,
            "Shadow Cull Test Matrix",
            bytemuck::cast_slice(&SHIFTED_VP),
        );
        let object_spatial = storage_buffer(
            &device,
            "Shadow Cull Test Spatial",
            bytemuck::cast_slice(&spatial),
        );
        let source_draws = storage_buffer(
            &device,
            "Shadow Cull Test Source Draws",
            bytemuck::cast_slice(&draws),
        );
        let output_draws = storage_buffer(
            &device,
            "Shadow Cull Test Output Draws",
            bytemuck::cast_slice(&[DrawIndexedIndirect::zeroed(); 4]),
        );
        let face_counts = storage_buffer(
            &device,
            "Shadow Cull Test Face Counts",
            bytemuck::cast_slice(&[0u32; 256]),
        );
        let face_dirty_buffer = storage_buffer(
            &device,
            "Shadow Cull Test Face Dirty",
            bytemuck::cast_slice(&face_dirty),
        );
        let coordinate_spaces = storage_buffer(
            &device,
            "Shadow Cull Test Coordinate Spaces",
            bytemuck::cast_slice(&[IDENTITY, SHEARED_SPACE]),
        );
        let source_indices = storage_buffer(
            &device,
            "Shadow Cull Test Source Indices",
            bytemuck::cast_slice(&[0u32, 1, 2, 3, 4]),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Batch Cull Test Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shadow_matrices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: object_spatial.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: source_draws.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_draws.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: face_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: face_dirty_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: coordinate_spaces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: source_indices.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Shadow Batch Cull Test Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Batch Cull Test Dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);

        let counts = read_buffer::<u32>(&device, &queue, &face_counts, 256);
        assert_eq!(counts[0], 3);
        assert!(counts[1..].iter().all(|&count| count == 0));

        let mut output = read_buffer::<DrawIndexedIndirect>(
            &device,
            &queue,
            &output_draws,
            4,
        );
        output.truncate(counts[0] as usize);
        output.sort_by_key(|draw| draw.first_instance);
        assert_eq!(output, vec![draws[0], draws[2], draws[3]]);
    });
}
