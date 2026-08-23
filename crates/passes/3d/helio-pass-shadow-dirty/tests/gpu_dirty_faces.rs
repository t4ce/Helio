use std::sync::{mpsc, Arc};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

// The x plane normals have length 2 before normalization. A sphere centred at
// x=0.575 with radius 0.1 intersects the true right plane at x=0.5; treating
// the raw plane distance as metres incorrectly rejects it.
const SCALED_X_VP: [f32; 16] = [
    2.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

const SHEARED_SPACE: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    2.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneObjectSpatial {
    model: [f32; 16],
    normal: [f32; 12],
    sphere: [f32; 4],
    flags: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectHistory {
    model: [f32; 16],
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

fn dispatch_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    face_dirty: &wgpu::Buffer,
) -> Vec<u32> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Shadow Dirty Test Encoder"),
    });
    encoder.clear_buffer(face_dirty, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shadow Dirty Test Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Test Readback"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(face_dirty, 0, &readback, 0, 256 * 4);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll shadow dirty readback");
    receiver
        .recv()
        .expect("shadow dirty map callback")
        .expect("map shadow dirty output");
    let mapped = slice
        .get_mapped_range()
        .expect("read mapped shadow dirty output");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
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
fn gpu_dirty_marks_the_old_face_and_forces_empty_topology() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("skipping shadow dirty regression: no GPU adapter available");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Shadow Dirty Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("adapter must create a shadow dirty test device");
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("shadow dirty GPU validation error: {error:?}");
        }));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Dirty Test Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_dirty.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow Dirty Test Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let current = SceneObjectSpatial {
            model: IDENTITY,
            normal: [0.0; 12],
            sphere: [100.0, 0.0, 0.5, 0.1],
            flags: 0,
            pad: [0; 3],
        };
        let previous = ObjectHistory {
            model: IDENTITY,
            sphere: [0.575, 0.0, 0.5, 0.1],
            flags: 0,
            pad: [0; 3],
        };
        let object_spatial = storage_buffer(
            &device,
            "Shadow Dirty Test Current Spatial",
            bytemuck::bytes_of(&current),
        );
        let source_indices = storage_buffer(
            &device,
            "Shadow Dirty Test Source Indices",
            bytemuck::cast_slice(&[0u32]),
        );
        let object_history = storage_buffer(
            &device,
            "Shadow Dirty Test Object History",
            bytemuck::bytes_of(&previous),
        );
        let shadow_matrices = storage_buffer(
            &device,
            "Shadow Dirty Test Matrix",
            bytemuck::cast_slice(&[SCALED_X_VP, SCALED_X_VP]),
        );
        let face_dirty = storage_buffer(
            &device,
            "Shadow Dirty Test Face Dirty",
            bytemuck::cast_slice(&[0u32; 256]),
        );
        let coordinate_spaces = storage_buffer(
            &device,
            "Shadow Dirty Test Coordinate Spaces",
            bytemuck::cast_slice(&[IDENTITY, SHEARED_SPACE]),
        );
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Dirty Test Uniforms"),
            contents: bytemuck::cast_slice(&[1u32, 1, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let light_dirty = storage_buffer(
            &device,
            "Shadow Dirty Test Light Dirty",
            bytemuck::cast_slice(&[0u32]),
        );
        let coordinate_spaces_prev = storage_buffer(
            &device,
            "Shadow Dirty Test Previous Coordinate Spaces",
            bytemuck::cast_slice(&[IDENTITY, SHEARED_SPACE]),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Dirty Test Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: object_spatial.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: object_history.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: shadow_matrices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: face_dirty.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: coordinate_spaces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: light_dirty.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: coordinate_spaces_prev.as_entire_binding() },
            ],
        });

        let old_face = dispatch_and_read(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &face_dirty,
        );
        assert_eq!(old_face[0], 1, "the previous sphere must dirty the face it exited");
        assert!(old_face[1..].iter().all(|&dirty| dirty == 0));

        let sheared_current = SceneObjectSpatial {
            model: IDENTITY,
            normal: [0.0; 12],
            sphere: [100.0, 0.0, 0.5, 0.1],
            flags: 1 << 8,
            pad: [0; 3],
        };
        let sheared_previous = ObjectHistory {
            model: IDENTITY,
            sphere: [0.735, 0.0, 0.5, 0.1],
            flags: 1 << 8,
            pad: [0; 3],
        };
        queue.write_buffer(&object_spatial, 0, bytemuck::bytes_of(&sheared_current));
        queue.write_buffer(&object_history, 0, bytemuck::bytes_of(&sheared_previous));
        queue.write_buffer(&uniforms, 0, bytemuck::cast_slice(&[1u32, 1, 0, 0]));
        let sheared_old_face = dispatch_and_read(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &face_dirty,
        );
        assert_eq!(
            sheared_old_face[0],
            1,
            "affine shear must not make the previous sphere radius non-conservative",
        );
        assert!(sheared_old_face[1..].iter().all(|&dirty| dirty == 0));

        // ALWAYS_VISIBLE is also a dirty-selection escape hatch: deliberately
        // unusable bounds cannot safely choose a subset of shadow faces.
        let always_visible = SceneObjectSpatial {
            model: IDENTITY,
            normal: [0.0; 12],
            sphere: [100.0, 0.0, 0.5, 0.0],
            flags: 4,
            pad: [0; 3],
        };
        let previous_outside = ObjectHistory {
            model: IDENTITY,
            sphere: [200.0, 0.0, 0.5, 0.1],
            flags: 0,
            pad: [0; 3],
        };
        queue.write_buffer(&object_spatial, 0, bytemuck::bytes_of(&always_visible));
        queue.write_buffer(&object_history, 0, bytemuck::bytes_of(&previous_outside));
        queue.write_buffer(&uniforms, 0, bytemuck::cast_slice(&[1u32, 2, 0, 0]));
        let no_cull_contract = dispatch_and_read(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &face_dirty,
        );
        assert_eq!(&no_cull_contract[..2], &[1, 1]);
        assert!(no_cull_contract[2..].iter().all(|&dirty| dirty == 0));

        // Removing the last movable caster produces no object invocation. The
        // forced topology thread must still dirty active faces for ShadowPass.
        queue.write_buffer(&uniforms, 0, bytemuck::cast_slice(&[0u32, 1, 1, 0]));
        let empty_topology = dispatch_and_read(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &face_dirty,
        );
        assert_eq!(empty_topology[0], 1);
        assert!(empty_topology[1..].iter().all(|&dirty| dirty == 0));
    });
}
