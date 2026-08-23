use std::sync::Arc;

use helio_pass_decal::DecalPass;

async fn create_pass() -> Option<DecalPass> {
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
            label: Some("Decal Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("adapter must create a device");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("decal GPU validation error: {error:?}");
    }));
    let placeholder = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Decal Portability Placeholder"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    Some(DecalPass::new(
        &device,
        &queue,
        &placeholder,
        &placeholder,
        1,
        1,
    ))
}

#[test]
fn pipelines_compile_on_the_available_backend() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping decal portability test: no GPU adapter available");
        return;
    };
}
