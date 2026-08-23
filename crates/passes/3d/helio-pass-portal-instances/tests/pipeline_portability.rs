use std::sync::Arc;

use helio_pass_portal_instances::PortalInstancePass;

async fn create_pass() -> Option<PortalInstancePass> {
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
    let (device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Portal Instance Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("adapter must create a device");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("portal-instance GPU validation error: {error:?}");
    }));

    let indirect = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Portal Instance Portability Indirect"),
        size: 20,
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }));
    let projections = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Portal Instance Portability Projections"),
        size: 12,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    }));

    Some(PortalInstancePass::new(&device, indirect, projections))
}

#[test]
fn pipeline_compiles_on_the_available_backend_without_bindless_features() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping portal-instance portability test: no GPU adapter available");
        return;
    };
}
