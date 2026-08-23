use std::sync::Arc;

use helio_pass_portal_cull::PortalCullPass;

async fn create_pass() -> Option<PortalCullPass> {
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
            label: Some("Portal Cull Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("adapter must create a device");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("portal-cull GPU validation error: {error:?}");
    }));

    Some(PortalCullPass::new(&device))
}

#[test]
fn pipelines_compile_on_the_available_backend() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping portal-cull portability test: no GPU adapter available");
        return;
    };
}
