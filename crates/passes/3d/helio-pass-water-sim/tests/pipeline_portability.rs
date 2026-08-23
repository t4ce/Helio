use std::sync::Arc;

use helio_pass_water_sim::WaterSimPass;

async fn create_pass() -> Option<WaterSimPass> {
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
            label: Some("WaterSim Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("water-sim GPU validation error: {error:?}");
    }));
    let camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("WaterSim Portability Camera"),
        size: 736,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    Some(WaterSimPass::new(
        &device,
        &camera,
        16,
        16,
        wgpu::TextureFormat::Rgba16Float,
    ))
}

#[test]
fn pipelines_compile_with_downlevel_binding_limits() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping water-sim portability test: no GPU adapter available");
        return;
    };
}
