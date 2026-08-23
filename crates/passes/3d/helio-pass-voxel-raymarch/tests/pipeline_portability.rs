use std::sync::Arc;

use helio_pass_voxel_raymarch::VoxelRayMarchPass;

async fn create_pass() -> Option<VoxelRayMarchPass> {
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
            label: Some("Voxel Raymarch Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must create a device");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("voxel-raymarch GPU validation error: {error:?}");
    }));

    Some(VoxelRayMarchPass::new_composited(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
    ))
}

#[test]
fn pipelines_compile_on_the_available_backend() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping voxel-raymarch portability test: no GPU adapter available");
        return;
    };
}
