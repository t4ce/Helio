use std::sync::Arc;

use helio_pass_sdf::{
    SdfPass, SdfPassConfigError, REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
};

async fn device() -> Option<wgpu::Device> {
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
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage =
        REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE;
    limits.max_storage_buffer_binding_size = adapter
        .limits()
        .max_storage_buffer_binding_size
        .min(limits.max_storage_buffer_binding_size.max(32 * 1024 * 1024));
    limits.max_buffer_size = adapter
        .limits()
        .max_buffer_size
        .min(limits.max_buffer_size.max(32 * 1024 * 1024));
    let (device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("SDF Eight-Storage Portability Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .expect("adapter advertising the requested limits must create a device");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("SDF pipeline validation error: {error:?}");
    }));
    Some(device)
}

#[test]
fn pipelines_compile_at_the_exact_eight_storage_tier() {
    let Some(device) = pollster::block_on(device()) else {
        eprintln!("skipping SDF portability test: no eight-storage adapter available");
        return;
    };
    let _pass = SdfPass::new(&device, wgpu::TextureFormat::Rgba8Unorm);
}

#[test]
fn custom_grid_ceiling_and_alignment_are_checked_before_allocation() {
    let Some(device) = pollster::block_on(device()) else {
        eprintln!("skipping SDF grid test: no eight-storage adapter available");
        return;
    };
    assert!(matches!(
        SdfPass::with_grid(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            130,
            [-1.0; 3],
            [1.0; 3],
        ),
        Err(SdfPassConfigError::GridNotBrickAligned),
    ));
    assert!(matches!(
        SdfPass::with_grid(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            136,
            [-1.0; 3],
            [1.0; 3],
        ),
        Err(SdfPassConfigError::BrickCapacityExceeded {
            requested: 4913,
            maximum: 4096,
        }),
    ));
}
