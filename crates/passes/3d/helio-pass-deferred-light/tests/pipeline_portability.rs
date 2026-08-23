use std::sync::Arc;

use helio_pass_deferred_light::{
    DeferredLightPass, BASE_SAMPLED_TEXTURE_COUNT, REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
};

async fn create_pass() -> Option<DeferredLightPass> {
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
        || adapter.limits().max_sampled_textures_per_shader_stage < BASE_SAMPLED_TEXTURE_COUNT
    {
        eprintln!(
            "skipping deferred-light portability test: adapter lacks 8-storage/16-texture tier"
        );
        return None;
    }
    let mut required_limits = wgpu::Limits::downlevel_defaults();
    required_limits.max_storage_buffers_per_shader_stage =
        REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE;
    required_limits.max_sampled_textures_per_shader_stage = BASE_SAMPLED_TEXTURE_COUNT;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Deferred Light Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("deferred-light GPU validation error: {error:?}");
    }));
    let camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Deferred Light Portability Camera"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    Some(DeferredLightPass::new(
        &device,
        &queue,
        &camera,
        wgpu::TextureFormat::Rgba16Float,
    ))
}

#[test]
fn pipelines_compile_at_the_documented_eight_storage_tier() {
    let Some(_pass) = pollster::block_on(create_pass()) else {
        eprintln!("skipping deferred-light portability test: no GPU adapter available");
        return;
    };
}
