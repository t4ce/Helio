use std::sync::Arc;

use helio_pass_postprocess::{PostProcessPass, PostProcessVolumeBlendPass};

async fn create_passes() -> Option<(PostProcessPass, PostProcessVolumeBlendPass)> {
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
            label: Some("PostProcess Portability Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("post-process GPU validation error: {error:?}");
    }));

    Some((
        PostProcessPass::new(
            &device,
            &queue,
            16,
            16,
            wgpu::TextureFormat::Rgba16Float,
        ),
        PostProcessVolumeBlendPass::new(&device),
    ))
}

#[test]
fn pipelines_compile_with_downlevel_binding_limits() {
    let Some(_passes) = pollster::block_on(create_passes()) else {
        eprintln!("skipping post-process portability test: no GPU adapter available");
        return;
    };
}
