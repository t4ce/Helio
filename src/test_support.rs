use std::sync::{Arc, OnceLock};

struct TestGpuContext {
    // Keep the full backend ownership chain alive until process exit. Several
    // native Vulkan implementations are not robust when independent instances
    // and devices are created and torn down concurrently by the test harness.
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

static TEST_GPU: OnceLock<Option<TestGpuContext>> = OnceLock::new();

pub(crate) fn test_gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    TEST_GPU
        .get_or_init(|| {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let mut adapter = None;
            for force_fallback_adapter in [false, true] {
                if let Ok(candidate) = pollster::block_on(instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter,
                        apply_limit_buckets: false,
                    },
                )) {
                    adapter = Some(candidate);
                    break;
                }
            }
            let adapter = adapter?;
            let required_limits = adapter.limits();
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Helio Shared Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    ..Default::default()
                },
            ))
            .ok()?;
            Some(TestGpuContext {
                _instance: instance,
                _adapter: adapter,
                device: Arc::new(device),
                queue: Arc::new(queue),
            })
        })
        .as_ref()
        .map(|gpu| (Arc::clone(&gpu.device), Arc::clone(&gpu.queue)))
}
