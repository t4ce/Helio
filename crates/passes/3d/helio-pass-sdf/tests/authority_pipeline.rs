use std::sync::Arc;

use glam::Mat4;
use helio_core::{GpuScene, RenderGraph};
use helio_pass_sdf::{
    BooleanOp, SdfEdit, SdfPass, SdfShapeParams, SdfShapeType,
    REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
};
use helio_scenedb::SdfAuthority;

async fn context() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
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
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage =
        REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("SDF Authority Pipeline Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .expect("adapter must provide the default eight-storage tier");
    Some((Arc::new(device), Arc::new(queue)))
}

#[test]
fn standalone_authority_publication_drives_the_pass_without_shadow_authored_state() {
    let Some((device, queue)) = pollster::block_on(context()) else {
        eprintln!("skipping SDF authority pipeline: no eight-storage adapter available");
        return;
    };
    let mut authority = SdfAuthority::new(Arc::clone(&device), Arc::clone(&queue));
    authority
        .add(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::IDENTITY,
            params: SdfShapeParams::sphere(4.0),
            blend_radius: 0.0,
        })
        .unwrap();
    let mut scene = GpuScene::new(Arc::clone(&device), Arc::clone(&queue));
    let publication = authority.publication();
    scene.publish_sdf_authority(
        publication.edit_buffer,
        publication.edit_allocation_epoch,
        publication.edit_count,
        publication.terrain_buffer,
        publication.terrain_allocation_epoch,
        publication.content_generation,
        bytemuck::cast_slice(publication.bounds),
        publication.terrain_y_bounds,
        publication.requires_canonical_scan,
    );

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SDF Authority Pipeline Target"),
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SDF Authority Pipeline Depth"),
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let mut graph = RenderGraph::new(&device, &queue);
    graph.add_pass(Box::new(SdfPass::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
    )));
    graph.lock(16, 16);
    graph
        .execute(
            &scene,
            &target.create_view(&wgpu::TextureViewDescriptor::default()),
            &depth.create_view(&wgpu::TextureViewDescriptor::default()),
        )
        .expect("SDF graph execution must accept direct SceneDB publications");
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("complete SDF authority pipeline work");
}
