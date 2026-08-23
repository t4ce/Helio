use helio_pass_sprite_batch::{SpriteBatchPass, SpriteError, SpriteInstance};
use helio_pass_sprite_cull::SpriteCullPass;
use helio_pass_sprite_simulate::SpriteSimulatePass;
use std::sync::Arc;

async fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    gpu_with_atlas_limit(None).await
}

async fn gpu_with_atlas_limit(
    maximum_layers: Option<u32>,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapter = None;
    for fallback in [false, true] {
        if let Ok(candidate) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: fallback,
                apply_limit_buckets: false,
            })
            .await
        {
            adapter = Some(candidate);
            break;
        }
    }
    let adapter = adapter?;
    let mut limits = adapter.limits();
    if let Some(maximum_layers) = maximum_layers {
        limits.max_texture_array_layers = maximum_layers;
    }
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Sprite SceneDB Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .ok()
}

fn rgba(value: u8) -> [u8; 16] {
    [value; 16]
}

fn read_indirect(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> [u32; 6] {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sprite SceneDB Indirect Readback"),
        size: 24,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Sprite SceneDB Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, 24);
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .expect("map callback runs")
        .expect("indirect readback maps");
    let mapped = staging.slice(..).get_mapped_range().expect("mapped bytes");
    let mut words = [0; 6];
    words.copy_from_slice(bytemuck::cast_slice(&mapped));
    drop(mapped);
    staging.unmap();
    words
}

#[test]
fn crud_clear_and_recycled_rows_reject_stale_handles() {
    let Some((device, queue)) = pollster::block_on(gpu()) else {
        eprintln!("skipping sprite SceneDB CRUD test: no GPU adapter");
        return;
    };
    let mut pass = SpriteBatchPass::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
    let atlas = pass.add_atlas_layer(&device, &queue, 2, 2, &rgba(17));
    let first = pass.insert_sprite(
        SpriteInstance::new([1.0, 2.0], [3.0, 4.0]).with_atlas_layer(atlas),
    );
    assert_eq!(pass.sprite_count(), 1);
    assert_eq!(pass.sprite(first).unwrap().position, [1.0, 2.0]);

    pass.try_update_sprite(
        first,
        SpriteInstance::new([8.0, 9.0], [3.0, 4.0]).with_atlas_layer(atlas),
    )
    .unwrap();
    assert_eq!(pass.sprite(first).unwrap().position, [8.0, 9.0]);
    pass.try_remove_sprite(first).unwrap();
    assert_eq!(pass.sprite_count(), 0);
    assert!(matches!(
        pass.try_update_sprite(first, SpriteInstance::new([0.0; 2], [1.0; 2])),
        Err(SpriteError::StaleSprite(handle)) if handle == first
    ));

    let replacement = pass.insert_sprite(SpriteInstance::new([0.0; 2], [1.0; 2]));
    assert_ne!(replacement, first, "recycled entity must carry a new generation");
    pass.clear_sprites();
    assert_eq!(pass.sprite_count(), 0);
    assert!(pass.sprite(replacement).is_none());
}

#[test]
fn atlas_identity_is_generation_checked_and_removal_is_transactional() {
    let Some((device, queue)) = pollster::block_on(gpu()) else {
        eprintln!("skipping sprite atlas lifecycle test: no GPU adapter");
        return;
    };
    let mut pass = SpriteBatchPass::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
    let first = pass.add_atlas_layer(&device, &queue, 2, 2, &rgba(31));
    let sprite = pass.insert_sprite(
        SpriteInstance::new([0.0; 2], [1.0; 2]).with_atlas_layer(first),
    );
    assert!(matches!(
        pass.try_remove_atlas_layer(first),
        Err(SpriteError::Atlas(helio_scenedb::SpriteAtlasError::LayerInUse { references: 1 }))
    ));
    assert_eq!(pass.atlas_layer_count(), 1);

    let transient = pass.add_atlas_layer(&device, &queue, 2, 2, &rgba(39));
    pass.remove_atlas_layer(transient);
    assert!(matches!(
        pass.try_update_sprite(
            sprite,
            SpriteInstance::new([4.0; 2], [1.0; 2]).with_atlas_layer(transient),
        ),
        Err(SpriteError::StaleAtlas(handle)) if handle == transient
    ));
    assert_eq!(pass.sprite(sprite).unwrap().atlas(), Some(first));
    assert!(matches!(
        pass.try_remove_atlas_layer(first),
        Err(SpriteError::Atlas(helio_scenedb::SpriteAtlasError::LayerInUse { references: 1 }))
    ));

    pass.remove_sprite(sprite);
    pass.try_remove_atlas_layer(first).unwrap();
    let replacement = pass.add_atlas_layer(&device, &queue, 2, 2, &rgba(47));
    assert_ne!(replacement, first);
    assert!(matches!(
        pass.try_insert_sprite(
            SpriteInstance::new([0.0; 2], [1.0; 2]).with_atlas_layer(first)
        ),
        Err(SpriteError::StaleAtlas(handle)) if handle == first
    ));
    pass.clear_atlas_layers();
    assert_eq!(pass.atlas_layer_count(), 0);
}

#[test]
fn failed_atlas_import_and_hardware_growth_leave_scene_authority_unchanged() {
    let Some((device, queue)) = pollster::block_on(gpu_with_atlas_limit(Some(4))) else {
        eprintln!("skipping sprite atlas capacity test: no GPU adapter");
        return;
    };
    let mut pass = SpriteBatchPass::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
    let initial_capacity = pass.atlas_capacity();
    assert!(matches!(
        pass.try_add_atlas_layer(&device, &queue, 2, 2, &[0; 15]),
        Err(SpriteError::Atlas(
            helio_scenedb::SpriteAtlasError::InvalidByteLength {
                expected: 16,
                actual: 15
            }
        ))
    ));
    assert_eq!(pass.atlas_layer_count(), 0);
    assert_eq!(pass.atlas_capacity(), initial_capacity);

    let layers: Vec<_> = (0..3)
        .map(|value| pass.add_atlas_layer(&device, &queue, 2, 2, &rgba(value)))
        .collect();
    assert_eq!(pass.atlas_layer_count(), 3);
    assert_eq!(pass.atlas_capacity(), 4);
    assert!(matches!(
        pass.try_add_atlas_layer(&device, &queue, 2, 2, &rgba(99)),
        Err(SpriteError::Atlas(
            helio_scenedb::SpriteAtlasError::HardwareCapacityExceeded {
                maximum_layers: 3
            }
        ))
    ));
    assert_eq!(pass.atlas_layer_count(), 3);
    assert_eq!(pass.atlas_capacity(), 4);

    // This is the typed replacement for the removed raw TextureView/u32
    // escape hatch: every imported layer yields an Entity-generation handle
    // which composes directly into the authored SceneSprite row.
    let sprite = pass.insert_sprite(
        SpriteInstance::new([0.0; 2], [1.0; 2]).with_atlas_layer(layers[2]),
    );
    assert_eq!(pass.sprite(sprite).unwrap().atlas(), Some(layers[2]));
}

#[test]
fn component_and_atlas_growth_publish_new_epochs_without_stale_bindings() {
    let Some((device, queue)) = pollster::block_on(gpu()) else {
        eprintln!("skipping sprite growth test: no GPU adapter");
        return;
    };
    let mut pass = SpriteBatchPass::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
    let before = pass.buffer_source().snapshot();

    // Construct every consumer before SceneDB grows. A cached bind group must
    // not turn this initial one-row allocation into an authored capacity.
    let mut cull = SpriteCullPass::from_source(&device, &queue, pass.buffer_source(), 512);
    cull.set_view_rect([0.0, 0.0], [1_000.0, 1_000.0]);
    let simulate = SpriteSimulatePass::new(
        &device,
        pass.buffer_source(),
        [-1_000.0, -1_000.0],
        [1_000.0, 1_000.0],
    );
    assert!(SpriteSimulatePass::try_new(
        &device,
        pass.buffer_source(),
        [-1_000.0, -1_000.0],
        [1_000.0, 1_000.0],
    )
    .is_err(), "one source cannot silently acquire two runtime writers");

    let handles: Vec<_> = (0..300)
        .map(|index| {
            pass.insert_sprite(SpriteInstance::new([index as f32, 0.0], [1.0; 2]))
        })
        .collect();
    pass.flush_scene_gpu();
    let after = pass.buffer_source().snapshot();
    assert_eq!(after.row_span, 300);
    assert!(after.instances_epoch > before.instances_epoch);
    assert!(after.presence_epoch > before.presence_epoch);

    let initial_atlas_capacity = pass.atlas_capacity();
    for value in 1..=5 {
        pass.add_atlas_layer(&device, &queue, 2, 2, &rgba(value));
    }
    assert!(pass.atlas_capacity() > initial_atlas_capacity);

    cull.run_once_for_testing(&device, &queue);
    let indirect = read_indirect(&device, &queue, &cull.indirect_buf);
    assert_eq!(indirect[1], 300, "cull rebound both grown SceneDB columns");
    assert_eq!(indirect[5], 0);

    pass.remove_sprite(handles[127]);
    pass.flush_scene_gpu();
    cull.run_once_for_testing(&device, &queue);
    assert_eq!(read_indirect(&device, &queue, &cull.indirect_buf)[1], 299);

    pass.clear_sprites();
    pass.flush_scene_gpu();
    cull.run_once_for_testing(&device, &queue);
    assert_eq!(read_indirect(&device, &queue, &cull.indirect_buf)[1], 0);
    drop(simulate);
}

#[test]
fn typed_atlas_scene_row_renders_through_the_real_batch_pipeline() {
    let Some((device, queue)) = pollster::block_on(gpu()) else {
        eprintln!("skipping sprite batch render test: no GPU adapter");
        return;
    };
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut batch = SpriteBatchPass::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
    let atlas = batch.add_atlas_layer(
        &device,
        &queue,
        1,
        1,
        &[255, 0, 0, 255],
    );
    batch.insert_sprite(
        SpriteInstance::new([0.0, 0.0], [4.0, 4.0]).with_atlas_layer(atlas),
    );
    let mut cull = SpriteCullPass::from_source(&device, &queue, batch.buffer_source(), 1);
    cull.set_view_rect([0.0, 0.0], [2.0, 2.0]);
    batch.use_gpu_culling(cull.draw_order_buf.clone(), cull.indirect_buf.clone());

    let mut graph = helio_core::RenderGraph::new(&device, &queue);
    graph.add_pass(Box::new(cull));
    graph.add_pass(Box::new(batch));
    graph.lock(4, 4);
    let scene = helio_core::GpuScene::new(Arc::clone(&device), Arc::clone(&queue));
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sprite Batch Test Target"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sprite Batch Test Dummy Depth"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    graph.execute(&scene, &target_view, &depth_view).unwrap();

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sprite Batch Pixel Readback"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Sprite Batch Pixel Copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let center = 2 * 256 + 2 * 4;
    assert_eq!(&mapped[center..center + 4], &[255, 0, 0, 255]);
}

#[test]
fn integrated_helio_scene_injects_one_shared_sprite_authority_publication() {
    let Some((device, queue)) = pollster::block_on(gpu()) else {
        eprintln!("skipping integrated Helio sprite authority test: no GPU adapter");
        return;
    };
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut scene = helio::Scene::new(Arc::clone(&device), Arc::clone(&queue));
    let (buffer_source, atlas_source) = scene.sprite_publications();
    let before = buffer_source.snapshot();
    let atlas_epoch = atlas_source.snapshot().epoch;
    let atlas = scene.add_sprite_atlas_layer(1, 1, &[0, 255, 0, 255]);
    assert!(atlas_source.snapshot().epoch > atlas_epoch);

    for index in 0..1_100 {
        scene.insert_sprite(
            helio::SceneSpriteRow::new([index as f32, 0.0], [1.0; 2])
                .with_atlas_layer(atlas),
        );
    }
    scene.flush();
    let after = buffer_source.snapshot();
    assert_eq!(after.row_span, 1_100);
    assert!(after.instances_epoch > before.instances_epoch);
    assert!(after.presence_epoch > before.presence_epoch);

    let mut batch = SpriteBatchPass::from_publications(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        buffer_source.clone(),
        atlas_source,
    );
    assert!(!batch.owns_scene_authority());
    assert!(matches!(
        batch.try_insert_sprite(SpriteInstance::new([0.0; 2], [1.0; 2])),
        Err(SpriteError::ExternalAuthority)
    ));

    let mut cull = SpriteCullPass::from_source(&device, &queue, buffer_source, 1_200);
    cull.set_view_rect([0.0, 0.0], [2_000.0; 2]);
    cull.run_once_for_testing(&device, &queue);
    assert_eq!(read_indirect(&device, &queue, &cull.indirect_buf)[1], 1_100);
}
