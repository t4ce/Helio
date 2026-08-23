//! Graph textures expose one full array view plus stable single-layer render
//! views for every allocated layer. Explicit arrays and XR's implicit two-view
//! allocation must follow the same contract.

use std::sync::Arc;

use helio_core::graph::{GraphTexturePool, TextureDescriptor};

async fn validate_array_views() -> Option<()> {
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
            label: Some("Graph Texture Array Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("adapter must support WebGPU downlevel limits");
    device.on_uncaptured_error(Arc::new(|error| {
        panic!("graph texture array GPU validation error: {error:?}");
    }));

    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    let mut pool = GraphTexturePool::new();
    pool.allocate(
        &device,
        TextureDescriptor {
            name: "explicit_array".to_string(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            depth_or_array_layers: 4,
            mip_level_count: 1,
            sample_count: 1,
            usage,
            alias_group: None,
        },
    );
    assert!(pool.get_layer_view("explicit_array", 0).is_some());
    assert!(pool.get_layer_view("explicit_array", 3).is_some());
    assert!(pool.get_layer_view("explicit_array", 4).is_none());

    let array_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Graph Texture Array Test BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }],
    });
    let _explicit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Graph Explicit Array BG"),
        layout: &array_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(
                pool.get_view("explicit_array").expect("full array view"),
            ),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    for layer in 0..4 {
        let view = pool
            .get_layer_view("explicit_array", layer)
            .expect("declared layer view");
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Graph Texture Array Layer Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: (layer + 1) as f64 / 5.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Graph Texture Array Readback"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        pool.get_texture("explicit_array")
            .expect("explicit array texture")
            .as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 4 },
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map callback").expect("map succeeds");
    let data = slice.get_mapped_range().expect("mapped range");
    for layer in 0..4usize {
        let expected = (((layer + 1) as f32 / 5.0) * 255.0).round() as i32;
        let actual = data[layer * 256] as i32;
        assert!(
            (actual - expected).abs() <= 1,
            "layer {layer} clear landed in the wrong slice: expected {expected}, got {actual}"
        );
    }
    drop(data);
    readback.unmap();

    let mut xr_pool = GraphTexturePool::new();
    xr_pool.set_xr_mode(true);
    xr_pool.allocate(
        &device,
        TextureDescriptor {
            name: "xr_pair".to_string(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            usage,
            alias_group: None,
        },
    );
    assert!(xr_pool.get_layer_view("xr_pair", 0).is_some());
    assert!(xr_pool.get_layer_view("xr_pair", 1).is_some());
    assert!(xr_pool.get_layer_view("xr_pair", 2).is_none());
    let _xr_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Graph XR Pair BG"),
        layout: &array_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(
                xr_pool.get_view("xr_pair").expect("XR array view"),
            ),
        }],
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    Some(())
}

#[test]
fn graph_array_view_covers_declared_layers_and_preserves_xr_pairing() {
    if pollster::block_on(validate_array_views()).is_none() {
        eprintln!("skipping graph texture array test: no GPU adapter available");
    }
}
