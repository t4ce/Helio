use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    GpuLookupKey, GpuLookupQuery, GpuLookupResult, GpuResidencyCounters, GpuResidencyError,
    GpuUploadOutcome, PlanetaryVoxelGpuConfig, PlanetaryVoxelResidency, RESIDENCY_WGSL,
};
use helio_planet_voxel_core::{
    CellWord, EvictOutcome, GpuPageMeta, PageEvict, PageKey, PageUpload,
    PlanetFrameProjection, PlanetFrameUniform, PlanetId, PlanetPageKey, PlanetPosition,
    SourceGeneration, UploadOutcome, VisiblePage, VisiblePageSet, PAGE_CELL_BYTES,
    PAGE_CELL_COUNT,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[test]
fn headless_residency_round_trips_cells_metadata_lookup_and_rebuild() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = request_test_adapter(&instance).await;
        let Some(adapter) = adapter else {
            eprintln!(
                "GPU_VALIDATION_SKIPPED_NO_ADAPTER: no primary or fallback adapter available"
            );
            return;
        };
        let adapter_info = adapter.get_info();
        eprintln!(
            "GPU_VALIDATION_ADAPTER: name={:?} backend={:?} device_type={:?}",
            adapter_info.name, adapter_info.backend, adapter_info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Planetary Voxel Residency Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary voxel GPU validation error: {error:?}");
        }));

        let config = PlanetaryVoxelGpuConfig::new(4, 16, 16, 4, 8).unwrap();
        let mut residency = PlanetaryVoxelResidency::new(&device, &queue, config).unwrap();
        let initial_resources = residency.resource_stats();
        assert_eq!(initial_resources.buffers, 4);
        assert_eq!(initial_resources.textures, 1);

        let planet_a = PlanetId([0x11; 16]);
        let planet_b = PlanetId([0x82; 16]);
        let key_a = PlanetPageKey::new(planet_a, PageKey::new(0, [-2, 1, -3]));
        let key_b = PlanetPageKey::new(planet_b, PageKey::new(2, [3, -4, 5]));
        let frame_a_initial = PlanetFrameUniform::from_camera(
            planet_a,
            PlanetPosition::from_meters([0.25, 1.0, 2.0]).unwrap(),
            1,
        );
        let frame_b_initial = PlanetFrameUniform::from_camera(
            planet_b,
            PlanetPosition::from_meters([2.0, 3.0, 1.0]).unwrap(),
            1,
        );
        let projection_a_initial = frame_projection(1, 0, frame_a_initial);
        let projection_b_initial = frame_projection(2, 1, frame_b_initial);
        residency
            .synchronize_planet_frames(
                &queue,
                1,
                1,
                &[projection_a_initial, projection_b_initial],
            )
            .unwrap();

        let cell_a = CellWord::new(-120, 7, 3);
        let cell_b = CellWord::new(-8, 19, 1);
        let outcomes = residency
            .apply_upload_batch(
                &device,
                &queue,
                vec![upload(key_a, 9, cell_a), upload(key_b, 11, cell_b)],
            )
            .unwrap();
        assert!(matches!(
            outcomes.as_slice(),
            [
                GpuUploadOutcome::Residency(UploadOutcome::Inserted { slot: 0, .. }),
                GpuUploadOutcome::Residency(UploadOutcome::Inserted { slot: 1, .. })
            ]
        ));

        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 8, CellWord::AIR)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Stale {
                newest_generation
            })] if *newest_generation == source(9)
        ));
        assert_eq!(
            residency
                .apply_evict_batch(
                    &device,
                    &queue,
                    vec![PageEvict {
                        key: key_a,
                        generation: source(8),
                    }],
                )
                .unwrap(),
            vec![EvictOutcome::Recorded { removed: None }]
        );
        assert_eq!(
            residency.cache().resident(key_a).unwrap().generation,
            source(9)
        );

        assert!(matches!(
            residency.apply_upload_batch(
                &device,
                &queue,
                (0..5)
                    .map(|index| {
                        upload(
                            PlanetPageKey::new(planet_a, PageKey::new(0, [index, 0, 0])),
                            1,
                            CellWord::AIR,
                        )
                    })
                    .collect(),
            ),
            Err(GpuResidencyError::BatchCapacity {
                actual: 5,
                maximum: 4
            })
        ));
        assert_eq!(residency.cache().counters().resident_pages, 2);

        assert!(!residency
            .synchronize_planet_frames(
                &queue,
                1,
                1,
                &[projection_a_initial, projection_b_initial],
            )
            .unwrap()
            .changed);
        let outside_origin = [i64::from(i32::MAX).div_euclid(32) * 32 + 32, 0, 0];
        assert!(matches!(
            residency.synchronize_planet_frames(
                &queue,
                1,
                2,
                &[
                    frame_projection(
                        1,
                        0,
                        PlanetFrameUniform::from_camera(
                            planet_a,
                            PlanetPosition::from_lod0_cell(outside_origin),
                            2,
                        ),
                    ),
                    projection_b_initial,
                ],
            ),
            Err(GpuResidencyError::Address(_))
        ));
        assert!(residency
            .synchronize_planet_frames(
                &queue,
                1,
                2,
                &[
                    frame_projection(
                        1,
                        0,
                        PlanetFrameUniform::from_camera(
                            planet_a,
                            PlanetPosition::from_lod0_cell([32, 0, 0]),
                            2,
                        ),
                    ),
                    projection_b_initial,
                ],
            )
            .unwrap()
            .changed);
        residency
            .apply_visible_set(
                &queue,
                VisiblePageSet {
                    frame_index: 3,
                    pages: vec![VisiblePage {
                        key: key_a,
                        generation: source(9),
                        transition_mask: 0b10_0101,
                    }],
                },
            )
            .unwrap();

        residency.resize(1920, 1080);
        assert_eq!(residency.resource_stats(), initial_resources);
        residency.recreate_gpu_resources(&device, &queue).unwrap();
        assert_eq!(residency.resource_stats(), initial_resources);

        let frame_a = [32_i64, 0, 0];
        let frame_b = [0_i64; 3];
        let missing = PlanetPageKey::new(planet_a, PageKey::new(1, [-99, 4, 2]));
        let queries = [
            GpuLookupQuery::from(GpuLookupKey::from_planet_page(key_a, frame_a).unwrap()),
            GpuLookupQuery::from(GpuLookupKey::from_planet_page(key_b, frame_b).unwrap()),
            GpuLookupQuery::from(GpuLookupKey::from_planet_page(missing, frame_a).unwrap()),
        ];
        let results = dispatch_lookup(&device, &queue, &residency, &queries);
        assert!(results[0].found());
        assert_eq!(results[0].slot, 0);
        assert_eq!(results[0].generation(), 1);
        assert!(results[1].found());
        assert_eq!(results[1].slot, 1);
        assert_eq!(results[1].generation(), 2);
        assert!(!results[2].found());

        assert_eq!(read_atlas_cell(&device, &queue, &residency, 0), cell_a);
        let metadata: Vec<GpuPageMeta> = read_buffer_range(
            &device,
            &queue,
            residency.metadata_buffer(),
            0,
            size_of::<GpuPageMeta>() as u64,
        );
        assert_eq!(metadata[0].slot, 0);
        assert_eq!(metadata[0].generation(), 1);
        assert_eq!(metadata[0].relative_lod0_cell_min, [-96, 32, -96]);
        assert_eq!(metadata[0].transition_mask, 0b10_0101);
        let counters: Vec<GpuResidencyCounters> = read_buffer_range(
            &device,
            &queue,
            residency.counters_buffer(),
            0,
            size_of::<GpuResidencyCounters>() as u64,
        );
        assert_eq!(counters[0].resident_pages, 2);
        assert_eq!(counters[0].peak_resident_pages, 2);
        assert_eq!(counters[0].device_rebuilds, 1);
        assert_eq!(counters[0].uploads_published, 2);
        assert_eq!(counters[0].batches_submitted, 2);
        assert_eq!(
            u64::from(counters[0].cell_bytes_uploaded_low)
                | (u64::from(counters[0].cell_bytes_uploaded_high) << 32),
            4 * PAGE_CELL_BYTES as u64
        );
        assert_eq!(counters[0].resource_buffers, initial_resources.buffers);
        assert_eq!(counters[0].resource_textures, initial_resources.textures);
        assert_eq!(
            counters[0].atlas_capacity_pages,
            residency.allocation_plan().atlas.capacity_pages
        );
        assert_eq!(
            u64::from(counters[0].resident_cell_bytes_low)
                | (u64::from(counters[0].resident_cell_bytes_high) << 32),
            2 * PAGE_CELL_BYTES as u64
        );
        assert_eq!(
            u64::from(counters[0].allocated_gpu_bytes_low)
                | (u64::from(counters[0].allocated_gpu_bytes_high) << 32),
            initial_resources.allocated_bytes
        );

        assert!(matches!(
            residency
                .apply_evict_batch(
                    &device,
                    &queue,
                    vec![PageEvict {
                        key: key_a,
                        generation: source(9),
                    }],
                )
                .unwrap()
                .as_slice(),
            [EvictOutcome::Recorded { removed: Some(_) }]
        ));
        let result = dispatch_lookup(&device, &queue, &residency, &queries[..1]);
        assert!(!result[0].found());
        // Eviction makes the old tile unreachable through the page table. The
        // tile is deliberately not cleared; the next occupant overwrites the
        // complete 32^3 page before its table entry is published.
        assert_eq!(read_atlas_cell(&device, &queue, &residency, 0), cell_a);

        let replacement = CellWord::new(-512, 33, 5);
        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 9, cell_a)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Stale {
                newest_generation
            })] if *newest_generation == source(9)
        ));
        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 10, replacement)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Inserted {
                slot: 0,
                ..
            })]
        ));
        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 10, cell_a)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(
                UploadOutcome::GenerationConflict { slot: 0 }
            )]
        ));
        let replacement_result = dispatch_lookup(&device, &queue, &residency, &queries[..1]);
        assert!(replacement_result[0].found());
        assert_eq!(replacement_result[0].generation(), 3);
        assert_eq!(read_atlas_cell(&device, &queue, &residency, 0), replacement);

        let replacement_planet_cell = CellWord::new(-900, 44, 6);
        assert!(matches!(
            residency
                .apply_upload_batch(
                    &device,
                    &queue,
                    vec![PageUpload::new(
                        key_a,
                        SourceGeneration::new(2, 0),
                        vec![replacement_planet_cell; PAGE_CELL_COUNT],
                    )
                    .unwrap()],
                )
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Replaced { .. })]
        ));
        assert!(matches!(
            residency
                .apply_upload_batch(
                    &device,
                    &queue,
                    vec![PageUpload::new(
                        key_a,
                        SourceGeneration::new(1, u64::MAX),
                        vec![CellWord::AIR; PAGE_CELL_COUNT],
                    )
                    .unwrap()],
                )
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Stale {
                newest_generation
            })] if *newest_generation == SourceGeneration::new(2, 0)
        ));
        let replacement_planet_result = dispatch_lookup(&device, &queue, &residency, &queries[..1]);
        assert!(replacement_planet_result[0].found());
        assert_eq!(replacement_planet_result[0].generation(), 4);
        assert_eq!(
            read_atlas_cell(&device, &queue, &residency, 0),
            replacement_planet_cell
        );

        validate_table_probe_backpressure(&device, &queue, planet_a);
        validate_frame_authority_is_not_capped_by_page_residency(&device, &queue);
    });
}

fn validate_table_probe_backpressure(device: &wgpu::Device, queue: &wgpu::Queue, planet: PlanetId) {
    let mut residency = PlanetaryVoxelResidency::new(
        device,
        queue,
        PlanetaryVoxelGpuConfig::new(3, 8, 1, 3, 4).unwrap(),
    )
    .unwrap();
    residency
        .synchronize_planet_frames(
            queue,
            1,
            1,
            &[frame_projection(
                1,
                0,
                PlanetFrameUniform::from_camera(
                    planet,
                    PlanetPosition::from_lod0_cell([0; 3]),
                    1,
                ),
            )],
        )
        .unwrap();
    let mut collisions = Vec::new();
    for x in -10_000..10_000 {
        let key = PlanetPageKey::new(planet, PageKey::new(0, [x, 0, 0]));
        let lookup = GpuLookupKey::from_planet_page(key, [0; 3]).unwrap();
        if lookup.hash() & 7 == 0 {
            collisions.push(key);
            if collisions.len() == 2 {
                break;
            }
        }
    }
    assert_eq!(collisions.len(), 2);
    residency
        .apply_upload_batch(device, queue, vec![upload(collisions[0], 1, CellWord::AIR)])
        .unwrap();
    assert_eq!(
        residency
            .apply_upload_batch(device, queue, vec![upload(collisions[1], 1, CellWord::AIR)])
            .unwrap(),
        vec![GpuUploadOutcome::PageTableBackpressure]
    );
    assert!(residency.cache().resident(collisions[1]).is_none());
    assert_eq!(residency.counters().table_saturation_events, 1);
}

fn validate_frame_authority_is_not_capped_by_page_residency(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut residency = PlanetaryVoxelResidency::new(
        device,
        queue,
        PlanetaryVoxelGpuConfig::new(2, 8, 8, 2, 4).unwrap(),
    )
    .unwrap();
    let frames = (1..=9_u8)
        .map(|index| {
            frame_projection(
                u64::from(index),
                u32::from(index - 1),
                PlanetFrameUniform::from_camera(
                    PlanetId([index; 16]),
                    PlanetPosition::from_lod0_cell([0; 3]),
                    1,
                ),
            )
        })
        .collect::<Vec<_>>();
    residency
        .synchronize_planet_frames(queue, 1, 1, &frames)
        .unwrap();
    assert_eq!(residency.planet_frame_count(), 9);
    let mut invalid_frames = frames.clone();
    invalid_frames[3].frame.page_edge_cells = 0;
    assert!(matches!(
        residency.synchronize_planet_frames(queue, 1, 2, &invalid_frames),
        Err(GpuResidencyError::PlanetFrame(_))
    ));
    assert_eq!(
        residency.planet_frame_count(),
        9,
        "invalid generic snapshots must not partially replace canonical projection state",
    );

    let removed_planet = frames[0].frame.planet_id();
    let removed_key = PlanetPageKey::new(removed_planet, PageKey::new(0, [0; 3]));
    residency
        .apply_upload_batch(device, queue, vec![upload(removed_key, 1, CellWord::AIR)])
        .unwrap();
    let mut replacement_identity = frames.clone();
    replacement_identity[0].identity = 100;
    let replacement = residency
        .synchronize_planet_frames(queue, 1, 2, &replacement_identity)
        .unwrap();
    assert_eq!(replacement.invalidated_planets, vec![removed_planet]);
    assert_eq!(replacement.removed_pages.len(), 1);
    assert!(residency.cache().resident(removed_key).is_none());
    assert!(matches!(
        residency.synchronize_planet_frames(queue, 1, 1, &frames),
        Err(GpuResidencyError::StalePlanetFrameSnapshot {
            current: 2,
            incoming: 1,
        })
    ));
    let mut conflicting_snapshot = replacement_identity.clone();
    conflicting_snapshot[1].frame = PlanetFrameUniform::from_camera(
        conflicting_snapshot[1].frame.planet_id(),
        PlanetPosition::from_lod0_cell([32, 0, 0]),
        2,
    );
    assert!(matches!(
        residency.synchronize_planet_frames(queue, 1, 2, &conflicting_snapshot),
        Err(GpuResidencyError::PlanetFrameSnapshotConflict { generation: 2 })
    ));
    residency
        .apply_upload_batch(device, queue, vec![upload(removed_key, 1, CellWord::AIR)])
        .unwrap();

    let source_replacement = residency
        .synchronize_planet_frames(queue, 2, 2, &replacement_identity)
        .unwrap();
    assert_eq!(source_replacement.removed_pages.len(), 1);
    assert!(source_replacement
        .invalidated_planets
        .contains(&removed_planet));
    assert!(residency.cache().resident(removed_key).is_none());
    residency
        .apply_upload_batch(device, queue, vec![upload(removed_key, 1, CellWord::AIR)])
        .unwrap();
    let outcome = residency
        .synchronize_planet_frames(queue, 2, 3, &replacement_identity[1..])
        .unwrap();
    assert_eq!(outcome.removed_pages.len(), 1);
    assert_eq!(outcome.removed_pages[0].key, removed_key);
    assert!(residency.cache().resident(removed_key).is_none());
    assert_eq!(residency.planet_frame_count(), 8);
}

const fn frame_projection(
    identity: u64,
    gpu_row: u32,
    frame: PlanetFrameUniform,
) -> PlanetFrameProjection {
    PlanetFrameProjection {
        identity,
        gpu_row,
        frame,
    }
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}

fn upload(key: PlanetPageKey, generation: u64, cell: CellWord) -> PageUpload {
    PageUpload::new(key, source(generation), vec![cell; PAGE_CELL_COUNT]).unwrap()
}

const fn source(page: u64) -> SourceGeneration {
    SourceGeneration::new(1, page)
}

fn dispatch_lookup(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    residency: &PlanetaryVoxelResidency,
    queries: &[GpuLookupQuery],
) -> Vec<GpuLookupResult> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Planetary Voxel Lookup Validation Shader"),
        source: wgpu::ShaderSource::Wgsl(RESIDENCY_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Planetary Voxel Lookup Validation Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("validate_lookup"),
        compilation_options: Default::default(),
        cache: None,
    });
    let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Planetary Voxel Lookup Queries"),
        contents: bytemuck::cast_slice(queries),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_bytes = std::mem::size_of_val(queries) as u64 / size_of::<GpuLookupQuery>() as u64
        * size_of::<GpuLookupResult>() as u64;
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Voxel Lookup Results"),
        size: result_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Planetary Voxel Lookup Validation Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: residency.page_table_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: residency.residency_uniform_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: query_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Voxel Lookup Validation Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planetary Voxel Lookup Validation Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((queries.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    read_buffer_range(device, queue, &result_buffer, 0, result_bytes)
}

fn read_buffer_range<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<T> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Voxel Validation Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Voxel Validation Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, offset, &readback, 0, size);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("GPU readback callback must run")
        .expect("GPU readback mapping must succeed");
    let mapped = slice
        .get_mapped_range()
        .expect("GPU readback range must be available");
    let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn read_atlas_cell(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    residency: &PlanetaryVoxelResidency,
    slot: u32,
) -> CellWord {
    let origin = residency
        .allocation_plan()
        .atlas
        .origin_for_slot(slot)
        .expect("resident slot must map to the atlas");
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Voxel Atlas Cell Readback"),
        size: size_of::<CellWord>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Voxel Atlas Cell Readback Encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: residency.atlas_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: origin[0],
                y: origin[1],
                z: origin[2],
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("GPU atlas readback callback must run")
        .expect("GPU atlas readback mapping must succeed");
    let mapped = slice
        .get_mapped_range()
        .expect("GPU atlas readback range must be available");
    let cell = bytemuck::cast_slice::<u8, CellWord>(&mapped)[0];
    drop(mapped);
    readback.unmap();
    cell
}

fn size_of<T>() -> usize {
    core::mem::size_of::<T>()
}
