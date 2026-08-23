#[path = "support/mod.rs"]
mod support;

use std::sync::Arc;

use bytemuck::Zeroable;
use helio_planet_voxel_core::{PlanetFrameUniform, PlanetId, PlanetPosition};
use helio_scenedb::{
    PlanetFrameAuthority, PlanetFrameAuthorityError, PlanetFrameUpdateOutcome,
};

fn frame(index: u8, frame_number: u64) -> PlanetFrameUniform {
    PlanetFrameUniform::from_camera(
        PlanetId([index; 16]),
        PlanetPosition::from_lod0_cell([i64::from(index) * 64, -32, 96]),
        frame_number,
    )
}

fn gpu_rows(bytes: &[u8]) -> Vec<PlanetFrameUniform> {
    bytes
        .chunks_exact(std::mem::size_of::<PlanetFrameUniform>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn frame_limited_context(maximum_rows: u32) -> pulsar_scenedb::gpu::EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("planet-frame authority tests need a local GPU adapter");
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = u64::from(maximum_rows)
        .checked_mul(std::mem::size_of::<PlanetFrameUniform>() as u64)
        .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("planet-frame-authority-limited-device"),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("limited planet-frame device");
    pulsar_scenedb::gpu::EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

#[test]
fn sparse_removal_reuses_only_with_a_new_generation_and_clears_rows() {
    let ctx = support::test_context();
    let mut authority =
        PlanetFrameAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let first = authority.insert(frame(1, 1)).unwrap();
    let removed = authority.insert(frame(2, 1)).unwrap();
    let trailing = authority.insert(frame(3, 1)).unwrap();

    assert_eq!(authority.remove(removed).unwrap(), frame(2, 1));
    assert_eq!(authority.get(removed), None);
    assert_eq!(
        authority.remove(removed),
        Err(PlanetFrameAuthorityError::StaleFrame),
    );
    let replacement = authority.insert(frame(4, 1)).unwrap();
    assert_eq!(replacement.slot(), removed.slot());
    assert_ne!(replacement.generation(), removed.generation());
    assert_eq!(authority.get(first), Some(&frame(1, 1)));
    assert_eq!(authority.get(trailing), Some(&frame(3, 1)));

    authority.remove(trailing).unwrap();
    let publication = authority.publication();
    assert_eq!(publication.row_span, 2);
    let bytes = support::readback(
        &ctx,
        &publication.buffer,
        3 * std::mem::size_of::<PlanetFrameUniform>() as u64,
    );
    let rows = gpu_rows(&bytes);
    assert_eq!(rows[0], frame(1, 1));
    assert_eq!(rows[1], frame(4, 1));
    assert_eq!(rows[2], PlanetFrameUniform::zeroed());

    authority.clear();
    assert!(authority.is_empty());
    assert_eq!(authority.row_span(), 0);
    let publication = authority.publication();
    let bytes = support::readback(
        &ctx,
        &publication.buffer,
        2 * std::mem::size_of::<PlanetFrameUniform>() as u64,
    );
    assert!(gpu_rows(&bytes)
        .into_iter()
        .all(|row| row == PlanetFrameUniform::zeroed()));
}

#[test]
fn growth_preserves_stable_rows_and_device_rebuild_changes_only_allocation_epoch() {
    let ctx = support::test_context();
    let mut authority =
        PlanetFrameAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let authority_epoch = authority.authority_epoch();
    let distinct_authority =
        PlanetFrameAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    assert_ne!(distinct_authority.authority_epoch(), authority_epoch);
    let initial_epoch = authority.allocation_epoch();
    let mut ids = Vec::new();
    for index in 1..=24 {
        ids.push(authority.insert(frame(index, 1)).unwrap());
    }
    assert!(authority.allocation_epoch() > initial_epoch);
    assert_eq!(authority.row_span(), 24);
    for (row, id) in ids.iter().copied().enumerate() {
        assert_eq!(id.slot(), row as u32);
        assert_eq!(authority.get(id), Some(&frame(row as u8 + 1, 1)));
    }

    let authored_generation = authority.content_generation();
    let grown_epoch = authority.allocation_epoch();
    authority
        .recreate_gpu_resources(Arc::clone(ctx.device()), Arc::clone(ctx.queue()))
        .unwrap();
    assert_eq!(authority.authority_epoch(), authority_epoch);
    assert!(authority.allocation_epoch() > grown_epoch);
    assert_eq!(authority.content_generation(), authored_generation);
    let publication = authority.publication();
    let bytes = support::readback(
        &ctx,
        &publication.buffer,
        24 * std::mem::size_of::<PlanetFrameUniform>() as u64,
    );
    for (index, row) in gpu_rows(&bytes).into_iter().enumerate() {
        assert_eq!(row, frame(index as u8 + 1, 1));
    }
}

#[test]
fn device_limit_failure_does_not_partially_allocate_or_publish() {
    let ctx = frame_limited_context(16);
    let mut authority =
        PlanetFrameAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let ids = (1..=16_u8)
        .map(|index| authority.insert(frame(index, 1)).unwrap())
        .collect::<Vec<_>>();
    let allocation_epoch = authority.allocation_epoch();
    let content_generation = authority.content_generation();

    assert_eq!(
        authority.insert(frame(17, 1)),
        Err(PlanetFrameAuthorityError::CapacityExceeded),
    );
    assert_eq!(authority.len(), 16);
    assert_eq!(authority.row_span(), 16);
    assert_eq!(authority.allocation_epoch(), allocation_epoch);
    assert_eq!(authority.content_generation(), content_generation);
    for (index, id) in ids.into_iter().enumerate() {
        assert_eq!(authority.get(id), Some(&frame(index as u8 + 1, 1)));
    }
}

#[test]
fn stale_conflicting_and_invalid_updates_are_transactional() {
    let ctx = support::test_context();
    let mut authority =
        PlanetFrameAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let id = authority.insert(frame(7, 5)).unwrap();
    let generation = authority.content_generation();

    assert_eq!(
        authority.set(id, frame(7, 4)).unwrap(),
        PlanetFrameUpdateOutcome::Stale { newest_frame: 5 },
    );
    let mut conflict = frame(7, 5);
    conflict.origin_x[0] ^= 32;
    assert_eq!(
        authority.set(id, conflict).unwrap(),
        PlanetFrameUpdateOutcome::FrameConflict,
    );
    assert_eq!(
        authority.insert(frame(7, 6)),
        Err(PlanetFrameAuthorityError::DuplicatePlanet(PlanetId([7; 16]))),
    );
    let mut invalid = frame(8, 1);
    invalid.page_edge_cells = 0;
    assert_eq!(
        authority.insert(invalid),
        Err(PlanetFrameAuthorityError::InvalidFrame),
    );
    assert_eq!(authority.content_generation(), generation);
    assert_eq!(authority.len(), 1);
    assert_eq!(authority.get(id), Some(&frame(7, 5)));

    assert_eq!(
        authority.set(id, frame(8, 6)),
        Err(PlanetFrameAuthorityError::PlanetIdentityMismatch {
            expected: PlanetId([7; 16]),
            actual: PlanetId([8; 16]),
        }),
    );
    assert_eq!(authority.content_generation(), generation);
}
