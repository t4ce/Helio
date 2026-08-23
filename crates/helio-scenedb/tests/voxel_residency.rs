#[path = "support/mod.rs"]
mod support;

use std::sync::Arc;

use helio_scenedb::{Entity, VoxelResidency, VoxelResidencyError};
use helio_voxel_core::GpuVoxelMaterial;

fn u32_at(bytes: &[u8], word: usize) -> u32 {
    u32::from_le_bytes(bytes[word * 4..word * 4 + 4].try_into().unwrap())
}

#[test]
fn voxel_regions_are_distinct_preserved_and_cleared_before_reuse() {
    let ctx = support::test_context();
    let mut residency = VoxelResidency::new(
        Arc::clone(ctx.device()),
        Arc::clone(ctx.queue()),
    );
    let first = Entity::from_bits((1u64 << 32) | 17);
    let second = Entity::from_bits((3u64 << 32) | 91);

    assert_eq!(residency.allocate(first, 512).unwrap(), 0);
    assert_eq!(residency.brick_base(first), Some(0));
    let first_epoch = residency.epoch();
    assert_eq!(residency.allocate(second, 512).unwrap(), 512);
    assert_eq!(residency.brick_base(second), Some(512));
    assert!(residency.epoch() > first_epoch);

    let first_words = [0x0102_0304; 128];
    let second_words = [0xa1b2_c3d4; 128];
    residency.write_brick(first, 0, true, &first_words).unwrap();
    residency.write_brick(second, 0, true, &second_words).unwrap();

    let brick_bytes = support::readback(&ctx, residency.brick_buffer(), 513 * 8);
    let data_bytes = support::readback(
        &ctx,
        residency.data_buffer(),
        (65_536 + 128) * 4,
    );
    assert_eq!(u32_at(&brick_bytes, 0), 1 << 24);
    assert_eq!(u32_at(&brick_bytes, 512 * 2), (1 << 24) | 65_536);
    assert_eq!(u32_at(&data_bytes, 0), first_words[0]);
    assert_eq!(u32_at(&data_bytes, 65_536), second_words[0]);

    assert_eq!(residency.release(first).unwrap(), 0);
    let replacement = Entity::from_bits((9u64 << 32) | 17);
    assert_eq!(residency.allocate(replacement, 256).unwrap(), 0);
    assert_eq!(residency.brick_base(replacement), Some(0));

    let brick_bytes = support::readback(&ctx, residency.brick_buffer(), 513 * 8);
    let data_bytes = support::readback(
        &ctx,
        residency.data_buffer(),
        (65_536 + 1) * 4,
    );
    assert_eq!(u32_at(&brick_bytes, 0), 0, "reused metadata must be tombstoned");
    assert_eq!(u32_at(&data_bytes, 0), 0, "reused voxel bytes must be cleared");
    assert_eq!(
        u32_at(&brick_bytes, 512 * 2),
        (1 << 24) | 65_536,
        "growth and reuse must preserve the surviving volume",
    );
    assert_eq!(u32_at(&data_bytes, 65_536), second_words[0]);

    assert_eq!(residency.release(replacement).unwrap(), 0);
    let coalesced = Entity::from_bits((10u64 << 32) | 17);
    assert_eq!(
        residency.allocate(coalesced, 512).unwrap(),
        0,
        "adjacent free ranges must coalesce without relocating the survivor",
    );
}

#[test]
fn packed_offsets_define_an_exact_addressable_brick_ceiling() {
    assert_eq!(VoxelResidency::MAX_ADDRESSABLE_BRICKS, 131_072);
    let final_region_word = (VoxelResidency::MAX_ADDRESSABLE_BRICKS - 1) * 128;
    assert!(final_region_word <= 0x00ff_ffff);
    assert_eq!(VoxelResidency::MAX_ADDRESSABLE_BRICKS * 128, 0x0100_0000);
}

#[test]
fn palette_regions_scale_relocate_and_coalesce_without_fixed_per_volume_waste() {
    let ctx = support::test_context();
    let mut residency = VoxelResidency::new(
        Arc::clone(ctx.device()),
        Arc::clone(ctx.queue()),
    );
    let first = Entity::from_bits((2u64 << 32) | 4);
    let second = Entity::from_bits((2u64 << 32) | 5);
    let replacement = Entity::from_bits((7u64 << 32) | 4);
    let material = |value: f32| GpuVoxelMaterial {
        color: [value, value + 1.0, value + 2.0],
        roughness: value + 3.0,
        metalness: value + 4.0,
        emissive: value + 5.0,
        _pad: [0; 2],
    };

    let first_allocation = residency
        .allocate_with_palette(first, 1, &[material(1.0)])
        .unwrap();
    let second_allocation = residency
        .allocate_with_palette(second, 1, &[material(10.0)])
        .unwrap();
    assert_eq!(first_allocation.palette_base, 0);
    assert_eq!(second_allocation.palette_base, 1);

    let grown = [material(20.0), material(30.0), material(40.0)];
    let (grown_base, grown_count) = residency.write_palette(first, &grown).unwrap();
    assert_eq!((grown_base, grown_count), (2, 3));
    assert_eq!(residency.palette_base(second), Some(1));
    let bytes = support::readback(
        &ctx,
        residency.palette_buffer(),
        ((grown_base as usize + grown.len()) * std::mem::size_of::<GpuVoxelMaterial>()) as u64,
    );
    let row_bytes = std::mem::size_of::<GpuVoxelMaterial>();
    let expected_bytes: &[u8] = bytemuck::cast_slice(&grown);
    assert_eq!(
        &bytes[grown_base as usize * row_bytes..(grown_base as usize + 3) * row_bytes],
        expected_bytes,
    );

    residency.release(second).unwrap();
    residency.release(first).unwrap();
    let replacement_palette = vec![material(50.0); 6];
    let allocation = residency
        .allocate_with_palette(replacement, 1, &replacement_palette)
        .unwrap();
    assert_eq!(allocation.palette_base, 0, "released power-of-two ranges coalesce");
    assert_eq!(allocation.palette_count, 6);

    let boundary = vec![material(60.0); helio_voxel_core::MAX_PALETTE_SIZE as usize];
    let (_, boundary_count) = residency.write_palette(replacement, &boundary).unwrap();
    assert_eq!(boundary_count, helio_voxel_core::MAX_PALETTE_SIZE);

    let too_large = vec![material(0.0); helio_voxel_core::MAX_PALETTE_SIZE as usize + 1];
    assert_eq!(
        residency.write_palette(replacement, &too_large),
        Err(VoxelResidencyError::PaletteTooLarge)
    );
}
