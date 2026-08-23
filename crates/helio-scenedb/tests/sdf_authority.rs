#[path = "support/mod.rs"]
mod support;

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio_scenedb::{
    BooleanOp, GpuSdfEdit, SdfAuthority, SdfAuthorityError, SdfEdit, SdfShapeParams,
    SdfShapeType, TerrainConfig,
};

fn edit(shape: SdfShapeType, x: f32) -> SdfEdit {
    let params = match shape {
        SdfShapeType::Sphere => SdfShapeParams::sphere(1.0),
        SdfShapeType::Cube => SdfShapeParams::cube(1.0, 2.0, 3.0),
        SdfShapeType::Capsule => SdfShapeParams::capsule(1.0, 2.0),
        SdfShapeType::Torus => SdfShapeParams::torus(2.0, 0.5),
        SdfShapeType::Cylinder => SdfShapeParams::cylinder(1.0, 2.0),
    };
    SdfEdit {
        shape,
        op: BooleanOp::Union,
        transform: Mat4::from_translation(Vec3::new(x, 0.0, 0.0)),
        params,
        blend_radius: 0.0,
    }
}

fn gpu_rows(authority: &SdfAuthority, bytes: &[u8]) -> Vec<GpuSdfEdit> {
    bytes
        .chunks_exact(std::mem::size_of::<GpuSdfEdit>())
        .take(authority.len())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

#[test]
fn ordered_crud_preserves_identity_and_reuses_only_with_a_new_generation() {
    let ctx = support::test_context();
    let mut authority = SdfAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));

    let sphere = authority.add(edit(SdfShapeType::Sphere, 0.0)).unwrap();
    let cube = authority.add(edit(SdfShapeType::Cube, 10.0)).unwrap();
    let capsule = authority.add(edit(SdfShapeType::Capsule, 20.0)).unwrap();
    assert_eq!(authority.id_at(1), Some(cube));

    authority.move_edit(cube, 0).unwrap();
    authority
        .set(sphere, edit(SdfShapeType::Cylinder, 3.0))
        .unwrap();
    assert_eq!(authority.id_at(0), Some(cube));
    assert_eq!(authority.id_at(1), Some(sphere));
    assert_eq!(authority.id_at(2), Some(capsule));

    assert_eq!(authority.remove(cube).unwrap().shape, SdfShapeType::Cube);
    assert_eq!(authority.get(cube), None);
    assert_eq!(
        authority.remove(cube),
        Err(SdfAuthorityError::StaleEdit),
    );
    let replacement = authority.add(edit(SdfShapeType::Torus, 30.0)).unwrap();
    assert_eq!(replacement.slot(), cube.slot());
    assert_ne!(replacement.generation(), cube.generation());

    let publication = authority.publication();
    let bytes = support::readback(
        &ctx,
        &publication.edit_buffer,
        authority.len() as u64 * std::mem::size_of::<GpuSdfEdit>() as u64,
    );
    let rows = gpu_rows(&authority, &bytes);
    assert_eq!(
        rows.iter().map(|row| row.shape_type).collect::<Vec<_>>(),
        vec![
            SdfShapeType::Cylinder as u32,
            SdfShapeType::Capsule as u32,
            SdfShapeType::Torus as u32,
        ],
        "GPU rows must follow authored boolean order, not stable-id slot order",
    );
    assert!(rows
        .iter()
        .all(|row| (row.distance_scale - 1.0).abs() < 1.0e-6));

    authority.clear();
    assert!(authority.is_empty());
    assert_eq!(authority.get(sphere), None);
    assert_eq!(authority.get(capsule), None);
    assert_eq!(authority.get(replacement), None);
}

#[test]
fn edit_growth_preserves_rows_and_changes_only_the_allocation_epoch() {
    let ctx = support::test_context();
    let mut authority = SdfAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let initial_epoch = authority.publication().edit_allocation_epoch;
    for index in 0..20 {
        authority
            .add(edit(SdfShapeType::Sphere, index as f32))
            .unwrap();
    }
    let publication = authority.publication();
    assert!(publication.edit_allocation_epoch > initial_epoch);
    assert_eq!(publication.edit_count, 20);
    assert_eq!(publication.bounds.len(), 20);
    let bytes = support::readback(
        &ctx,
        &publication.edit_buffer,
        20 * std::mem::size_of::<GpuSdfEdit>() as u64,
    );
    let rows = gpu_rows(&authority, &bytes);
    for (index, row) in rows.iter().enumerate() {
        // Stored matrices are inverses, so translation has the opposite sign.
        assert_eq!(row.transform[12], -(index as f32));
    }
}

#[test]
fn terrain_and_edit_validation_are_generation_tracked_without_noop_churn() {
    let ctx = support::test_context();
    let mut authority = SdfAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let initial = authority.publication().content_generation;
    authority
        .set_terrain(Some(TerrainConfig::canyons()))
        .unwrap();
    let terrain_generation = authority.publication().content_generation;
    assert!(terrain_generation > initial);
    assert_eq!(
        authority.publication().terrain_y_bounds,
        Some([-20.0, 16.0]),
        "canyon detail adds three world units beyond authored amplitude",
    );
    authority
        .set_terrain(Some(TerrainConfig::canyons()))
        .unwrap();
    assert_eq!(
        authority.publication().content_generation,
        terrain_generation,
        "identical config must not invalidate clipmaps",
    );

    let invalid = SdfEdit {
        transform: Mat4::from_scale(Vec3::ZERO),
        ..edit(SdfShapeType::Sphere, 0.0)
    };
    assert_eq!(
        authority.add(invalid),
        Err(SdfAuthorityError::InvalidEdit),
    );
    let non_uniform = SdfEdit {
        transform: Mat4::from_scale(Vec3::new(1.0, 2.0, 1.0)),
        ..edit(SdfShapeType::Sphere, 0.0)
    };
    assert_eq!(
        authority.add(non_uniform),
        Err(SdfAuthorityError::InvalidEdit),
        "non-uniform transforms need a shape-specific exact distance solver",
    );
    let mut invalid_terrain = TerrainConfig::rolling();
    invalid_terrain.octaves = 0;
    assert_eq!(
        authority.set_terrain(Some(invalid_terrain)),
        Err(SdfAuthorityError::InvalidTerrain),
    );
}

#[test]
fn intersection_streams_publish_the_required_canonical_scan_mode() {
    let ctx = support::test_context();
    let mut authority = SdfAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let mut intersection = edit(SdfShapeType::Cube, 0.0);
    intersection.op = BooleanOp::Intersection;
    let id = authority.add(intersection).unwrap();
    assert!(authority.publication().requires_canonical_scan);

    authority
        .set(id, edit(SdfShapeType::Sphere, 0.0))
        .unwrap();
    assert!(!authority.publication().requires_canonical_scan);

    authority.set(id, intersection).unwrap();
    assert!(authority.publication().requires_canonical_scan);
    authority.remove(id).unwrap();
    assert!(!authority.publication().requires_canonical_scan);
}

#[test]
fn cpu_queries_read_the_same_canonical_stream() {
    let ctx = support::test_context();
    let mut authority = SdfAuthority::new(Arc::clone(ctx.device()), Arc::clone(ctx.queue()));
    let scaled_sphere = SdfEdit {
        transform: Mat4::from_scale(Vec3::splat(2.0)),
        ..edit(SdfShapeType::Sphere, 0.0)
    };
    assert!((scaled_sphere.to_gpu().distance_scale - 2.0).abs() < 1.0e-6);
    authority.add(scaled_sphere).unwrap();
    assert!((authority.evaluate_sdf(Vec3::ZERO) + 2.0).abs() < 1.0e-6);
    assert!((authority.evaluate_sdf(Vec3::X * 3.0) - 1.0).abs() < 1.0e-6);

    let hit = authority
        .pick_surface(Vec3::new(-4.0, 0.0, 0.0), Vec3::X * 4.0, 10.0)
        .expect("sphere should be hit");
    assert!((hit.distance - 2.0).abs() < 0.03);
    assert!(hit.normal.dot(-Vec3::X) > 0.99);
}
