use glam::{Mat4, Vec3};
use helio_scenedb::{
    ScenePostProcessVolume, SceneReflectionCapture, SceneWaterHitbox, SceneWaterVolume,
};

use super::{
    ReflectionCaptureDescriptor, Scene, WaterHitboxDescriptor, WaterVolumeDescriptor,
};

fn create_test_scene() -> Option<Scene> {
    let (device, queue) = crate::test_support::test_gpu()?;
    Some(Scene::new(device, queue))
}

#[test]
fn environment_crud_tags_queries_and_gpu_rows_stay_in_lockstep() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping environment SceneDB test: no GPU adapter");
        return;
    };

    let first_water = scene
        .insert_water_volume_with_tag(WaterVolumeDescriptor::default(), 101)
        .unwrap();
    let mut second_desc = WaterVolumeDescriptor::default();
    second_desc.water_color = [0.1, 0.3, 0.8];
    let second_water = scene
        .insert_water_volume_with_tag(second_desc, 102)
        .unwrap();
    let hitbox = scene
        .insert_water_hitbox_with_tag(
            WaterHitboxDescriptor {
                old_min: [-1.0; 3],
                old_max: [1.0; 3],
                new_min: [-0.5; 3],
                new_max: [0.5; 3],
                edge_softness: 0.4,
                strength: 1.25,
            },
            201,
        )
        .unwrap();
    let pp = scene
        .insert_post_process_volume_with_tag(
            libhelio::PostProcessVolumeDescriptor::default(),
            301,
        )
        .unwrap();
    let capture = scene
        .insert_reflection_capture_with_tag(ReflectionCaptureDescriptor::default(), 401)
        .unwrap();

    assert_eq!(scene.water_volume_by_tag(101), Some(first_water));
    assert_eq!(scene.water_volume_by_tag(102), Some(second_water));
    assert_eq!(scene.water_hitbox_by_tag(201), Some(hitbox));
    assert_eq!(scene.post_process_volume_by_tag(301), Some(pp));
    assert_eq!(scene.reflection_capture_by_tag(401), Some(capture));
    assert_eq!(scene.iter_water_volumes().count(), 2);
    assert_eq!(scene.iter_water_hitboxes().count(), 1);
    assert_eq!(scene.iter_post_process_volumes().count(), 1);
    assert_eq!(scene.iter_reflection_captures().count(), 1);

    let second_entity = crate::handles::entity_from_handle(second_water);
    let second_row = scene
        .authority
        .gpu_row::<SceneWaterVolume>(second_entity)
        .unwrap();
    let first_slot_generation = scene.gpu_scene.water_sim_slot_generations[0];
    scene.remove_water_volume(first_water).unwrap();
    // Swap-removing compact membership must not move the surviving volume's
    // persistent heightfield residency from slot 1 into the vacated slot 0.
    assert_eq!(
        scene.gpu_scene.water_volume_projections.as_slice(),
        &[[second_row, 1]]
    );

    // A new occupant may reuse slot 0, but its reset epoch must change even if
    // SceneDB also reuses the removed component row. WaterSim observes this
    // epoch and clears both ping-pong textures before the first simulation.
    let replacement = scene
        .insert_water_volume_with_tag(WaterVolumeDescriptor::default(), 103)
        .unwrap();
    assert_eq!(scene.gpu_scene.water_volume_projections.as_slice()[1][1], 0);
    assert_ne!(
        scene.gpu_scene.water_sim_slot_generations[0],
        first_slot_generation
    );
    scene.remove_water_volume(replacement).unwrap();

    scene.assign_reflection_capture_layers();
    assert_eq!(scene.get_reflection_capture(capture).unwrap().cubemap_index, 0);

    let mut wind = libhelio::Wind::default();
    wind.direction = Vec3::new(3.0, 0.0, 4.0);
    wind.speed = 7.0;
    scene.set_wind(wind);
    scene.advance_wind(0.25);
    let queried_wind = scene.wind();
    assert_eq!(queried_wind.direction, wind.direction);
    assert_eq!(queried_wind.speed, 7.0);
    assert_eq!(queried_wind.time, 0.25);
    assert_eq!(queried_wind.prev_time, 0.0);

    scene.flush();
    let resources = scene.gpu_scene.resources();
    assert_eq!(resources.water_volume_count, 1);
    assert_eq!(resources.water_hitbox_count, 1);
    assert_eq!(resources.post_process_volume_count, 1);
    assert_eq!(resources.reflection_capture_count, 1);
    assert!(scene.gpu_scene.canonical.water_volumes.is_some());
    assert!(scene.gpu_scene.canonical.water_hitboxes.is_some());
    assert!(scene.gpu_scene.canonical.post_process_volumes.is_some());
    assert!(scene.gpu_scene.canonical.reflection_captures.is_some());

    scene.clear();
    assert_eq!(scene.water_volumes_count(), 0);
    assert_eq!(scene.water_hitboxes_count(), 0);
    assert_eq!(scene.post_process_volumes_count(), 0);
    assert_eq!(scene.reflection_capture_count(), 0);
    assert_eq!(scene.water_volume_by_tag(102), None);
    assert_eq!(scene.water_hitbox_by_tag(201), None);
    assert_eq!(scene.post_process_volume_by_tag(301), None);
    assert_eq!(scene.reflection_capture_by_tag(401), None);
    assert!(scene.get_water_volume(second_water).is_none());
    assert!(scene.get_water_hitbox(hitbox).is_none());
    assert!(scene.get_post_process_volume(pp).is_none());
    assert!(scene.get_reflection_capture(capture).is_none());

    // Clear removes authored scene entities but intentionally preserves the
    // global wind subsystem just like the pre-SceneDB Scene field did.
    assert_eq!(scene.wind().time, 0.25);
    assert_eq!(scene.authority.gpu_live_count::<SceneWaterVolume>(), 0);
    assert_eq!(scene.authority.gpu_live_count::<SceneWaterHitbox>(), 0);
    assert_eq!(scene.authority.gpu_live_count::<ScenePostProcessVolume>(), 0);
    assert_eq!(scene.authority.gpu_live_count::<SceneReflectionCapture>(), 0);
}

#[test]
fn water_simulation_authorship_is_per_volume_and_updates_transactionally() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping water simulation authority test: no GPU adapter");
        return;
    };

    let id = scene
        .insert_water_volume(WaterVolumeDescriptor::default())
        .unwrap();
    let before = scene.get_water_volume(id).unwrap();

    let mut invalid = WaterVolumeDescriptor::default();
    invalid.wave_scale = 0.0;
    assert!(scene.update_water_volume(id, invalid).is_err());
    let after_rejection = scene.get_water_volume(id).unwrap();
    assert_eq!(
        bytemuck::bytes_of(&before),
        bytemuck::bytes_of(&after_rejection),
        "a rejected authored edit must not partially mutate the canonical row"
    );

    let mut edited = WaterVolumeDescriptor::default();
    edited.wave_spring = 1.75;
    edited.wave_damping = 0.91;
    edited.wave_speed = 2.5;
    edited.wave_scale = 0.35;
    edited.wind_direction = [3.0, 4.0];
    edited.wind_strength = 4.25;
    scene.update_water_volume(id, edited).unwrap();

    let canonical = scene.get_water_volume(id).unwrap();
    assert_eq!(canonical.sim_dynamics, [1.75, 0.91, 0.35, 0.0]);
    assert_eq!(canonical.wave_params[2], 2.5);
    assert!((canonical.wind_params[0] - 0.6).abs() < 1e-6);
    assert!((canonical.wind_params[1] - 0.8).abs() < 1e-6);
    assert_eq!(canonical.wind_params[2], 4.25);
}

#[test]
fn water_simulation_targets_reject_stale_and_unsimulated_handles() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping water simulation target test: no GPU adapter");
        return;
    };

    let mut ids = Vec::new();
    for _ in 0..=helio_core::WATER_SIM_SLOT_COUNT {
        ids.push(
            scene
                .insert_water_volume(WaterVolumeDescriptor::default())
                .unwrap(),
        );
    }
    let first_target = scene.water_volume_sim_target(ids[0]).unwrap();
    let first_drop = scene.water_drop_target(ids[0], [0.0, 0.0]).unwrap();
    assert_eq!(first_drop.simulation(), first_target);
    assert_eq!(first_drop.world_center(), [0.0, 0.0]);
    assert!(matches!(
        scene.water_drop_target(ids[0], [10_000.0, 0.0]),
        Err(super::SceneError::InvalidOperation { .. })
    ));
    assert!(matches!(
        scene.water_volume_sim_target(ids[helio_core::WATER_SIM_SLOT_COUNT]),
        Err(super::SceneError::InvalidOperation { .. })
    ));
    assert!(matches!(
        scene.water_drop_target(
            ids[helio_core::WATER_SIM_SLOT_COUNT],
            [0.0, 0.0]
        ),
        Err(super::SceneError::InvalidOperation { .. })
    ));

    scene.remove_water_volume(ids[0]).unwrap();
    assert!(matches!(
        scene.water_volume_sim_target(ids[0]),
        Err(super::SceneError::InvalidHandle { .. })
    ));
    assert!(matches!(
        scene.water_drop_target(ids[0], [0.0, 0.0]),
        Err(super::SceneError::InvalidHandle { .. })
    ));

    let promoted = scene
        .water_volume_sim_target(ids[helio_core::WATER_SIM_SLOT_COUNT])
        .unwrap();
    assert_eq!(promoted.sim_slot(), first_target.sim_slot());
    assert_ne!(
        promoted.residency_generation(),
        first_target.residency_generation(),
        "slot reuse must invalidate already queued transient targets"
    );
    let promoted_drop = scene
        .water_drop_target(ids[helio_core::WATER_SIM_SLOT_COUNT], [0.0, 0.0])
        .unwrap();
    assert_eq!(promoted_drop.simulation(), promoted);
}

#[test]
fn post_process_shader_projection_keeps_high_priority_rows_without_truncating_authority() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping post-process projection test: no GPU adapter");
        return;
    };

    let mut ids = Vec::new();
    for priority in 0..=libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS {
        let mut desc = libhelio::PostProcessVolumeDescriptor::default();
        desc.priority = priority as f32;
        ids.push(scene.insert_post_process_volume(desc).unwrap());
    }

    assert_eq!(
        scene.iter_post_process_volumes().count(),
        libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS + 1,
    );
    assert_eq!(
        scene.post_process_volumes_count() as usize,
        libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS + 1,
    );
    assert_eq!(
        scene.gpu_scene.post_process_volume_indices.len(),
        libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS,
    );

    let lowest_row = scene
        .authority
        .gpu_row::<ScenePostProcessVolume>(crate::handles::entity_from_handle(ids[0]))
        .unwrap();
    let highest_row = scene
        .authority
        .gpu_row::<ScenePostProcessVolume>(crate::handles::entity_from_handle(*ids.last().unwrap()))
        .unwrap();
    assert!(!scene
        .gpu_scene
        .post_process_volume_indices
        .as_slice()
        .contains(&lowest_row));
    assert!(scene
        .gpu_scene
        .post_process_volume_indices
        .as_slice()
        .contains(&highest_row));

    let mut promoted = libhelio::PostProcessVolumeDescriptor::default();
    promoted.priority = 10_000.0;
    scene
        .update_post_process_volume(ids[0], promoted)
        .unwrap();
    assert!(scene
        .gpu_scene
        .post_process_volume_indices
        .as_slice()
        .contains(&lowest_row));
    assert_eq!(scene.iter_post_process_volumes().count(), ids.len());
}

#[test]
fn post_process_priority_validation_is_transactional_for_insert_and_update() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping post-process priority validation test: no GPU adapter");
        return;
    };

    let mut invalid_insert = libhelio::PostProcessVolumeDescriptor::default();
    invalid_insert.priority = f32::NAN;
    assert!(matches!(
        scene.insert_post_process_volume(invalid_insert),
        Err(super::SceneError::InvalidOperation { .. })
    ));
    assert_eq!(scene.iter_post_process_volumes().count(), 0);
    assert!(scene.gpu_scene.post_process_volume_indices.is_empty());

    let mut authored = libhelio::PostProcessVolumeDescriptor::default();
    authored.priority = 7.0;
    let id = scene.insert_post_process_volume(authored).unwrap();
    let entity = crate::handles::entity_from_handle(id);
    let row = scene
        .authority
        .gpu_row::<ScenePostProcessVolume>(entity)
        .unwrap();
    let old_component = *scene.authority.get::<ScenePostProcessVolume>(entity).unwrap();
    let old_projection = scene.gpu_scene.post_process_volume_indices.as_slice().to_vec();
    let old_generation = id.generation();

    let mut invalid_update = libhelio::PostProcessVolumeDescriptor::default();
    invalid_update.priority = f32::INFINITY;
    assert!(matches!(
        scene.update_post_process_volume(id, invalid_update),
        Err(super::SceneError::InvalidOperation { .. })
    ));

    let component = scene.authority.get::<ScenePostProcessVolume>(entity).unwrap();
    assert_eq!(component.user_tag, old_component.user_tag);
    assert_eq!(
        bytemuck::bytes_of(&component.volume),
        bytemuck::bytes_of(&old_component.volume),
    );
    assert_eq!(component._reserved, old_component._reserved);
    assert_eq!(
        scene.gpu_scene.post_process_volume_indices.as_slice(),
        old_projection,
    );
    assert_eq!(
        scene
            .authority
            .gpu_row::<ScenePostProcessVolume>(entity),
        Some(row),
    );
    assert_eq!(id.generation(), old_generation);
}

#[test]
fn static_reflection_crud_invalidates_bakes_but_dynamic_only_edits_do_not() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping reflection bake invalidation test: no GPU adapter");
        return;
    };

    let mut static_desc = ReflectionCaptureDescriptor::default();
    let static_capture = scene
        .insert_reflection_capture(static_desc.clone())
        .unwrap();
    assert!(scene.is_bake_invalidated());

    // Model the clean baseline established after a successful configured bake.
    scene.bake_invalidated = false;
    static_desc.transform = Mat4::from_translation(Vec3::new(4.0, 2.0, -3.0));
    scene
        .update_reflection_capture(static_capture, &static_desc)
        .unwrap();
    assert!(scene.is_bake_invalidated());

    scene.bake_invalidated = false;
    let mut dynamic_desc = ReflectionCaptureDescriptor::default().dynamic();
    let dynamic_capture = scene
        .insert_reflection_capture(dynamic_desc.clone())
        .unwrap();
    assert!(!scene.is_bake_invalidated());
    dynamic_desc.brightness = 2.0;
    scene
        .update_reflection_capture(dynamic_capture, &dynamic_desc)
        .unwrap();
    assert!(!scene.is_bake_invalidated());
    assert!(scene.remove_reflection_capture(dynamic_capture));
    assert!(!scene.is_bake_invalidated());

    assert!(scene.remove_reflection_capture(static_capture));
    assert!(scene.is_bake_invalidated());
}

#[test]
fn reflection_descriptor_validation_is_transactional_for_insert_and_update() {
    let Some(mut scene) = create_test_scene() else {
        eprintln!("skipping reflection validation test: no GPU adapter");
        return;
    };

    let invalid_inserts = [
        ReflectionCaptureDescriptor {
            transform: Mat4::from_cols_array(&[f32::NAN; 16]),
            ..ReflectionCaptureDescriptor::default()
        },
        ReflectionCaptureDescriptor {
            transform: Mat4::ZERO,
            ..ReflectionCaptureDescriptor::default()
        },
        ReflectionCaptureDescriptor {
            influence_radius: -1.0,
            ..ReflectionCaptureDescriptor::default()
        },
        ReflectionCaptureDescriptor {
            transition_distance: f32::INFINITY,
            ..ReflectionCaptureDescriptor::default()
        },
    ];
    for invalid in invalid_inserts {
        assert!(matches!(
            scene.insert_reflection_capture(invalid),
            Err(super::SceneError::InvalidOperation { .. })
        ));
    }
    assert_eq!(scene.reflection_capture_count(), 0);
    assert_eq!(scene.authority.gpu_live_count::<SceneReflectionCapture>(), 0);
    assert!(scene.gpu_scene.reflection_capture_projections.is_empty());
    assert!(!scene.is_bake_invalidated());

    let id = scene
        .insert_reflection_capture(ReflectionCaptureDescriptor::default())
        .unwrap();
    scene.assign_reflection_capture_layers();
    scene.bake_invalidated = false;

    let entity = crate::handles::entity_from_handle(id);
    let row = scene
        .authority
        .gpu_row::<SceneReflectionCapture>(entity)
        .unwrap();
    let old_component = *scene.authority.get::<SceneReflectionCapture>(entity).unwrap();
    let old_projection = scene.gpu_scene.reflection_capture_projections.as_slice().to_vec();
    let old_generation = id.generation();
    let invalid_updates = [
        ReflectionCaptureDescriptor {
            transform: Mat4::ZERO,
            ..ReflectionCaptureDescriptor::default()
        },
        ReflectionCaptureDescriptor {
            extents: [1.0, -0.5, 1.0],
            ..ReflectionCaptureDescriptor::default()
        },
        ReflectionCaptureDescriptor {
            influence_radius: f32::NAN,
            ..ReflectionCaptureDescriptor::default()
        },
        ReflectionCaptureDescriptor {
            brightness: f32::NEG_INFINITY,
            ..ReflectionCaptureDescriptor::default()
        },
    ];
    for invalid in &invalid_updates {
        assert!(matches!(
            scene.update_reflection_capture(id, invalid),
            Err(super::SceneError::InvalidOperation { .. })
        ));
    }

    let component = scene.authority.get::<SceneReflectionCapture>(entity).unwrap();
    assert_eq!(component.user_tag, old_component.user_tag);
    assert_eq!(
        bytemuck::bytes_of(&component.capture),
        bytemuck::bytes_of(&old_component.capture),
    );
    assert_eq!(component._reserved, old_component._reserved);
    assert_eq!(
        scene.gpu_scene.reflection_capture_projections.as_slice(),
        old_projection,
    );
    assert_eq!(
        scene.authority.gpu_row::<SceneReflectionCapture>(entity),
        Some(row),
    );
    assert_eq!(id.generation(), old_generation);
    assert_eq!(scene.get_reflection_capture(id).unwrap().cubemap_index, 0);
    assert!(!scene.is_bake_invalidated());
}
