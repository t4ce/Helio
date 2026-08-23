#[path = "support/mod.rs"]
mod support;

use std::sync::Arc;

use bytemuck::Zeroable;
use helio_scenedb::{
    register_scene_component_buffers, SceneAssetError, SceneAuthority, SceneAuthorityConfig,
    SceneAuthoritySubsystemConfig,
    SceneMaterial, SceneMaterialTextureRef, SceneMaterialTextureRefs,
    RadiantGraphRegistry, SceneMaterialTextureSlotRow, SceneMaterialTexturesRow, SceneTexture,
    SceneTextureAssetKey, SceneTextureSampler, SceneTextureTransform, TextureResidency,
    MATERIAL_BUFFER_KEY, MATERIAL_TEXTURES_BUFFER_KEY,
};
use pulsar_scenedb::{
    component_id, gpu_column_descs_for_component, GpuColumnSet, MirrorMode,
};

fn texture(key: u128) -> SceneTexture {
    SceneTexture::sampled_2d(
        SceneTextureAssetKey(key),
        1,
        1,
        wgpu::TextureFormat::Rgba8Unorm,
        SceneTextureSampler::default(),
    )
}

fn material_with_base_color(texture: pulsar_scenedb::Entity) -> SceneMaterialTextureRefs {
    SceneMaterialTextureRefs {
        base_color: Some(SceneMaterialTextureRef {
            texture,
            uv_channel: 1,
            transform: SceneTextureTransform {
                offset: [0.25, 0.5],
                scale: [2.0, 3.0],
                rotation_radians: 0.75,
            },
        }),
        ..Default::default()
    }
}

#[test]
fn material_texture_rows_have_named_padding_safe_reflection() {
    fn assert_pod<T: bytemuck::Pod>() {}
    assert_pod::<SceneMaterialTextureSlotRow>();
    assert_pod::<SceneMaterialTexturesRow>();

    assert_eq!(std::mem::size_of::<SceneMaterialTextureSlotRow>(), 48);
    assert_eq!(std::mem::offset_of!(SceneMaterialTextureSlotRow, offset_scale), 16);
    assert_eq!(std::mem::offset_of!(SceneMaterialTextureSlotRow, rotation), 32);
    assert_eq!(std::mem::size_of::<SceneMaterialTexturesRow>(), 352);
    assert_eq!(std::mem::offset_of!(SceneMaterialTexturesRow, params), 336);
    assert_eq!(std::mem::size_of::<SceneMaterial>(), 464);

    let columns = SceneMaterial::gpu_columns();
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[0].buffer_key, Some(MATERIAL_BUFFER_KEY));
    assert_eq!(columns[0].mode, MirrorMode::DirtyTracked);
    assert_eq!(columns[0].value_token.desc().size, 96);
    assert_eq!(columns[1].buffer_key, Some(MATERIAL_TEXTURES_BUFFER_KEY));
    assert_eq!(columns[1].mode, MirrorMode::DirtyTracked);
    assert_eq!(columns[1].value_token.desc().size, 352);

    let reflected = gpu_column_descs_for_component(component_id::<SceneMaterial>())
        .expect("SceneMaterial derive must publish World reflection");
    assert_eq!(reflected, columns);
}

#[test]
fn authority_subsystem_config_keeps_full_default_and_sprite_authority_narrow() {
    let ctx = support::test_context();
    let full = SceneAuthority::new(
        Arc::clone(ctx.device()),
        Arc::clone(ctx.queue()),
        SceneAuthorityConfig::default(),
        |_, _| {},
    );
    assert!(full.subsystem::<TextureResidency>().is_some());
    assert!(full.subsystem::<RadiantGraphRegistry>().is_some());

    let mut sprite_config = SceneAuthorityConfig::default();
    sprite_config.subsystems = SceneAuthoritySubsystemConfig::SPRITE_STANDALONE;
    let sprite = SceneAuthority::new(
        Arc::clone(ctx.device()),
        Arc::clone(ctx.queue()),
        sprite_config,
        |_, _| {},
    );
    assert!(sprite.subsystem::<TextureResidency>().is_none());
    assert!(sprite.subsystem::<RadiantGraphRegistry>().is_none());
}

#[test]
fn texture_asset_key_allocator_is_monotonic_collision_safe_and_non_recycling() {
    let ctx = support::test_context();
    let mut authority = SceneAuthority::new(
        Arc::clone(ctx.device()),
        Arc::clone(ctx.queue()),
        SceneAuthorityConfig::default(),
        |store, device| register_scene_component_buffers(store, 32, device),
    );

    let explicit = authority
        .insert_texture_asset(texture(1), Some("explicit-1"), &[255, 0, 0, 255])
        .expect("explicit key registration");
    let first_automatic = authority
        .subsystem_mut::<TextureResidency>()
        .unwrap()
        .allocate_asset_key()
        .expect("automatic key after a live explicit collision");
    assert_eq!(first_automatic, SceneTextureAssetKey(2));
    let automatic = authority
        .insert_texture_asset(
            texture(first_automatic.0),
            Some("automatic-2"),
            &[0, 255, 0, 255],
        )
        .expect("automatic key registration");

    assert_eq!(
        authority.insert_texture_asset(
            texture(1),
            Some("duplicate-explicit-1"),
            &[0, 0, 255, 255],
        ),
        Err(SceneAssetError::DuplicateTextureAsset {
            asset_key: SceneTextureAssetKey(1),
            existing: explicit,
        })
    );
    authority
        .remove_texture_asset(explicit)
        .expect("remove explicit asset");
    authority
        .remove_texture_asset(automatic)
        .expect("remove automatic asset");

    let after_removal = authority
        .subsystem_mut::<TextureResidency>()
        .unwrap()
        .allocate_asset_key()
        .expect("removed keys must not reset the high-water mark");
    assert_eq!(after_removal, SceneTextureAssetKey(3));

    authority
        .insert_texture_asset(
            texture(u128::MAX),
            Some("explicit-max"),
            &[255, 255, 255, 255],
        )
        .expect("the final valid explicit key remains representable");
    assert_eq!(
        authority
            .subsystem_mut::<TextureResidency>()
            .unwrap()
            .allocate_asset_key(),
        Ok(SceneTextureAssetKey(4)),
        "an unordered explicit maximum must not exhaust the automatic domain"
    );
}

#[test]
fn pinned_texture_slot_cannot_be_reused_under_a_live_material() {
    let ctx = support::test_context();
    let mut authority = SceneAuthority::new(
        Arc::clone(ctx.device()),
        Arc::clone(ctx.queue()),
        SceneAuthorityConfig::default(),
        |store, device| register_scene_component_buffers(store, 32, device),
    );

    let texture_a = authority
        .insert_texture_asset(texture(1), Some("texture-a"), &[255, 0, 0, 255])
        .expect("insert texture A");
    let texture_b = authority
        .insert_texture_asset(texture(2), Some("texture-b"), &[0, 255, 0, 255])
        .expect("insert texture B");
    let residency = authority.subsystem::<TextureResidency>().unwrap();
    let slot_a = residency.slot_for(texture_a).unwrap();
    let slot_b = residency.slot_for(texture_b).unwrap();
    assert_eq!((slot_a, slot_b), (0, 1));
    assert_eq!(residency.entity_for_asset(SceneTextureAssetKey(1)), Some(texture_a));
    assert!(residency.view_for_slot(slot_a).is_some());
    assert!(residency.sampler_for_slot(slot_a).is_some());
    assert_eq!(
        authority.insert_texture_asset(texture(1), Some("duplicate-a"), &[255, 0, 0, 255]),
        Err(SceneAssetError::DuplicateTextureAsset {
            asset_key: SceneTextureAssetKey(1),
            existing: texture_a,
        })
    );

    let mut authored = libhelio::GpuMaterial::zeroed();
    authored.tex_base_color = 77;
    authored.tex_normal = 88;
    let material = authority
        .insert_material_asset(authored, material_with_base_color(texture_a), 0x1234)
        .expect("insert material");
    let material_row = authority.gpu_row::<SceneMaterial>(material).unwrap();
    assert_eq!(material_row, 0);
    assert_ne!(material_row, material.index());
    assert_eq!(authority.gpu_live_count::<SceneMaterial>(), 1);
    assert_eq!(authority.gpu_row_span::<SceneMaterial>(), 1);
    let canonical = authority.get::<SceneMaterial>(material).unwrap();
    assert_eq!(canonical.material.0.tex_base_color, slot_a);
    assert_eq!(canonical.material.0.tex_normal, libhelio::GpuMaterial::NO_TEXTURE);
    assert_eq!(canonical.textures.base_color.texture_index, slot_a);
    assert_eq!(canonical.textures.base_color.uv_channel, 1);
    assert_eq!(canonical.textures.base_color.offset_scale, [0.25, 0.5, 2.0, 3.0]);
    assert_eq!(authority.get::<SceneTexture>(texture_a).unwrap().ref_count, 1);
    assert_eq!(
        authority
            .subsystem::<TextureResidency>()
            .unwrap()
            .material_pin_count(texture_a),
        Some(1)
    );

    assert_eq!(
        authority.remove_texture_asset(texture_a),
        Err(SceneAssetError::TextureInUse {
            entity: texture_a,
            ref_count: 1,
        })
    );

    authority.flush_gpu();
    let idle = authority.flush_gpu();
    assert_eq!((idle.ranges, idle.bytes), (0, 0));

    authority
        .update_material_asset(material, authored, material_with_base_color(texture_b), 0x1234)
        .expect("move material reference to B");
    let updated = authority.flush_gpu();
    assert_eq!(updated.ranges, 2);
    assert_eq!(updated.bytes, 96 + 352);
    assert_eq!(authority.get::<SceneTexture>(texture_a).unwrap().ref_count, 0);
    assert_eq!(authority.get::<SceneTexture>(texture_b).unwrap().ref_count, 1);

    authority
        .remove_texture_asset(texture_a)
        .expect("A is unpinned after material update");
    let texture_c = authority
        .insert_texture_asset(texture(3), Some("texture-c"), &[0, 0, 255, 255])
        .expect("insert texture C");
    {
        let residency = authority.subsystem::<TextureResidency>().unwrap();
        assert_eq!(residency.slot_for(texture_c), Some(slot_a));
        assert_eq!(residency.slot_for(texture_a), None);
        assert_eq!(residency.entity_for_slot(slot_a), Some(texture_c));
    }

    // Generation-bearing identity prevents the stale A reference from
    // resolving to C even though SceneDB deliberately recycled A's slot.
    assert_eq!(
        authority.update_material_asset(material, authored, material_with_base_color(texture_a), 0),
        Err(SceneAssetError::TextureNotResident(texture_a))
    );
    assert_eq!(
        authority
            .get::<SceneMaterial>(material)
            .unwrap()
            .material
            .0
            .tex_base_color,
        slot_b
    );
    assert_eq!(
        authority.remove_texture_asset(texture_b),
        Err(SceneAssetError::TextureInUse {
            entity: texture_b,
            ref_count: 1,
        })
    );

    let (epoch_before_sampler, slot_before_sampler) = {
        let residency = authority.subsystem::<TextureResidency>().unwrap();
        (residency.binding_epoch(), residency.slot_for(texture_c).unwrap())
    };
    authority
        .update_texture_sampler(
            texture_c,
            SceneTextureSampler {
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
            Some("texture-c-nearest"),
        )
        .expect("replace sampler");
    let residency = authority.subsystem::<TextureResidency>().unwrap();
    assert_eq!(residency.slot_for(texture_c), Some(slot_before_sampler));
    assert_eq!(residency.binding_epoch(), epoch_before_sampler.wrapping_add(1));

    authority.retain_material(material).expect("retain material");
    let retain_only = authority.flush_gpu();
    assert_eq!((retain_only.ranges, retain_only.bytes), (0, 0));
    assert!(matches!(
        authority.remove_material_asset(material),
        Err(SceneAssetError::MaterialInUse {
            entity,
            ref_count: 1,
        }) if entity == material
    ));
    authority.release_material(material).expect("release material");
    let ref_count_only = authority.flush_gpu();
    assert_eq!((ref_count_only.ranges, ref_count_only.bytes), (0, 0));
    authority
        .remove_material_asset(material)
        .expect("remove unreferenced material");
    assert_eq!(authority.gpu_row::<SceneMaterial>(material), None);
    assert_eq!(authority.gpu_live_count::<SceneMaterial>(), 0);
    assert_eq!(authority.gpu_row_span::<SceneMaterial>(), 0);
    authority
        .remove_texture_asset(texture_b)
        .expect("B pin released with material");
    authority
        .remove_texture_asset(texture_c)
        .expect("C was never referenced");
}
