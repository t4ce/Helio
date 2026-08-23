//! Texture resource management backed by SceneDB's registered residency.

use helio_scenedb::{SceneTexture, SceneTextureAssetKey, SceneTextureSampler, TextureResidency};

use crate::handles::{entity_from_handle, handle_from_entity, TextureId};
use crate::material::TextureUpload;

use super::super::errors::{scene_asset, Result};

impl super::super::Scene {
    /// Insert a texture into SceneDB. The texture object is created exactly
    /// once by `TextureResidency`/SceneDB `TextureStore`; Helio retains only
    /// its renderer placeholder and borrowed view/sampler access.
    pub fn insert_texture(&mut self, texture: TextureUpload) -> Result<TextureId> {
        let asset_key = self
            .authority
            .subsystem_mut::<TextureResidency>()
            .expect("TextureResidency is registered during SceneAuthority construction")
            .allocate_asset_key()
            .map_err(scene_asset)?;
        self.insert_texture_with_asset_key(asset_key.0, texture)
    }

    /// Insert a texture with an authored identity stable across scene reloads.
    /// The key is independent of the public Entity handle and physical binding
    /// slot; duplicate live keys are rejected by SceneDB.
    pub fn insert_texture_with_asset_key(
        &mut self,
        asset_key: u128,
        texture: TextureUpload,
    ) -> Result<TextureId> {
        let mut metadata = SceneTexture::sampled_2d(
            SceneTextureAssetKey(asset_key),
            texture.width,
            texture.height,
            texture.format,
            SceneTextureSampler {
                address_mode_u: texture.sampler.address_mode_u,
                address_mode_v: texture.sampler.address_mode_v,
                address_mode_w: texture.sampler.address_mode_w,
                mag_filter: texture.sampler.mag_filter,
                min_filter: texture.sampler.min_filter,
                mipmap_filter: texture.sampler.mipmap_filter,
            },
        );
        metadata.mip_level_count = 1 + texture.mip_data.len() as u32;
        let upload_bytes =
            texture.data.len() + texture.mip_data.iter().map(Vec::len).sum::<usize>();
        helio_core::upload::record_upload_bytes(upload_bytes as u64);
        let entity = self
            .authority
            .insert_texture_asset_with_mips(
                metadata,
                texture.label.as_deref(),
                &texture.data,
                &texture.mip_data,
            )
            .map_err(scene_asset)?;
        Ok(handle_from_entity(entity))
    }

    /// Remove an unreferenced texture. SceneDB checks both the canonical
    /// component count and residency pins before releasing its physical slot.
    pub fn remove_texture(&mut self, id: TextureId) -> Result<()> {
        self.authority
            .remove_texture_asset(entity_from_handle(id))
            .map_err(scene_asset)?;
        Ok(())
    }

    /// Resolve a live texture by its stable authored asset identity without a
    /// Helio-side lookup table.
    pub fn texture_by_asset_key(&self, asset_key: u128) -> Option<TextureId> {
        self.authority
            .subsystem::<TextureResidency>()
            .expect("TextureResidency is registered during SceneAuthority construction")
            .entity_for_asset(SceneTextureAssetKey(asset_key))
            .map(handle_from_entity)
    }

    /// Return the stable authored identity behind a live texture handle.
    pub fn texture_asset_key(&self, id: TextureId) -> Option<u128> {
        self.authority
            .get::<SceneTexture>(entity_from_handle(id))
            .map(|texture| texture.asset_key.0)
    }

    /// Return the exact non-compacting SceneDB residency slot for a live
    /// texture handle. This is deliberately distinct from `TextureId::slot()`:
    /// the latter is the generational SceneDB entity identity, not the index
    /// used by renderer texture-view and sampler arrays.
    pub fn texture_residency_slot(&self, id: TextureId) -> Option<u32> {
        self.authority
            .subsystem::<TextureResidency>()
            .and_then(|residency| residency.slot_for(entity_from_handle(id)))
    }

    /// Version of the exact SceneDB slot-ordered texture/view/sampler table.
    pub fn texture_binding_version(&self) -> u64 {
        self.authority
            .subsystem::<TextureResidency>()
            .expect("TextureResidency is registered during SceneAuthority construction")
            .binding_epoch()
    }

    /// Material binding representation shared by every pass for this scene.
    pub fn material_binding_config(&self) -> libhelio::MaterialBindingConfig {
        self.material_binding
    }

    /// Return the view at the exact non-compacting SceneDB residency slot, or
    /// Helio's placeholder for a vacant/out-of-range row.
    pub fn texture_view_for_slot(&self, slot: usize) -> &wgpu::TextureView {
        self.authority
            .subsystem::<TextureResidency>()
            .and_then(|residency| residency.view_for_slot(slot as u32))
            .unwrap_or(&self.placeholder_view)
    }

    /// Return the sampler paired with the exact SceneDB residency slot, or
    /// Helio's placeholder for a vacant/out-of-range row.
    pub fn texture_sampler_for_slot(&self, slot: usize) -> &wgpu::Sampler {
        self.authority
            .subsystem::<TextureResidency>()
            .and_then(|residency| residency.sampler_for_slot(slot as u32))
            .unwrap_or(&self.placeholder_sampler)
    }
}

#[cfg(test)]
mod tests {
    use helio_scenedb::SceneTexture;

    use crate::handles::entity_from_handle;
    use crate::{Scene, SceneError, TextureSamplerDesc, TextureUpload};

    fn upload(label: &str) -> TextureUpload {
        TextureUpload::rgba8(
            label,
            1,
            1,
            false,
            vec![255, 255, 255, 255],
            TextureSamplerDesc::default(),
        )
    }

    #[test]
    fn scene_automatic_texture_keys_follow_scenedb_high_water_across_clear() {
        let (device, queue) = crate::test_support::test_gpu().expect("No test GPU adapter found");
        let mut scene = Scene::new(device, queue);

        let explicit = scene
            .insert_texture_with_asset_key(1, upload("explicit"))
            .expect("explicit texture");
        assert_eq!(scene.texture_by_asset_key(1), Some(explicit));
        assert_eq!(scene.texture_asset_key(explicit), Some(1));
        let explicit_residency_slot = scene
            .texture_residency_slot(explicit)
            .expect("inserted texture must be resident");
        assert!(matches!(
            scene.insert_texture_with_asset_key(1, upload("duplicate")),
            Err(SceneError::DuplicateTextureAssetKey { asset_key: 1 })
        ));
        let automatic = scene.insert_texture(upload("automatic")).unwrap();
        assert_eq!(scene.texture_asset_key(automatic), Some(2));
        let automatic_residency_slot = scene
            .texture_residency_slot(automatic)
            .expect("inserted texture must be resident");
        assert_ne!(explicit_residency_slot, automatic_residency_slot);
        assert_eq!(
            scene
                .authority
                .get::<SceneTexture>(entity_from_handle(automatic))
                .unwrap()
                .asset_key
                .0,
            2
        );
        assert_ne!(explicit, automatic);

        scene.clear();
        assert_eq!(scene.texture_by_asset_key(1), None);
        assert_eq!(scene.texture_asset_key(explicit), None);
        assert_eq!(scene.texture_residency_slot(explicit), None);

        let after_clear = scene.insert_texture(upload("after-clear")).unwrap();
        assert_eq!(
            scene
                .authority
                .get::<SceneTexture>(entity_from_handle(after_clear))
                .unwrap()
                .asset_key
                .0,
            3,
            "clear must preserve SceneDB's non-recycling asset-key history"
        );

        scene
            .insert_texture_with_asset_key(u128::MAX, upload("last-explicit-key"))
            .expect("u128::MAX remains a valid explicit identity");
        let after_explicit_max = scene
            .insert_texture(upload("automatic-after-explicit-max"))
            .expect("explicit maximum must not exhaust automatic identities");
        assert_eq!(scene.texture_asset_key(after_explicit_max), Some(4));
    }

    #[test]
    fn clear_preserves_scenedb_radiant_assets_and_render_projection() {
        let (device, queue) = crate::test_support::test_gpu().expect("No test GPU adapter found");
        let mut scene = Scene::new(device, queue);
        const HASH: u64 = 0xA55E_7001;
        const FIRST_WGSL: &str = "fn radiant_test_asset() -> f32 { return 1.0; }";
        const REPLACEMENT_WGSL: &str = "fn radiant_test_asset() -> f32 { return 2.0; }";

        scene
            .radiant_graphs_mut()
            .register(HASH, FIRST_WGSL.to_owned())
            .unwrap();
        scene.flush();
        let first_epoch = scene.radiant_graphs().epoch();
        assert_eq!(
            scene
                .gpu_scene
                .graph_wgsl_snippets
                .get(&HASH)
                .map(String::as_str),
            Some(FIRST_WGSL)
        );
        scene
            .radiant_graphs_mut()
            .register(HASH, REPLACEMENT_WGSL.to_owned())
            .unwrap();
        scene.flush();
        let replacement_epoch = scene.radiant_graphs().epoch();
        assert_eq!(replacement_epoch, first_epoch.wrapping_add(1));
        assert_eq!(scene.gpu_scene.graph_wgsl_epoch, replacement_epoch);
        assert_eq!(
            scene
                .gpu_scene
                .graph_wgsl_snippets
                .get(&HASH)
                .map(String::as_str),
            Some(REPLACEMENT_WGSL)
        );

        scene.clear();

        assert_eq!(scene.radiant_graphs().get(HASH), Some(REPLACEMENT_WGSL));
        assert_eq!(scene.radiant_graphs().epoch(), replacement_epoch);
        assert_eq!(
            scene
                .gpu_scene
                .graph_wgsl_snippets
                .get(&HASH)
                .map(String::as_str),
            Some(REPLACEMENT_WGSL),
            "clear keeps the renderer projection coherent with retained graph assets"
        );
    }
}
