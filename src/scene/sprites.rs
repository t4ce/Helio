//! High-level CRUD for the main SceneDB sprite authority.

use helio_scenedb::{
    sprite_content_hash, SceneSprite, SceneSpriteAtlasId, SceneSpriteAtlasLayer, SceneSpriteId,
    SceneSpriteRow, SpriteAtlasError, SpriteAtlasResidency, SpriteAuthorityError,
};

use super::Scene;

impl Scene {
    fn allocate_sprite_authored_epoch(&mut self) -> Result<u32, SpriteAuthorityError> {
        let epoch = self.next_sprite_authored_epoch;
        if epoch == 0 {
            return Err(SpriteAuthorityError::AuthoredEpochExhausted);
        }
        self.next_sprite_authored_epoch = self
            .next_sprite_authored_epoch
            .checked_add(1)
            .unwrap_or(0);
        Ok(epoch)
    }

    fn resolve_sprite_atlas(
        &mut self,
        row: &mut SceneSpriteRow,
        retain: bool,
    ) -> Result<(), SpriteAuthorityError> {
        let Some(atlas) = row.atlas() else {
            row.clear_atlas();
            return Ok(());
        };
        if self
            .authority
            .get::<SceneSpriteAtlasLayer>(atlas.entity())
            .is_none()
        {
            return Err(SpriteAuthorityError::StaleAtlas(atlas));
        }
        let residency = self
            .authority
            .subsystem_mut::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered at Scene construction");
        let physical_layer = if retain {
            residency.retain(atlas.entity())?
        } else {
            residency
                .resolve(atlas.entity())
                .ok_or(SpriteAuthorityError::StaleAtlas(atlas))?
        };
        row.resolve_atlas_for_gpu(atlas, physical_layer);
        Ok(())
    }

    pub fn try_insert_sprite(
        &mut self,
        mut row: SceneSpriteRow,
    ) -> Result<SceneSpriteId, SpriteAuthorityError> {
        let row_span = self.authority.gpu_row_span::<SceneSprite>();
        let live_count = self.authority.gpu_live_count::<SceneSprite>();
        let required = if live_count < row_span {
            row_span.max(1)
        } else {
            row_span
                .checked_add(1)
                .ok_or(SpriteAuthorityError::CapacityRequestTooLarge(usize::MAX))?
        };
        self.authority
            .reserve_gpu_component_capacity::<SceneSprite>(required)?;
        let epoch = self.allocate_sprite_authored_epoch()?;
        self.resolve_sprite_atlas(&mut row, true)?;
        row.set_authored_epoch(epoch);
        let entity = self.authority.insert(SceneSprite { sprite: row });
        Ok(SceneSpriteId(entity))
    }

    pub fn insert_sprite(&mut self, row: SceneSpriteRow) -> SceneSpriteId {
        self.try_insert_sprite(row)
            .expect("Scene::insert_sprite failed")
    }

    pub fn try_update_sprite(
        &mut self,
        id: SceneSpriteId,
        mut row: SceneSpriteRow,
    ) -> Result<(), SpriteAuthorityError> {
        let old = self
            .authority
            .get::<SceneSprite>(id.entity())
            .copied()
            .ok_or(SpriteAuthorityError::StaleSprite(id))?;
        let old_atlas = old.sprite.atlas();
        let new_atlas = row.atlas();
        let epoch = self.allocate_sprite_authored_epoch()?;
        if old_atlas != new_atlas {
            if let Some(old_atlas) = old_atlas {
                self.authority
                    .subsystem::<SpriteAtlasResidency>()
                    .expect("sprite atlas residency is registered")
                    .validate_release_reference(old_atlas.entity())?;
            }
        }
        self.resolve_sprite_atlas(&mut row, old_atlas != new_atlas)?;
        row.set_authored_epoch(epoch);
        if !self
            .authority
            .replace_gpu(id.entity(), SceneSprite { sprite: row })
        {
            if old_atlas != new_atlas {
                if let Some(new_atlas) = new_atlas {
                    let _ = self
                        .authority
                        .subsystem_mut::<SpriteAtlasResidency>()
                        .expect("sprite atlas residency is registered")
                        .release_reference(new_atlas.entity());
                }
            }
            return Err(SpriteAuthorityError::StaleSprite(id));
        }
        if old_atlas != new_atlas {
            if let Some(old_atlas) = old_atlas {
                self.authority
                    .subsystem_mut::<SpriteAtlasResidency>()
                    .expect("sprite atlas residency is registered")
                    .release_reference(old_atlas.entity())?;
            }
        }
        Ok(())
    }

    pub fn update_sprite(&mut self, id: SceneSpriteId, row: SceneSpriteRow) {
        self.try_update_sprite(id, row)
            .expect("Scene::update_sprite failed");
    }

    pub fn try_remove_sprite(
        &mut self,
        id: SceneSpriteId,
    ) -> Result<SceneSpriteRow, SpriteAuthorityError> {
        let sprite = self
            .authority
            .get::<SceneSprite>(id.entity())
            .copied()
            .ok_or(SpriteAuthorityError::StaleSprite(id))?;
        if let Some(atlas) = sprite.sprite.atlas() {
            self.authority
                .subsystem::<SpriteAtlasResidency>()
                .expect("sprite atlas residency is registered")
                .validate_release_reference(atlas.entity())?;
            self.authority
                .subsystem_mut::<SpriteAtlasResidency>()
                .expect("sprite atlas residency is registered")
                .release_reference(atlas.entity())?;
        }
        if !self.authority.despawn(id.entity()) {
            if let Some(atlas) = sprite.sprite.atlas() {
                self.authority
                    .subsystem_mut::<SpriteAtlasResidency>()
                    .expect("sprite atlas residency is registered")
                    .retain(atlas.entity())
                    .expect("sprite removal rollback restores the released reference");
            }
            return Err(SpriteAuthorityError::StaleSprite(id));
        }
        Ok(sprite.sprite)
    }

    pub fn remove_sprite(&mut self, id: SceneSpriteId) {
        self.try_remove_sprite(id)
            .expect("Scene::remove_sprite failed");
    }

    pub fn sprite(&self, id: SceneSpriteId) -> Option<SceneSpriteRow> {
        self.authority
            .get::<SceneSprite>(id.entity())
            .map(|sprite| sprite.sprite)
    }

    pub fn try_clear_sprites(&mut self) -> Result<(), SpriteAuthorityError> {
        let ids: Vec<_> = self
            .authority
            .query::<SceneSprite>()
            .map(|(entity, _)| SceneSpriteId(entity))
            .collect();
        for id in ids {
            self.try_remove_sprite(id)?;
        }
        Ok(())
    }

    pub fn clear_sprites(&mut self) {
        self.try_clear_sprites().expect("Scene::clear_sprites failed");
    }

    pub fn try_reserve_sprites(
        &mut self,
        capacity: usize,
    ) -> Result<(), SpriteAuthorityError> {
        let capacity = u32::try_from(capacity)
            .map_err(|_| SpriteAuthorityError::CapacityRequestTooLarge(capacity))?;
        self.authority.reserve_entity_capacity(capacity);
        self.authority
            .reserve_gpu_component_capacity::<SceneSprite>(capacity)?;
        Ok(())
    }

    pub fn reserve_sprites(&mut self, capacity: usize) {
        self.try_reserve_sprites(capacity)
            .expect("Scene::reserve_sprites failed");
    }

    pub fn try_add_sprite_atlas_layer(
        &mut self,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<SceneSpriteAtlasId, SpriteAuthorityError> {
        self.authority
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .validate(width, height, rgba8)?;
        let entity = self.authority.insert(SceneSpriteAtlasLayer {
            width,
            height,
            content_hash: sprite_content_hash(rgba8),
        });
        if let Err(error) = self
            .authority
            .subsystem_mut::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .insert(entity, width, height, rgba8)
        {
            let _ = self.authority.despawn(entity);
            return Err(error.into());
        }
        Ok(SceneSpriteAtlasId(entity))
    }

    pub fn add_sprite_atlas_layer(
        &mut self,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> SceneSpriteAtlasId {
        self.try_add_sprite_atlas_layer(width, height, rgba8)
            .expect("Scene::add_sprite_atlas_layer failed")
    }

    pub fn try_remove_sprite_atlas_layer(
        &mut self,
        id: SceneSpriteAtlasId,
    ) -> Result<(), SpriteAuthorityError> {
        if self
            .authority
            .get::<SceneSpriteAtlasLayer>(id.entity())
            .is_none()
        {
            return Err(SpriteAuthorityError::StaleAtlas(id));
        }
        self.authority
            .subsystem_mut::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .remove(id.entity())?;
        if !self.authority.despawn(id.entity()) {
            return Err(SpriteAuthorityError::StaleAtlas(id));
        }
        Ok(())
    }

    pub fn remove_sprite_atlas_layer(&mut self, id: SceneSpriteAtlasId) {
        self.try_remove_sprite_atlas_layer(id)
            .expect("Scene::remove_sprite_atlas_layer failed");
    }

    pub fn try_clear_sprite_atlas_layers(&mut self) -> Result<(), SpriteAuthorityError> {
        let ids: Vec<_> = self
            .authority
            .query::<SceneSpriteAtlasLayer>()
            .map(|(entity, _)| SceneSpriteAtlasId(entity))
            .collect();
        let residency = self
            .authority
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered");
        for id in &ids {
            let references = residency
                .references(id.entity())
                .ok_or(SpriteAuthorityError::StaleAtlas(*id))?;
            if references != 0 {
                return Err(SpriteAtlasError::LayerInUse { references }.into());
            }
        }
        for id in ids {
            self.try_remove_sprite_atlas_layer(id)?;
        }
        Ok(())
    }

    pub fn clear_sprite_atlas_layers(&mut self) {
        self.try_clear_sprite_atlas_layers()
            .expect("Scene::clear_sprite_atlas_layers failed");
    }

    pub fn sprite_count(&self) -> usize {
        self.authority.gpu_live_count::<SceneSprite>() as usize
    }

    pub fn sprite_atlas_layer_count(&self) -> usize {
        self.authority
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .live_count() as usize
    }

    pub fn sprite_atlas_limits(&self) -> (u32, u32) {
        let residency = self
            .authority
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered");
        (
            residency.maximum_imported_layers(),
            residency.maximum_dimension(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_scenedb::{MATERIAL_BUFFER_KEY, SPRITE_BUFFER_KEY};
    use std::sync::Arc;

    fn gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        crate::test_support::test_gpu()
    }

    #[test]
    fn sprite_reserve_does_not_grow_unrelated_scene_partners() {
        let Some((device, queue)) = gpu() else {
            eprintln!("skipping sprite reserve-domain test: no GPU adapter");
            return;
        };
        let mut scene = Scene::new(device, queue);
        let material_before = scene
            .authority
            .partner_buffer_snapshot(MATERIAL_BUFFER_KEY)
            .expect("material partner is registered")
            .epoch;
        let sprite_before = scene
            .authority
            .partner_buffer_snapshot(SPRITE_BUFFER_KEY)
            .expect("sprite partner is registered")
            .epoch;

        scene.try_reserve_sprites(2_048).unwrap();

        let material_after = scene
            .authority
            .partner_buffer_snapshot(MATERIAL_BUFFER_KEY)
            .expect("material partner remains registered")
            .epoch;
        let sprite_after = scene
            .authority
            .partner_buffer_snapshot(SPRITE_BUFFER_KEY)
            .expect("sprite partner remains registered")
            .epoch;
        assert_eq!(material_after, material_before);
        assert!(sprite_after > sprite_before);
    }

    #[test]
    fn scene_clear_releases_sprites_before_atlas_residency() {
        let Some((device, queue)) = gpu() else {
            eprintln!("skipping sprite clear-lifecycle test: no GPU adapter");
            return;
        };
        let mut scene = Scene::new(device, queue);
        let referenced_atlas = scene.add_sprite_atlas_layer(1, 1, &[255, 255, 255, 255]);
        let unreferenced_atlas = scene.add_sprite_atlas_layer(1, 1, &[0, 0, 0, 255]);
        let sprite = scene.insert_sprite(
            SceneSpriteRow::new([4.0, 8.0], [16.0, 16.0])
                .with_atlas_layer(referenced_atlas),
        );
        assert_eq!(scene.sprite_count(), 1);
        assert_eq!(scene.sprite_atlas_layer_count(), 2);

        scene.clear();

        assert_eq!(scene.sprite_count(), 0);
        assert_eq!(scene.sprite_atlas_layer_count(), 0);
        assert!(scene.sprite(sprite).is_none());
        assert!(matches!(
            scene.try_remove_sprite(sprite),
            Err(SpriteAuthorityError::StaleSprite(id)) if id == sprite
        ));
        assert!(matches!(
            scene.try_remove_sprite_atlas_layer(referenced_atlas),
            Err(SpriteAuthorityError::StaleAtlas(id)) if id == referenced_atlas
        ));
        assert!(matches!(
            scene.try_remove_sprite_atlas_layer(unreferenced_atlas),
            Err(SpriteAuthorityError::StaleAtlas(id)) if id == unreferenced_atlas
        ));
    }
}
