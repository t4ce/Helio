//! SceneDB-backed decal CRUD and compact active-row projection.
//!
//! Decal shader rows stay in SceneDB's component-local canonical buffer. The
//! renderer uploads only one `u32` row index per active decal, avoiding the
//! legacy 128-byte row clone/rebuild while retaining dense shader iteration.

use helio_scenedb::{SceneDecal, SceneDecalRow, SceneIndices};

use crate::handles::{entity_from_handle, handle_from_entity, DecalId};
use super::resources::entity_projection::EntityRowProjection;

use super::Scene;

pub(in crate::scene) type DecalProjection = EntityRowProjection<DecalId>;

impl Scene {
    pub fn insert_decal(&mut self, decal: libhelio::GpuDecal) -> DecalId {
        self.insert_decal_with_tag(decal, 0, None)
    }

    pub fn insert_decal_with_tag(
        &mut self,
        decal: libhelio::GpuDecal,
        user_tag: u64,
        movability: Option<libhelio::Movability>,
    ) -> DecalId {
        let entity = self.authority.insert(SceneDecal {
            user_tag,
            decal: SceneDecalRow::from(decal),
            movability: movability.unwrap_or(libhelio::Movability::Movable) as u32,
            _pad: 0,
        });
        let id = handle_from_entity(entity);
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_decal(user_tag, entity);
        let gpu_row = self
            .authority
            .gpu_row::<SceneDecal>(entity)
            .expect("inserted GPU component must own a mirror row");
        let compact_slot = self.decal_projection.insert(id, gpu_row);
        let gpu_slot = self.gpu_scene.decal_indices.push(gpu_row);
        debug_assert_eq!(gpu_slot, compact_slot);
        id
    }

    pub fn remove_decal(&mut self, id: DecalId) -> bool {
        let entity = entity_from_handle(id);
        let Some(record) = self.authority.get::<SceneDecal>(entity).copied() else {
            return false;
        };
        if !self.authority.despawn(entity) {
            return false;
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_decal(record.user_tag, entity);

        let compact_slot = self
            .decal_projection
            .remove(id)
            .expect("live canonical decal must have an active projection");
        let removed = self.gpu_scene.decal_indices.swap_remove(compact_slot);
        debug_assert!(removed.is_some());
        self.detach_actors_for_target(crate::scene::actor::SceneActorId::Decal(id));
        true
    }

    pub fn update_decal(&mut self, id: DecalId, decal: libhelio::GpuDecal) -> bool {
        let entity = entity_from_handle(id);
        let new_row = SceneDecalRow::from(decal);
        let Some(old_row) = self
            .authority
            .get::<SceneDecal>(entity)
            .map(|record| record.decal)
        else {
            return false;
        };
        if bytemuck::bytes_of(&old_row) == bytemuck::bytes_of(&new_row) {
            return true;
        }
        self.authority
            .edit_gpu::<SceneDecal, _>(entity, |record| {
                record.decal = new_row;
            })
            .is_some()
    }

    pub fn decal_count(&self) -> usize {
        self.decal_projection.len()
    }

    pub fn iter_decals(
        &self,
    ) -> impl Iterator<Item = (DecalId, &libhelio::GpuDecal, u64)> + '_ {
        self.authority.query::<SceneDecal>().map(|(entity, record)| {
            (
                handle_from_entity(entity),
                // SceneDecalRow is transparent over GpuDecal.
                &record.decal.0,
                record.user_tag,
            )
        })
    }

    pub fn get_decal(&self, id: DecalId) -> Option<libhelio::GpuDecal> {
        self.authority
            .get::<SceneDecal>(entity_from_handle(id))
            .map(|record| record.decal.0)
    }

    pub fn decal_by_tag(&self, user_tag: u64) -> Option<DecalId> {
        self.authority
            .subsystem::<SceneIndices>()
            .and_then(|indices| indices.decal_by_tag(user_tag))
            .filter(|entity| self.authority.get::<SceneDecal>(*entity).is_some())
            .map(handle_from_entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_remove_repairs_sparse_lookup_and_stale_generation() {
        let mut projection = DecalProjection::default();
        let first = DecalId::from_raw(11, 3);
        let middle = DecalId::from_raw(2, 4);
        let last = DecalId::from_raw(80, 1);

        assert_eq!(projection.insert(first, 11), 0);
        assert_eq!(projection.insert(middle, 3), 1);
        assert_eq!(projection.insert(last, 8), 2);
        assert_eq!(projection.remove(middle), Some(1));
        assert_eq!(projection.rows(), &[11, 8]);
        assert_eq!(projection.slot(last), Some(1));
        assert_eq!(projection.slot(DecalId::from_raw(last.slot(), 0)), None);
    }
}
