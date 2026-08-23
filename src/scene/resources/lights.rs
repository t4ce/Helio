//! SceneDB-backed light resource management.
//!
//! Authored rows live only in `SceneAuthority`. Realtime ordering and assigned
//! shadow atlas slices are an 8-byte-per-light Helio projection.

use helio_core::GpuLight;
use helio_scenedb::{SceneIndices, SceneLight, SceneLightRow};

use crate::handles::{entity_from_handle, handle_from_entity, LightId};

use super::super::errors::{invalid, Result};

/// Whether an authored edit can change membership/order in the shadow budget.
/// Position, direction, flare, IES, and volumetric parameters still invalidate
/// their own consumers, but do not justify re-sorting every realtime light.
fn shadow_budget_changed(old: &SceneLightRow, new: &SceneLightRow) -> bool {
    if old.requests_shadow() != new.requests_shadow() {
        return true;
    }
    if !new.requests_shadow() {
        return false;
    }

    let old_directional = old.light_type == 0;
    let new_directional = new.light_type == 0;
    if old_directional != new_directional {
        return true;
    }
    if new_directional {
        // Every directional light scores infinity regardless of range/intensity.
        return false;
    }

    old.position_range[3].to_bits() != new.position_range[3].to_bits()
        || old.color_intensity[3].to_bits() != new.color_intensity[3].to_bits()
}

impl super::super::Scene {
    /// Insert a movable realtime light with no application tag.
    pub fn insert_light(&mut self, light: GpuLight) -> LightId {
        self.insert_light_with_movability(light, None, 0)
    }

    /// Insert a canonical authored light with explicit movability and user tag.
    pub fn insert_light_with_movability(
        &mut self,
        light: GpuLight,
        movability: Option<libhelio::Movability>,
        user_tag: u64,
    ) -> LightId {
        // Preserve Helio's longstanding light default: realtime unless callers
        // explicitly opt into baked Static/Stationary behavior.
        let movability = movability.unwrap_or(libhelio::Movability::Movable);
        let entity = self.authority.insert(SceneLight {
            user_tag,
            light: SceneLightRow::from(light),
            movability: movability as u32,
            _pad: 0,
        });
        let id = handle_from_entity(entity);
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_light(user_tag, entity);

        if movability.can_move() {
            let gpu_row = self
                .authority
                .gpu_row::<SceneLight>(entity)
                .expect("inserted GPU component must own a mirror row");
            let compact_slot = self.light_projection.insert(id, gpu_row);
            let projection_slot = self
                .gpu_scene
                .light_projections
                .push([gpu_row, u32::MAX]);
            debug_assert_eq!(projection_slot, compact_slot);
            self.gpu_scene.movable_light_count = self.light_projection.len() as u32;
            self.movable_lights_generation = self.movable_lights_generation.wrapping_add(1);
            self.gpu_scene.movable_lights_generation = self.movable_lights_generation;
        } else {
            self.bake_invalidated = true;
        }

        id
    }

    /// Replace authored light parameters through SceneDB's mirror-aware edit.
    pub fn update_light(&mut self, id: LightId, light: GpuLight) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(old_record) = self.authority.get::<SceneLight>(entity).copied() else {
            return Err(invalid("light"));
        };
        let old_light = GpuLight::from(old_record.light);
        let new_row = SceneLightRow::from(light);
        if bytemuck::bytes_of(&old_record.light) == bytemuck::bytes_of(&new_row) {
            return Ok(());
        }
        let shadow_budget_changed = shadow_budget_changed(&old_record.light, &new_row);
        let is_movable = old_record.movability == libhelio::Movability::Movable as u32;

        if !is_movable
            && (old_light.position_range != light.position_range
                || old_light.direction_outer != light.direction_outer)
        {
            log::warn!(
                "Attempted to update position/direction on Static light {:?}. Set movability to Movable to allow updates.",
                id
            );
            return Ok(());
        }

        self.authority
            .edit_gpu::<SceneLight, _>(entity, |record| {
                record.light = new_row;
            })
            .ok_or_else(|| invalid("light"))?;

        if is_movable {
            debug_assert!(self.light_projection.slot(id).is_some());
            if shadow_budget_changed {
                self.light_projection.mark_atlas_dirty();
            }
            self.movable_lights_generation = self.movable_lights_generation.wrapping_add(1);
            self.gpu_scene.movable_lights_generation = self.movable_lights_generation;
        } else {
            self.bake_invalidated = true;
        }
        Ok(())
    }

    /// Remove canonical persistence and its compact realtime projection.
    pub fn remove_light(&mut self, id: LightId) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(record) = self.authority.get::<SceneLight>(entity).copied() else {
            return Err(invalid("light"));
        };
        if !self.authority.despawn(entity) {
            return Err(invalid("light"));
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_light(record.user_tag, entity);

        if record.movability == libhelio::Movability::Movable as u32 {
            let compact_slot = self
                .light_projection
                .remove(id)
                .expect("live movable light must have a compact projection");
            let removed = self.gpu_scene.light_projections.swap_remove(compact_slot);
            debug_assert!(removed.is_some());
            self.gpu_scene.movable_light_count = self.light_projection.len() as u32;
            self.movable_lights_generation = self.movable_lights_generation.wrapping_add(1);
            self.gpu_scene.movable_lights_generation = self.movable_lights_generation;
        } else {
            self.bake_invalidated = true;
        }
        self.detach_actors_for_target(crate::scene::actor::SceneActorId::Light(id));
        Ok(())
    }

    /// Realtime authored lights in the exact order of the compact GPU projection.
    pub(crate) fn iter_realtime_lights(&self) -> impl Iterator<Item = GpuLight> + '_ {
        self.light_projection.ids().iter().filter_map(|&id| {
            self.authority
                .get::<SceneLight>(entity_from_handle(id))
                .map(|record| GpuLight::from(record.light))
        })
    }

    pub(crate) fn realtime_light_count(&self) -> usize {
        self.light_projection.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_shadow_budget_inputs_request_an_atlas_rerank() {
        let original = SceneLightRow::from(GpuLight {
            position_range: [1.0, 2.0, 3.0, 10.0],
            color_intensity: [1.0, 1.0, 1.0, 4.0],
            shadow_index: 0,
            light_type: 1,
            ..Default::default()
        });

        let mut cosmetic = original;
        cosmetic.flare_intensity = 2.0;
        cosmetic.position_range[0] = 9.0;
        assert!(!shadow_budget_changed(&original, &cosmetic));

        let mut score = original;
        score.position_range[3] = 20.0;
        assert!(shadow_budget_changed(&original, &score));

        let mut eligibility = original;
        eligibility.shadow_requested = u32::MAX;
        assert!(shadow_budget_changed(&original, &eligibility));

        let mut directional = original;
        directional.light_type = 0;
        assert!(shadow_budget_changed(&original, &directional));
    }
}
