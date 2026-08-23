//! SceneDB-backed water-volume and hitbox CRUD.
//!
//! Authored shader rows live only in SceneDB. Helio retains a compact
//! `[component row, simulation slot]` projection because water shaders iterate
//! a dense active domain while SceneDB rows are stable and may contain holes.

use helio_scenedb::{
    SceneIndices, SceneWaterHitbox, SceneWaterHitboxRow, SceneWaterVolume,
    SceneWaterVolumeRow,
};
use libhelio::{GpuWaterHitbox, GpuWaterVolume};
use helio_core::{WATER_SIM_SLOT_COUNT, WATER_SIM_SLOT_UNASSIGNED};

use crate::handles::{
    entity_from_handle, handle_from_entity, WaterHitboxId, WaterVolumeId,
};
use crate::scene::actor::{WaterHitboxDescriptor, WaterVolumeDescriptor};
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::Scene;

use super::resources::entity_projection::EntityRowProjection;

fn validate_water_simulation(desc: &WaterVolumeDescriptor) -> Result<()> {
    let finite = desc.wave_spring.is_finite()
        && desc.wave_damping.is_finite()
        && desc.wave_speed.is_finite()
        && desc.wave_scale.is_finite()
        && desc.wind_strength.is_finite()
        && desc.wind_direction.into_iter().all(f32::is_finite);
    if !finite
        || !(0.1..=2.0).contains(&desc.wave_spring)
        || !(0.0..=1.0).contains(&desc.wave_damping)
        || desc.wave_speed < 0.0
        || desc.wave_scale < 0.01
        || desc.wind_strength < 0.0
    {
        return Err(SceneError::InvalidOperation {
            reason: "water simulation dynamics must be finite and within shader bounds",
        });
    }
    Ok(())
}

pub(in crate::scene) struct WaterVolumeProjection {
    active: EntityRowProjection<WaterVolumeId>,
    /// Stable simulation residency parallel to `active`. Volumes beyond the
    /// fixed pass capacity carry WATER_SIM_SLOT_UNASSIGNED.
    sim_slots: Vec<u32>,
    /// Reversed so `pop()` assigns slots 0, 1, ... deterministically.
    free_sim_slots: Vec<u32>,
}

impl Default for WaterVolumeProjection {
    fn default() -> Self {
        Self {
            active: EntityRowProjection::default(),
            sim_slots: Vec::new(),
            free_sim_slots: (0..WATER_SIM_SLOT_COUNT as u32).rev().collect(),
        }
    }
}

struct WaterVolumeRemoval {
    compact: usize,
    /// An existing unsimulated volume promoted into the released residency.
    promoted: Option<(usize, [u32; 2])>,
}

impl WaterVolumeProjection {
    fn insert(&mut self, id: WaterVolumeId, row: u32) -> (usize, [u32; 2]) {
        let compact = self.active.insert(id, row);
        let sim_slot = self
            .free_sim_slots
            .pop()
            .unwrap_or(WATER_SIM_SLOT_UNASSIGNED);
        debug_assert_eq!(compact, self.sim_slots.len());
        self.sim_slots.push(sim_slot);
        (compact, [row, sim_slot])
    }

    fn remove(&mut self, id: WaterVolumeId) -> Option<WaterVolumeRemoval> {
        let compact = self.active.remove(id)?;
        let released = self.sim_slots.swap_remove(compact);
        let mut promoted = None;
        if released != WATER_SIM_SLOT_UNASSIGNED {
            if let Some(promoted_compact) = self
                .sim_slots
                .iter()
                .position(|slot| *slot == WATER_SIM_SLOT_UNASSIGNED)
            {
                self.sim_slots[promoted_compact] = released;
                promoted = Some((
                    promoted_compact,
                    [
                        self.active
                            .row(promoted_compact)
                            .expect("parallel water projection row is missing"),
                        released,
                    ],
                ));
            } else {
                self.free_sim_slots.push(released);
            }
        }
        Some(WaterVolumeRemoval { compact, promoted })
    }

    fn ids(&self) -> &[WaterVolumeId] {
        self.active.ids()
    }

    fn sim_slot(&self, id: WaterVolumeId) -> Option<u32> {
        let compact = self.active.slot(id)?;
        self.sim_slots
            .get(compact)
            .copied()
            .filter(|slot| *slot != WATER_SIM_SLOT_UNASSIGNED)
    }

    fn len(&self) -> usize {
        self.active.len()
    }
}

impl Scene {
    pub fn insert_water_volume(&mut self, desc: WaterVolumeDescriptor) -> Result<WaterVolumeId> {
        self.insert_water_volume_with_tag(desc, 0)
    }

    pub fn insert_water_volume_with_tag(
        &mut self,
        desc: WaterVolumeDescriptor,
        user_tag: u64,
    ) -> Result<WaterVolumeId> {
        validate_water_simulation(&desc)?;
        let entity = self.authority.insert(SceneWaterVolume {
            user_tag,
            volume: SceneWaterVolumeRow::from(desc.to_gpu()),
            _reserved: 0,
        });
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_water_volume(user_tag, entity);
        let row = self
            .authority
            .gpu_row::<SceneWaterVolume>(entity)
            .expect("inserted GPU component must own a mirror row");
        let id = handle_from_entity(entity);
        let (compact, projection) = self.water_volume_projection.insert(id, row);
        let gpu_slot = self.gpu_scene.water_volume_projections.push(projection);
        debug_assert_eq!(compact, gpu_slot);
        if projection[1] != WATER_SIM_SLOT_UNASSIGNED {
            self.gpu_scene.reset_water_sim_slot(projection[1]);
        }
        Ok(id)
    }

    pub fn remove_water_volume(&mut self, id: WaterVolumeId) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(record) = self.authority.get::<SceneWaterVolume>(entity).copied() else {
            return Err(invalid("water volume"));
        };
        if !self.authority.despawn(entity) {
            return Err(invalid("water volume"));
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_water_volume(record.user_tag, entity);
        let removal = self
            .water_volume_projection
            .remove(id)
            .expect("live water volume must have an active projection");
        debug_assert!(self
            .gpu_scene
            .water_volume_projections
            .swap_remove(removal.compact)
            .is_some());
        if let Some((compact, projection)) = removal.promoted {
            debug_assert!(self
                .gpu_scene
                .water_volume_projections
                .update(compact, projection));
            self.gpu_scene.reset_water_sim_slot(projection[1]);
        }
        self.detach_actors_for_target(crate::scene::actor::SceneActorId::WaterVolume(id));
        Ok(())
    }

    pub fn update_water_volume(
        &mut self,
        id: WaterVolumeId,
        desc: WaterVolumeDescriptor,
    ) -> Result<()> {
        validate_water_simulation(&desc)?;
        let entity = entity_from_handle(id);
        let new_row = SceneWaterVolumeRow::from(desc.to_gpu());
        let Some(old_row) = self
            .authority
            .get::<SceneWaterVolume>(entity)
            .map(|record| record.volume)
        else {
            return Err(invalid("water volume"));
        };
        if bytemuck::bytes_of(&old_row) == bytemuck::bytes_of(&new_row) {
            return Ok(());
        }
        self.authority
            .edit_gpu::<SceneWaterVolume, _>(entity, |record| record.volume = new_row)
            .ok_or_else(|| invalid("water volume"))?;
        Ok(())
    }

    pub fn get_water_volume(&self, id: WaterVolumeId) -> Option<GpuWaterVolume> {
        self.authority
            .get::<SceneWaterVolume>(entity_from_handle(id))
            .map(|record| record.volume.0)
    }

    pub fn iter_water_volumes(
        &self,
    ) -> impl Iterator<Item = (WaterVolumeId, &GpuWaterVolume, u64)> + '_ {
        self.authority.query::<SceneWaterVolume>().map(|(entity, record)| {
            (handle_from_entity(entity), &record.volume.0, record.user_tag)
        })
    }

    /// Compatibility snapshot in compact projection order. Rendering binds the
    /// canonical buffer plus `water_volume_projections` directly and does not call
    /// this allocation-producing facade.
    pub fn get_water_volumes_gpu(&self) -> Vec<GpuWaterVolume> {
        self.water_volume_projection
            .ids()
            .iter()
            .filter_map(|&id| self.get_water_volume(id))
            .collect()
    }

    pub fn water_volumes_count(&self) -> u32 {
        self.water_volume_projection.len() as u32
    }

    /// Stable heightfield residency for this volume. The first eight active
    /// volumes receive a slot; additional authored volumes remain queryable and
    /// GPU-resident but return `None` until a slot becomes available.
    pub fn water_volume_sim_slot(&self, id: WaterVolumeId) -> Option<u32> {
        self.water_volume_projection.sim_slot(id)
    }

    /// Resolve a live authored volume to its pass-owned heightfield residency.
    ///
    /// The generation makes queued transient effects safe across removal and
    /// slot reuse. Volumes beyond the fixed simulation capacity remain valid
    /// authored records but return an explicit error until promoted.
    pub fn water_volume_sim_target(
        &self,
        id: WaterVolumeId,
    ) -> Result<helio_core::WaterSimulationTarget> {
        let entity = entity_from_handle(id);
        if self.authority.get::<SceneWaterVolume>(entity).is_none() {
            return Err(invalid("water volume"));
        }
        let sim_slot = self.water_volume_projection.sim_slot(id).ok_or(
            SceneError::InvalidOperation {
                reason: "water volume has no simulation residency",
            },
        )?;
        let canonical_row = self
            .authority
            .gpu_row::<SceneWaterVolume>(entity)
            .expect("live water volume must own a canonical GPU row");
        let residency_generation = self.gpu_scene.water_sim_slot_generations[sim_slot as usize];
        Ok(helio_core::WaterSimulationTarget::from_parts(
            sim_slot,
            canonical_row,
            residency_generation,
        ))
    }

    /// Validate a world-space impulse centre against one authored volume and
    /// resolve its stable simulation residency.
    ///
    /// The pass maps the point with cascade 0's periodic world tile, matching
    /// surface sampling. Canonical bounds are still the ownership gate: a point
    /// outside this volume is rejected rather than aliasing through the tile.
    pub fn water_drop_target(
        &self,
        id: WaterVolumeId,
        world_center: [f32; 2],
    ) -> Result<helio_core::WaterDropTarget> {
        let entity = entity_from_handle(id);
        let volume = self
            .authority
            .get::<SceneWaterVolume>(entity)
            .ok_or_else(|| invalid("water volume"))?
            .volume
            .0;
        let [x, z] = world_center;
        let min = volume.bounds_min;
        let max = volume.bounds_max;
        if !x.is_finite()
            || !z.is_finite()
            || !min[0].is_finite()
            || !min[2].is_finite()
            || !max[0].is_finite()
            || !max[2].is_finite()
            || max[0] <= min[0]
            || max[2] <= min[2]
            || x < min[0]
            || x > max[0]
            || z < min[2]
            || z > max[2]
        {
            return Err(SceneError::InvalidOperation {
                reason: "water drop centre must be finite and inside non-degenerate volume bounds",
            });
        }
        Ok(helio_core::WaterDropTarget::from_parts(
            self.water_volume_sim_target(id)?,
            world_center,
        ))
    }

    pub fn water_volume_by_tag(&self, user_tag: u64) -> Option<WaterVolumeId> {
        self.authority
            .subsystem::<SceneIndices>()
            .and_then(|indices| indices.water_volume_by_tag(user_tag))
            .filter(|entity| self.authority.get::<SceneWaterVolume>(*entity).is_some())
            .map(handle_from_entity)
    }

    pub fn insert_water_hitbox(&mut self, desc: WaterHitboxDescriptor) -> Result<WaterHitboxId> {
        self.insert_water_hitbox_with_tag(desc, 0)
    }

    pub fn insert_water_hitbox_with_tag(
        &mut self,
        desc: WaterHitboxDescriptor,
        user_tag: u64,
    ) -> Result<WaterHitboxId> {
        let entity = self.authority.insert(SceneWaterHitbox {
            user_tag,
            hitbox: SceneWaterHitboxRow::from(desc.to_gpu()),
            _reserved: 0,
        });
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_water_hitbox(user_tag, entity);
        let row = self
            .authority
            .gpu_row::<SceneWaterHitbox>(entity)
            .expect("inserted GPU component must own a mirror row");
        let id = handle_from_entity(entity);
        let compact = self.water_hitbox_projection.insert(id, row);
        let gpu_slot = self.gpu_scene.water_hitbox_indices.push(row);
        debug_assert_eq!(compact, gpu_slot);
        Ok(id)
    }

    pub fn remove_water_hitbox(&mut self, id: WaterHitboxId) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(record) = self.authority.get::<SceneWaterHitbox>(entity).copied() else {
            return Err(invalid("water hitbox"));
        };
        if !self.authority.despawn(entity) {
            return Err(invalid("water hitbox"));
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_water_hitbox(record.user_tag, entity);
        let compact = self
            .water_hitbox_projection
            .remove(id)
            .expect("live water hitbox must have an active projection");
        debug_assert!(self.gpu_scene.water_hitbox_indices.swap_remove(compact).is_some());
        self.detach_actors_for_target(crate::scene::actor::SceneActorId::WaterHitbox(id));
        Ok(())
    }

    pub fn update_water_hitbox(
        &mut self,
        id: WaterHitboxId,
        desc: WaterHitboxDescriptor,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let new_row = SceneWaterHitboxRow::from(desc.to_gpu());
        let Some(old_row) = self
            .authority
            .get::<SceneWaterHitbox>(entity)
            .map(|record| record.hitbox)
        else {
            return Err(invalid("water hitbox"));
        };
        if bytemuck::bytes_of(&old_row) == bytemuck::bytes_of(&new_row) {
            return Ok(());
        }
        self.authority
            .edit_gpu::<SceneWaterHitbox, _>(entity, |record| record.hitbox = new_row)
            .ok_or_else(|| invalid("water hitbox"))?;
        Ok(())
    }

    pub fn get_water_hitbox(&self, id: WaterHitboxId) -> Option<GpuWaterHitbox> {
        self.authority
            .get::<SceneWaterHitbox>(entity_from_handle(id))
            .map(|record| record.hitbox.0)
    }

    pub fn iter_water_hitboxes(
        &self,
    ) -> impl Iterator<Item = (WaterHitboxId, &GpuWaterHitbox, u64)> + '_ {
        self.authority.query::<SceneWaterHitbox>().map(|(entity, record)| {
            (handle_from_entity(entity), &record.hitbox.0, record.user_tag)
        })
    }

    pub fn get_water_hitboxes_gpu(&self) -> Vec<GpuWaterHitbox> {
        self.water_hitbox_projection
            .ids()
            .iter()
            .filter_map(|&id| self.get_water_hitbox(id))
            .collect()
    }

    pub fn water_hitboxes_count(&self) -> u32 {
        self.water_hitbox_projection.len() as u32
    }

    pub fn water_hitbox_by_tag(&self, user_tag: u64) -> Option<WaterHitboxId> {
        self.authority
            .subsystem::<SceneIndices>()
            .and_then(|indices| indices.water_hitbox_by_tag(user_tag))
            .filter(|entity| self.authority.get::<SceneWaterHitbox>(*entity).is_some())
            .map(handle_from_entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ninth_volume_is_unsimulated_and_promoted_without_moving_survivor_slots() {
        let mut projection = WaterVolumeProjection::default();
        let ids: Vec<_> = (0..WATER_SIM_SLOT_COUNT + 1)
            .map(|index| WaterVolumeId::from_raw(index as u32 + 1, 1))
            .collect();

        for (index, id) in ids.iter().copied().enumerate() {
            let (_, row) = projection.insert(id, index as u32 + 20);
            assert_eq!(
                row[1],
                if index < WATER_SIM_SLOT_COUNT {
                    index as u32
                } else {
                    WATER_SIM_SLOT_UNASSIGNED
                }
            );
        }
        assert_eq!(projection.sim_slot(ids[WATER_SIM_SLOT_COUNT]), None);

        let removal = projection.remove(ids[3]).expect("projected volume");
        let (promoted_compact, promoted) = removal.promoted.expect("ninth volume promoted");
        assert_eq!(promoted[1], 3);
        assert_eq!(projection.sim_slot(ids[WATER_SIM_SLOT_COUNT]), Some(3));
        assert_eq!(projection.sim_slots[promoted_compact], 3);
        for (id, expected_slot) in ids[..WATER_SIM_SLOT_COUNT]
            .iter()
            .copied()
            .zip(0u32..)
            .filter(|(id, _)| *id != ids[3])
        {
            let compact = projection.active.slot(id).expect("survivor remains projected");
            assert_eq!(projection.sim_slots[compact], expected_slot);
        }
    }
}
