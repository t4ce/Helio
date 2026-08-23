//! Stable Auto-voxel output slots and coalesced extraction work.
//!
//! Canonical voxel bytes remain in SceneDB. This projection owns only the
//! renderer's fixed mesh-output address space and the latest operation for each
//! output slot. Exact `Entity` generations prevent a removed volume's pending
//! clear/extract from being applied to a replacement that reused its slot.

use std::collections::HashMap;

use helio_scenedb::Entity;
use helio_voxel_core::{GpuVoxelMeshWork, VOXEL_MESH_WORK_ALLOCATED};

pub const VOXEL_MESH_OUTPUT_CAPACITY: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotRegion {
    base: u32,
    count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelMeshProjectionError {
    AlreadyResident,
    NotResident,
    InvalidBrick,
    CapacityExceeded,
}

pub struct VoxelMeshProjection {
    rows: Vec<GpuVoxelMeshWork>,
    owners: Vec<Option<Entity>>,
    occupied: Vec<bool>,
    dirty: Vec<bool>,
    regions: HashMap<Entity, SlotRegion>,
    free_regions: Vec<SlotRegion>,
    next_slot: u32,
}

impl Default for VoxelMeshProjection {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            owners: Vec::new(),
            occupied: Vec::new(),
            dirty: Vec::new(),
            regions: HashMap::new(),
            free_regions: Vec::new(),
            next_slot: 0,
        }
    }
}

impl VoxelMeshProjection {
    fn take_free_region(&mut self, count: u32) -> Option<SlotRegion> {
        let index = self
            .free_regions
            .iter()
            .position(|region| region.count >= count)?;
        let free = self.free_regions[index];
        let allocation = SlotRegion {
            base: free.base,
            count,
        };
        if free.count == count {
            self.free_regions.remove(index);
        } else {
            self.free_regions[index].base += count;
            self.free_regions[index].count -= count;
        }
        Some(allocation)
    }

    fn recycle_region(&mut self, region: SlotRegion) {
        self.free_regions.push(region);
        self.free_regions.sort_unstable_by_key(|entry| entry.base);
        let mut merged: Vec<SlotRegion> = Vec::with_capacity(self.free_regions.len());
        for entry in self.free_regions.drain(..) {
            if let Some(previous) = merged.last_mut() {
                if previous.base + previous.count == entry.base {
                    previous.count += entry.count;
                    continue;
                }
            }
            merged.push(entry);
        }
        self.free_regions = merged;
        while self
            .free_regions
            .last()
            .is_some_and(|tail| tail.base + tail.count == self.next_slot)
        {
            let tail = self.free_regions.pop().expect("output tail was just observed");
            self.next_slot = tail.base;
        }
    }

    pub fn allocate(
        &mut self,
        entity: Entity,
        brick_count: u32,
        volume_row: u32,
    ) -> Result<u32, VoxelMeshProjectionError> {
        if self.regions.contains_key(&entity) {
            return Err(VoxelMeshProjectionError::AlreadyResident);
        }
        if brick_count == 0 || brick_count > VOXEL_MESH_OUTPUT_CAPACITY {
            return Err(VoxelMeshProjectionError::CapacityExceeded);
        }
        let region = match self.take_free_region(brick_count) {
            Some(region) => region,
            None => {
                let end = self
                    .next_slot
                    .checked_add(brick_count)
                    .ok_or(VoxelMeshProjectionError::CapacityExceeded)?;
                if end > VOXEL_MESH_OUTPUT_CAPACITY {
                    return Err(VoxelMeshProjectionError::CapacityExceeded);
                }
                let region = SlotRegion {
                    base: self.next_slot,
                    count: brick_count,
                };
                self.next_slot = end;
                region
            }
        };

        let end = (region.base + region.count) as usize;
        if self.rows.len() < end {
            self.rows.resize(end, GpuVoxelMeshWork::default());
            self.owners.resize(end, None);
            self.occupied.resize(end, false);
            self.dirty.resize(end, false);
        }
        for local_brick in 0..region.count {
            let slot = (region.base + local_brick) as usize;
            self.owners[slot] = Some(entity);
            self.occupied[slot] = false;
            self.rows[slot] = GpuVoxelMeshWork {
                volume_row,
                local_brick,
                flags: VOXEL_MESH_WORK_ALLOCATED,
                generation: 0,
            };
            self.dirty[slot] = true;
        }
        self.regions.insert(entity, region);
        Ok(region.base)
    }

    pub fn release(&mut self, entity: Entity) -> Result<u32, VoxelMeshProjectionError> {
        let region = self
            .regions
            .remove(&entity)
            .ok_or(VoxelMeshProjectionError::NotResident)?;
        for local_brick in 0..region.count {
            let slot = (region.base + local_brick) as usize;
            debug_assert_eq!(self.owners[slot], Some(entity));
            self.owners[slot] = None;
            self.occupied[slot] = false;
            self.rows[slot] = GpuVoxelMeshWork::default();
            self.dirty[slot] = true;
        }
        self.recycle_region(region);
        Ok(region.base)
    }

    pub fn mark_uploaded(
        &mut self,
        entity: Entity,
        local_brick: u32,
        occupied: bool,
        brick_grid_dim: u32,
    ) -> Result<(), VoxelMeshProjectionError> {
        let region = *self
            .regions
            .get(&entity)
            .ok_or(VoxelMeshProjectionError::NotResident)?;
        if local_brick >= region.count || brick_grid_dim == 0 {
            return Err(VoxelMeshProjectionError::InvalidBrick);
        }
        let slot = (region.base + local_brick) as usize;
        if self.owners.get(slot).copied().flatten() != Some(entity) {
            return Err(VoxelMeshProjectionError::NotResident);
        }
        self.occupied[slot] = occupied;
        self.dirty[slot] = true;

        // A changed raw brick affects its own cells and any neighbouring
        // brick whose +halo/central-difference sample reaches into it.
        let plane = brick_grid_dim
            .checked_mul(brick_grid_dim)
            .ok_or(VoxelMeshProjectionError::InvalidBrick)?;
        let z = local_brick / plane;
        let y = (local_brick / brick_grid_dim) % brick_grid_dim;
        let x = local_brick % brick_grid_dim;
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= brick_grid_dim as i32
                        || ny >= brick_grid_dim as i32
                        || nz >= brick_grid_dim as i32
                    {
                        continue;
                    }
                    let neighbour = nz as u32 * plane
                        + ny as u32 * brick_grid_dim
                        + nx as u32;
                    if neighbour < region.count {
                        self.dirty[(region.base + neighbour) as usize] = true;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn take_batch(&mut self, generation: u64) -> Option<Vec<GpuVoxelMeshWork>> {
        if !self.dirty.iter().any(|dirty| *dirty) {
            return None;
        }
        let generation = generation as u32;
        for (row, dirty) in self.rows.iter_mut().zip(&mut self.dirty) {
            if *dirty {
                row.generation = generation;
                *dirty = false;
            }
        }
        Some(self.rows.clone())
    }

    pub fn draw_count(&self) -> u32 {
        self.occupied
            .iter()
            .rposition(|occupied| *occupied)
            .map_or(0, |slot| slot as u32 + 1)
    }

    #[cfg(test)]
    pub fn row(&self, slot: u32) -> Option<GpuVoxelMeshWork> {
        self.rows.get(slot as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removal_and_same_batch_reuse_overwrite_stale_clears() {
        let old = Entity::from_bits((1u64 << 32) | 7);
        let replacement = Entity::from_bits((2u64 << 32) | 7);
        let mut projection = VoxelMeshProjection::default();
        assert_eq!(projection.allocate(old, 2, 19).unwrap(), 0);
        projection.mark_uploaded(old, 0, true, 1).unwrap();
        projection.release(old).unwrap();
        assert_eq!(projection.allocate(replacement, 2, 31).unwrap(), 0);
        projection.mark_uploaded(replacement, 0, true, 1).unwrap();

        let rows = projection.take_batch(9).unwrap();
        assert_eq!(rows[0].volume_row, 31);
        assert_eq!(rows[0].flags, VOXEL_MESH_WORK_ALLOCATED);
        assert_eq!(rows[0].generation, 9);
    }

    #[test]
    fn output_ceiling_is_checked_and_free_ranges_coalesce() {
        let first = Entity::from_bits((1u64 << 32) | 1);
        let second = Entity::from_bits((1u64 << 32) | 2);
        let third = Entity::from_bits((1u64 << 32) | 3);
        let mut projection = VoxelMeshProjection::default();
        assert_eq!(projection.allocate(first, 512, 4).unwrap(), 0);
        assert_eq!(projection.allocate(second, 512, 5).unwrap(), 512);
        assert_eq!(
            projection.allocate(third, 1, 6),
            Err(VoxelMeshProjectionError::CapacityExceeded)
        );
        projection.release(first).unwrap();
        projection.release(second).unwrap();
        assert_eq!(projection.allocate(third, 1024, 6).unwrap(), 0);
    }

    #[test]
    fn upload_marks_the_full_halo_neighbourhood_but_preserves_occupancy() {
        let entity = Entity::from_bits((4u64 << 32) | 2);
        let mut projection = VoxelMeshProjection::default();
        projection.allocate(entity, 27, 8).unwrap();
        projection.take_batch(1).unwrap();
        projection.mark_uploaded(entity, 13, true, 3).unwrap();
        let rows = projection.take_batch(2).unwrap();
        assert!(rows.iter().all(|row| row.generation == 2));
        assert_eq!(projection.draw_count(), 14);
    }
}
