//! Generic compact projection from stable public handles to SceneDB GPU rows.
//!
//! The sparse side is indexed by the public handle's entity slot solely for
//! O(1) CPU lookup. Shader addressing always uses the component-local row
//! supplied by `SceneAuthority::gpu_row`, never `Entity.index`/handle slot.

use crate::handles::Handle;

pub(in crate::scene) struct EntityRowProjection<H> {
    slot_by_entity: Vec<u32>,
    entity_by_slot: Vec<H>,
    row_by_slot: Vec<u32>,
}

impl<H> Default for EntityRowProjection<H> {
    fn default() -> Self {
        Self {
            slot_by_entity: Vec::new(),
            entity_by_slot: Vec::new(),
            row_by_slot: Vec::new(),
        }
    }
}

impl<H: Handle + Eq> EntityRowProjection<H> {
    pub fn insert(&mut self, id: H, gpu_row: u32) -> usize {
        let sparse_slot = id.slot() as usize;
        if self.slot_by_entity.len() <= sparse_slot {
            self.slot_by_entity.resize(sparse_slot + 1, u32::MAX);
        }
        debug_assert!(self.slot(id).is_none(), "entity already projected");

        let compact_slot = self.entity_by_slot.len();
        let compact_slot_u32 =
            u32::try_from(compact_slot).expect("active projection exceeds u32 rows");
        self.entity_by_slot.push(id);
        self.row_by_slot.push(gpu_row);
        self.slot_by_entity[sparse_slot] = compact_slot_u32;
        compact_slot
    }

    pub fn remove(&mut self, id: H) -> Option<usize> {
        let compact_slot = self.slot(id)?;
        self.entity_by_slot.swap_remove(compact_slot);
        self.row_by_slot.swap_remove(compact_slot);
        self.slot_by_entity[id.slot() as usize] = u32::MAX;

        if let Some(&moved) = self.entity_by_slot.get(compact_slot) {
            self.slot_by_entity[moved.slot() as usize] =
                u32::try_from(compact_slot).expect("active projection exceeds u32 rows");
        }
        Some(compact_slot)
    }

    pub fn slot(&self, id: H) -> Option<usize> {
        let compact_slot = *self.slot_by_entity.get(id.slot() as usize)?;
        if compact_slot == u32::MAX {
            return None;
        }
        let compact_slot = compact_slot as usize;
        (self.entity_by_slot.get(compact_slot) == Some(&id)).then_some(compact_slot)
    }

    pub fn ids(&self) -> &[H] {
        &self.entity_by_slot
    }

    pub fn rows(&self) -> &[u32] {
        &self.row_by_slot
    }

    pub fn row(&self, compact_slot: usize) -> Option<u32> {
        self.row_by_slot.get(compact_slot).copied()
    }

    pub fn len(&self) -> usize {
        self.entity_by_slot.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entity_by_slot.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handles::WaterVolumeId;

    #[test]
    fn swap_remove_repairs_handle_lookup_without_conflating_gpu_rows() {
        let mut projection = EntityRowProjection::default();
        let first = WaterVolumeId::from_raw(9, 1);
        let middle = WaterVolumeId::from_raw(40, 2);
        let last = WaterVolumeId::from_raw(3, 7);

        assert_eq!(projection.insert(first, 2), 0);
        assert_eq!(projection.insert(middle, 99), 1);
        assert_eq!(projection.insert(last, 4), 2);
        assert_eq!(projection.remove(middle), Some(1));
        assert_eq!(projection.ids(), &[first, last]);
        assert_eq!(projection.rows(), &[2, 4]);
        assert_eq!(projection.slot(last), Some(1));
        assert_eq!(projection.slot(WaterVolumeId::from_raw(last.slot(), 6)), None);
    }
}
