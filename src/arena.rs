//! Arena-backed collections used by scene resources and handle-based storage.
//!
//! This module provides two data structures:
//!
//! - [`DenseArena`] for compact dense storage of live items with O(1) insert/remove
//!   and handle-based lookup through a generation-protected slot handle.
//! - [`SparsePool`] for sparse slot-based storage with stable handles and free-list reuse.

use std::marker::PhantomData;
use crate::handles::Handle;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseSlotMeta {
    generation: u32,
    dense_index: u32,
    occupied: bool,
}

/// Result returned by [`DenseArena::remove`] when an item is deleted.
///
/// Contains the removed value and auxilliary information needed by callers
/// to update moved items after a swap-remove.
pub struct DenseRemove<T> {
    /// The removed value.
    pub removed: T,

}

/// Compact dense storage for items keyed by handle.
///
/// `DenseArena` keeps a contiguous `dense` vector of live items and a parallel
/// slot table that maps stable handles to the current dense index.
/// Removals are implemented with swap-remove to maintain O(1) behavior.
pub struct DenseArena<T, H> {
    pub slots: Vec<DenseSlotMeta>,
    pub dense: Vec<T>,
    pub dense_to_slot: Vec<u32>,
    pub free_list: Vec<u32>,
    pub marker: PhantomData<H>,
}

impl<T, H: Handle> DenseArena<T, H> {
    /// Create an empty dense arena.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            dense: Vec::new(),
            dense_to_slot: Vec::new(),
            free_list: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Number of live items currently stored in the arena.
    pub fn dense_len(&self) -> usize {
        self.dense.len()
    }

    /// Immutable access by dense-array index.
    ///
    /// This is useful for bulk operations that iterate over all live items in
    /// dense storage order.
    pub fn get_dense(&self, index: usize) -> Option<&T> {
        self.dense.get(index)
    }

    /// Mutable access by dense-array index.
    ///
    /// Used after rebuilds or when the caller needs to patch an item in place.
    pub fn get_dense_mut(&mut self, index: usize) -> Option<&mut T> {
        self.dense.get_mut(index)
    }

    /// Insert a new value and return its handle and dense index.
    ///
    /// Reuses a freed slot when available; otherwise it appends a new slot.
    pub fn insert(&mut self, value: T) -> (H, usize) {
        let dense_index = self.dense.len();
        let slot_index = if let Some(slot) = self.free_list.pop() {
            let meta = &mut self.slots[slot as usize];
            meta.occupied = true;
            meta.dense_index = dense_index as u32;
            slot
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(DenseSlotMeta {
                generation: 1,
                dense_index: dense_index as u32,
                occupied: true,
            });
            slot
        };

        self.dense.push(value);
        self.dense_to_slot.push(slot_index);
        let generation = self.slots[slot_index as usize].generation;
        (H::from_parts(slot_index, generation), dense_index)
    }

    /// Mutable lookup by handle, returning the current dense index and reference.
    pub fn get_mut_with_index(&mut self, handle: H) -> Option<(usize, &mut T)> {
        let meta = *self.slots.get(handle.slot() as usize)?;
        if !meta.occupied || meta.generation != handle.generation() {
            return None;
        }
        let dense_index = meta.dense_index as usize;
        self.dense
            .get_mut(dense_index)
            .map(|value| (dense_index, value))
    }

    /// Immutable lookup by handle, returning the current dense index and reference.
    /// Remove an item by handle and return its removal metadata.
    ///
    /// If the removed item is not the last element in dense storage, the last
    /// element is moved into the freed slot and its dense index is updated.
    pub fn remove(&mut self, handle: H) -> Option<DenseRemove<T>> {
        let slot_index = handle.slot() as usize;
        let meta = self.slots.get(slot_index).copied()?;
        if !meta.occupied || meta.generation != handle.generation() {
            return None;
        }

        let dense_index = meta.dense_index as usize;
        let removed = self.dense.swap_remove(dense_index);
        self.dense_to_slot.swap_remove(dense_index);

        if dense_index < self.dense.len() {
            let moved_slot = self.dense_to_slot[dense_index] as usize;
            self.slots[moved_slot].dense_index = dense_index as u32;
        }

        let slot = &mut self.slots[slot_index];
        slot.occupied = false;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_list.push(slot_index as u32);

        Some(DenseRemove {
            removed,
        })
    }

    /// Iterate all live items, yielding `(handle, &value)` in dense-array order.
    ///
    /// Reconstructs each handle from the stored slot index and current generation.
    /// This is O(N) over live items.
    pub fn iter_with_handles(&self) -> impl Iterator<Item = (H, &T)> + '_ {
        self.dense.iter().enumerate().map(|(dense_idx, value)| {
            let slot_idx = self.dense_to_slot[dense_idx];
            let gen = self.slots[slot_idx as usize].generation;
            (H::from_parts(slot_idx, gen), value)
        })
    }

    /// Immutable lookup by handle.
    pub fn get(&self, handle: H) -> Option<&T> {
        let meta = *self.slots.get(handle.slot() as usize)?;
        if !meta.occupied || meta.generation != handle.generation() {
            return None;
        }
        self.dense.get(meta.dense_index as usize)
    }

    /// Iterate all live items yielding `(handle, &value)` with a simpler API.
    pub fn iter(&self) -> impl Iterator<Item = (H, &T)> + '_ {
        self.iter_with_handles()
    }

}

#[derive(Debug)]
struct SparseSlot<T> {
    generation: u32,
    value: Option<T>,
}

/// Simple sparse slot pool with handle-based lookup.
///
/// Each slot contains an optional value and a generation counter. Handles are
/// validated against the current generation to prevent use-after-free.
pub struct SparsePool<T, H> {
    slots: Vec<SparseSlot<T>>,
    free_list: Vec<u32>,
    live_count: usize,
    marker: PhantomData<H>,
}

impl<T, H: Handle> SparsePool<T, H> {
    /// Create an empty sparse pool.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            live_count: 0,
            marker: PhantomData,
        }
    }

    /// Insert a value and return its new handle, slot index, and whether it was a fresh slot.
    pub fn insert(&mut self, value: T) -> (H, usize, bool) {
        if let Some(slot) = self.free_list.pop() {
            let entry = &mut self.slots[slot as usize];
            entry.value = Some(value);
            self.live_count += 1;
            return (H::from_parts(slot, entry.generation), slot as usize, false);
        }

        let slot = self.slots.len() as u32;
        self.slots.push(SparseSlot {
            generation: 1,
            value: Some(value),
        });
        self.live_count += 1;
        (H::from_parts(slot, 1), slot as usize, true)
    }

    /// Immutable lookup by handle.
    pub fn get(&self, handle: H) -> Option<&T> {
        let slot = self.slots.get(handle.slot() as usize)?;
        if slot.generation != handle.generation() {
            return None;
        }
        slot.value.as_ref()
    }

    /// Mutable lookup by handle, returning the slot index and reference.
    pub fn get_mut_with_slot(&mut self, handle: H) -> Option<(usize, &mut T)> {
        let slot_index = handle.slot() as usize;
        let slot = self.slots.get_mut(slot_index)?;
        if slot.generation != handle.generation() {
            return None;
        }
        slot.value.as_mut().map(|value| (slot_index, value))
    }

    /// Remove a value by handle and free its slot for reuse.
    pub fn remove(&mut self, handle: H) -> Option<(usize, T)> {
        let slot_index = handle.slot() as usize;
        let slot = self.slots.get_mut(slot_index)?;
        if slot.generation != handle.generation() {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_list.push(slot_index as u32);
        self.live_count = self.live_count.saturating_sub(1);
        Some((slot_index, value))
    }

    /// Number of currently live values stored in the pool.
    pub fn live_len(&self) -> usize {
        self.live_count
    }

    /// Iterate live values with their current generation-bearing handles.
    ///
    /// Handles are rebuilt from the slot's stored generation rather than from
    /// a dense position, so callers can safely collect IDs before a later
    /// mutation pass without reviving stale slots.
    pub fn iter(&self) -> impl Iterator<Item = (H, &T)> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value
                .as_ref()
                .map(|value| (H::from_parts(index as u32, slot.generation), value))
        })
    }

}
