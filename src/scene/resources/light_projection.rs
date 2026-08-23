//! Compact realtime-light topology derived from canonical SceneDB entities.
//!
//! The sparse map makes `LightId -> compact slot` O(1), while the parallel
//! compact vector preserves the renderer's insertion/swap-remove ordering and
//! supplies `compact slot -> LightId` for canonical authored-row lookups.

use crate::handles::LightId;
use super::entity_projection::EntityRowProjection;

#[derive(Default)]
pub(in crate::scene) struct LightProjection {
    rows: EntityRowProjection<LightId>,
    atlas_dirty: bool,
}

impl LightProjection {
    pub fn insert(&mut self, id: LightId, gpu_row: u32) -> usize {
        let compact_slot = self.rows.insert(id, gpu_row);
        self.atlas_dirty = true;
        compact_slot
    }

    pub fn remove(&mut self, id: LightId) -> Option<usize> {
        let compact_slot = self.rows.remove(id)?;
        self.atlas_dirty = true;
        Some(compact_slot)
    }

    pub fn slot(&self, id: LightId) -> Option<usize> {
        self.rows.slot(id)
    }

    pub fn ids(&self) -> &[LightId] {
        self.rows.ids()
    }

    pub fn row(&self, compact_slot: usize) -> Option<u32> {
        self.rows.row(compact_slot)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn mark_atlas_dirty(&mut self) {
        self.atlas_dirty = true;
    }

    pub fn take_atlas_dirty(&mut self) -> bool {
        std::mem::take(&mut self.atlas_dirty)
    }
}

/// Select shadow winners by importance while retaining every continuing
/// winner's valid atlas base. Only entrants consume freed bases, so removing
/// an earlier compact light cannot invalidate all later cached shadow faces.
pub(in crate::scene) fn stable_shadow_assignments(
    light_count: usize,
    scores: &[(f32, usize)],
    previous: &[[u32; 2]],
    max_casters: usize,
) -> Vec<u32> {
    const FACES_PER_LIGHT: u32 = 6;

    let mut ranked = scores.to_vec();
    ranked.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let winners: Vec<usize> = ranked
        .into_iter()
        .take(max_casters)
        .map(|(_, compact_slot)| compact_slot)
        .collect();

    let mut assignments = vec![u32::MAX; light_count];
    let mut used_bases = vec![false; max_casters];
    let mut needs_base = Vec::new();
    for &compact_slot in &winners {
        let previous_base = previous
            .get(compact_slot)
            .map(|projection| projection[1])
            .unwrap_or(u32::MAX);
        let base_slot = (previous_base / FACES_PER_LIGHT) as usize;
        if previous_base != u32::MAX
            && previous_base % FACES_PER_LIGHT == 0
            && base_slot < max_casters
            && !used_bases[base_slot]
        {
            assignments[compact_slot] = previous_base;
            used_bases[base_slot] = true;
        } else {
            needs_base.push(compact_slot);
        }
    }

    // Compact order is the deterministic secondary policy for brand-new
    // winners; retained winners never move merely because this ordering does.
    needs_base.sort_unstable();
    let mut free_bases = used_bases
        .iter()
        .enumerate()
        .filter_map(|(base_slot, used)| (!*used).then_some(base_slot as u32 * FACES_PER_LIGHT));
    for compact_slot in needs_base {
        assignments[compact_slot] = free_bases
            .next()
            .expect("winner count never exceeds shadow caster capacity");
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_remove_repairs_both_directions_and_rejects_stale_generation() {
        let mut projection = LightProjection::default();
        let first = LightId::from_raw(9, 1);
        let middle = LightId::from_raw(40, 2);
        let last = LightId::from_raw(3, 7);

        assert_eq!(projection.insert(first, 12), 0);
        assert_eq!(projection.insert(middle, 2), 1);
        assert_eq!(projection.insert(last, 7), 2);
        assert_eq!(projection.remove(middle), Some(1));
        assert_eq!(projection.ids(), &[first, last]);
        assert_eq!(projection.row(1), Some(7));
        assert_eq!(projection.slot(last), Some(1));
        assert_eq!(projection.slot(LightId::from_raw(last.slot(), 6)), None);
    }

    #[test]
    fn continuing_winners_keep_atlas_bases_when_an_earlier_winner_exits() {
        let previous = [[10, 0], [11, 6], [12, 12], [13, u32::MAX]];
        let scores = [(30.0, 1), (20.0, 2), (10.0, 3)];
        assert_eq!(
            stable_shadow_assignments(4, &scores, &previous, 3),
            vec![u32::MAX, 6, 12, 0]
        );
    }
}
