//! Portals: a pair of poses whose `pair_map_inverse` is one more coordinate
//! space, used to draw a clipped duplicate of nearby geometry.
//!
//! Unlike a sublevel, a portal tags no member objects. What's visible through
//! a portal is decided fresh every frame by `helio-pass-portal-cull`: it
//! frustum-tests every draw-call group's bounds *mapped through the portal's
//! `pair_map_inverse`* against the main camera, exactly like the ordinary
//! frustum cull already does for world space — content that isn't actually
//! near the portal's other side maps nowhere near the camera and is rejected
//! by that same test. Survivors are drawn a second time by
//! `helio-pass-portal-instances`, through the *same* main camera (no separate
//! "eye" — see `docs/` for why that's mathematically equivalent to one), and
//! clipped in the fragment shader to the portal's opening.
//!
//! The CPU-side pose algebra (`pair_map`, crossing detection, teleport) is
//! `helio_portal_core`, re-exported here — it is unaffected by any of the
//! above and unchanged from the design's first version.

use glam::Vec2;
use helio_scenedb::{CpuOnlyComponent, SceneCoordinateSpace, SceneCoordinateSpaceRow};
use helio_portal_core::{PortalPair, PortalPose};

use crate::handles::{entity_from_handle, handle_from_entity, PortalId};
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::Scene;

/// Configuration for [`Scene::add_portal`].
#[derive(Debug, Clone, Copy)]
pub struct PortalDescriptor {
    /// The "near" surface — the one the viewer looks through.
    pub a: PortalPose,
    /// The "far" surface — what's actually near `b` is what gets duplicated
    /// and drawn as if seen through `a`.
    pub b: PortalPose,
    /// Half-extent of the portal opening in `a`'s local X/Y — the fragment
    /// clip test's bounds. Content mapped outside this (or behind the
    /// surface) is discarded rather than drawn.
    pub half_extent: Vec2,
}

/// Internal record for a portal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortalRecord {
    pub pair: PortalPair,
    pub half_extent: Vec2,
}

impl CpuOnlyComponent for PortalRecord {}

impl Scene {
    /// Create a portal: allocate a coordinate-space slot and write
    /// `pair_map_inverse()` (B's frame → A's frame) into it.
    ///
    /// # Errors
    /// [`SceneError::CoordinateSpaceCapacityExceeded`] if all coordinate-space
    /// slots are already claimed by other sublevels/portals.
    pub fn add_portal(&mut self, desc: PortalDescriptor) -> Result<PortalId> {
        if self.authority.gpu_live_count::<SceneCoordinateSpace>()
            >= libhelio::MAX_COORDINATE_SPACES
        {
            return Err(SceneError::CoordinateSpaceCapacityExceeded);
        }
        let pair = PortalPair { a: desc.a, b: desc.b };
        let candidate = PortalRecord {
            pair,
            half_extent: desc.half_extent,
        };
        self.validate_portal_chain_capacity(None, candidate)?;
        let inverse = pair.pair_map_inverse().to_cols_array();
        let entity = self.authority.insert(candidate);
        let attached = self.authority.replace_gpu(
            entity,
            SceneCoordinateSpace {
                transform: SceneCoordinateSpaceRow(inverse),
            },
        );
        debug_assert!(attached, "fresh SceneDB portal entity must be alive");
        let slot = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .expect("attached coordinate-space component must have a GPU row");
        assert!(
            slot < libhelio::MAX_COORDINATE_SPACES,
            "controlled coordinate-space population exceeded the shader ABI"
        );
        self.gpu_scene
            .coordinate_space_history
            .stage_new(slot, inverse);
        self.republish_portal_views();
        // Adding a portal changes which chains exist (a new leaf appears at
        // every depth). Pose and extent edits also rebuild because pruning is
        // based on authored placement and opening size.
        self.republish_portal_chains();
        Ok(handle_from_entity(entity))
    }

    /// Update a portal's pose pair. **O(1)** — one coordinate-space matrix
    /// write, no scene object is touched.
    pub fn update_portal_pose(&mut self, portal: PortalId, a: PortalPose, b: PortalPose) -> Result<()> {
        let entity = entity_from_handle(portal);
        let space = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .ok_or_else(|| invalid("portal"))?;
        let pair = PortalPair { a, b };
        let mut candidate = self
            .authority
            .get::<PortalRecord>(entity)
            .copied()
            .ok_or_else(|| invalid("portal"))?;
        candidate.pair = pair;
        self.validate_portal_chain_capacity(Some(entity), candidate)?;
        self.authority
            .edit_cpu::<PortalRecord, _>(entity, |record| record.pair = pair)
            .ok_or_else(|| invalid("portal"))?;
        let inverse = pair.pair_map_inverse().to_cols_array();
        self.authority
            .edit_gpu::<SceneCoordinateSpace, _>(entity, |coordinate| {
                coordinate.transform = SceneCoordinateSpaceRow(inverse);
            })
            .ok_or_else(|| invalid("portal"))?;
        self.gpu_scene
            .coordinate_space_history
            .stage_current(space, inverse);
        self.republish_portal_views();
        // Reachability pruning compares portal poses, so a pose edit can add
        // or remove deeper chains even when the portal count is unchanged.
        self.republish_portal_chains();
        Ok(())
    }

    /// Update a portal's clip half-extent (the opening's size in `a`'s local X/Y).
    pub fn update_portal_half_extent(&mut self, portal: PortalId, half_extent: Vec2) -> Result<()> {
        let entity = entity_from_handle(portal);
        if self.authority.gpu_row::<SceneCoordinateSpace>(entity).is_none() {
            return Err(invalid("portal"));
        }
        let mut candidate = self
            .authority
            .get::<PortalRecord>(entity)
            .copied()
            .ok_or_else(|| invalid("portal"))?;
        candidate.half_extent = half_extent;
        self.validate_portal_chain_capacity(Some(entity), candidate)?;
        self.authority
            .edit_cpu::<PortalRecord, _>(entity, |record| record.half_extent = half_extent)
            .ok_or_else(|| invalid("portal"))?;
        self.republish_portal_views();
        // Half extents are part of the reachability predicate, so widening or
        // narrowing an opening can add or remove recursive chains.
        self.republish_portal_chains();
        Ok(())
    }

    /// Current pose pair for a portal — e.g. to run
    /// [`helio_portal_core::crossing_detected`] against either surface, or
    /// `pair.teleport_ray(..)` to carry a camera/player through.
    pub fn portal_pair(&self, portal: PortalId) -> Option<PortalPair> {
        let entity = entity_from_handle(portal);
        self.authority
            .get::<SceneCoordinateSpace>(entity)?;
        self.authority.get::<PortalRecord>(entity).map(|record| record.pair)
    }

    /// Current clip half-extent for a portal.
    pub fn portal_half_extent(&self, portal: PortalId) -> Option<Vec2> {
        let entity = entity_from_handle(portal);
        self.authority
            .get::<SceneCoordinateSpace>(entity)?;
        self.authority
            .get::<PortalRecord>(entity)
            .map(|record| record.half_extent)
    }

    /// Remove a portal and free its GPU coordinate-space slot for reuse.
    pub fn remove_portal(&mut self, portal: PortalId) -> Result<()> {
        let entity = entity_from_handle(portal);
        if self.authority.get::<PortalRecord>(entity).is_none() {
            return Err(invalid("portal"));
        }
        let space = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .ok_or_else(|| invalid("portal"))?;
        self.gpu_scene
            .coordinate_space_history
            .stage_new(space, glam::Mat4::IDENTITY.to_cols_array());
        let removed = self.authority.despawn(entity);
        debug_assert!(removed, "validated portal entity must despawn");
        self.republish_portal_views();
        self.republish_portal_chains();
        Ok(())
    }

    /// Rewrites the GPU-facing portal view list from scratch.
    ///
    /// Portal counts are always small (a handful at most), so republishing
    /// the whole list on every add/remove/pose-update is simpler than
    /// dirty-tracking it and costs nothing measurable — this is not a
    /// per-frame cost, only a per-edit one.
    fn republish_portal_views(&mut self) {
        let views: Vec<libhelio::GpuPortalView> = self
            .active_portal_records()
            .into_iter()
            .map(|(coordinate_space, record)| libhelio::GpuPortalView {
                transform: record.pair.a.transform.to_cols_array(),
                inverse_transform: record.pair.a.transform.inverse().to_cols_array(),
                half_extent: record.half_extent.to_array(),
                coordinate_space,
                _pad: 0,
            })
            .collect();
        self.gpu_scene.portal_views.set_data(views);
    }

    /// Rebuilds every valid portal chain from scratch — this is what makes
    /// portals reflect each other recursively with zero manual authoring.
    /// See `libhelio::GpuPortalChain`'s docs for the shape and
    /// `helio-pass-portal-cull`/`helio-pass-portal-instances` for how the
    /// list gets consumed. Called on add/remove and pose edits because the
    /// reachability pruning below is explicitly pose-dependent.
    ///
    /// Generates every sequence of active-portal indices, length
    /// `1..=MAX_CHAIN_DEPTH`, **allowing repeats** — `[P, P, P]` (the same
    /// portal three times over) is exactly what makes a single mirror-style
    /// portal, or a room whose portal faces its own reflection, read as
    /// infinite. A plain depth-first walk of the "append any portal" tree,
    /// stopping once `MAX_PORTAL_CHAINS` is hit (see that constant's docs —
    /// scenes are expected to stay well under it).
    fn republish_portal_chains(&mut self) {
        let records: Vec<PortalRecord> = self
            .active_portal_records()
            .into_iter()
            .map(|(_, record)| record)
            .collect();

        let chains = build_portal_chains(&records)
            .expect("portal mutations are capacity-checked before publication");
        self.gpu_scene.portal_chains.set_data(chains);
    }

    /// Validate a proposed addition or replacement before mutating SceneDB.
    /// This keeps the fixed render budget explicit and transactional: callers
    /// receive an error instead of observing an arbitrarily truncated chain
    /// set after an otherwise successful edit.
    fn validate_portal_chain_capacity(
        &self,
        replacement_entity: Option<helio_scenedb::Entity>,
        candidate: PortalRecord,
    ) -> Result<()> {
        let mut replaced = false;
        let mut records: Vec<_> = self
            .authority
            .query::<PortalRecord>()
            .map(|(entity, record)| {
                if replacement_entity == Some(entity) {
                    replaced = true;
                    candidate
                } else {
                    *record
                }
            })
            .collect();
        if replacement_entity.is_none() {
            records.push(candidate);
        } else if !replaced {
            return Err(invalid("portal"));
        }
        build_portal_chains(&records)
            .map(|_| ())
            .ok_or(SceneError::PortalChainCapacityExceeded)
    }

    /// Snapshot portals in stable coordinate-row order so independently
    /// rebuilt view and chain projections always agree on compact indices.
    fn active_portal_records(&self) -> Vec<(u32, PortalRecord)> {
        let mut records: Vec<_> = self
            .authority
            .query::<PortalRecord>()
            .filter_map(|(entity, record)| {
                self.authority
                    .gpu_row::<SceneCoordinateSpace>(entity)
                    .map(|row| (row, *record))
            })
            .collect();
        records.sort_unstable_by_key(|(row, _)| *row);
        records
    }
}

/// Depth-first: append `chains` with every non-empty prefix reachable by
/// picking `0..records.len()` at each of up to `MAX_CHAIN_DEPTH` steps,
/// pruned to steps where the next portal is actually reachable through the
/// current innermost one — see `portal_reachable_through`.
fn build_portal_chains(records: &[PortalRecord]) -> Option<Vec<libhelio::GpuPortalChain>> {
    let mut chains = Vec::new();
    if records.is_empty() {
        return Some(chains);
    }
    let mut prefix = Vec::with_capacity(libhelio::MAX_CHAIN_DEPTH);
    generate_chains(records, &mut prefix, &mut chains).then_some(chains)
}

fn generate_chains(
    records: &[PortalRecord],
    prefix: &mut Vec<u32>,
    chains: &mut Vec<libhelio::GpuPortalChain>,
) -> bool {
    if !prefix.is_empty() {
        if chains.len() == libhelio::MAX_PORTAL_CHAINS {
            return false;
        }
        let mut portals = [0u32; libhelio::MAX_CHAIN_DEPTH];
        portals[..prefix.len()].copy_from_slice(prefix);
        chains.push(libhelio::GpuPortalChain { portals, depth: prefix.len() as u32 });
    }
    if prefix.len() >= libhelio::MAX_CHAIN_DEPTH {
        return true;
    }
    for p in 0..records.len() as u32 {
        // Only extend the chain with `p` as the new innermost portal when
        // its own opening is plausibly reachable through whichever portal
        // is currently innermost (`prefix`'s last entry) — i.e. `p.a` sits
        // inside that portal's `b` window. Without this, chain generation
        // combinatorially pairs up completely unrelated portals (any two
        // portals leading to two unrelated, non-recursive rooms, say), and
        // the per-instance cull/clip tests can't fully catch the result:
        // the outermost stage is deliberately loose on X/Y (see
        // `gbuffer_portal.wgsl`'s module doc — wide content behind a
        // narrow opening needs to still show), so a nonsense multi-hop
        // transform can still slip through the outer portal's mask as
        // faint, wrongly-positioned "ghost" content. A scene where a
        // portal genuinely IS reachable through the previous one —
        // `portal_cube`'s opposite-wall doors, or a portal paired with
        // itself — passes this check fine and keeps recursing exactly as
        // before; a scene of unrelated single-hop portals (`portal_rooms`)
        // now simply never generates the bogus deeper chains at all.
        if let Some(&prev_idx) = prefix.last() {
            if !portal_reachable_through(&records[prev_idx as usize], &records[p as usize]) {
                continue;
            }
        }
        prefix.push(p);
        if !generate_chains(records, prefix, chains) {
            prefix.pop();
            return false;
        }
        prefix.pop();
    }
    true
}

/// True when `next`'s real opening (`next.a`) plausibly sits within `prev`'s
/// far surface (`prev.b`)'s window — i.e. composing `prev` after `next` in a
/// chain corresponds to an actual physical adjacency, not an arbitrary
/// pairing of unrelated portals. Position-only (orientation isn't checked):
/// two surfaces occupying the same physical opening but facing opposite
/// ways — exactly `prev.b` and the opposite-facing portal whose `a` sits at
/// that same spot, as in `portal_cube` — is precisely the legitimate case
/// this needs to keep allowing.
fn portal_reachable_through(prev: &PortalRecord, next: &PortalRecord) -> bool {
    let local = prev.pair.b.transform.inverse().transform_point3(next.pair.a.position());
    let tolerance = prev.half_extent.x.max(prev.half_extent.y);
    local.x.abs() <= prev.half_extent.x && local.y.abs() <= prev.half_extent.y && local.z.abs() <= tolerance
}

/// Placement helper: build a rigid [`PortalPose`] at `position` looking toward
/// `forward` (its outward normal), with `up` as the surface's up vector.
///
/// Thin convenience wrapper — [`PortalPose::from_look_at`] takes a look-at
/// *target*, not a direction; this is the more natural call shape for placing
/// a portal in a level (`portal_pose_facing(pos, Vec3::Z, Vec3::Y)` rather
/// than having to compute `pos + forward` at every call site).
pub fn portal_pose_facing(position: glam::Vec3, forward: glam::Vec3, up: glam::Vec3) -> PortalPose {
    PortalPose::from_look_at(position, position + forward, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colocated_record() -> PortalRecord {
        let pose = PortalPose::from_look_at(glam::Vec3::ZERO, glam::Vec3::Z, glam::Vec3::Y);
        PortalRecord {
            pair: PortalPair { a: pose, b: pose },
            half_extent: Vec2::splat(2.0),
        }
    }

    #[test]
    fn portal_chain_budget_is_exact_instead_of_silently_truncated() {
        let six = vec![colocated_record(); 6];
        let chains = build_portal_chains(&six).expect("6 + 36 + 216 chains fit");
        assert_eq!(chains.len(), 258);

        let seven = vec![colocated_record(); 7];
        assert!(
            build_portal_chains(&seven).is_none(),
            "7 + 49 + 343 chains must report the 300-row ABI overflow"
        );
    }
}
