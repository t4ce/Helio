//! SceneDB-backed reflection-capture authority and compact render projection.
//!
//! Authored capture rows are sparse component-local SceneDB rows. Influence
//! ordering and cube-array layer residency are renderer policy, so Helio owns
//! only one compact `[row, layer]` pair per active capture.

use helio_scenedb::{
    SceneIndices, SceneReflectionCapture, SceneReflectionCaptureRow,
};
use libhelio::{GpuReflectionCapture, ReflectionCaptureMobility};

use crate::handles::{
    entity_from_handle, handle_from_entity, ReflectionCaptureId,
};
use crate::scene::actor::ReflectionCaptureDescriptor;
use crate::scene::errors::{invalid, Result, SceneError};

use super::entity_projection::EntityRowProjection;

fn select_resident_reflection_projections(
    mut candidates: Vec<(f32, u32, i32)>,
) -> (Vec<[u32; 2]>, usize) {
    candidates.retain(|(_, _, layer)| *layer >= 0);
    let compare = |a: &(f32, u32, i32), b: &(f32, u32, i32)| {
        a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
    };
    let overflow = candidates
        .len()
        .saturating_sub(libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS);
    if overflow != 0 {
        candidates.select_nth_unstable_by(
            libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS,
            compare,
        );
    }
    candidates.truncate(libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS);
    candidates.sort_by(compare);
    (
        candidates
            .into_iter()
            .map(|(_, row, layer)| [row, layer as u32])
            .collect(),
        overflow,
    )
}

pub(in crate::scene) struct ReflectionProjection {
    active: EntityRowProjection<ReflectionCaptureId>,
    /// Cube-array layer parallel to `active` compact slots. `-1` means the
    /// capture has no resident probe and therefore contributes nothing.
    layers: Vec<i32>,
}

impl Default for ReflectionProjection {
    fn default() -> Self {
        Self {
            active: EntityRowProjection::default(),
            layers: Vec::new(),
        }
    }
}

impl ReflectionProjection {
    fn insert(&mut self, id: ReflectionCaptureId, gpu_row: u32) {
        let compact = self.active.insert(id, gpu_row);
        debug_assert_eq!(compact, self.layers.len());
        self.layers.push(-1);
    }

    fn remove(&mut self, id: ReflectionCaptureId) -> bool {
        let Some(compact) = self.active.remove(id) else {
            return false;
        };
        self.layers.swap_remove(compact);
        true
    }

    fn layer(&self, id: ReflectionCaptureId) -> Option<i32> {
        self.active.slot(id).map(|slot| self.layers[slot])
    }

    fn set_layer(&mut self, id: ReflectionCaptureId, layer: i32) -> bool {
        let Some(slot) = self.active.slot(id) else {
            return false;
        };
        self.layers[slot] = layer;
        true
    }

    fn ids(&self) -> &[ReflectionCaptureId] {
        self.active.ids()
    }

    fn rows(&self) -> &[u32] {
        self.active.rows()
    }

    fn len(&self) -> usize {
        self.active.len()
    }
}

/// Validate and flatten authored parameters into the shader-layout canonical
/// row. Cube layer residency deliberately remains the fixed `-1` sentinel.
fn build_row(desc: &ReflectionCaptureDescriptor) -> Result<SceneReflectionCaptureRow> {
    if !desc.transform.is_finite() {
        return Err(SceneError::InvalidOperation {
            reason: "reflection capture transform must be finite",
        });
    }
    let world_to_local = desc.transform.inverse();
    if !world_to_local.is_finite() {
        return Err(SceneError::InvalidOperation {
            reason: "reflection capture transform must be invertible",
        });
    }
    if !desc.influence_radius.is_finite() || desc.influence_radius < 0.0 {
        return Err(SceneError::InvalidOperation {
            reason: "reflection capture influence radius must be finite and non-negative",
        });
    }
    if !desc
        .extents
        .into_iter()
        .all(|extent| extent.is_finite() && extent >= 0.0)
    {
        return Err(SceneError::InvalidOperation {
            reason: "reflection capture extents must be finite and non-negative",
        });
    }
    if !desc.transition_distance.is_finite() || desc.transition_distance < 0.0 {
        return Err(SceneError::InvalidOperation {
            reason: "reflection capture transition distance must be finite and non-negative",
        });
    }
    if !desc.brightness.is_finite() {
        return Err(SceneError::InvalidOperation {
            reason: "reflection capture brightness must be finite",
        });
    }

    let pos = desc.position();
    Ok(SceneReflectionCaptureRow::from(GpuReflectionCapture {
        position_radius: [pos[0], pos[1], pos[2], desc.influence_radius],
        extents_transition: [
            desc.extents[0],
            desc.extents[1],
            desc.extents[2],
            desc.transition_distance,
        ],
        world_to_local: world_to_local.to_cols_array_2d(),
        cubemap_index: -1,
        shape: desc.shape as u32,
        mobility: desc.mobility as u32,
        brightness: desc.brightness,
    }))
}

impl super::super::Scene {
    pub fn insert_reflection_capture(
        &mut self,
        desc: ReflectionCaptureDescriptor,
    ) -> Result<ReflectionCaptureId> {
        self.insert_reflection_capture_with_tag(desc, 0)
    }

    pub fn insert_reflection_capture_with_tag(
        &mut self,
        desc: ReflectionCaptureDescriptor,
        user_tag: u64,
    ) -> Result<ReflectionCaptureId> {
        let invalidates_bake = desc.mobility == ReflectionCaptureMobility::Static;
        let capture = build_row(&desc)?;
        let entity = self.authority.insert(SceneReflectionCapture {
            user_tag,
            capture,
            _reserved: 0,
        });
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_reflection_capture(user_tag, entity);
        let gpu_row = self
            .authority
            .gpu_row::<SceneReflectionCapture>(entity)
            .expect("inserted GPU component must own a mirror row");
        let id = handle_from_entity(entity);
        self.reflection_projection.insert(id, gpu_row);
        self.rebuild_reflection_capture_projection();
        if invalidates_bake {
            self.bake_invalidated = true;
        }
        Ok(id)
    }

    pub fn update_reflection_capture(
        &mut self,
        id: ReflectionCaptureId,
        desc: &ReflectionCaptureDescriptor,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(old_row) = self
            .authority
            .get::<SceneReflectionCapture>(entity)
            .map(|record| record.capture)
        else {
            return Err(invalid("reflection capture"));
        };
        let new_row = build_row(desc)?;
        if bytemuck::bytes_of(&old_row) == bytemuck::bytes_of(&new_row) {
            return Ok(());
        }
        self.authority
            .edit_gpu::<SceneReflectionCapture, _>(entity, |record| {
                record.capture = new_row;
            })
            .ok_or_else(|| invalid("reflection capture"))?;
        if old_row.mobility == ReflectionCaptureMobility::Static as u32
            || new_row.mobility == ReflectionCaptureMobility::Static as u32
        {
            self.bake_invalidated = true;
        }
        if new_row.mobility != ReflectionCaptureMobility::Static as u32 {
            let cleared = self.reflection_projection.set_layer(id, -1);
            debug_assert!(cleared);
        }
        self.rebuild_reflection_capture_projection();
        Ok(())
    }

    pub fn remove_reflection_capture(&mut self, id: ReflectionCaptureId) -> bool {
        let entity = entity_from_handle(id);
        let Some(record) = self
            .authority
            .get::<SceneReflectionCapture>(entity)
            .copied()
        else {
            return false;
        };
        if !self.authority.despawn(entity) {
            return false;
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_reflection_capture(record.user_tag, entity);
        debug_assert!(self.reflection_projection.remove(id));
        self.rebuild_reflection_capture_projection();
        if record.capture.mobility == ReflectionCaptureMobility::Static as u32 {
            self.bake_invalidated = true;
        }
        self.detach_actors_for_target(
            crate::scene::actor::SceneActorId::ReflectionCapture(id),
        );
        true
    }

    pub fn get_reflection_capture(
        &self,
        id: ReflectionCaptureId,
    ) -> Option<GpuReflectionCapture> {
        let mut capture = self
            .authority
            .get::<SceneReflectionCapture>(entity_from_handle(id))?
            .capture
            .as_authored_gpu_capture();
        capture.cubemap_index = self.reflection_projection.layer(id)?;
        Some(capture)
    }

    pub fn iter_reflection_captures(
        &self,
    ) -> impl Iterator<Item = (ReflectionCaptureId, GpuReflectionCapture, u64)> + '_ {
        self.authority
            .query::<SceneReflectionCapture>()
            .filter_map(|(entity, record)| {
                let id = handle_from_entity(entity);
                let mut capture = record.capture.as_authored_gpu_capture();
                capture.cubemap_index = self.reflection_projection.layer(id)?;
                Some((id, capture, record.user_tag))
            })
    }

    pub fn reflection_capture_by_tag(&self, user_tag: u64) -> Option<ReflectionCaptureId> {
        self.authority
            .subsystem::<SceneIndices>()
            .and_then(|indices| indices.reflection_capture_by_tag(user_tag))
            .filter(|entity| {
                self.authority
                    .get::<SceneReflectionCapture>(*entity)
                    .is_some()
            })
            .map(handle_from_entity)
    }

    pub fn reflection_capture_count(&self) -> usize {
        self.reflection_projection.len()
    }

    /// World positions of every static capture, in the stable projection order
    /// used by [`Self::assign_reflection_capture_layers`].
    pub fn static_reflection_capture_positions(&self) -> Vec<[f32; 3]> {
        let mut out: Vec<_> = self
            .reflection_projection
            .ids()
            .iter()
            .filter_map(|&id| {
                let row = self
                    .authority
                    .get::<SceneReflectionCapture>(entity_from_handle(id))?
                    .capture;
                (row.mobility == ReflectionCaptureMobility::Static as u32).then(|| {
                    let p = row.position_radius;
                    [p[0], p[1], p[2]]
                })
            })
            .collect();
        out.shrink_to_fit();
        out
    }

    /// Assign static captures to cube-array layers in exactly the order
    /// [`Self::static_reflection_capture_positions`] reports them.
    pub fn assign_reflection_capture_layers(&mut self) {
        let ids: Vec<_> = self
            .reflection_projection
            .ids()
            .iter()
            .copied()
            .filter(|&id| {
                self.authority
                    .get::<SceneReflectionCapture>(entity_from_handle(id))
                    .is_some_and(|record| {
                        record.capture.mobility == ReflectionCaptureMobility::Static as u32
                    })
            })
            .collect();
        for (layer, id) in ids.into_iter().enumerate() {
            let assigned = self.reflection_projection.set_layer(id, layer as i32);
            debug_assert!(assigned);
        }
        self.rebuild_reflection_capture_projection();
    }

    /// Rebuild the small derived projection, ordered smallest influence first.
    /// Canonical 112-byte rows are never cloned into a Helio-owned buffer.
    fn rebuild_reflection_capture_projection(&mut self) {
        let candidates: Vec<(f32, u32, i32)> = self
            .reflection_projection
            .ids()
            .iter()
            .copied()
            .zip(self.reflection_projection.rows().iter().copied())
            .filter_map(|(id, row)| {
                let capture = self
                    .authority
                    .get::<SceneReflectionCapture>(entity_from_handle(id))?;
                let layer = self.reflection_projection.layer(id)?;
                Some((capture.capture.influence_size(), row, layer))
            })
            .collect();
        let (projections, overflow) = select_resident_reflection_projections(candidates);
        if overflow != 0 {
            log::warn!(
                "reflection capture projection limit {} exceeded by {} resident captures; dropping the largest-influence captures",
                libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS,
                overflow,
            );
        }
        self.gpu_scene
            .reflection_capture_projections
            .set_data(projections);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_preserves_layers_through_swap_remove() {
        let first = ReflectionCaptureId::from_raw(5, 1);
        let middle = ReflectionCaptureId::from_raw(42, 2);
        let last = ReflectionCaptureId::from_raw(3, 7);
        let mut projection = ReflectionProjection::default();
        projection.insert(first, 4);
        projection.insert(middle, 9);
        projection.insert(last, 2);
        assert!(projection.set_layer(first, 11));
        assert!(projection.set_layer(last, 17));

        assert!(projection.remove(middle));
        assert_eq!(projection.layer(first), Some(11));
        assert_eq!(projection.layer(last), Some(17));
        assert_eq!(projection.rows(), &[4, 2]);
    }

    #[test]
    fn unbaked_captures_are_filtered_before_the_shader_cap() {
        let mut candidates = (0..libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS)
            .map(|row| (row as f32, row as u32, -1))
            .collect::<Vec<_>>();
        candidates.push((10_000.0, 777, 3));

        let (selected, overflow) = select_resident_reflection_projections(candidates);

        assert_eq!(overflow, 0);
        assert_eq!(selected, vec![[777, 3]]);
    }

    #[test]
    fn resident_capture_projection_keeps_the_64_smallest_influences() {
        let candidates = (0..=libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS)
            .rev()
            .map(|row| (row as f32, row as u32, row as i32))
            .collect();

        let (selected, overflow) = select_resident_reflection_projections(candidates);

        assert_eq!(overflow, 1);
        assert_eq!(selected.len(), libhelio::MAX_REFLECTION_CAPTURE_PROJECTIONS);
        assert_eq!(selected.first(), Some(&[0, 0]));
        assert_eq!(selected.last(), Some(&[63, 63]));
    }
}
