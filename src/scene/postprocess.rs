//! SceneDB-backed post-process-volume CRUD and compact active-row projection.

use helio_scenedb::{
    SceneIndices, ScenePostProcessVolume, ScenePostProcessVolumeRow,
};
use libhelio::{GpuPostProcessVolume, PostProcessVolumeDescriptor};

use crate::handles::{
    entity_from_handle, handle_from_entity, PostProcessVolumeId,
};
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::Scene;

fn validate_post_process_volume_priority(desc: &PostProcessVolumeDescriptor) -> Result<()> {
    if !desc.priority.is_finite() {
        return Err(SceneError::InvalidOperation {
            reason: "post-process volume priority must be finite",
        });
    }
    Ok(())
}

fn select_post_process_volume_rows(
    mut candidates: Vec<(f32, u32)>,
) -> (Vec<u32>, usize) {
    let compare = |a: &(f32, u32), b: &(f32, u32)| {
        b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    };
    let overflow = candidates
        .len()
        .saturating_sub(libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS);
    if overflow != 0 {
        candidates.select_nth_unstable_by(
            libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS,
            compare,
        );
    }
    candidates.truncate(libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS);
    candidates.sort_by(compare);
    (
        candidates.into_iter().map(|(_, row)| row).collect(),
        overflow,
    )
}

impl Scene {
    pub fn insert_post_process_volume(
        &mut self,
        desc: PostProcessVolumeDescriptor,
    ) -> Result<PostProcessVolumeId> {
        self.insert_post_process_volume_with_tag(desc, 0)
    }

    pub fn insert_post_process_volume_with_tag(
        &mut self,
        desc: PostProcessVolumeDescriptor,
        user_tag: u64,
    ) -> Result<PostProcessVolumeId> {
        validate_post_process_volume_priority(&desc)?;
        let entity = self.authority.insert(ScenePostProcessVolume {
            user_tag,
            volume: ScenePostProcessVolumeRow::from(desc.to_gpu()),
            _reserved: 0,
        });
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_post_process_volume(user_tag, entity);
        let row = self
            .authority
            .gpu_row::<ScenePostProcessVolume>(entity)
            .expect("inserted GPU component must own a mirror row");
        let id = handle_from_entity(entity);
        self.post_process_volume_projection.insert(id, row);
        self.rebuild_post_process_volume_projection();
        Ok(id)
    }

    pub fn remove_post_process_volume(&mut self, id: PostProcessVolumeId) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(record) = self
            .authority
            .get::<ScenePostProcessVolume>(entity)
            .copied()
        else {
            return Err(invalid("post-process volume"));
        };
        if !self.authority.despawn(entity) {
            return Err(invalid("post-process volume"));
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_post_process_volume(record.user_tag, entity);
        self.post_process_volume_projection
            .remove(id)
            .expect("live post-process volume must have an active projection");
        self.rebuild_post_process_volume_projection();
        self.detach_actors_for_target(
            crate::scene::actor::SceneActorId::PostProcessVolume(id),
        );
        Ok(())
    }

    pub fn update_post_process_volume(
        &mut self,
        id: PostProcessVolumeId,
        desc: PostProcessVolumeDescriptor,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let Some(old_row) = self
            .authority
            .get::<ScenePostProcessVolume>(entity)
            .map(|record| record.volume)
        else {
            return Err(invalid("post-process volume"));
        };
        validate_post_process_volume_priority(&desc)?;
        let new_row = ScenePostProcessVolumeRow::from(desc.to_gpu());
        if bytemuck::bytes_of(&old_row) == bytemuck::bytes_of(&new_row) {
            return Ok(());
        }
        self.authority
            .edit_gpu::<ScenePostProcessVolume, _>(entity, |record| record.volume = new_row)
            .ok_or_else(|| invalid("post-process volume"))?;
        if old_row.0.priority.to_bits() != new_row.0.priority.to_bits() {
            self.rebuild_post_process_volume_projection();
        }
        Ok(())
    }

    pub fn get_post_process_volume(
        &self,
        id: PostProcessVolumeId,
    ) -> Option<GpuPostProcessVolume> {
        self.authority
            .get::<ScenePostProcessVolume>(entity_from_handle(id))
            .map(|record| record.volume.0)
    }

    pub fn iter_post_process_volumes(
        &self,
    ) -> impl Iterator<Item = (PostProcessVolumeId, &GpuPostProcessVolume, u64)> + '_ {
        self.authority
            .query::<ScenePostProcessVolume>()
            .map(|(entity, record)| {
                (handle_from_entity(entity), &record.volume.0, record.user_tag)
            })
    }

    pub fn post_process_volumes_count(&self) -> u32 {
        self.post_process_volume_projection.len() as u32
    }

    pub fn post_process_volume_by_tag(
        &self,
        user_tag: u64,
    ) -> Option<PostProcessVolumeId> {
        self.authority
            .subsystem::<SceneIndices>()
            .and_then(|indices| indices.post_process_volume_by_tag(user_tag))
            .filter(|entity| {
                self.authority
                    .get::<ScenePostProcessVolume>(*entity)
                    .is_some()
            })
            .map(handle_from_entity)
    }

    /// Rebuild the shader-bounded row projection from canonical priorities.
    /// Authored SceneDB membership remains unbounded by this render ABI.
    fn rebuild_post_process_volume_projection(&mut self) {
        let candidates = self
            .post_process_volume_projection
            .ids()
            .iter()
            .copied()
            .zip(
                self.post_process_volume_projection
                    .rows()
                    .iter()
                    .copied(),
            )
            .filter_map(|(id, row)| {
                let component = self
                    .authority
                    .get::<ScenePostProcessVolume>(entity_from_handle(id))?;
                Some((component.volume.0.priority, row))
            })
            .collect();
        let (rows, overflow) = select_post_process_volume_rows(candidates);
        if overflow != 0 {
            log::warn!(
                "post-process volume projection limit {} exceeded by {} authored volumes; dropping the lowest-priority rows from the shader projection",
                libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS,
                overflow,
            );
        }
        self.gpu_scene.post_process_volume_indices.set_data(rows);
    }
}

#[cfg(test)]
mod tests {
    use super::select_post_process_volume_rows;

    #[test]
    fn shader_projection_keeps_highest_priorities_with_a_stable_row_tie_break() {
        let mut candidates = (0..=libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS)
            .map(|row| (row as f32, row as u32))
            .collect::<Vec<_>>();
        candidates.extend([(10_000.0, 900), (10_000.0, 800)]);

        let (selected, overflow) = select_post_process_volume_rows(candidates);

        assert_eq!(selected.len(), libhelio::MAX_POST_PROCESS_VOLUME_PROJECTIONS);
        assert_eq!(overflow, 3);
        assert_eq!(&selected[..2], &[800, 900]);
        assert!(selected.contains(&256));
        assert!(!selected.contains(&0));
    }
}
