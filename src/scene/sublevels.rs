//! Sublevels: a group of objects rendered through one shared, cheaply movable
//! coordinate-space transform.
//!
//! A sublevel's members keep their ordinary *local* (sublevel-relative)
//! `SceneObjectSpatialRow::model`, exactly as an ungrouped object would — nothing
//! about how an object is authored changes. What differs is that the GPU
//! vertex/cull shaders read one extra transform, `coordinate_spaces[slot]`
//! (see `libhelio::{coordinate_space, set_coordinate_space}`), and apply it
//! on top. The table is SceneDB's `SceneCoordinateSpace` GPU partner; moving the whole
//! sublevel is therefore **O(1)**: one matrix write via
//! [`Scene::move_sublevel`]/[`Scene::update_sublevel`], never a walk over
//! member objects. Assigning membership (tagging each member with the
//! sublevel's slot) is the one O(N) step, and it happens only once, at
//! [`Scene::add_sublevel`] — not per frame.
//!
//! Unlike portals, sublevel content needs no clipping: it isn't a duplicate
//! draw of anything, just the same geometry rendered in a different place.

use glam::Mat4;
use helio_scenedb::{CpuOnlyComponent, SceneCoordinateSpace, SceneCoordinateSpaceRow, SceneObject};

use crate::groups::GroupId;
use crate::handles::{entity_from_handle, handle_from_entity, SublevelId};
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::helpers::{object_groups, object_movability};
use crate::scene::Scene;

/// Configuration for [`Scene::add_sublevel`].
#[derive(Debug, Clone, Copy)]
pub struct SublevelDescriptor {
    /// The group whose *current* members become this sublevel's members.
    /// Membership is captured once, at creation — objects added to `group`
    /// afterward are not automatically included (see
    /// [`Scene::refresh_sublevel_membership`] to re-capture explicitly).
    pub group: GroupId,

    /// World-space placement of the sublevel's local origin.
    pub placement: Mat4,
}

/// Internal record for a sublevel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SublevelRecord {
    pub group: GroupId,
}

impl CpuOnlyComponent for SublevelRecord {}

impl Scene {
    /// Create a sublevel: allocate a coordinate-space slot, place it, and tag
    /// `desc.group`'s current members with that slot.
    ///
    /// # Performance
    /// - CPU cost: **O(N)** over the SceneDB object query (N = total scene
    ///   objects, not group size) — a one-time membership walk, same shape as
    ///   [`Scene::move_group`](crate::Scene::move_group) but paid once here
    ///   instead of on every call.
    /// - GPU cost: one coordinate-space slot write, plus one spatial-row write
    ///   per member (their `flags` changed) — deferred to the next `flush()`
    ///   if a full rebuild is already pending.
    ///
    /// # Errors
    /// [`SceneError::CoordinateSpaceCapacityExceeded`] if all coordinate-space
    /// slots are already claimed by other sublevels/portals.
    pub fn add_sublevel(&mut self, desc: SublevelDescriptor) -> Result<SublevelId> {
        if self.authority.gpu_live_count::<SceneCoordinateSpace>()
            >= libhelio::MAX_COORDINATE_SPACES
        {
            return Err(SceneError::CoordinateSpaceCapacityExceeded);
        }

        let entity = self.authority.insert(SublevelRecord { group: desc.group });
        let attached = self.authority.replace_gpu(
            entity,
            SceneCoordinateSpace {
                transform: SceneCoordinateSpaceRow(desc.placement.to_cols_array()),
            },
        );
        debug_assert!(attached, "fresh SceneDB sublevel entity must be alive");
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
            .stage_new(slot, desc.placement.to_cols_array());

        self.tag_group_with_coordinate_space(desc.group, slot);

        Ok(handle_from_entity(entity))
    }

    /// Re-walks `sublevel`'s group and tags any current member that isn't
    /// already tagged with this sublevel's coordinate space.
    ///
    /// [`Scene::add_sublevel`] captures membership once, at creation; call
    /// this after adding more objects to the same group if they should join
    /// the sublevel too. Same O(N) shape as `add_sublevel`'s initial walk.
    pub fn refresh_sublevel_membership(&mut self, sublevel: SublevelId) -> Result<()> {
        let entity = entity_from_handle(sublevel);
        let record = *self
            .authority
            .get::<SublevelRecord>(entity)
            .ok_or_else(|| invalid("sublevel"))?;
        let slot = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .ok_or_else(|| invalid("sublevel"))?;
        self.tag_group_with_coordinate_space(record.group, slot);
        Ok(())
    }

    fn tag_group_with_coordinate_space(&mut self, group: GroupId, slot: u32) {
        let mut changed_static = false;
        let mut changed_movable = false;
        {
            let authority = &mut self.authority;
            let object_history = &mut self.gpu_scene.object_history;
            authority.edit_gpu_each::<SceneObject>(|_, gpu_row, object| {
                if !object_groups(object).contains(group)
                    || libhelio::coordinate_space(object.spatial.flags) == slot
                {
                    return false;
                }
                let can_move = object_movability(object).can_move();
                object.spatial.flags = libhelio::set_coordinate_space(object.spatial.flags, slot);
                object_history.stage_current(
                    gpu_row,
                    object.spatial.model,
                    object.spatial.sphere,
                    object.spatial.flags,
                );
                changed_movable |= can_move;
                changed_static |= !can_move;
                true
            });
        }
        if changed_movable {
            self.movable_objects_generation = self.movable_objects_generation.wrapping_add(1);
            self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        }
        if changed_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }
    }

    /// Apply a transform delta to a sublevel's placement. **O(1)** — one
    /// coordinate-space matrix write, no member object is touched.
    ///
    /// `delta` is pre-multiplied: `new_placement = delta * old_placement`,
    /// same convention as [`Scene::move_group`](crate::Scene::move_group).
    pub fn move_sublevel(&mut self, sublevel: SublevelId, delta: Mat4) -> Result<()> {
        let entity = entity_from_handle(sublevel);
        if self.authority.get::<SublevelRecord>(entity).is_none() {
            return Err(invalid("sublevel"));
        }
        let space = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .ok_or_else(|| invalid("sublevel"))?;
        let placement = self
            .authority
            .edit_gpu::<SceneCoordinateSpace, _>(entity, |coordinate| {
                let current = Mat4::from_cols_array(&coordinate.transform.0);
                coordinate.transform = SceneCoordinateSpaceRow((delta * current).to_cols_array());
                coordinate.transform.0
            })
            .ok_or_else(|| invalid("sublevel"))?;
        self.gpu_scene
            .coordinate_space_history
            .stage_current(space, placement);
        self.static_objects_dirty = true;
        self.bake_invalidated = true;
        self.movable_objects_generation = self.movable_objects_generation.wrapping_add(1);
        self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        Ok(())
    }

    /// Set a sublevel's placement directly. **O(1)** — one coordinate-space
    /// matrix write, no member object is touched.
    pub fn update_sublevel(&mut self, sublevel: SublevelId, placement: Mat4) -> Result<()> {
        let entity = entity_from_handle(sublevel);
        if self.authority.get::<SublevelRecord>(entity).is_none() {
            return Err(invalid("sublevel"));
        }
        let space = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .ok_or_else(|| invalid("sublevel"))?;
        let matrix = placement.to_cols_array();
        self.authority
            .edit_gpu::<SceneCoordinateSpace, _>(entity, |coordinate| {
                coordinate.transform = SceneCoordinateSpaceRow(matrix);
            })
            .ok_or_else(|| invalid("sublevel"))?;
        self.gpu_scene
            .coordinate_space_history
            .stage_current(space, matrix);
        self.static_objects_dirty = true;
        self.bake_invalidated = true;
        self.movable_objects_generation = self.movable_objects_generation.wrapping_add(1);
        self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        Ok(())
    }

    /// Current world-space placement of a sublevel, e.g. to compose with a
    /// member object's local transform for CPU-side picking.
    pub fn sublevel_placement(&self, sublevel: SublevelId) -> Option<Mat4> {
        let entity = entity_from_handle(sublevel);
        self.authority.get::<SublevelRecord>(entity)?;
        self.authority
            .get::<SceneCoordinateSpace>(entity)
            .map(|coordinate| Mat4::from_cols_array(&coordinate.transform.0))
    }

    /// Remove a sublevel: untag its members (they fall back to world space,
    /// coordinate space 0) and free its GPU slot for reuse.
    ///
    /// # Performance
    /// CPU cost: O(N) over the SceneDB object query, same as `add_sublevel`.
    pub fn remove_sublevel(&mut self, sublevel: SublevelId) -> Result<()> {
        let entity = entity_from_handle(sublevel);
        let record = *self
            .authority
            .get::<SublevelRecord>(entity)
            .ok_or_else(|| invalid("sublevel"))?;
        let coordinate_space = self
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity)
            .ok_or_else(|| invalid("sublevel"))?;

        let mut changed_static = false;
        let mut changed_movable = false;
        {
            let authority = &mut self.authority;
            let object_history = &mut self.gpu_scene.object_history;
            authority.edit_gpu_each::<SceneObject>(|_, gpu_row, object| {
                if !object_groups(object).contains(record.group)
                    || libhelio::coordinate_space(object.spatial.flags) != coordinate_space
                {
                    return false;
                }
                let can_move = object_movability(object).can_move();
                object.spatial.flags = libhelio::set_coordinate_space(object.spatial.flags, 0);
                object_history.stage_current(
                    gpu_row,
                    object.spatial.model,
                    object.spatial.sphere,
                    object.spatial.flags,
                );
                changed_movable |= can_move;
                changed_static |= !can_move;
                true
            });
        }
        if changed_movable {
            self.movable_objects_generation = self.movable_objects_generation.wrapping_add(1);
            self.gpu_scene.movable_objects_generation = self.movable_objects_generation;
        }
        if changed_static {
            self.static_objects_dirty = true;
            self.bake_invalidated = true;
        }

        self.gpu_scene
            .coordinate_space_history
            .stage_new(coordinate_space, Mat4::IDENTITY.to_cols_array());
        let removed = self.authority.despawn(entity);
        debug_assert!(removed, "validated sublevel entity must despawn");
        Ok(())
    }
}
