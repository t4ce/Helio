//! Group membership stored on SceneDB objects with derived visibility updates.

use helio_scenedb::SceneObject;

use crate::groups::{GroupId, GroupMask};
use crate::handles::{entity_from_handle, ObjectId};

use super::super::errors::{invalid, Result};
use super::super::helpers::object_is_visible;

impl super::super::Scene {
    fn publish_object_group_visibility(&mut self, id: ObjectId, mask: GroupMask) {
        if self.objects_dirty {
            return;
        }
        if let Some(slot) = self.object_projection_slot(id) {
            let visible = u32::from(object_is_visible(mask, self.group_hidden()));
            self.gpu_scene.visibility.update(slot, visible);
        }
    }

    pub fn set_object_groups(&mut self, id: ObjectId, mask: GroupMask) -> Result<()> {
        let entity = entity_from_handle(id);
        self.authority
            .edit_gpu::<SceneObject, _>(entity, |record| record.groups = mask.0)
            .ok_or_else(|| invalid("object"))?;
        self.publish_object_group_visibility(id, mask);
        Ok(())
    }

    pub fn add_object_to_group(&mut self, id: ObjectId, group: GroupId) -> Result<()> {
        let entity = entity_from_handle(id);
        let mask = self
            .authority
            .edit_gpu::<SceneObject, _>(entity, |record| {
                let mask = GroupMask(record.groups).with(group);
                record.groups = mask.0;
                mask
            })
            .ok_or_else(|| invalid("object"))?;
        self.publish_object_group_visibility(id, mask);
        Ok(())
    }

    pub fn remove_object_from_group(&mut self, id: ObjectId, group: GroupId) -> Result<()> {
        let entity = entity_from_handle(id);
        let mask = self
            .authority
            .edit_gpu::<SceneObject, _>(entity, |record| {
                let mask = GroupMask(record.groups).without(group);
                record.groups = mask.0;
                mask
            })
            .ok_or_else(|| invalid("object"))?;
        self.publish_object_group_visibility(id, mask);
        Ok(())
    }

    pub fn object_groups(&self, id: ObjectId) -> Result<GroupMask> {
        self.object_record(id)
            .map(|record| GroupMask(record.groups))
            .ok_or_else(|| invalid("object"))
    }
}
