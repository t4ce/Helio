//! Group visibility control for batch show/hide operations.
//!
//! Provides methods for hiding/showing entire groups of classic and virtual
//! objects with efficient renderer-derived cull updates.

use crate::groups::{GroupId, GroupMask};
use helio_scenedb::{SceneObject, SceneVisibilityState};

use super::super::helpers::{object_groups, object_is_visible};
use super::super::objects::NO_OBJECT_PROJECTION_SLOT;

impl super::super::Scene {
    /// Canonical authored hidden-group mask from SceneDB's registered
    /// singleton. The GPU visibility buffer is only a compact renderer
    /// projection derived from this value and object membership components.
    pub(in crate::scene) fn group_hidden(&self) -> GroupMask {
        GroupMask(
            self.authority
                .subsystem::<SceneVisibilityState>()
                .expect("visibility subsystem is registered at Scene construction")
                .hidden_groups(),
        )
    }

    fn replace_group_hidden(&mut self, hidden: GroupMask) -> bool {
        self.authority
            .subsystem_mut::<SceneVisibilityState>()
            .expect("visibility subsystem is registered at Scene construction")
            .replace_hidden_groups(hidden.0)
    }

    pub(in crate::scene) fn clear_group_visibility(&mut self) {
        self.authority
            .subsystem_mut::<SceneVisibilityState>()
            .expect("visibility subsystem is registered at Scene construction")
            .clear();
    }

    /// Hide all objects that belong to a group.
    ///
    /// Sets the group's hidden bit, making all objects in this group invisible.
    /// Objects in multiple groups are hidden if **any** of their groups is hidden.
    ///
    /// # Parameters
    /// - `group`: Group to hide
    ///
    /// # Performance
    /// - CPU cost: O(1) if already hidden, O(N + V) if state changes
    /// - GPU cost: dirty-tracked classic scalar writes plus one compact VG
    ///   cull-projection refresh when effective VG visibility changes
    /// - Memory: No allocations
    ///
    /// # Idempotent
    ///
    /// Calling this on an already-hidden group is a no-op (O(1)).
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// // Hide all enemies
    /// let group_enemies = GroupId(0);
    /// scene.hide_group(group_enemies);
    ///
    /// // All objects with group_enemies in their mask are now hidden
    /// ```
    pub fn hide_group(&mut self, group: GroupId) {
        let hidden = self.group_hidden();
        if hidden.contains(group) {
            return; // already hidden — nothing to do
        }
        self.replace_group_hidden(hidden.with(group));
        self.flush_group_visibility();
    }

    /// Show all objects in a group (unless another one of their groups is hidden).
    ///
    /// Clears the group's hidden bit, making objects in this group visible again
    /// (unless they belong to another hidden group).
    ///
    /// # Parameters
    /// - `group`: Group to show
    ///
    /// # Performance
    /// - CPU cost: O(1) if already visible, O(N + V) if state changes
    /// - GPU cost: dirty-tracked classic scalar writes plus one compact VG
    ///   cull-projection refresh when effective VG visibility changes
    /// - Memory: No allocations
    ///
    /// # Multi-Group Objects
    ///
    /// If an object belongs to groups A and B, and both are hidden:
    /// 1. `show_group(A)` will **not** make the object visible (group B is still hidden)
    /// 2. `show_group(B)` will make the object visible (both groups are now shown)
    ///
    /// # Idempotent
    ///
    /// Calling this on an already-visible group is a no-op (O(1)).
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// // Show all UI elements
    /// let group_ui = GroupId(1);
    /// scene.show_group(group_ui);
    /// ```
    pub fn show_group(&mut self, group: GroupId) {
        let hidden = self.group_hidden();
        if !hidden.contains(group) {
            return; // already visible — nothing to do
        }
        self.replace_group_hidden(hidden.without(group));
        self.flush_group_visibility();
    }

    /// Return `true` if a group is currently hidden.
    ///
    /// Queries the hidden state of a specific group.
    ///
    /// # Parameters
    /// - `group`: Group to query
    ///
    /// # Returns
    /// `true` if the group is hidden, `false` if visible.
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// let group_enemies = GroupId(0);
    /// if scene.is_group_hidden(group_enemies) {
    ///     println!("Enemies are hidden");
    /// }
    /// ```
    pub fn is_group_hidden(&self, group: GroupId) -> bool {
        self.group_hidden().contains(group)
    }

    /// Set visibility for multiple groups at once via bitmask.
    ///
    /// Only the bits in `mask` are affected; all other groups keep their current state.
    ///
    /// # Parameters
    /// - `mask`: Bitmask of groups to modify
    /// - `visible`: `true` to show these groups, `false` to hide them
    ///
    /// # Performance
    /// - CPU cost: O(1) if no state change, O(N + V) if visibility changes
    /// - GPU cost: dirty-tracked classic scalar writes plus one compact VG
    ///   cull-projection refresh when effective VG visibility changes
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::{GroupId, GroupMask};
    ///
    /// // Hide groups 0, 1, and 2 in one call
    /// let mask = GroupMask::from_id(GroupId(0))
    ///     .with(GroupId(1))
    ///     .with(GroupId(2));
    /// scene.set_group_visibility(mask, false);
    ///
    /// // Show them all again
    /// scene.set_group_visibility(mask, true);
    /// ```
    pub fn set_group_visibility(&mut self, mask: GroupMask, visible: bool) {
        let hidden = self.group_hidden();
        let new_hidden = if visible {
            GroupMask(hidden.0 & !mask.0) // clear bits → visible
        } else {
            GroupMask(hidden.0 | mask.0) // set bits → hidden
        };
        if !self.replace_group_hidden(new_hidden) {
            return;
        }
        self.flush_group_visibility();
    }

    /// Internal: re-evaluate derived visibility for classic and virtual objects
    /// when SceneDB's canonical hidden-group mask changes.
    ///
    /// Iterates over both canonical object stores. Classic objects update the
    /// dirty-tracked visibility buffer in place; virtual objects update the
    /// CPU-side input to the pass-owned instance-cull projection. Either path
    /// is skipped independently when its own topology rebuild is pending.
    ///
    /// # Performance
    /// - CPU cost: O(N + V) over classic and virtual SceneDB records
    /// - GPU cost: O(N) classic scalar updates plus one compact 16-byte-per-VG
    ///   cull-projection refresh when any virtual object's result changes
    /// - Memory: No allocations
    ///
    /// # Optimization
    ///
    /// A pending classic rebuild must not suppress a clean VG projection (or
    /// vice versa), so each category makes its skip decision independently.
    pub(in crate::scene) fn flush_group_visibility(&mut self) {
        let group_hidden = self.group_hidden();
        if !self.objects_dirty {
            let authority = &self.authority;
            let object_projection_slots = &self.object_projection_slots;
            let visibility = &mut self.gpu_scene.visibility;
            for (entity, object) in authority.query::<SceneObject>() {
                let Some(slot) = object_projection_slots
                    .get(entity.index() as usize)
                    .copied()
                    .filter(|&slot| slot != NO_OBJECT_PROJECTION_SLOT)
                else {
                    continue;
                };
                visibility.update(
                    slot as usize,
                    u32::from(object_is_visible(object_groups(object), group_hidden)),
                );
            }
        }
        self.refresh_vg_group_visibility(group_hidden);
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use glam::Mat4;
    use helio_scenedb::SceneVisibilityState;

    use crate::groups::{GroupId, GroupMask};
    use crate::mesh::{MeshUpload, PackedVertex};
    use crate::scene::{ObjectDescriptor, Scene};
    use crate::vg::{VirtualMeshUpload, VirtualObjectDescriptor};

    fn insert_grouped_object(scene: &mut Scene, groups: GroupMask) {
        let vertex = |position| {
            PackedVertex::from_components(
                position,
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            )
        };
        let mesh = scene.insert_mesh(MeshUpload {
            vertices: vec![
                vertex([-1.0, -1.0, 0.0]),
                vertex([1.0, -1.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        });
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        scene
            .insert_object(ObjectDescriptor {
                mesh,
                material,
                transform: Mat4::IDENTITY,
                bounds: [0.0, 0.0, 0.0, 2.0],
                flags: 0,
                groups,
                movability: None,
                user_tag: 0,
            })
            .expect("valid grouped object");
    }

    fn insert_grouped_virtual_object(scene: &mut Scene, groups: GroupMask) {
        let vertex = |position| {
            PackedVertex::from_components(
                position,
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            )
        };
        let mesh = scene.insert_virtual_mesh(VirtualMeshUpload {
            vertices: vec![
                vertex([-1.0, -1.0, 0.0]),
                vertex([1.0, -1.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        });
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        scene
            .insert_virtual_object(VirtualObjectDescriptor {
                virtual_mesh: mesh,
                material_id: material,
                transform: Mat4::IDENTITY,
                bounds: [0.0, 0.0, 0.0, 2.0],
                flags: 0,
                groups,
                movability: None,
            })
            .expect("valid grouped virtual object");
    }

    fn test_scene() -> Scene {
        let (device, queue) =
            crate::test_support::test_gpu().expect("no test GPU adapter found");
        Scene::new(device, queue)
    }

    #[test]
    fn scenedb_visibility_authority_drives_the_classic_gpu_projection() {
        let mut scene = test_scene();
        let groups = GroupMask::from(GroupId::DEBUG).with(GroupId::DEFAULT);
        insert_grouped_object(&mut scene, groups);
        scene.flush();

        assert_eq!(scene.gpu_scene.visibility.as_slice(), &[1]);
        assert_eq!(
            scene
                .authority
                .subsystem::<SceneVisibilityState>()
                .expect("registered visibility authority")
                .hidden_groups(),
            GroupMask::NONE.0,
        );

        scene.set_group_visibility(groups, false);
        assert_eq!(scene.gpu_scene.visibility.as_slice(), &[0]);
        assert_eq!(
            scene
                .authority
                .subsystem::<SceneVisibilityState>()
                .expect("registered visibility authority")
                .hidden_groups(),
            groups.0,
        );

        scene.show_group(GroupId::DEBUG);
        assert_eq!(scene.gpu_scene.visibility.as_slice(), &[0]);
        scene.show_group(GroupId::DEFAULT);
        assert_eq!(scene.gpu_scene.visibility.as_slice(), &[1]);
    }

    #[test]
    fn clear_resets_visibility_authority_before_new_scene_content() {
        let mut scene = test_scene();
        scene.hide_group(GroupId::DEBUG);
        assert!(scene.is_group_hidden(GroupId::DEBUG));

        scene.clear();

        assert!(!scene.is_group_hidden(GroupId::DEBUG));
        assert_eq!(
            scene
                .authority
                .subsystem::<SceneVisibilityState>()
                .expect("registered visibility authority")
                .hidden_groups(),
            GroupMask::NONE.0,
        );
        assert!(scene.gpu_scene.visibility.is_empty());

        insert_grouped_object(&mut scene, GroupMask::from(GroupId::DEBUG));
        scene.flush();
        assert_eq!(scene.gpu_scene.visibility.as_slice(), &[1]);
    }

    #[test]
    fn scenedb_visibility_authority_drives_virtual_geometry_without_topology_rebuild() {
        let mut scene = test_scene();
        let groups = GroupMask::from(GroupId::DEBUG).with(GroupId::DEFAULT);
        insert_grouped_virtual_object(&mut scene, groups);
        scene.flush();

        assert_eq!(
            scene
                .vg_frame_data()
                .expect("VG frame publication")
                .object_visibility,
            &[1],
        );
        let topology_version = scene.vg_buffer_version;
        let initial_cull_version = scene.vg_cull_signature_version;

        scene.set_group_visibility(groups, false);
        assert_eq!(scene.vg_cpu_visibility, [0]);
        assert_eq!(scene.vg_buffer_version, topology_version);
        assert_eq!(scene.vg_cull_signature_version, initial_cull_version + 1);

        // One hidden membership remains, so this canonical policy change does
        // not trigger an unnecessary pass projection upload.
        scene.show_group(GroupId::DEBUG);
        assert_eq!(scene.vg_cpu_visibility, [0]);
        assert_eq!(scene.vg_cull_signature_version, initial_cull_version + 1);

        scene.show_group(GroupId::DEFAULT);
        assert_eq!(scene.vg_cpu_visibility, [1]);
        assert_eq!(scene.vg_buffer_version, topology_version);
        assert_eq!(scene.vg_cull_signature_version, initial_cull_version + 2);
        assert_eq!(
            scene
                .authority
                .subsystem::<SceneVisibilityState>()
                .expect("registered visibility authority")
                .hidden_groups(),
            GroupMask::NONE.0,
        );
    }
}
