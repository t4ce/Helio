//! Multi-material (sectioned) mesh support.
//!
//! Mirrors Unreal Engine's Static Mesh section model: a single asset with one shared
//! vertex buffer and N index ranges, each drawn with an independent material.
//! All sections of a placed instance share the same world-space transform.
//!
//! # GPU architecture
//!
//! - **One vertex buffer region** per sectioned mesh asset (uploaded once).
//! - **N index buffer regions**, one per section (one draw call each).
//! - **N `ObjectId`s** per placed instance, related by one SceneDB
//!   [`SectionedInstanceId`] entity.
//!
//! Moving/removing a [`SectionedInstanceId`] updates all N draw calls atomically, so
//! the object behaves as a single unit from the caller's perspective.  The scene also
//! maintains a SceneDB-registered query index so the picker can automatically
//! resolve a section hit back to its parent [`SectionedInstanceId`].

use std::collections::HashMap;

use glam::Mat4;
use helio_scenedb::{CpuOnlyComponent, Entity, SceneMaterial, SceneObject, Subsystem};

use crate::groups::GroupMask;
use crate::handles::{
    entity_from_handle, handle_from_entity, MaterialId, MeshId, MultiMeshId, ObjectId,
    SectionedInstanceId,
};
use crate::mesh::{MultiMeshRecord, SectionedMeshUpload};
use crate::scene::types::ObjectDescriptor;

use super::errors::{invalid, Result, SceneError};

// ─── Internal instance record ─────────────────────────────────────────────────

/// Internal record for a placed sectioned mesh instance.
///
/// Stored as a CPU-only SceneDB component. The Vec is canonical relationship
/// data; it is not mirrored because render passes consume the resulting
/// SceneObject rows rather than the aggregate itself.
pub(crate) struct SectionedInstanceRecord {
    /// One draw-call `ObjectId` per material section (same order as the `materials`
    /// slice passed to [`Scene::insert_sectioned_object`]).
    pub section_objects: Vec<Entity>,
    /// Asset handle — used when decrementing the ref-count on removal.
    pub multi_mesh: Entity,
}

impl CpuOnlyComponent for MultiMeshRecord {}
impl CpuOnlyComponent for SectionedInstanceRecord {}

/// SceneDB-registered query index for the reverse section relationship.
///
/// The authoritative relationship remains [`SectionedInstanceRecord`]; this
/// index only makes picking and editor lookup O(1), just like SceneIndices
/// does for tags.
#[derive(Default)]
pub(crate) struct SectionRelations {
    instance_by_object: HashMap<Entity, Entity>,
}

impl SectionRelations {
    fn insert(&mut self, object: Entity, instance: Entity) {
        self.instance_by_object.insert(object, instance);
    }

    fn remove(&mut self, object: Entity, instance: Entity) {
        if self.instance_by_object.get(&object) == Some(&instance) {
            self.instance_by_object.remove(&object);
        }
    }

    pub(crate) fn instance_for_object(&self, object: Entity) -> Option<Entity> {
        self.instance_by_object.get(&object).copied()
    }
}

impl Subsystem for SectionRelations {
    fn name(&self) -> &'static str {
        "helio.scene.section_relations"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ─── Scene methods ────────────────────────────────────────────────────────────

impl super::Scene {
    // ── Asset queries ─────────────────────────────────────────────────────────

    /// Return the per-section [`MeshId`]s for a previously uploaded sectioned mesh.
    ///
    /// Needed when registering the mesh geometry with [`crate::ScenePicker`] so
    /// that each section participates in BVH ray-picking.  The returned slice is
    /// in the same order as the `sections` array passed to [`insert_sectioned_mesh`].
    ///
    /// Returns `None` if the handle is stale or has already been removed.
    pub fn sectioned_section_mesh_ids(&self, id: MultiMeshId) -> Option<&[MeshId]> {
        self.authority
            .get::<MultiMeshRecord>(entity_from_handle(id))
            .map(|r| r.section_mesh_ids.as_slice())
    }

    // ── Asset upload / removal ────────────────────────────────────────────────

    /// Upload a multi-material mesh to the GPU.
    ///
    /// Vertices are pushed **once** into the shared vertex buffer.
    /// Each element of `upload.sections` is an independent index list that will be
    /// rendered as a separate draw call with its own material.
    ///
    /// Returns a [`MultiMeshId`] asset handle. The asset persists until
    /// [`remove_sectioned_mesh`](Self::remove_sectioned_mesh) is called.
    pub fn insert_sectioned_mesh(&mut self, upload: SectionedMeshUpload) -> MultiMeshId {
        let record = self.mesh_pool_mut().insert_sectioned(upload);
        handle_from_entity(self.authority.insert(record))
    }

    /// Remove a multi-material mesh asset.
    ///
    /// Fails if any live instances still reference this mesh.
    /// Its vertex/index ranges are returned to the geometry allocator.
    pub fn remove_sectioned_mesh(&mut self, id: MultiMeshId) -> Result<()> {
        let entity = entity_from_handle(id);
        let (ref_count, section_ids) = {
            let record = self
                .authority
                .get::<MultiMeshRecord>(entity)
                .ok_or_else(|| invalid("multi_mesh"))?;
            (record.ref_count, record.section_mesh_ids.clone())
        };
        if ref_count > 0 {
            return Err(SceneError::ResourceInUse {
                resource: "multi_mesh",
            });
        }

        // Each section owns exactly the asset's one retained reference at
        // this point. Validate all rows before mutating so removal is atomic
        // even if a caller separately placed one section as a normal object.
        for &mesh_id in &section_ids {
            let mesh = self.mesh_pool().get(mesh_id).ok_or_else(|| invalid("mesh"))?;
            if mesh.ref_count != 1 {
                return Err(SceneError::ResourceInUse {
                    resource: "sectioned mesh section",
                });
            }
        }
        for mesh_id in section_ids {
            self.mesh_pool_mut()
                .get_mut(mesh_id)
                .expect("validated section mesh")
                .ref_count = 0;
            self.mesh_pool_mut()
                .remove(mesh_id)
                .expect("validated section mesh remains live");
        }
        self.authority
            .remove::<MultiMeshRecord>(entity)
            .ok_or_else(|| invalid("multi_mesh"))?;
        debug_assert!(self.authority.despawn(entity));
        Ok(())
    }

    // ── Instance placement ────────────────────────────────────────────────────

    /// Place a multi-material mesh instance into the scene.
    ///
    /// Creates **one GPU draw call per section**, all sharing the same `transform`.
    /// The number of `materials` must exactly match the number of sections the
    /// mesh was uploaded with.
    ///
    /// Returns a [`SectionedInstanceId`] — a lightweight `Copy` handle that the
    /// scene stores internally.  Pass it to [`update_sectioned_object_transform`],
    /// [`remove_sectioned_object`], or the editor.  The picker automatically maps
    /// any section hit back to this handle.
    ///
    /// # Errors
    /// - `InvalidHandle` if `multi_mesh` is not a valid handle.
    /// - `InvalidHandle` if `materials.len()` ≠ section count.
    /// - `InvalidHandle` if any `MaterialId` in `materials` is invalid.
    pub fn insert_sectioned_object(
        &mut self,
        multi_mesh: MultiMeshId,
        materials: &[MaterialId],
        transform: Mat4,
        bounds: [f32; 4],
        movability: Option<libhelio::Movability>,
    ) -> Result<SectionedInstanceId> {
        let multi_mesh_entity = entity_from_handle(multi_mesh);
        // Snapshot the section mesh IDs so no SceneDB query borrow is held
        // while inserting the section objects.
        let section_mesh_ids = {
            let record = self
                .authority
                .get::<MultiMeshRecord>(multi_mesh_entity)
                .ok_or_else(|| invalid("multi_mesh"))?;
            if record.section_mesh_ids.len() != materials.len() {
                return Err(invalid("material count must match mesh section count"));
            }
            record.section_mesh_ids.clone()
        };

        if self
            .authority
            .get::<MultiMeshRecord>(multi_mesh_entity)
            .is_some_and(|record| record.ref_count == u32::MAX)
        {
            return Err(SceneError::InvalidOperation {
                reason: "sectioned mesh instance reference count overflow",
            });
        }
        for (&mesh_id, &material_id) in section_mesh_ids.iter().zip(materials.iter()) {
            self.mesh_pool().get(mesh_id).ok_or_else(|| invalid("mesh"))?;
            let material_entity = entity_from_handle(material_id);
            if self.authority.get::<SceneMaterial>(material_entity).is_none()
                || self.authority.gpu_row::<SceneMaterial>(material_entity).is_none()
            {
                return Err(invalid("material"));
            }
        }

        let mut section_objects: Vec<Entity> = Vec::with_capacity(section_mesh_ids.len());
        for (&mesh_id, &material_id) in section_mesh_ids.iter().zip(materials.iter()) {
            let object = match self.insert_object(ObjectDescriptor {
                mesh: mesh_id,
                material: material_id,
                transform,
                bounds,
                flags: 0b11, // casts + receives shadows
                groups: GroupMask::NONE,
                movability,
                user_tag: 0,
            }) {
                Ok(id) => entity_from_handle(id),
                Err(error) => {
                    for inserted in section_objects.drain(..) {
                        let _ = self.remove_object(handle_from_entity(inserted));
                    }
                    return Err(error);
                }
            };
            section_objects.push(object);
        }

        // Increment the asset ref-count.
        self.authority
            .edit_cpu::<MultiMeshRecord, _>(multi_mesh_entity, |record| record.ref_count += 1)
            .expect("validated multi-mesh component remains live");

        // Store the canonical relationship component and update its query index.
        let record = SectionedInstanceRecord {
            section_objects: section_objects.clone(),
            multi_mesh: multi_mesh_entity,
        };
        let instance_entity = self.authority.insert(record);
        let relations = self
            .authority
            .subsystem_mut::<SectionRelations>()
            .expect("SectionRelations is registered during Scene construction");
        for &object in &section_objects {
            relations.insert(object, instance_entity);
        }

        Ok(handle_from_entity(instance_entity))
    }

    /// Update the world transform of all sections in a placed instance.
    ///
    /// O(N) where N = section count (typically 2–8).
    pub fn update_sectioned_object_transform(
        &mut self,
        id: SectionedInstanceId,
        transform: Mat4,
    ) -> Result<()> {
        let section_objects = self
            .authority
            .get::<SectionedInstanceRecord>(entity_from_handle(id))
            .ok_or_else(|| invalid("sectioned_instance"))?
            .section_objects
            .clone();
        for object in section_objects {
            self.update_object_transform(handle_from_entity(object), transform)?;
        }
        Ok(())
    }

    /// Remove a placed sectioned mesh instance.
    ///
    /// Removes all GPU draw calls, cleans up the reverse map, and decrements the
    /// asset ref-count.  The [`MultiMeshId`] asset is unaffected.
    pub fn remove_sectioned_object(&mut self, id: SectionedInstanceId) -> Result<()> {
        let instance_entity = entity_from_handle(id);
        let (section_objects, multi_mesh) = {
            let record = self
                .authority
                .get::<SectionedInstanceRecord>(instance_entity)
                .ok_or_else(|| invalid("sectioned_instance"))?;
            (record.section_objects.clone(), record.multi_mesh)
        };
        if section_objects
            .iter()
            .any(|&object| self.authority.get::<SceneObject>(object).is_none())
        {
            return Err(SceneError::InvalidOperation {
                reason: "sectioned instance contains a missing object component",
            });
        }

        {
            let relations = self
                .authority
                .subsystem_mut::<SectionRelations>()
                .expect("SectionRelations is registered during Scene construction");
            for &object in &section_objects {
                relations.remove(object, instance_entity);
            }
        }
        for object in section_objects {
            self.remove_object(handle_from_entity(object))?;
        }

        self.authority
            .edit_cpu::<MultiMeshRecord, _>(multi_mesh, |record| {
                record.ref_count = record.ref_count.saturating_sub(1);
            })
            .ok_or(SceneError::InvalidOperation {
                reason: "sectioned instance references a missing multi-mesh component",
            })?;
        self.authority
            .remove::<SectionedInstanceRecord>(instance_entity)
            .ok_or_else(|| invalid("sectioned_instance"))?;
        debug_assert!(self.authority.despawn(instance_entity));
        self.detach_actors_for_target(
            crate::scene::actor::SceneActorId::SectionedObject(id),
        );
        Ok(())
    }

    // ── Instance queries (used by picker + editor) ────────────────────────────

    /// Return the `SectionedInstanceId` that owns the given section `ObjectId`,
    /// or `None` if the object is not part of any sectioned instance.
    pub fn section_instance_for_object(&self, id: ObjectId) -> Option<SectionedInstanceId> {
        let instance = self
            .authority
            .subsystem::<SectionRelations>()?
            .instance_for_object(entity_from_handle(id))?;
        self.authority
            .get::<SectionedInstanceRecord>(instance)
            .map(|_| handle_from_entity(instance))
    }

    /// Return the world transform of a sectioned instance (taken from section 0).
    ///
    /// Returns `None` if the handle is stale.
    pub fn get_sectioned_instance_transform(&self, id: SectionedInstanceId) -> Option<Mat4> {
        let first = *self
            .authority
            .get::<SectionedInstanceRecord>(entity_from_handle(id))?
            .section_objects
            .first()?;
        self.get_object_transform(handle_from_entity(first)).ok()
    }

    /// Return the bounding sphere `[cx, cy, cz, radius]` of a sectioned instance
    /// (taken from section 0 — all sections share the same bounds).
    ///
    /// Returns `None` if the handle is stale.
    pub fn get_sectioned_instance_bounds(&self, id: SectionedInstanceId) -> Option<[f32; 4]> {
        let first = *self
            .authority
            .get::<SectionedInstanceRecord>(entity_from_handle(id))?
            .section_objects
            .first()?;
        self.get_object_bounds(handle_from_entity(first)).ok()
    }

    /// Duplicate a placed sectioned mesh instance, preserving its transform and materials.
    ///
    /// Returns the [`SectionedInstanceId`] of the new copy, or an error if the
    /// source handle is stale.
    pub fn duplicate_sectioned_object(&mut self, id: SectionedInstanceId) -> Result<SectionedInstanceId> {
        // Snapshot what we need before any mutable borrows.
        let (multi_mesh, section_objects) = {
            let rec = self
                .authority
                .get::<SectionedInstanceRecord>(entity_from_handle(id))
                .ok_or_else(|| invalid("sectioned_instance"))?;
            (rec.multi_mesh, rec.section_objects.clone())
        };

        // Collect per-section descriptors (material + bounds) from section 0 for bounds.
        let mut materials: Vec<MaterialId> = Vec::with_capacity(section_objects.len());
        let mut bounds = [0.0f32; 4];
        let mut transform = Mat4::IDENTITY;
        let mut movability = None;
        for (i, &object) in section_objects.iter().enumerate() {
            let desc = self.get_object_descriptor(handle_from_entity(object))?;
            materials.push(desc.material);
            if i == 0 {
                bounds     = desc.bounds;
                transform  = desc.transform;
                movability = desc.movability;
            }
        }

        self.insert_sectioned_object(
            handle_from_entity(multi_mesh),
            &materials,
            transform,
            bounds,
            movability,
        )
    }
}
