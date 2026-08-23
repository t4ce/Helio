//! Scene frame-lifecycle methods.
//!
//! This module contains methods that advance the scene through its frame lifecycle,
//! such as [`advance_frame`](Scene::advance_frame), [`set_render_size`](Scene::set_render_size),
//! and [`clear`](Scene::clear).

#[cfg(feature = "bake")]
use glam::Mat4;
use helio_scenedb::{
    SceneMaterial, SceneObject, ScenePlanarReflector, ScenePostProcessVolume,
    SceneReflectionCapture, SceneTexture, SceneSky, SceneVoxelVolume, SceneWaterHitbox,
    SceneWaterVolume,
};
use libhelio::sky::SkyContext;

use crate::handles::handle_from_entity;
#[cfg(feature = "bake")]
use crate::scene::helpers::{object_mesh, object_movability};
use crate::scene::multi_mesh::{SectionedInstanceRecord};
use crate::scene::portals::PortalRecord;
use crate::scene::sublevels::SublevelRecord;
use crate::mesh::MultiMeshRecord;
use crate::scene::Scene;
use crate::scene::SceneActorTrait;
use crate::scene::actor::{RetainedSceneActor, SceneActorId};

impl Scene {
    /// Remove every scene-content authored resource and resident asset.
    ///
    /// This is the efficient path for batch renderers that swap the entire scene
    /// between frames. Objects are removed first, then zero-reference meshes,
    /// materials, and textures are explicitly swept in dependency order, so no
    /// manual ID tracking is required by the caller.
    /// Global authored policy, including group visibility, is reset to its
    /// default so content loaded afterward cannot inherit state from this scene.
    /// Reusable asset-registry state intentionally survives: compiled Radiant
    /// graph sources remain registered, and SceneDB's automatic texture-asset
    /// key high-water mark is not reset, so a later automatic texture cannot
    /// reuse an identity from the cleared content.
    ///
    /// Calls `flush()` before returning so GPU buffers are synchronised.
    pub fn clear(&mut self) {
        // Coordinate-space owners must be removed through their feature APIs:
        // sublevels first retag member objects to row 0, and both paths reset
        // temporal history before SceneDB releases a reusable component row.
        let sublevel_ids: Vec<_> = self
            .authority
            .query::<SublevelRecord>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        for id in sublevel_ids {
            let _ = self.remove_sublevel(id);
        }
        let portal_ids: Vec<_> = self
            .authority
            .query::<PortalRecord>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        for id in portal_ids {
            let _ = self.remove_portal(id);
        }

        // Sectioned instances are atomic aggregate objects. Remove them first
        // so their reverse relation index and asset reference counts cannot
        // outlive a generic per-object sweep.
        let sectioned_instance_ids: Vec<_> = self
            .authority
            .query::<SectionedInstanceRecord>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        for id in sectioned_instance_ids {
            let _ = self.remove_sectioned_object(id);
        }

        // Collect all remaining handles before mutating — iterators are
        // invalidated by removal.
        let object_ids: Vec<_> = self
            .authority
            .query::<SceneObject>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let light_ids: Vec<_> = self.iter_lights().map(|(id, _, _)| id).collect();
        let decal_ids: Vec<_> = self.iter_decals().map(|(id, _, _)| id).collect();
        let water_volume_ids: Vec<_> = self
            .authority
            .query::<SceneWaterVolume>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let water_hitbox_ids: Vec<_> = self
            .authority
            .query::<SceneWaterHitbox>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let post_process_volume_ids: Vec<_> = self
            .authority
            .query::<ScenePostProcessVolume>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let reflection_capture_ids: Vec<_> = self
            .authority
            .query::<SceneReflectionCapture>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let planar_reflector_ids: Vec<_> = self
            .authority
            .query::<ScenePlanarReflector>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let voxel_volume_ids: Vec<_> = self
            .authority
            .query::<SceneVoxelVolume>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        let virtual_object_ids: Vec<_> = self
            .virtual_geometry()
            .objects
            .iter()
            .map(|(id, _)| id)
            .collect();
        let foliage_type_ids = self.foliage_type_projection.ids().to_vec();
        let foliage_layer_ids = self.foliage_layer_projection.ids().to_vec();
        let foliage_interactor_ids = self.foliage_interactor_projection.ids().to_vec();

        // Objects first: the cascade frees meshes, materials, and textures.
        for id in object_ids {
            let _ = self.remove_object(id);
        }

        for id in light_ids {
            let _ = self.remove_light(id);
        }

        for id in decal_ids {
            let _ = self.remove_decal(id);
        }
        for id in water_volume_ids {
            let _ = self.remove_water_volume(id);
        }
        for id in water_hitbox_ids {
            let _ = self.remove_water_hitbox(id);
        }
        for id in post_process_volume_ids {
            let _ = self.remove_post_process_volume(id);
        }
        for id in reflection_capture_ids {
            let _ = self.remove_reflection_capture(id);
        }
        for id in planar_reflector_ids {
            let _ = self.remove_planar_reflector(id);
        }
        for id in voxel_volume_ids {
            let _ = self.remove_voxel_volume(id);
        }

        // These derived render categories also retain canonical materials.
        // Release them before sweeping standalone material entities.
        for id in virtual_object_ids {
            let _ = self.remove_virtual_object(id);
        }
        for id in foliage_layer_ids {
            let _ = self.remove_foliage_layer(id);
        }
        for id in foliage_interactor_ids {
            let _ = self.remove_foliage_interactor(id);
        }
        for id in foliage_type_ids {
            let _ = self.remove_foliage_type(id);
        }

        // Sprites retain atlas-residency references. Remove every canonical
        // sprite first, then retire all atlas identities/physical layers. This
        // ordering makes `clear()` a true scene reset and keeps the residency
        // subsystem's reference accounting transactional.
        self.try_clear_sprites()
            .expect("Scene::clear sprite reference teardown must remain valid");
        self.try_clear_sprite_atlas_layers()
            .expect("Scene::clear sprite atlas teardown must remain valid");

        // Virtual mesh assets may be authored without a placed object. After
        // the object sweep, remove every generation-stable asset whose
        // canonical refcount reached zero; this also releases its Once-handed
        // off MeshPool LOD allocations through remove_virtual_mesh.
        let virtual_mesh_ids: Vec<_> = self
            .virtual_geometry()
            .meshes
            .iter()
            .filter_map(|(&id, record)| (record.ref_count == 0).then_some(id))
            .collect();
        for id in virtual_mesh_ids {
            let _ = self.remove_virtual_mesh(id);
        }

        // Sectioned mesh assets retain their shared section geometry even
        // with no placed instance. Release those explicit asset references
        // after all ordinary/derived objects have gone away.
        let multi_mesh_ids: Vec<_> = self
            .authority
            .query::<MultiMeshRecord>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        for id in multi_mesh_ids {
            let _ = self.remove_sectioned_mesh(id);
        }

        // Ordinary meshes may be authored before any object references them.
        // Object removal therefore cannot discover these standalone assets.
        // Sectioned/virtual owners have already released their references
        // above, so every remaining zero-ref record can now be removed through
        // the normal allocator/residency teardown path.
        let mesh_ids: Vec<_> = self
            .mesh_pool()
            .iter()
            .filter_map(|(id, record)| (record.ref_count == 0).then_some(id))
            .collect();
        for id in mesh_ids {
            let _ = self.remove_mesh(id);
        }

        // Standalone materials/textures are legal: they may have been authored
        // ahead of any object, so object removal alone cannot guarantee their
        // cleanup. Remove them in dependency order while preserving SceneDB's
        // generation/refcount validation and residency-slot release rules.
        let material_ids: Vec<_> = self
            .authority
            .query::<SceneMaterial>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        for id in material_ids {
            let _ = self.remove_material(id);
        }

        let texture_ids: Vec<_> = self
            .authority
            .query::<SceneTexture>()
            .map(|(entity, _)| handle_from_entity(entity))
            .collect();
        for id in texture_ids {
            let _ = self.remove_texture(id);
        }

        // Execution actors and their transient identities never survive a
        // canonical scene reset. An actor currently being ticked is held in
        // advance_frame's local snapshot; clearing this map makes that actor
        // and every not-yet-ticked snapshot entry retire safely afterward.
        self.clear_extension_data();
        self.custom_actors.clear();
        self.custom_actor_targets.clear();
        {
            // SDF edits and terrain configuration are authored scene state in
            // the registered SceneDB subsystem. Leaving either behind would
            // make `clear()` retain canonical geometry even though every
            // entity-backed scene component had been removed.
            let sdf = self.sdf_authority_mut();
            sdf.clear();
            sdf.set_terrain(None)
                .expect("disabling canonical SDF terrain is always valid");
        }
        self.clear_planet_frames();
        self.clear_presentation();
        // Global visibility policy is authored scene state too. A clear must
        // not make newly loaded objects inherit hidden groups from the prior
        // scene, while the GPU visibility buffer remains a derived projection.
        self.clear_group_visibility();
        self.authority
            .edit_cpu::<SceneSky, _>(self.sky_entity, |sky| sky.context = None)
            .expect("canonical sky entity is permanent for the Scene lifetime");

        self.flush();
    }

    /// Insert a custom trait-based scene actor.
    ///
    /// This can be e.g. `SceneActor::Sky`, `MeshActor`, `LightActor`, or other custom actors.
    /// Descriptor-only actors hand their payload to SceneDB and are discarded;
    /// only actors whose [`SceneActorTrait::retain_after_attach`] returns true
    /// remain in Helio as per-frame behavior containers.
    pub fn insert_actor<A: SceneActorTrait + 'static>(&mut self, mut actor: A) -> SceneActorId {
        actor.on_attach(self);
        if let Some(context) = actor.sky_context() {
            self.claim_sky_context(context);
        }

        let target = actor.inserted_id();
        if !actor.retain_after_attach() {
            return target;
        }

        let execution_id = if target != SceneActorId::None
            && !self.custom_actor_targets.contains_key(&target)
        {
            target
        } else {
            self.allocate_custom_actor_identity()
        };
        self.custom_actor_targets.insert(execution_id, target);
        self.custom_actors.push(RetainedSceneActor {
            id: execution_id,
            actor: Box::new(actor),
        });

        if target == SceneActorId::None {
            execution_id
        } else {
            target
        }
    }

    fn allocate_custom_actor_identity(&mut self) -> SceneActorId {
        SceneActorId::Custom(self.spawn_extension_entity())
    }

    fn claim_sky_context(&mut self, context: SkyContext) -> bool {
        self.authority
            .edit_cpu::<SceneSky, _>(self.sky_entity, |sky| {
                if sky.context.is_some() {
                    false
                } else {
                    sky.context = Some(context);
                    true
                }
            })
            .expect("canonical sky entity is permanent for the Scene lifetime")
    }

    /// Replace the winning actor's canonical SceneDB sky payload.
    ///
    /// `sky_context()` on a custom actor is sampled only during insertion. A
    /// retained actor with dynamic sky parameters must call this from
    /// `on_tick`. Returns `false` when no sky actor has claimed the scene yet.
    pub fn update_sky_context(&mut self, context: SkyContext) -> bool {
        self.authority
            .edit_cpu::<SceneSky, _>(self.sky_entity, |sky| {
                if sky.context.is_none() {
                    false
                } else {
                    sky.context = Some(context);
                    true
                }
            })
            .expect("canonical sky entity is permanent for the Scene lifetime")
    }

    /// Returns the canonical SceneDB sky context for the current frame.
    pub fn sky_context(&self) -> SkyContext {
        self.authority
            .get::<SceneSky>(self.sky_entity)
            .and_then(|sky| sky.context)
            .unwrap_or_default()
    }

    /// Remove an actor and, when it owns a typed scene target, remove that
    /// canonical target through its normal lifecycle API as well.
    pub fn remove_actor(&mut self, id: SceneActorId) -> crate::scene::Result<()> {
        match id {
            SceneActorId::None => return Err(crate::scene::errors::invalid("actor")),
            SceneActorId::Custom(_) => {
                return if self.detach_actors_for_target(id) {
                    Ok(())
                } else {
                    Err(crate::scene::errors::invalid("actor"))
                };
            }
            SceneActorId::Decal(id) => {
                if !self.remove_decal(id) {
                    return Err(crate::scene::errors::invalid("decal"));
                }
            }
            SceneActorId::Mesh(id) => self.remove_mesh(id)?,
            SceneActorId::Light(id) => self.remove_light(id)?,
            SceneActorId::ReflectionCapture(id) => {
                if !self.remove_reflection_capture(id) {
                    return Err(crate::scene::errors::invalid("reflection capture"));
                }
            }
            SceneActorId::VirtualMesh(id) => self.remove_virtual_mesh(id)?,
            SceneActorId::VirtualObject(id) => self.remove_virtual_object(id)?,
            SceneActorId::Object(id) => self.remove_object(id)?,
            SceneActorId::SectionedObject(id) => self.remove_sectioned_object(id)?,
            SceneActorId::WaterVolume(id) => self.remove_water_volume(id)?,
            SceneActorId::WaterHitbox(id) => self.remove_water_hitbox(id)?,
            SceneActorId::PostProcessVolume(id) => self.remove_post_process_volume(id)?,
        }
        self.detach_actors_for_target(id);
        Ok(())
    }

    /// Detach execution wrappers associated with a removed canonical target.
    /// This remains safe while `advance_frame` owns its tick snapshot because
    /// the identity map, rather than the vector borrow, decides survival.
    pub(in crate::scene) fn detach_actors_for_target(&mut self, target: SceneActorId) -> bool {
        let ids: Vec<_> = self
            .custom_actor_targets
            .iter()
            .filter_map(|(&id, &actor_target)| {
                (id == target || actor_target == target).then_some(id)
            })
            .collect();
        if ids.is_empty() {
            return false;
        }
        for id in &ids {
            self.custom_actor_targets.remove(id);
        }
        for entity in ids.iter().filter_map(|id| id.as_custom()) {
            self.authority.despawn(entity.0);
        }
        self.custom_actors
            .retain(|entry| self.custom_actor_targets.contains_key(&entry.id));
        true
    }

    /// Set the render target size for camera calculations.
    ///
    /// Updates the internal width/height used for aspect ratio calculations
    /// and viewport-dependent effects.
    ///
    /// # Parameters
    /// - `width`: Render target width in pixels
    /// - `height`: Render target height in pixels
    ///
    /// # Example
    /// ```ignore
    /// scene.set_render_size(1920, 1080);
    /// ```
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.gpu_scene.width = width;
        self.gpu_scene.height = height;
    }

    /// Advance the frame counter.
    ///
    /// Increments the internal frame counter used for temporal effects and shader logic.
    /// `Renderer` calls this automatically after a successful frame. Custom
    /// executors must call it exactly once after rendering, not after each
    /// upload-only [`Scene::flush`].
    ///
    /// # Frame Counter Uses
    /// - Temporal anti-aliasing (TAA) - jitter pattern sequencing
    /// - Temporal dithering - noise pattern variation
    /// - Shader debugging - frame-dependent visualization
    ///
    pub fn advance_frame(&mut self) {
        // Own the current tick set outside Scene while callbacks receive
        // `&mut Scene`. Actors inserted by a callback land in custom_actors
        // and begin ticking next frame; removals erase their identity from the
        // map, so current or not-yet-ticked snapshot entries retire safely.
        let actors = std::mem::take(&mut self.custom_actors);
        let mut survivors = Vec::with_capacity(actors.len());
        for mut entry in actors {
            if !self.custom_actor_targets.contains_key(&entry.id) {
                continue;
            }
            if entry.actor.is_active() {
                entry.actor.on_tick(self);
            }
            if self.custom_actor_targets.contains_key(&entry.id) {
                survivors.push(entry);
            }
        }
        survivors.append(&mut self.custom_actors);
        self.custom_actors = survivors;

        self.gpu_scene.advance_frame();
    }

    /// Build a [`SceneGeometry`](helio_bake::SceneGeometry) from all static objects and lights.
    ///
    /// Automatically extracts all objects and lights marked as Static or Stationary
    /// (i.e., not Movable) and converts them to bake-ready geometry. This eliminates
    /// the need to manually duplicate scene information for baking.
    ///
    /// # Returns
    /// A `SceneGeometry` containing:
    /// - All static object meshes with their world transforms applied
    /// - All static lights configured for baking
    ///
    /// # Example
    /// ```ignore
    /// // After building your scene normally...
    /// let bake_scene = scene.build_static_bake_scene();
    /// renderer.configure_bake(BakeRequest {
    ///     scene: bake_scene,
    ///     config: BakeConfig::fast("my_scene"),
    /// });
    /// ```
    #[cfg(feature = "bake")]
    pub fn build_static_bake_scene(&mut self) -> helio_bake::SceneGeometry {
        use helio_bake::{LightSource, LightSourceKind, SceneGeometry};
        use libhelio::{LightType, Movability};

        let mut bake_scene = SceneGeometry::new();
        let mut static_object_count = 0;
        let mut static_light_count = 0;

        let objects: Vec<_> = self
            .authority
            .query::<SceneObject>()
            .map(|(_, object)| *object)
            .collect();
        for object_record in &objects {

            // Skip movable objects - only bake static and stationary geometry
            if object_movability(object_record) == Movability::Movable {
                continue;
            }

            // Extract mesh data from the pool
            let mesh_id = object_mesh(object_record);
            let Some(mesh_upload) = self.mesh_pool().extract_mesh_data(mesh_id) else {
                continue;
            };

            // Convert to bake mesh with world transform applied
            // Pass mesh slot to generate deterministic UUID for lightmap region mapping
            let local_transform = Mat4::from_cols_array(&object_record.spatial.model);
            let coordinate_space = libhelio::coordinate_space(object_record.spatial.flags);
            let transform = if coordinate_space == 0 {
                local_transform
            } else {
                Mat4::from_cols_array(
                    &self
                        .gpu_scene
                        .coordinate_space_history
                        .slot(coordinate_space),
                )
                    * local_transform
            };
            let mesh_slot = mesh_id.slot();
            let bake_mesh = crate::mesh_upload_to_bake(&mesh_upload, transform, Some(mesh_slot));
            bake_scene.add_mesh(bake_mesh);
            static_object_count += 1;
        }

        // Extract all static lights
        for (_, gpu_light, _) in self.iter_lights() {
            // Include ALL lights in the bake regardless of movability.
            // Lights default to Movable even for static scenes; filtering them out
            // would result in a zero-light bake and an all-black lightmap.
            // If a user wants a light to be purely dynamic (never baked), they
            // should set bake_enabled = false on the BakeMesh's LightSource.
            let light_type = gpu_light.light_type;

            // Determine light kind from type
            let kind = if light_type == LightType::Directional as u32 {
                LightSourceKind::Directional {
                    direction: [
                        gpu_light.direction_outer[0],
                        gpu_light.direction_outer[1],
                        gpu_light.direction_outer[2],
                    ],
                }
            } else if light_type == LightType::Point as u32 {
                LightSourceKind::Point {
                    position: [
                        gpu_light.position_range[0],
                        gpu_light.position_range[1],
                        gpu_light.position_range[2],
                    ],
                    range: gpu_light.position_range[3],
                }
            } else if light_type == LightType::Spot as u32 {
                LightSourceKind::Spot {
                    position: [
                        gpu_light.position_range[0],
                        gpu_light.position_range[1],
                        gpu_light.position_range[2],
                    ],
                    direction: [
                        gpu_light.direction_outer[0],
                        gpu_light.direction_outer[1],
                        gpu_light.direction_outer[2],
                    ],
                    range: gpu_light.position_range[3],
                    inner_angle: gpu_light.inner_angle.acos(),
                    outer_angle: gpu_light.direction_outer[3].acos(),
                }
            } else {
                continue; // Unknown light type
            };

            bake_scene.add_light(LightSource {
                kind,
                color: [
                    gpu_light.color_intensity[0],
                    gpu_light.color_intensity[1],
                    gpu_light.color_intensity[2],
                ],
                intensity: gpu_light.color_intensity[3],
                bake_enabled: true,
                casts_shadows: gpu_light.shadow_index != u32::MAX,
            });
            static_light_count += 1;
        }

        // ── Transform lightmap UVs into atlas space ────────────────────────────
        //
        // Nebula's `build_atlas_regions` assigns each mesh an equal-area cell in
        // the atlas using a ceil(sqrt(N)) × ceil(sqrt(N)) grid.  The bake WGSL
        // shader at each texel searches ALL mesh triangles to find which triangle
        // contains that atlas-space `lm_uv`.  For correctness, vertex `lm_uv`
        // values must therefore be in ATLAS UV space, NOT in per-mesh [0,1]² UV
        // space.
        //
        // Without this transform every mesh's UV0 covers [0,1]², so for every
        // texel all N meshes' triangles match — mesh 0 always wins (listed first),
        // its lighting bleeds into every other mesh's atlas cell, and meshes 1…N-1
        // all show mesh 0's lighting at runtime.  Three-way correctness chain:
        //   bake:    `lm_uv_atlas = uv_offset + UV0 * uv_scale`  → unique range per mesh
        //   runtime: `atlas_uv   = uv_offset + UV0 * uv_scale`   → same atlas address
        //   result:  runtime UV  == bake UV                        → correct texel lookup
        let n = bake_scene.meshes.len();
        if n > 1 {
            let cols = (n as f64).sqrt().ceil() as u32;
            let rows = (n as u32).div_ceil(cols);
            let cell_w = 1.0_f32 / cols as f32;
            let cell_h = 1.0_f32 / rows as f32;
            for (i, mesh) in bake_scene.meshes.iter_mut().enumerate() {
                let col = (i as u32) % cols;
                let row = (i as u32) / cols;
                let uo = col as f32 * cell_w;
                let vo = row as f32 * cell_h;
                if let Some(uvs) = mesh.lightmap_uvs.as_mut() {
                    for uv in uvs.iter_mut() {
                        uv[0] = uo + uv[0] * cell_w;
                        uv[1] = vo + uv[1] * cell_h;
                    }
                }
            }
            log::debug!(
                "[helio-bake] Transformed lightmap UVs to atlas space: {} meshes → {}×{} grid ({:.4}×{:.4} cells)",
                n, cols, rows, cell_w, cell_h
            );
        }

        log::info!(
            "[helio-bake] Auto-extracted {} static/stationary objects and {} lights for baking",
            static_object_count,
            static_light_count
        );

        // Clear the invalidation flag - scene is now synced with bake data
        self.bake_invalidated = false;

        bake_scene
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::handles::{entity_from_handle, handle_from_entity, LightId};
    use crate::mesh::{MeshUpload, PackedVertex};
    use crate::scene::actor::VirtualMeshActor;
    use crate::{
        BillboardInstance, BooleanOp, FoliageTypeDescriptor, GroupId, PlanetFrameUniform,
        PlanetId, PlanetPosition, PortalDescriptor, SceneActor, SceneError, SdfEdit,
        SdfShapeParams, SdfShapeType, SublevelDescriptor, TerrainConfig,
    };
    use glam::{Mat4, Vec2, Vec3};
    use helio_portal_core::PortalPose;
    use helio_scenedb::SceneCoordinateSpace;

    use super::*;
    use libhelio::{SkyActor, VolumetricClouds};

    fn create_test_device() -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
        crate::test_support::test_gpu().expect("No test GPU adapter found")
    }

    struct PassiveDynamicSky {
        context: SkyContext,
        next_context: SkyContext,
    }

    impl SceneActorTrait for PassiveDynamicSky {
        fn sky_context(&self) -> Option<SkyContext> {
            Some(self.context)
        }

        fn on_tick(&mut self, _scene: &mut Scene) {
            // Mutating only the boxed behavior payload is intentionally not a
            // canonical scene edit. Dynamic skies must call the Scene API.
            self.context = self.next_context;
        }
    }

    struct OrderedActor {
        label: u32,
        spawn_label: Option<u32>,
        log: Arc<Mutex<Vec<u32>>>,
    }

    impl SceneActorTrait for OrderedActor {
        fn on_tick(&mut self, scene: &mut Scene) {
            self.log.lock().unwrap().push(self.label);
            if let Some(label) = self.spawn_label.take() {
                scene.insert_actor(Self {
                    label,
                    spawn_label: None,
                    log: Arc::clone(&self.log),
                });
            }
        }
    }

    struct TargetLightActor {
        id: Option<LightId>,
        ticks: Arc<AtomicUsize>,
    }

    impl SceneActorTrait for TargetLightActor {
        fn on_attach(&mut self, scene: &mut Scene) {
            self.id = Some(scene.insert_light(helio_core::GpuLight::default()));
        }

        fn on_tick(&mut self, _scene: &mut Scene) {
            self.ticks.fetch_add(1, Ordering::Relaxed);
        }

        fn inserted_id(&self) -> SceneActorId {
            self.id
                .map(SceneActorId::Light)
                .unwrap_or(SceneActorId::None)
        }
    }

    #[test]
    fn test_sky_actor_detection_default() {
        let (device, queue) = create_test_device();
        let scene = Scene::new(device, queue);

        let sky_ctx = scene.sky_context();
        assert!(!sky_ctx.has_sky, "Default scene should have no sky");
        assert!(sky_ctx.clouds.is_none(), "Default scene should have no clouds");
    }

    #[test]
    fn test_sky_actor_detection_with_clouds() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // Insert sky actor with clouds
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new()
                .with_sky_color([0.5, 0.7, 1.0])
                .with_clouds(VolumetricClouds {
                    coverage: 0.6,
                    density: 0.8,
                    ..Default::default()
                })
        ));

        assert!(
            scene.custom_actors.is_empty(),
            "built-in Sky is snapshotted into SceneDB, not retained as an actor payload"
        );
        assert!(scene.authority.get::<SceneSky>(scene.sky_entity).unwrap().context.is_some());

        let sky_ctx = scene.sky_context();
        assert!(sky_ctx.has_sky, "Sky actor should be detected");
        assert!(sky_ctx.clouds.is_some(), "Cloud settings should be detected");

        if let Some(clouds) = sky_ctx.clouds {
            assert!((clouds.coverage - 0.6).abs() < 0.01, "Coverage should match");
            assert!((clouds.density - 0.8).abs() < 0.01, "Density should match");
        }
    }

    #[test]
    fn test_multiple_sky_actors_first_wins() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // Insert first sky actor
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new().with_sky_color([1.0, 0.0, 0.0])
        ));

        // Insert second sky actor (should be ignored)
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new().with_sky_color([0.0, 1.0, 0.0])
        ));

        let sky_ctx = scene.sky_context();
        // First actor wins
        assert!((sky_ctx.sky_color[0] - 1.0).abs() < 0.01, "Should use first actor's color");
    }

    #[test]
    fn dynamic_sky_changes_require_explicit_canonical_mutation() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let initial = SkyActor::new().with_sky_color([1.0, 0.0, 0.0]).context();
        let next = SkyActor::new().with_sky_color([0.0, 1.0, 0.0]).context();

        scene.insert_actor(PassiveDynamicSky {
            context: initial,
            next_context: next,
        });
        scene.advance_frame();
        assert_eq!(scene.sky_context().sky_color, initial.sky_color);
        assert!(scene.update_sky_context(next));
        assert_eq!(scene.sky_context().sky_color, next.sky_color);

        scene.clear();
        assert!(!scene.sky_context().has_sky);
        assert!(!scene.update_sky_context(next));
    }

    #[test]
    fn actor_tick_is_ordered_and_insertions_start_next_frame() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let log = Arc::new(Mutex::new(Vec::new()));
        let first = scene.insert_actor(OrderedActor {
            label: 1,
            spawn_label: Some(2),
            log: Arc::clone(&log),
        });
        assert!(matches!(first, SceneActorId::Custom(_)));
        let first_entity = first.as_custom().expect("targetless actor extension identity");
        assert!(scene.scene_data().is_alive(first_entity));

        scene.advance_frame();
        assert_eq!(&*log.lock().unwrap(), &[1]);
        scene.advance_frame();
        assert_eq!(&*log.lock().unwrap(), &[1, 1, 2]);

        scene.remove_actor(first).unwrap();
        assert!(!scene.scene_data().is_alive(first_entity));
        scene.advance_frame();
        assert_eq!(
            &*log.lock().unwrap(),
            &[1, 1, 2, 2],
            "the surviving child keeps stable insertion order"
        );
    }

    #[test]
    fn direct_target_removal_retires_its_execution_actor() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let ticks = Arc::new(AtomicUsize::new(0));
        let actor = scene.insert_actor(TargetLightActor {
            id: None,
            ticks: Arc::clone(&ticks),
        });
        let light = actor.as_light().expect("custom actor target");
        assert_eq!(scene.custom_actors.len(), 1);

        scene.remove_light(light).unwrap();
        assert!(scene.custom_actors.is_empty());
        scene.advance_frame();
        assert_eq!(ticks.load(Ordering::Relaxed), 0);

        let descriptor_only = scene.insert_actor(SceneActor::light(
            helio_core::GpuLight::default(),
        ));
        assert!(scene.custom_actors.is_empty());
        scene.remove_actor(descriptor_only).unwrap();
        assert!(scene.remove_actor(descriptor_only).is_err());
    }

    #[test]
    fn extension_entity_despawn_retires_its_execution_actor() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let log = Arc::new(Mutex::new(Vec::new()));
        let actor = scene.insert_actor(OrderedActor {
            label: 7,
            spawn_label: None,
            log: Arc::clone(&log),
        });
        let entity = actor.as_custom().expect("targetless actor identity");

        scene.scene_data_mut().despawn(entity).unwrap();
        assert!(scene.custom_actors.is_empty());
        assert!(!scene.scene_data().is_alive(entity));
        scene.advance_frame();
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn presentation_lists_are_scenedb_owned_and_generation_tracked() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        assert_eq!(scene.presentation_generations(), (0, 0));

        let billboard = BillboardInstance {
            world_pos: [1.0, 2.0, 3.0, 0.0],
            scale_flags: [0.5, 0.5, 0.0, 0.0],
            color: [1.0, 0.5, 0.25, 1.0],
        };
        scene.set_billboard_instances(&[billboard]);
        let billboard_generation = scene.presentation_generations().0;
        assert_eq!(scene.authored_billboards().len(), 1);
        scene.set_billboard_instances(&[billboard]);
        assert_eq!(scene.presentation_generations().0, billboard_generation);

        let emitter = libhelio::GpuCoronaEmitter::zeroed();
        scene.set_corona_emitters(&[emitter]).unwrap();
        let corona_generation = scene.presentation_generations().1;
        assert_eq!(scene.corona_emitters().len(), 1);
        scene.set_corona_emitters(&[emitter]).unwrap();
        assert_eq!(scene.presentation_generations().1, corona_generation);

        let too_many = vec![emitter; libhelio::CORONA_MAX_EMITTERS as usize + 1];
        assert!(matches!(
            scene.set_corona_emitters(&too_many),
            Err(SceneError::CoronaEmitterCapacityExceeded { .. })
        ));
        assert_eq!(scene.corona_emitters().len(), 1);
        assert_eq!(scene.presentation_generations().1, corona_generation);

        scene.clear();
        assert!(scene.authored_billboards().is_empty());
        assert!(scene.corona_emitters().is_empty());
        assert!(scene.presentation_generations().0 > billboard_generation);
        assert!(scene.presentation_generations().1 > corona_generation);
    }

    #[test]
    fn corona_clear_readd_without_render_advances_simulation_lifetime() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let emitter = libhelio::GpuCoronaEmitter::zeroed();
        scene.set_corona_emitters(&[emitter]).unwrap();
        let before_clear = scene.corona_reset_epoch();

        scene.clear();
        scene.set_corona_emitters(&[emitter]).unwrap();

        assert_eq!(scene.corona_emitters().len(), 1);
        assert_ne!(
            scene.corona_reset_epoch(),
            before_clear,
            "SceneDB must preserve a clear boundary even when the pass never observes the empty list"
        );
    }

    #[test]
    fn virtual_mesh_actor_consumes_the_upload_on_attach() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let vertex = |position| {
            PackedVertex::from_components(
                position,
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            )
        };
        let mut actor = VirtualMeshActor::new(crate::VirtualMeshUpload {
            vertices: vec![
                vertex([-1.0, -1.0, 0.0]),
                vertex([1.0, -1.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        });

        actor.on_attach(&mut scene);
        assert!(actor.upload.is_none());
        let id = actor.id().expect("virtual mesh handoff id");
        scene.remove_virtual_mesh(id).unwrap();
    }

    #[test]
    fn clear_sweeps_unplaced_virtual_mesh_assets_and_once_handoff_lods() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let vertex = |position| {
            PackedVertex::from_components(
                position,
                [0.0, 0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
            )
        };
        let virtual_mesh = scene.insert_virtual_mesh(crate::VirtualMeshUpload {
            vertices: vec![
                vertex([-1.0, -1.0, 0.0]),
                vertex([1.0, -1.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            indices: vec![0, 1, 2],
        });
        let lod_meshes = scene
            .virtual_geometry()
            .meshes
            .get(&virtual_mesh)
            .unwrap()
            .mesh_ids
            .clone();
        assert!(lod_meshes.iter().all(|&mesh| scene.mesh_pool().get(mesh).is_some()));

        scene.clear();

        assert!(!scene.virtual_geometry().meshes.contains_key(&virtual_mesh));
        assert!(lod_meshes.iter().all(|&mesh| scene.mesh_pool().get(mesh).is_none()));
    }

    #[test]
    fn clear_sweeps_unplaced_static_and_dynamic_mesh_assets() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let upload = || MeshUpload {
            vertices: vec![
                PackedVertex::from_components(
                    [-1.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    1.0,
                ),
                PackedVertex::from_components(
                    [1.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    1.0,
                ),
                PackedVertex::from_components(
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 1.0],
                    [1.0, 0.0, 0.0],
                    1.0,
                ),
            ],
            indices: vec![0, 1, 2],
        };
        let static_mesh = scene.insert_mesh(upload());
        let dynamic_mesh = scene.insert_dynamic_mesh(upload());
        assert!(scene.mesh_pool().get(static_mesh).is_some());
        assert!(scene.mesh_pool().get(dynamic_mesh).is_some());

        scene.clear();

        assert!(scene.mesh_pool().get(static_mesh).is_none());
        assert!(scene.mesh_pool().get(dynamic_mesh).is_none());
    }

    #[test]
    fn repeated_flush_does_not_advance_coordinate_history() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let initial = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let edited = Mat4::from_translation(Vec3::new(9.0, 8.0, 7.0));
        let sublevel = scene
            .add_sublevel(SublevelDescriptor {
                group: GroupId::DEFAULT,
                placement: initial,
            })
            .unwrap();
        let row = scene
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity_from_handle(sublevel))
            .unwrap();
        scene.flush();
        scene.advance_frame();

        scene.update_sublevel(sublevel, edited).unwrap();
        scene.flush();
        scene.flush();
        assert_eq!(
            scene.gpu_scene.coordinate_space_history.previous_slot(row),
            initial.to_cols_array(),
            "upload-only flushes must not collapse temporal velocity"
        );
        assert_eq!(
            scene.gpu_scene.coordinate_space_history.slot(row),
            edited.to_cols_array()
        );

        scene.advance_frame();
        assert_eq!(
            scene.gpu_scene.coordinate_space_history.previous_slot(row),
            edited.to_cols_array()
        );
    }

    #[test]
    fn coordinate_space_entities_publish_and_reuse_rows_without_stale_history() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        assert_eq!(scene.authority.gpu_live_count::<SceneCoordinateSpace>(), 1);

        let first_placement = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        let sublevel = scene
            .add_sublevel(SublevelDescriptor {
                group: GroupId::DEFAULT,
                placement: first_placement,
            })
            .expect("first sublevel");
        let sublevel_entity = entity_from_handle(sublevel);
        let sublevel_row = scene
            .authority
            .gpu_row::<SceneCoordinateSpace>(sublevel_entity)
            .expect("sublevel coordinate row");
        assert_eq!(sublevel_row, 1);
        assert!(scene
            .authority
            .get::<SublevelRecord>(sublevel_entity)
            .is_some());

        let a = PortalPose::from_look_at(Vec3::ZERO, Vec3::Z, Vec3::Y);
        let b = PortalPose::from_look_at(Vec3::new(0.0, 0.0, 8.0), Vec3::ZERO, Vec3::Y);
        let portal = scene
            .add_portal(PortalDescriptor {
                a,
                b,
                half_extent: Vec2::splat(2.0),
            })
            .expect("first portal");
        let portal_entity = entity_from_handle(portal);
        assert_eq!(
            scene
                .authority
                .gpu_row::<SceneCoordinateSpace>(portal_entity),
            Some(2)
        );
        assert!(scene.authority.get::<PortalRecord>(portal_entity).is_some());

        let far_a = PortalPose::from_look_at(
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 1.0),
            Vec3::Y,
        );
        let far_b = PortalPose::from_look_at(
            Vec3::new(100.0, 0.0, 8.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::Y,
        );
        let moving_portal = scene
            .add_portal(PortalDescriptor {
                a: far_a,
                b: far_b,
                half_extent: Vec2::splat(2.0),
            })
            .expect("second portal");
        let unrelated_chain_count = scene.gpu_scene.portal_chains.len();
        scene
            .update_portal_pose(
                moving_portal,
                b,
                PortalPose::from_look_at(
                    Vec3::new(0.0, 0.0, 16.0),
                    Vec3::new(0.0, 0.0, 8.0),
                    Vec3::Y,
                ),
            )
            .expect("pose-dependent chain rebuild");
        assert!(
            scene.gpu_scene.portal_chains.len() > unrelated_chain_count,
            "moving a portal into another's far opening must republish reachable chains"
        );

        scene.flush();
        let publication = scene
            .gpu_scene
            .canonical
            .coordinate_spaces
            .as_ref()
            .expect("SceneDB coordinate partner must be published");
        assert_eq!(publication.len(), 4);
        assert!(scene.gpu_scene.coordinate_space_buffer_epoch().is_some());

        scene.remove_sublevel(sublevel).unwrap();
        assert!(scene.sublevel_placement(sublevel).is_none());
        let replacement_placement = Mat4::from_translation(Vec3::new(-3.0, 2.0, 11.0));
        let replacement = scene
            .add_sublevel(SublevelDescriptor {
                group: GroupId::STATIC,
                placement: replacement_placement,
            })
            .expect("replacement sublevel");
        let replacement_row = scene
            .authority
            .gpu_row::<SceneCoordinateSpace>(entity_from_handle(replacement))
            .expect("replacement coordinate row");
        assert_eq!(replacement_row, sublevel_row, "released component row is reused");
        assert_eq!(
            scene.gpu_scene.coordinate_space_history.slot(replacement_row),
            replacement_placement.to_cols_array(),
            "a new row lifetime must not retain the prior owner's CPU history"
        );

        scene.clear();
        assert!(scene.portal_pair(portal).is_none());
        assert!(scene.sublevel_placement(replacement).is_none());
        assert_eq!(scene.authority.gpu_live_count::<SceneCoordinateSpace>(), 1);
        assert_eq!(
            scene
                .gpu_scene
                .canonical
                .coordinate_spaces
                .as_ref()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn sublevels_and_portals_share_the_shader_coordinate_capacity() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let mut sublevels = Vec::new();
        for index in 0..(libhelio::MAX_COORDINATE_SPACES - 2) {
            sublevels.push(
                scene
                    .add_sublevel(SublevelDescriptor {
                        group: GroupId::new(index as u8),
                        placement: Mat4::from_translation(Vec3::new(index as f32, 0.0, 0.0)),
                    })
                    .expect("coordinate row below the shared cap"),
            );
        }
        let a = PortalPose::from_look_at(Vec3::ZERO, Vec3::Z, Vec3::Y);
        let b = PortalPose::from_look_at(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y);
        let portal = scene
            .add_portal(PortalDescriptor {
                a,
                b,
                half_extent: Vec2::ONE,
            })
            .expect("last shared coordinate row");
        assert_eq!(
            scene.authority.gpu_live_count::<SceneCoordinateSpace>(),
            libhelio::MAX_COORDINATE_SPACES
        );
        assert!(matches!(
            scene.add_sublevel(SublevelDescriptor {
                group: GroupId::DEBUG,
                placement: Mat4::IDENTITY,
            }),
            Err(SceneError::CoordinateSpaceCapacityExceeded)
        ));

        scene.remove_portal(portal).unwrap();
        let recovered = scene
            .add_sublevel(SublevelDescriptor {
                group: GroupId::DEBUG,
                placement: Mat4::IDENTITY,
            })
            .expect("removal returns one shared row to SceneDB's allocator");
        assert!(scene.sublevel_placement(recovered).is_some());
        assert_eq!(
            scene.authority.gpu_live_count::<SceneCoordinateSpace>(),
            libhelio::MAX_COORDINATE_SPACES
        );

        for sublevel in sublevels {
            scene.remove_sublevel(sublevel).unwrap();
        }
        scene.remove_sublevel(recovered).unwrap();
        assert_eq!(scene.authority.gpu_live_count::<SceneCoordinateSpace>(), 1);
    }

    #[test]
    fn voxel_volume_rows_are_scenedb_owned_and_projection_dense() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let descriptor = |translation: f32| crate::VoxelVolumeDescriptor {
            voxel_size: 0.5,
            root_extent: 32.0,
            local_to_world: glam::Mat4::from_translation(glam::Vec3::X * translation),
            movability: Some(libhelio::Movability::Stationary),
            mode: Some(crate::VoxelMode::Dynamic),
            material_palette: vec![helio_voxel_core::GpuVoxelMaterial::zeroed()],
        };

        let first = scene.insert_voxel_volume(descriptor(1.0)).unwrap();
        let second = scene.insert_voxel_volume(descriptor(2.0)).unwrap();
        let first_entity = entity_from_handle(first);
        let second_entity = entity_from_handle(second);
        let first_row = scene
            .authority
            .gpu_row::<SceneVoxelVolume>(first_entity)
            .unwrap();
        let second_row = scene
            .authority
            .gpu_row::<SceneVoxelVolume>(second_entity)
            .unwrap();
        assert_eq!((first_row, second_row), (0, 1));
        assert_eq!(scene.gpu_scene.voxel_volume_indices.as_slice(), &[0, 1]);
        let second_component = scene
            .authority
            .get::<SceneVoxelVolume>(second_entity)
            .unwrap();
        assert_eq!(second_component.volume.0.dimensions, [64; 3]);
        assert_eq!(second_component.volume.0.brick_grid_dim, 8);
        assert_eq!(second_component.volume.0.brick_offset, 512);
        assert_eq!(
            scene
                .authority
                .subsystem::<helio_scenedb::VoxelResidency>()
                .unwrap()
                .brick_base(first_entity),
            Some(0),
        );

        scene.remove_voxel_volume(first).unwrap();
        assert_eq!(scene.gpu_scene.voxel_volume_indices.as_slice(), &[second_row]);
        assert!(scene.remove_voxel_volume(first).is_err());

        // Stable component rows do not relocate survivors; the freed row is
        // reused by the next exact Entity+component lifetime.
        let third = scene.insert_voxel_volume(descriptor(3.0)).unwrap();
        let third_entity = entity_from_handle(third);
        assert_eq!(
            scene
                .authority
                .gpu_row::<SceneVoxelVolume>(third_entity),
            Some(first_row),
        );
        assert_eq!(
            scene.gpu_scene.voxel_volume_indices.as_slice(),
            &[second_row, first_row],
        );
        assert_eq!(
            scene
                .authority
                .get::<SceneVoxelVolume>(third_entity)
                .unwrap()
                .volume
                .0
                .brick_offset,
            0,
        );

        scene.authority
            .edit_cpu::<crate::scene::voxel::VoxelVolumeRecord, _>(third_entity, |record| {
                record.edit_cooldown = 9;
            })
            .unwrap();
        let work_generation_before_cpu_edit = scene.gpu_scene.voxel_mesh_work_generation;
        scene
            .edit_voxel_volume(
                third,
                helio_voxel_core::VoxelEdit {
                    op: helio_voxel_core::VoxelOp::Paint,
                    center: glam::Vec3::ZERO,
                    radius: 1.0,
                    material: 1,
                },
            )
            .unwrap();
        assert_eq!(
            scene
                .authority
                .get::<crate::scene::voxel::VoxelVolumeRecord>(third_entity)
                .unwrap()
                .edit_cooldown,
            0
        );
        assert_eq!(
            scene.gpu_scene.voxel_mesh_work_generation,
            work_generation_before_cpu_edit,
            "CPU octree dirty marking is not a fabricated GPU edit stream"
        );

        // Auto/mesh volumes use the same canonical residency as Dynamic, but
        // remain absent from the compact raymarch-volume projection.
        let mut auto_descriptor = descriptor(4.0);
        auto_descriptor.mode = Some(crate::VoxelMode::Auto);
        let auto = scene.insert_voxel_volume(auto_descriptor).unwrap();
        let auto_entity = entity_from_handle(auto);
        scene
            .update_voxel_material_palette(
                auto,
                vec![helio_voxel_core::GpuVoxelMaterial::zeroed(); 3],
            )
            .unwrap();
        let auto_component = scene
            .authority
            .get::<SceneVoxelVolume>(auto_entity)
            .unwrap();
        assert_ne!(auto_component.volume.0.brick_offset, u32::MAX);
        assert_eq!(auto_component.volume.0.palette_count, 3);
        assert_eq!(
            scene
                .authority
                .get::<crate::scene::voxel::VoxelVolumeRecord>(auto_entity)
                .unwrap()
                .material_palette
                .len(),
            3
        );
        assert_eq!(
            scene
                .authority
                .subsystem::<helio_scenedb::VoxelResidency>()
                .unwrap()
                .palette_base(auto_entity),
            Some(auto_component.volume.0.palette_offset)
        );
        assert!(scene
            .authority
            .subsystem::<helio_scenedb::VoxelResidency>()
            .unwrap()
            .brick_base(auto_entity)
            .is_some());
        assert_eq!(
            scene.gpu_scene.voxel_volume_indices.as_slice(),
            &[second_row, first_row],
        );
        scene
            .upload_voxel_terrain(auto, &crate::VoxelTerrain::empty())
            .unwrap();
        let auto_work_row = scene.voxel_mesh_projection.row(0).unwrap();
        assert_eq!(auto_work_row.volume_row, scene.authority.gpu_row::<SceneVoxelVolume>(auto_entity).unwrap());
        assert_eq!(
            auto_work_row.flags,
            helio_voxel_core::VOXEL_MESH_WORK_ALLOCATED
        );
        let work_generation_before_flush = scene.gpu_scene.voxel_mesh_work_generation;
        scene.flush();
        assert!(scene.gpu_scene.voxel_mesh_work_generation > work_generation_before_flush);
        assert_eq!(scene.gpu_scene.voxel_mesh_work.len(), 512);
        scene.remove_voxel_volume(auto).unwrap();

        scene.flush();
        assert_eq!(
            scene
                .gpu_scene
                .canonical
                .voxel_volumes
                .as_ref()
                .unwrap()
                .len(),
            2,
        );
        scene.clear();
        assert_eq!(scene.authority.gpu_live_count::<SceneVoxelVolume>(), 0);
        assert!(scene.gpu_scene.voxel_volume_indices.is_empty());
    }

    #[test]
    fn auto_voxel_output_ceiling_is_checked_and_reuse_overwrites_stale_work() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let descriptor = || crate::VoxelVolumeDescriptor {
            voxel_size: 0.5,
            root_extent: 32.0,
            local_to_world: glam::Mat4::IDENTITY,
            movability: Some(libhelio::Movability::Stationary),
            mode: Some(crate::VoxelMode::Auto),
            material_palette: vec![helio_voxel_core::GpuVoxelMaterial::zeroed()],
        };

        let first = scene.insert_voxel_volume(descriptor()).unwrap();
        let second = scene.insert_voxel_volume(descriptor()).unwrap();
        assert!(matches!(
            scene.insert_voxel_volume(descriptor()),
            Err(SceneError::VoxelMeshCapacityExceeded)
        ));
        assert_eq!(scene.voxel_mesh_projection.row(0).unwrap().local_brick, 0);
        assert_eq!(scene.voxel_mesh_projection.row(512).unwrap().local_brick, 0);

        scene.remove_voxel_volume(first).unwrap();
        let replacement = scene.insert_voxel_volume(descriptor()).unwrap();
        let replacement_entity = entity_from_handle(replacement);
        let replacement_row = scene
            .authority
            .gpu_row::<SceneVoxelVolume>(replacement_entity)
            .unwrap();
        assert_eq!(
            scene.voxel_mesh_projection.row(0).unwrap().volume_row,
            replacement_row,
            "same-batch reuse must replace the removed generation's clear row"
        );

        scene
            .upload_voxel_terrain(replacement, &crate::VoxelTerrain::empty())
            .unwrap();
        scene.flush();
        assert_eq!(scene.gpu_scene.voxel_mesh_work.len(), 1024);
        assert_eq!(scene.gpu_scene.voxel_mesh_draw_count, 0);
        scene.remove_voxel_volume(second).unwrap();
        scene.remove_voxel_volume(replacement).unwrap();
    }

    #[test]
    fn foliage_type_retains_and_releases_canonical_material() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());

        let foliage = scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                ..Default::default()
            })
            .expect("live material must resolve to a foliage shader row");
        let material_entity = entity_from_handle(material);
        assert_eq!(
            scene
                .authority
                .get::<SceneMaterial>(material_entity)
                .unwrap()
                .ref_count,
            1
        );
        assert!(matches!(
            scene.remove_material(material),
            Err(SceneError::ResourceInUse { resource: "material" })
        ));

        scene
            .remove_foliage_type(foliage)
            .expect("foliage removal releases its material");
        assert!(scene
            .authority
            .get::<SceneMaterial>(material_entity)
            .is_none());
    }

    #[test]
    fn sectioned_asset_keeps_shared_geometry_until_asset_removal() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let upload = crate::SectionedMeshUpload {
            vertices: vec![crate::PackedVertex::default(); 3],
            sections: vec![vec![0, 1, 2], vec![0, 2, 1]],
        };
        let multi_mesh = scene.insert_sectioned_mesh(upload);
        let section_meshes = scene
            .sectioned_section_mesh_ids(multi_mesh)
            .expect("sectioned asset component")
            .to_vec();
        assert_eq!(section_meshes.len(), 2);

        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        let instance = scene
            .insert_sectioned_object(
                multi_mesh,
                &[material, material],
                glam::Mat4::IDENTITY,
                [0.0, 0.0, 0.0, 1.0],
                None,
            )
            .expect("first placement");
        let first_section = scene
            .authority
            .get::<SectionedInstanceRecord>(entity_from_handle(instance))
            .unwrap()
            .section_objects[0];
        assert!(matches!(
            scene.remove_object(handle_from_entity(first_section)),
            Err(SceneError::InvalidOperation { .. })
        ));

        scene
            .remove_sectioned_object(instance)
            .expect("aggregate removal");
        assert!(section_meshes
            .iter()
            .all(|&mesh| scene.mesh_pool().get(mesh).is_some()));

        // Materials are independently authored resources, whereas section
        // geometry is owned by MultiMesh. Reusing the asset with a fresh
        // material proves its retained MeshIds stayed generation-valid.
        let replacement_material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        let replacement = scene
            .insert_sectioned_object(
                multi_mesh,
                &[replacement_material, replacement_material],
                glam::Mat4::IDENTITY,
                [0.0, 0.0, 0.0, 1.0],
                None,
            )
            .expect("second placement reuses retained geometry");
        scene.remove_sectioned_object(replacement).unwrap();
        scene.remove_sectioned_mesh(multi_mesh).unwrap();
        assert!(section_meshes
            .iter()
            .all(|&mesh| scene.mesh_pool().get(mesh).is_none()));
    }

    #[test]
    fn scenedb_light_and_decal_crud_keeps_compact_gpu_counts_and_tags() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        let movable = scene.insert_light_with_movability(
            libhelio::GpuLight {
                color_intensity: [1.0, 0.5, 0.25, 8.0],
                // Assigned atlas slices are render-derived. Any public
                // non-MAX input is normalized to the authored request sentinel.
                shadow_index: 37,
                ..Default::default()
            },
            Some(libhelio::Movability::Movable),
            101,
        );
        let static_light = scene.insert_light_with_movability(
            libhelio::GpuLight::default(),
            Some(libhelio::Movability::Static),
            202,
        );
        let first_decal = scene.insert_decal_with_tag(
            libhelio::GpuDecal::zeroed(),
            303,
            Some(libhelio::Movability::Movable),
        );
        let second_decal = scene.insert_decal_with_tag(
            libhelio::GpuDecal::zeroed(),
            404,
            None,
        );

        assert_eq!(scene.light_by_tag(101), Some(movable));
        assert_eq!(scene.light_by_tag(202), Some(static_light));
        assert_eq!(scene.decal_by_tag(303), Some(first_decal));
        assert_eq!(scene.decal_by_tag(404), Some(second_decal));
        assert_eq!(scene.iter_lights().count(), 2);
        assert_eq!(scene.decal_count(), 2);

        {
            let queried = scene
                .iter_lights()
                .find(|(id, _, _)| *id == movable)
                .map(|(_, light, _)| light)
                .expect("movable light must be visible to the query");
            let gotten = scene.get_light(movable).expect("movable light must be gettable");
            assert_eq!(queried.shadow_index, 0);
            assert_eq!(gotten.shadow_index, 0);
            assert_eq!(bytemuck::bytes_of(queried), bytemuck::bytes_of(&gotten));
        }

        let mut updated = scene.get_light(movable).expect("movable light should exist");
        updated.color_intensity[3] = 12.0;
        scene.update_light(movable, updated).unwrap();
        assert_eq!(scene.get_light(movable).unwrap().color_intensity[3], 12.0);

        scene.flush();
        {
            let resources = scene.gpu_scene().resources();
            assert_eq!(resources.light_count, 1, "static authored rows are not realtime slots");
            assert_eq!(resources.decal_count, 2, "decal count follows the active projection");
        }

        assert!(scene.remove_decal(first_decal));
        scene.remove_light(movable).unwrap();
        scene.flush();
        {
            let resources = scene.gpu_scene().resources();
            assert_eq!(resources.light_count, 0);
            assert_eq!(resources.decal_count, 1);
            assert_eq!(resources.shadow_count, 0);
        }
        assert_eq!(scene.light_by_tag(101), None);
        assert_eq!(scene.decal_by_tag(303), None);
        assert!(scene.get_light(movable).is_none());
        assert!(scene.get_decal(first_decal).is_none());

        scene.clear();
        assert_eq!(scene.iter_lights().count(), 0);
        assert_eq!(scene.decal_count(), 0);
        assert_eq!(scene.light_by_tag(202), None);
        assert_eq!(scene.decal_by_tag(404), None);
        let resources = scene.gpu_scene().resources();
        assert_eq!(resources.light_count, 0);
        assert_eq!(resources.decal_count, 0);
        assert_eq!(resources.shadow_count, 0);

        // Re-entry may reuse the same SceneDB rows, but public generations,
        // tag indices, and compact projections must all describe only the new
        // entities after clear.
        let reentered_light = scene.insert_light_with_movability(
            libhelio::GpuLight::default(),
            Some(libhelio::Movability::Movable),
            505,
        );
        let reentered_decal = scene.insert_decal_with_tag(
            libhelio::GpuDecal::zeroed(),
            606,
            Some(libhelio::Movability::Movable),
        );
        scene.flush();
        assert_eq!(scene.light_by_tag(505), Some(reentered_light));
        assert_eq!(scene.decal_by_tag(606), Some(reentered_decal));
        assert!(scene.get_light(static_light).is_none());
        assert!(scene.get_decal(second_decal).is_none());
        let resources = scene.gpu_scene().resources();
        assert_eq!(resources.light_count, 1);
        assert_eq!(resources.decal_count, 1);
        assert_eq!(resources.shadow_count, 0);
    }

    #[test]
    fn planetary_frames_publish_from_scenedb_and_clear_invalidates_stable_ids() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let make_frame = |byte, number| {
            PlanetFrameUniform::from_camera(
                PlanetId([byte; 16]),
                PlanetPosition::from_lod0_cell([i64::from(byte) * 32, 0, 0]),
                number,
            )
        };

        let (first, _) = scene.set_planet_frame(make_frame(1, 1)).unwrap();
        let (second, _) = scene.set_planet_frame(make_frame(2, 1)).unwrap();
        scene.update_planet_frame(first, make_frame(1, 2)).unwrap();
        scene.flush();
        assert_eq!(scene.gpu_scene().planet_frames().len(), 2);
        assert_eq!(scene.gpu_scene().planet_frame_row_span(), 2);
        assert!(scene.gpu_scene().planet_frame_buffer_epoch().is_some());

        scene.remove_planet_frame(second).unwrap();
        let replacement = scene.insert_planet_frame(make_frame(3, 1)).unwrap();
        assert_eq!(replacement.slot(), second.slot());
        assert_ne!(replacement.generation(), second.generation());
        scene.clear();
        assert!(scene.planet_frames().is_empty());
        assert!(scene.planet_frame(first).is_none());
        assert!(scene.planet_frame(replacement).is_none());
        assert!(scene.gpu_scene().planet_frames().is_empty());
        assert_eq!(scene.gpu_scene().planet_frame_row_span(), 0);
    }

    #[test]
    fn clear_resets_canonical_sdf_edits_terrain_and_publication() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);
        let edit = scene
            .add_sdf_edit(SdfEdit {
                shape: SdfShapeType::Sphere,
                op: BooleanOp::Union,
                transform: Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0)),
                params: SdfShapeParams::sphere(2.0),
                blend_radius: 0.0,
            })
            .expect("valid canonical SDF edit");
        scene
            .set_sdf_terrain(Some(TerrainConfig::rolling()))
            .expect("valid canonical terrain");
        scene.flush();
        assert_eq!(scene.sdf_edits().len(), 1);
        assert!(scene.sdf_terrain().is_some());
        assert_eq!(scene.gpu_scene().sdf_edit_count(), 1);

        scene.clear();

        assert!(scene.sdf_edits().is_empty());
        assert!(scene.sdf_edit(edit).is_none());
        assert!(scene.sdf_terrain().is_none());
        assert_eq!(scene.gpu_scene().sdf_edit_count(), 0);
    }
}
