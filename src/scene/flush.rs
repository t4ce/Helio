//! Flush (GPU upload) orchestration for the scene.
//!
//! This module contains the [`flush`](Scene::flush) method which synchronises all
//! pending CPU-side changes to GPU buffers, and the dirty-range tracking associated
//! with flush operations.

use bytemuck::Zeroable;
use helio_scenedb::components::{
    DECAL_BUFFER_KEY, LIGHT_BUFFER_KEY, MATERIAL_BUFFER_KEY, MATERIAL_TEXTURES_BUFFER_KEY,
    OBJECT_RENDER_BUFFER_KEY, OBJECT_SPATIAL_BUFFER_KEY, POST_PROCESS_VOLUME_BUFFER_KEY,
    PLANAR_REFLECTOR_BUFFER_KEY, REFLECTION_CAPTURE_BUFFER_KEY, WATER_HITBOX_BUFFER_KEY,
    WATER_VOLUME_BUFFER_KEY,
};
use helio_scenedb::{
    SceneCoordinateSpace, SceneCoordinateSpaceRow, SceneFoliageInteractor, SceneFoliageLayer,
    SceneFoliageType, SceneLight, SceneMaterial, SceneMaterialRow, SceneMaterialTexturesRow,
    SceneObjectRenderRow, SceneObjectSpatialRow, ScenePostProcessVolume, ScenePostProcessVolumeRow,
    ScenePlanarReflector, ScenePlanarReflectorRow, SceneReflectionCapture,
    SceneReflectionCaptureRow,
    SceneVoxelVolume, SceneVoxelVolumeRow, SceneWaterHitbox, SceneWaterHitboxRow,
    SceneWaterVolume, SceneWaterVolumeRow, COORDINATE_SPACE_BUFFER_KEY,
    FOLIAGE_INTERACTOR_BUFFER_KEY, FOLIAGE_LAYER_BUFFER_KEY, FOLIAGE_TYPE_BUFFER_KEY,
    PlanetFrameAuthority, SdfAuthority, VoxelResidency, VOXEL_VOLUME_BUFFER_KEY,
};
use libhelio::{GpuCameraUniforms, GpuShadowMatrix};

use crate::handles::entity_from_handle;
use crate::scene::Scene;
use crate::scene::resources::light_projection::stable_shadow_assignments;

/// FNV-1a hash over f32 bit patterns. Used for per-caster shadow dirty tracking.
/// Hashing bit patterns (not float values) ensures NaN and -0.0 are handled consistently.
#[inline]
fn fnv1a_f32s(vals: &[f32]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET;
    for &v in vals {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Hashes the exact unjittered camera inputs consumed by directional CSM.
///
/// The renderer left-multiplies projection by an NDC translation for temporal
/// jitter. Undoing that translation keeps a stationary camera's shadow cache
/// stable, while hashing the complete view/projection prevents FOV, aspect,
/// clip-plane, or pose changes from slipping past the CPU submission gate.
#[inline]
fn directional_camera_shadow_hash(camera: &GpuCameraUniforms) -> u64 {
    let jitter = camera.jitter_frame;
    let unjitter = glam::Mat4::from_translation(glam::Vec3::new(-jitter[0], -jitter[1], 0.0));
    let projection = unjitter * glam::Mat4::from_cols_array(&camera.proj);
    fnv1a_f32s(&camera.view) ^ fnv1a_f32s(&projection.to_cols_array())
}

impl Scene {
    /// Flush pending changes to GPU buffers.
    ///
    /// This method:
    /// 1. Assigns shadow atlas base layers to shadow-casting lights
    /// 2. Flushes mesh pool uploads (vertex/index data)
    /// 3. Flushes material texture buffer uploads
    /// 4. Rebuilds object instance buffers if dirty (persistent or optimized mode)
    /// 5. Rebuilds virtual geometry buffers if dirty
    /// 6. Flushes all GPU scene buffers (instances, draws, indirect, visibility, etc.)
    ///
    /// # Performance
    ///
    /// **Clean state (no topology changes):**
    /// - CPU cost: bounded by registered dirty-tracked buffers; no object/light scan
    /// - GPU cost: no canonical or projection upload when all epochs are unchanged
    ///
    /// **Dirty state (topology changed):**
    /// - CPU cost: O(N) for persistent rebuild, O(N log N) for optimized rebuild
    /// - GPU cost: O(N) buffer uploads for all object data
    ///
    /// # Shadow Management
    ///
    /// Automatically assigns shadow atlas layers to shadow-casting lights:
    /// - The configured atlas capacity determines the realtime caster limit
    /// - 6 slots per light (point = 6 faces, directional = 4 cascades + 2 padding, spot = 1 + 5 padding)
    /// - Lights beyond the cap have shadows disabled automatically
    ///
    /// # When to Call
    ///
    /// Call `flush()` after all scene modifications for the frame, before rendering:
    /// ```ignore
    /// // Modify scene
    /// scene.insert_object(desc)?;
    /// scene.update_object_transform(id, transform)?;
    /// scene.hide_group(group_id);
    ///
    /// // Flush changes
    /// scene.flush();
    ///
    /// // Render
    /// renderer.render(&scene, target)?;
    /// ```
    pub fn flush(&mut self) {
        // Assign shadow atlas slots to the highest-importance shadow-casting lights.
        //
        // Problem with sequential assignment: the first N lights inserted always win the
        // caster budget, regardless of how far away or how dim they are. A bright
        // close light inserted after the atlas is full gets no shadow.
        //
        // Solution — two-phase importance selection:
        //   Phase 1: Score every shadow-requesting light by VIEW-INDEPENDENT importance:
        //              intensity × range²
        //            Directional lights always score ∞ (global, never culled).
        //            Sort descending → top 42 are the frame's active casters.
        //   Phase 2: Keep every continuing winner's valid atlas base. Only new
        //            winners consume freed bases, preventing removal of an early
        //            compact light from shifting all later cached shadow faces.
        //
        // IMPORTANT: Camera distance is intentionally NOT used in scoring. Using camera
        // distance causes the budget to reshuffle every frame the camera moves, which
        // triggers shadow atlas re-renders (expensive with many draw calls). The budget
        // should only change when lights are added/removed or their properties change.
        if self.light_projection.take_atlas_dirty() {
            const FACES_PER_LIGHT: u32 = 6;
            let max_shadow_casters =
                ((self.shadow_face_capacity / FACES_PER_LIGHT) as usize).min(42);
            let light_count = self.light_projection.len();

            // Phase 1: score and select the top MAX_SHADOW_CASTERS.
            let mut scored: Vec<(f32, usize)> = Vec::with_capacity(light_count);
            for (compact_slot, &id) in self.light_projection.ids().iter().enumerate() {
                let Some(record) = self.authority.get::<SceneLight>(entity_from_handle(id)) else {
                    debug_assert!(false, "projected light must remain canonical");
                    continue;
                };
                let light = &record.light;
                if !light.requests_shadow() {
                    continue;
                }
                let score = if light.light_type == 0 {
                    // Directional: infinite range, always highest priority.
                    f32::MAX
                } else {
                    let range = light.position_range[3].max(0.001);
                    // intensity × range² — view-independent, stable across camera moves.
                    // Larger/brighter lights win the budget regardless of camera position.
                    light.color_intensity[3] * (range * range)
                };
                scored.push((score, compact_slot));
            }

            let assignments = stable_shadow_assignments(
                light_count,
                &scored,
                self.gpu_scene.light_projections.as_slice(),
                max_shadow_casters,
            );
            let active_face_count = assignments
                .iter()
                .copied()
                .filter(|&base| base != u32::MAX)
                .max()
                .map(|base| base + FACES_PER_LIGHT)
                .unwrap_or(0);
            // Keep allocation size and semantic activity separate: wgpu
            // storage bindings need a nonzero backing buffer, while passes
            // must see zero faces when no authored light won an atlas slot.
            let needed = active_face_count.max(1) as usize;
            self.gpu_scene.active_shadow_face_count = active_face_count;
            for (compact_slot, shadow_index) in assignments.into_iter().enumerate() {
                let entity_row = self
                    .light_projection
                    .row(compact_slot)
                    .expect("compact light projection must retain its SceneDB GPU row");
                let desired = [entity_row, shadow_index];
                if self.gpu_scene.light_projections.as_slice()[compact_slot] != desired {
                    let updated = self.gpu_scene.light_projections.update(compact_slot, desired);
                    debug_assert!(updated);
                }
            }
            if self.gpu_scene.shadow_matrices.len() != needed {
                self.gpu_scene
                    .shadow_matrices
                    .set_data(vec![GpuShadowMatrix::zeroed(); needed]);
            }
        }

        // ── Per-caster shadow dirty tracking ─────────────────────────────────
        // Hash authored light/camera state per caster for static-atlas cache
        // invalidation. Movable object dirtiness is detected per component row by
        // ShadowDirtyPass on the GPU; doing that work here was an unconditional
        // O(objects × casters) CPU scan every flush.
        {
            let mut new_hashes = [0u64; 42];

            for (compact_slot, &id) in self.light_projection.ids().iter().enumerate() {
                let shadow_index = self.gpu_scene.light_projections.as_slice()[compact_slot][1];
                if shadow_index == u32::MAX {
                    continue;
                }
                let Some(record) = self.authority.get::<SceneLight>(entity_from_handle(id)) else {
                    debug_assert!(false, "projected light must remain canonical");
                    continue;
                };
                let light = &record.light;
                let slot = (shadow_index / 6) as usize;
                if slot >= 42 {
                    continue;
                }
                let base_hash = fnv1a_f32s(&light.position_range)
                    ^ fnv1a_f32s(&light.direction_outer)
                    ^ (light.light_type as u64).wrapping_mul(2654435761);
                // Directional CSM depends on the complete unjittered camera
                // frustum. Hash the exact shader inputs so the CPU submission
                // gate can never hide a GPU `light_dirty` result. Temporal
                // jitter is removed on both sides and therefore stays cacheable.
                new_hashes[slot] = if light.light_type == 0 {
                    base_hash ^ directional_camera_shadow_hash(self.gpu_scene.camera.data())
                } else {
                    base_hash
                };
            }

            // ShadowPass compares these hashes against its last rendered state.
            for slot in 0..42usize {
                self.gpu_scene.per_caster_dirty_gen[slot] = new_hashes[slot];
            }
        }

        let queue = self.gpu_scene.queue.clone();
        self.mesh_pool_mut().flush(&queue);
        // Rebuild GPU buffers with automatic instancing when objects change.
        if self.objects_dirty {
            self.rebuild_instance_buffers();
            self.objects_dirty = false;
        }
        // Topology changes rebuild all mirrors. Transform-only changes publish
        // one bounded instance range without touching descriptors or work spans.
        if self.vg_objects_dirty {
            self.rebuild_vg_buffers();
            self.vg_objects_dirty = false;
        } else if let Some(range) = self.vg_instance_dirty_range.take() {
            self.vg_published_instance_dirty_range = Some(range);
            self.vg_instance_version = self.vg_instance_version.wrapping_add(1);
        }

        // ── Graph WGSL snippets ────────────────────────────────────────────────
        // Clone strings only when the registry itself changed, never during
        // clean frame flushes. Graph source publication happens only at this
        // boundary, keeping the cloned map and its cache-invalidation epoch
        // coherent; material flags have their own narrow mutation-time cache.
        let (graph_epoch, graph_snapshot) = {
            let registry = self.radiant_graphs();
            let epoch = registry.epoch();
            let snapshot = (epoch != self.published_radiant_graph_epoch)
                .then(|| registry.snapshot());
            (epoch, snapshot)
        };
        if let Some(snapshot) = graph_snapshot {
            self.gpu_scene.graph_wgsl_snippets = snapshot;
            self.gpu_scene.graph_wgsl_epoch = graph_epoch;
            self.published_radiant_graph_epoch = graph_epoch;
        }

        // One SceneDB publication boundary per frame. Persistent component
        // fields are committed before any Helio render-derived/pass buffers,
        // so passes see one coherent authority snapshot.
        self.authority.flush_gpu();
        assert!(
            helio_scenedb::refresh_sprite_buffer_source(
                &self.authority,
                &self.sprite_buffer_source,
            ),
            "main Scene sprite publication remains registered",
        );
        let active_object_count = self.gpu_scene.source_indices.len() as u32;
        let material_row_span = self.authority.gpu_row_span::<SceneMaterial>();
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(FOLIAGE_TYPE_BUFFER_KEY) {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<helio_foliage_core::GpuFoliageType>()
            );
            self.gpu_scene.publish_foliage_types(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneFoliageType>(),
            );
        }
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(FOLIAGE_LAYER_BUFFER_KEY) {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<helio_foliage_core::GpuFoliageLayer>()
            );
            self.gpu_scene.publish_foliage_layers(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneFoliageLayer>(),
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(FOLIAGE_INTERACTOR_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<helio_foliage_core::GpuFoliageInteractor>()
            );
            self.gpu_scene.publish_foliage_interactors(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneFoliageInteractor>(),
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(OBJECT_SPATIAL_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneObjectSpatialRow>()
            );
            self.gpu_scene.publish_object_spatial(
                snapshot.buffer,
                snapshot.epoch,
                active_object_count,
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(OBJECT_RENDER_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneObjectRenderRow>()
            );
            self.gpu_scene.publish_object_render(
                snapshot.buffer,
                snapshot.epoch,
                active_object_count,
            );
        }
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(LIGHT_BUFFER_KEY) {
            debug_assert_eq!(snapshot.row_stride as usize, std::mem::size_of::<libhelio::GpuLight>());
            self.gpu_scene.publish_lights(
                snapshot.buffer,
                snapshot.epoch,
                self.light_projection.len() as u32,
            );
        }
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(DECAL_BUFFER_KEY) {
            debug_assert_eq!(snapshot.row_stride as usize, std::mem::size_of::<libhelio::GpuDecal>());
            self.gpu_scene.publish_decals(
                snapshot.buffer,
                snapshot.epoch,
                self.decal_projection.len() as u32,
            );
        }
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(WATER_VOLUME_BUFFER_KEY) {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneWaterVolumeRow>()
            );
            self.gpu_scene.publish_water_volumes(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneWaterVolume>(),
            );
        }
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(WATER_HITBOX_BUFFER_KEY) {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneWaterHitboxRow>()
            );
            self.gpu_scene.publish_water_hitboxes(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneWaterHitbox>(),
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(POST_PROCESS_VOLUME_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<ScenePostProcessVolumeRow>()
            );
            self.gpu_scene.publish_post_process_volumes(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<ScenePostProcessVolume>(),
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(REFLECTION_CAPTURE_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneReflectionCaptureRow>()
            );
            self.gpu_scene.publish_reflection_captures(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneReflectionCapture>(),
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(PLANAR_REFLECTOR_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<ScenePlanarReflectorRow>()
            );
            self.gpu_scene.publish_planar_reflectors(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<ScenePlanarReflector>(),
            );
        }
        if let Some(snapshot) = self.authority.partner_buffer_snapshot(MATERIAL_BUFFER_KEY) {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneMaterialRow>()
            );
            self.gpu_scene.publish_materials(
                snapshot.buffer,
                snapshot.epoch,
                material_row_span,
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(MATERIAL_TEXTURES_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneMaterialTexturesRow>()
            );
            self.gpu_scene.publish_material_textures(
                snapshot.buffer,
                snapshot.epoch,
                material_row_span,
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(COORDINATE_SPACE_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneCoordinateSpaceRow>()
            );
            let row_span = self.authority.gpu_row_span::<SceneCoordinateSpace>();
            debug_assert!(row_span <= libhelio::MAX_COORDINATE_SPACES);
            self.gpu_scene.publish_coordinate_spaces(
                snapshot.buffer,
                snapshot.epoch,
                row_span,
            );
        }
        if let Some(snapshot) = self
            .authority
            .partner_buffer_snapshot(VOXEL_VOLUME_BUFFER_KEY)
        {
            debug_assert_eq!(
                snapshot.row_stride as usize,
                std::mem::size_of::<SceneVoxelVolumeRow>()
            );
            self.gpu_scene.publish_voxel_volumes(
                snapshot.buffer,
                snapshot.epoch,
                self.authority.gpu_row_span::<SceneVoxelVolume>(),
            );
        }
        let (
            voxel_bricks,
            voxel_data,
            voxel_palette,
            voxel_epoch,
            voxel_capacity,
            voxel_palette_capacity,
        ) = self
            .authority
            .subsystem::<VoxelResidency>()
            .expect("voxel residency subsystem is registered")
            .publication();
        self.gpu_scene.publish_voxel_residency(
            voxel_bricks,
            voxel_data,
            voxel_palette,
            voxel_epoch,
            voxel_capacity,
            voxel_palette_capacity,
        );
        let next_voxel_mesh_generation =
            self.gpu_scene.voxel_mesh_work_generation.wrapping_add(1);
        if let Some(rows) = self
            .voxel_mesh_projection
            .take_batch(next_voxel_mesh_generation)
        {
            self.gpu_scene.voxel_mesh_work.set_data(rows);
            self.gpu_scene.voxel_mesh_work_generation = next_voxel_mesh_generation;
        }
        self.gpu_scene.voxel_mesh_draw_count = self.voxel_mesh_projection.draw_count();
        let sdf = self
            .authority
            .subsystem::<SdfAuthority>()
            .expect("SDF authority is registered")
            .publication();
        self.gpu_scene.publish_sdf_authority(
            sdf.edit_buffer,
            sdf.edit_allocation_epoch,
            sdf.edit_count,
            sdf.terrain_buffer,
            sdf.terrain_allocation_epoch,
            sdf.content_generation,
            bytemuck::cast_slice(sdf.bounds),
            sdf.terrain_y_bounds,
            sdf.requires_canonical_scan,
        );
        let planet_frames = self
            .authority
            .subsystem::<PlanetFrameAuthority>()
            .expect("planet-frame authority is registered")
            .publication();
        self.gpu_scene.publish_planet_frames(
            planet_frames.buffer,
            planet_frames.authority_epoch,
            planet_frames.allocation_epoch,
            planet_frames.row_span,
            planet_frames.content_generation,
            planet_frames.entries.iter().map(|entry| entry.projection()),
        );
        self.gpu_scene.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::directional_camera_shadow_hash;
    use glam::{Mat4, Vec3};
    use libhelio::GpuCameraUniforms;

    fn camera(view: Mat4, projection: Mat4, jitter: [f32; 2]) -> GpuCameraUniforms {
        let jitter_transform = Mat4::from_translation(Vec3::new(jitter[0], jitter[1], 0.0));
        GpuCameraUniforms::new(
            view,
            jitter_transform * projection,
            Vec3::new(3.0, 4.0, 5.0),
            0.1,
            500.0,
            0,
            jitter,
            Mat4::IDENTITY,
        )
    }

    #[test]
    fn directional_shadow_gate_ignores_temporal_jitter_but_tracks_frustum_changes() {
        let view = Mat4::from_rotation_y(0.37)
            * Mat4::from_translation(Vec3::new(-3.0, -4.0, -5.0));
        let projection = Mat4::from_scale(Vec3::new(1.17, 1.83, 0.79));
        let a = camera(view, projection, [0.000_31, -0.000_47]);
        let b = camera(view, projection, [-0.000_53, 0.000_29]);
        assert_eq!(
            directional_camera_shadow_hash(&a),
            directional_camera_shadow_hash(&b),
            "temporal jitter must not invalidate a stationary CSM"
        );

        let changed_projection = Mat4::from_scale(Vec3::new(1.23, 1.83, 0.79));
        let changed = camera(view, changed_projection, [0.000_31, -0.000_47]);
        assert_ne!(
            directional_camera_shadow_hash(&a),
            directional_camera_shadow_hash(&changed),
            "projection/FOV changes must enter the shadow submission path"
        );
    }
}
