//! Core scene structure and constructor.
//!
//! This module contains the main [`Scene`] struct definition, constructor,
//! and trivial getters. Lifecycle methods, flush, camera, water, and stats
//! each live in their own sub-modules.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use helio_core::GpuScene;
use helio_scenedb::{
    register_coordinate_space_buffer, register_foliage_component_buffers,
    register_scene_component_buffers, register_sprite_component_buffer,
    register_voxel_volume_buffer, PlanetFrameAuthority, SceneAuthority, SceneAuthorityConfig,
    SceneCoordinateSpace, SceneIndices, SceneMaterial, SceneSky, SceneVisibilityState,
    SceneVoxelVolume, SceneVoxelVolumeRow, SceneWind, SceneWindRow, SdfAuthority, SpriteAtlasResidency,
    SpriteAtlasSource, SpriteBufferSource, VoxelResidency,
};
use helio_voxel_core::VoxelEdit;
use wgpu::util::DeviceExt;

use super::types::VoxelVolumeDescriptor;
use super::voxel::{VoxelMode, VoxelVolumeRecord};
use crate::terrain::{BrickRange, VoxelTerrain};
use crate::handles::{
    entity_from_handle, FoliageInteractorId, FoliageLayerId, FoliageTypeId, MaterialId,
    PlanarReflectorId, PostProcessVolumeId, VoxelVolumeId, WaterHitboxId,
};
use crate::mesh::MeshPool;
use crate::radiant::RadiantGraphRegistry;
use crate::scene::multi_mesh::SectionRelations;
use crate::scene::actor::{RetainedSceneActor, SceneActorId};
use crate::scene::extension::{
    ExtensionComponentPolicy, ExtensionSubsystemStore, SceneComponentRegistrar,
};
use crate::scene::presentation::ScenePresentationState;

use super::errors::{invalid, Result};
use super::decals::DecalProjection;
use super::resources::light_projection::LightProjection;
use super::resources::entity_projection::EntityRowProjection;
use super::resources::reflection::ReflectionProjection;
use super::resources::voxel_mesh_projection::{
    VoxelMeshProjection, VoxelMeshProjectionError,
};
use super::virtual_geometry::VirtualGeometryStorage;
use super::water::WaterVolumeProjection;

/// High-level scene management with persistent GPU-driven state.
///
/// See the [module-level documentation](crate::scene) for architecture details and usage examples.
pub struct Scene {
    /// Sole authority for persistent scene component data.
    pub(in crate::scene) authority: SceneAuthority,

    /// Shared publications for 2D passes integrated with this Scene. The
    /// render graph clones these sources; it never constructs another scene
    /// authority for the same authored rows.
    pub(in crate::scene) sprite_buffer_source: SpriteBufferSource,
    pub(in crate::scene) sprite_atlas_source: SpriteAtlasSource,
    pub(in crate::scene) next_sprite_authored_epoch: u32,

    /// GPU scene resources (buffers, bind groups, etc.)
    pub(in crate::scene) gpu_scene: GpuScene,

    /// Material texture representation and capacity selected for this device.
    pub(in crate::scene) material_binding: libhelio::MaterialBindingConfig,

    /// Placeholder texture (1x1 white)
    pub(in crate::scene) _placeholder_texture: wgpu::Texture,

    /// Placeholder texture view
    pub(in crate::scene) placeholder_view: wgpu::TextureView,

    /// Placeholder sampler
    pub(in crate::scene) placeholder_sampler: wgpu::Sampler,

    /// Last Radiant graph registry epoch copied into render-facing lookup
    /// state. Avoids cloning strings during clean frame flushes.
    pub(in crate::scene) published_radiant_graph_epoch: u64,

    /// Helio-derived compact realtime projections. Canonical authored rows and
    /// public handle generations remain exclusively in SceneDB.
    pub(in crate::scene) decal_projection: DecalProjection,
    pub(in crate::scene) light_projection: LightProjection,
    pub(in crate::scene) water_volume_projection: WaterVolumeProjection,
    pub(in crate::scene) water_hitbox_projection: EntityRowProjection<WaterHitboxId>,
    pub(in crate::scene) post_process_volume_projection:
        EntityRowProjection<PostProcessVolumeId>,
    pub(in crate::scene) planar_reflector_projection: EntityRowProjection<PlanarReflectorId>,
    pub(in crate::scene) reflection_projection: ReflectionProjection,
    /// Compact active voxel-volume slot -> stable SceneDB component row.
    pub(in crate::scene) voxel_volume_projection: EntityRowProjection<VoxelVolumeId>,
    /// Stable Auto-voxel output slots and coalesced extraction work.
    pub(in crate::scene) voxel_mesh_projection: VoxelMeshProjection,

    /// SceneDB `Entity::index()` -> current compact draw-order slot. This is a
    /// CPU-only renderer projection; canonical GPU rows are component-local and
    /// resolved separately through `SceneAuthority::gpu_row`. `u32::MAX` means
    /// no published slot.
    pub(in crate::scene) object_projection_slots: Vec<u32>,

    /// True when object topology or batching keys changed and Helio's derived
    /// draw/source/visibility projections must be rebuilt.
    pub(in crate::scene) objects_dirty: bool,

    /// True when a Static or Stationary object has been added or removed since the last
    /// shadow atlas render. Triggers a re-render of the static shadow atlas.
    pub(in crate::scene) static_objects_dirty: bool,

    /// True when static/stationary geometry or lights have been added since the last bake.
    /// When this is true and a bake was previously configured, the user must explicitly
    /// call auto_bake() again to rebake the scene with the new static content.
    pub(in crate::scene) bake_invalidated: bool,

    /// Previous frame's view-projection matrix (for temporal effects)
    pub(in crate::scene) prev_view_proj: glam::Mat4,

    /// Generation counter for movable objects - increments when any Movable object's transform changes.
    /// Used by shadow caching to detect when Movable objects move.
    pub(in crate::scene) movable_objects_generation: u64,

    /// Generation counter for movable lights - increments when any Movable light's position/direction changes.
    /// Used by shadow caching to detect when Movable lights move.
    pub(in crate::scene) movable_lights_generation: u64,

    /// Number of shadow-map array layers available in the active render graph.
    /// Six consecutive layers are reserved per realtime shadow caster.
    pub(in crate::scene) shadow_face_capacity: u32,

    /// Retained execution/behavior actors. Canonical descriptor payloads live
    /// in SceneDB components/subsystems and are never read from this list.
    pub(in crate::scene) custom_actors: Vec<RetainedSceneActor>,

    /// Active execution identity -> canonical target. Retained separately so
    /// actor removal is safe while the tick snapshot is outside the vector.
    pub(in crate::scene) custom_actor_targets: HashMap<SceneActorId, SceneActorId>,

    /// One-time reflection/registration validation cache for application-owned
    /// component types. This is access-policy metadata, never scene data.
    pub(in crate::scene) extension_component_policies:
        HashMap<TypeId, ExtensionComponentPolicy>,

    // ── Virtual geometry ──────────────────────────────────────────────────────
    /// Set when VG topology changes; triggers `rebuild_vg_buffers()`.
    pub(in crate::scene) vg_objects_dirty: bool,

    /// Monotonically increasing counter forwarded to `VgFrameData::buffer_version`.
    /// The VG pass re-uploads GPU buffers only when this advances.
    pub(in crate::scene) vg_buffer_version: u64,

    // ── Virtual geometry projections ─────────────────────────────────────────────
    /// Transform-only changes accumulated until the next scene flush (end exclusive).
    pub(in crate::scene) vg_instance_dirty_range: Option<(usize, usize)>,

    /// Last transform-only range published to the render pass (end exclusive).
    pub(in crate::scene) vg_published_instance_dirty_range: Option<(usize, usize)>,

    /// Monotonic version for published transform-only changes.
    pub(in crate::scene) vg_instance_version: u64,

    /// Monotonic version for SceneDB-authored state projected into VG cull records
    /// (currently material cull flags and group visibility). This is deliberately
    /// independent of topology and transform versions.
    pub(in crate::scene) vg_cull_signature_version: u64,

    /// Dense VirtualGeometryStorage object index -> compact published VG object
    /// slot. This Helio-derived projection keeps sparse/degenerate source records
    /// from shifting transform or visibility updates onto another GPU instance.
    /// `u32::MAX` means the canonical object has no renderable projection.
    pub(in crate::scene) vg_object_projection_slots: Vec<u32>,

    /// Unique meshlet entries for virtual meshes referenced by the current VG layout.
    pub(in crate::scene) vg_cpu_meshlets: Vec<libhelio::GpuMeshletEntry>,

    /// Object-level LOD ranges and bounds (one entry per VG object).
    pub(in crate::scene) vg_cpu_objects: Vec<libhelio::GpuVgObject>,

    /// Instance data for all VG objects (one entry per VG object, in order).
    pub(in crate::scene) vg_cpu_instances: Vec<helio_core::GpuInstanceData>,

    /// SceneDB-derived visibility scalar parallel to `vg_cpu_instances`.
    /// The VG pass folds this into its existing 16-byte instance-cull projection;
    /// it is never a second authored visibility authority or a GPU binding.
    pub(in crate::scene) vg_cpu_visibility: Vec<u32>,

    /// Immutable 64-meshlet expansion spans for the second GPU cull stage.
    pub(in crate::scene) vg_cpu_work_items: Vec<libhelio::GpuVgWorkItem>,

    /// Exact worst-case number of draws after choosing one LOD per object.
    pub(in crate::scene) vg_max_draw_count: u32,

    // ── Foliage ───────────────────────────────────────────────────────────────
    /// Compact active ids -> component-local canonical foliage-type rows.
    pub(in crate::scene) foliage_type_projection: EntityRowProjection<FoliageTypeId>,

    /// Version of authored foliage topology. Wind never advances it.
    pub(in crate::scene) foliage_generation: u64,
    pub(in crate::scene) foliage_layer_projection: EntityRowProjection<FoliageLayerId>,
    pub(in crate::scene) foliage_interactor_projection: EntityRowProjection<FoliageInteractorId>,
    pub(in crate::scene) foliage_max_height: f32,
    pub(in crate::scene) foliage_max_density: f32,

    /// Stable SceneDB entity carrying the one global wind component.
    pub(in crate::scene) wind_entity: helio_scenedb::Entity,

    /// Stable SceneDB entity carrying the canonical optional sky context.
    pub(in crate::scene) sky_entity: helio_scenedb::Entity,

    // ── Coordinate spaces (sublevels + portals) ──────────────────────────────────
}

impl Scene {
    /// Create a new empty scene.
    ///
    /// Initializes all resource pools, creates placeholder textures, and sets up
    /// GPU buffers with default capacities.
    ///
    /// # Parameters
    /// - `device`: GPU device for buffer/texture creation
    /// - `queue`: GPU queue for initial uploads
    ///
    /// # Returns
    /// A new [`Scene`] ready for resource insertion.
    ///
    /// # Initial State
    /// - All resource pools are empty
    /// - First `flush()` will rebuild GPU buffers with automatic instancing
    ///
    /// # Performance
    /// - CPU cost: O(1) struct initialization
    /// - GPU cost: Creates placeholder texture, allocates initial buffer capacity
    /// - Memory: Allocates arena/pool structures with default capacity
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use helio::Scene;
    ///
    /// let device = Arc::new(gpu_device);
    /// let queue = Arc::new(gpu_queue);
    /// let scene = Scene::new(device, queue);
    /// ```
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self::new_with_component_registration(device, queue, |_| {})
    }

    /// Create an empty scene and register application GPU component columns in
    /// SceneDB's only safe startup window.
    ///
    /// The callback runs after Helio's built-in columns are registered but
    /// before the World mirror is attached and before any entity exists. It
    /// receives only an opaque typed registrar: no `SceneGpuStore`,
    /// `SceneAuthority`, raw World, entity, built-in buffer, or subsystem hook
    /// escapes construction. Register a derived custom type with
    /// `registrar.register::<T>(initial_capacity)`.
    /// CPU-only custom components require no registration.
    pub fn new_with_component_registration(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        register_components: impl FnOnce(&mut SceneComponentRegistrar<'_>),
    ) -> Self {
        // Registration happens inside construction, before the first entity is
        // inserted, so GPU-partner generations can never miss a backfill.
        let authority_config = SceneAuthorityConfig::default();
        let component_capacity = authority_config.initial_entity_capacity;
        let mut authority = SceneAuthority::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            authority_config,
            |store, device| {
                register_scene_component_buffers(store, component_capacity, device);
                register_coordinate_space_buffer(store, device);
                register_voxel_volume_buffer(store, device);
                register_foliage_component_buffers(store, device);
                register_sprite_component_buffer(store, component_capacity, device);
                let mut registrar = SceneComponentRegistrar::new(store, device);
                register_components(&mut registrar);
            },
        );
        // Component-local row zero is the permanent world-space identity.
        // Keeping this entity alive for the Scene lifetime makes every later
        // sublevel/portal allocation naturally land in the shader's 1..31
        // domain without a second slot allocator.
        let world_space = authority.insert(SceneCoordinateSpace::IDENTITY);
        assert_eq!(
            authority.gpu_row::<SceneCoordinateSpace>(world_space),
            Some(0),
            "SceneDB coordinate-space row zero must be world identity",
        );
        authority.register_subsystem(SceneIndices::default());
        authority.register_subsystem(ExtensionSubsystemStore::default());
        authority.register_subsystem(SceneVisibilityState::default());
        authority.register_subsystem(ScenePresentationState::default());
        authority.register_subsystem(SectionRelations::default());
        authority.register_subsystem(VoxelResidency::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        ));
        authority.register_subsystem(SdfAuthority::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        ));
        authority.register_subsystem(PlanetFrameAuthority::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        ));
        authority.register_subsystem(MeshPool::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        ));
        authority.register_subsystem(VirtualGeometryStorage::default());
        authority.register_subsystem(SpriteAtlasResidency::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        ));
        let sprite_buffer_source = helio_scenedb::sprite_buffer_source_for(&authority)
            .expect("main Scene authority registers the sprite partner at construction");
        let sprite_atlas_source = authority
            .subsystem::<SpriteAtlasResidency>()
            .expect("main Scene authority registers sprite atlas residency")
            .publication_source();
        let wind_entity = authority.insert(SceneWind {
            wind: SceneWindRow::from(libhelio::Wind::default()),
        });
        let sky_entity = authority.insert(SceneSky::default());
        let material_binding = libhelio::MaterialBindingConfig::for_device(&device);
        helio_core::upload::record_upload_bytes(4);
        let placeholder_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Helio Placeholder Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let placeholder_view =
            placeholder_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Helio Placeholder Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            authority,
            sprite_buffer_source,
            sprite_atlas_source,
            next_sprite_authored_epoch: 1,
            gpu_scene: GpuScene::new(device.clone(), queue.clone()),
            material_binding,
            _placeholder_texture: placeholder_texture,
            placeholder_view,
            placeholder_sampler,
            published_radiant_graph_epoch: u64::MAX,
            decal_projection: DecalProjection::default(),
            light_projection: LightProjection::default(),
            water_volume_projection: WaterVolumeProjection::default(),
            water_hitbox_projection: EntityRowProjection::default(),
            post_process_volume_projection: EntityRowProjection::default(),
            planar_reflector_projection: EntityRowProjection::default(),
            reflection_projection: ReflectionProjection::default(),
            voxel_volume_projection: EntityRowProjection::default(),
            voxel_mesh_projection: VoxelMeshProjection::default(),
            object_projection_slots: Vec::new(),
            objects_dirty: true,        // rebuild on first flush
            static_objects_dirty: true, // rebuild static shadow atlas on first flush
            bake_invalidated: false,    // no bake configured yet
            prev_view_proj: glam::Mat4::IDENTITY,
            movable_objects_generation: 0,
            movable_lights_generation: 0,
            shadow_face_capacity: 32,
            custom_actors: Vec::new(),
            custom_actor_targets: HashMap::new(),
            extension_component_policies: HashMap::new(),
            vg_objects_dirty: false,
            vg_buffer_version: 0,
            vg_instance_dirty_range: None,
            vg_published_instance_dirty_range: None,
            vg_instance_version: 0,
            vg_cull_signature_version: 0,
            vg_object_projection_slots: Vec::new(),
            vg_cpu_meshlets: Vec::new(),
            vg_cpu_objects: Vec::new(),
            vg_cpu_instances: Vec::new(),
            vg_cpu_visibility: Vec::new(),
            vg_cpu_work_items: Vec::new(),
            vg_max_draw_count: 0,
            foliage_type_projection: EntityRowProjection::default(),
            foliage_generation: 0,
            foliage_layer_projection: EntityRowProjection::default(),
            foliage_interactor_projection: EntityRowProjection::default(),
            foliage_max_height: 0.0,
            foliage_max_density: 0.0,
            wind_entity,
            sky_entity,
        }
    }

    /// Allocation-epoch-aware SceneDB publications for constructing an
    /// integrated `SpriteBatchPass::from_publications`. Mutations and flushes
    /// remain owned by this Scene; the pass receives no second authority.
    pub fn sprite_publications(&self) -> (SpriteBufferSource, SpriteAtlasSource) {
        (
            self.sprite_buffer_source.clone(),
            self.sprite_atlas_source.clone(),
        )
    }

    /// SceneDB-owned registry of compiled Radiant graph assets.
    ///
    /// This reusable asset registry intentionally survives [`Scene::clear`];
    /// use [`Self::radiant_graphs_mut`] to register, hot-replace, or explicitly
    /// retire sources. Render passes see only the epoch-gated cloned projection
    /// published by [`Scene::flush`].
    pub fn radiant_graphs(&self) -> &RadiantGraphRegistry {
        self.authority
            .subsystem::<RadiantGraphRegistry>()
            .expect("Radiant graph registry is registered at Scene construction")
    }

    /// Mutable access to SceneDB's compiled Radiant graph asset registry.
    pub fn radiant_graphs_mut(&mut self) -> &mut RadiantGraphRegistry {
        self.authority
            .subsystem_mut::<RadiantGraphRegistry>()
            .expect("Radiant graph registry is registered at Scene construction")
    }

    /// SceneDB-owned general geometry residency and metadata subsystem.
    pub(in crate::scene) fn mesh_pool(&self) -> &MeshPool {
        self.authority
            .subsystem::<MeshPool>()
            .expect("general geometry subsystem is registered at Scene construction")
    }

    pub(in crate::scene) fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        self.authority
            .subsystem_mut::<MeshPool>()
            .expect("general geometry subsystem is registered at Scene construction")
    }

    pub(in crate::scene) fn virtual_geometry(&self) -> &VirtualGeometryStorage {
        self.authority
            .subsystem::<VirtualGeometryStorage>()
            .expect("virtual geometry subsystem is registered at Scene construction")
    }

    pub(in crate::scene) fn virtual_geometry_mut(&mut self) -> &mut VirtualGeometryStorage {
        self.authority
            .subsystem_mut::<VirtualGeometryStorage>()
            .expect("virtual geometry subsystem is registered at Scene construction")
    }

    pub(crate) fn set_shadow_face_capacity(&mut self, capacity: u32) {
        let capacity = capacity.clamp(1, 256);
        if self.shadow_face_capacity != capacity {
            self.shadow_face_capacity = capacity;
            self.light_projection.mark_atlas_dirty();
        }
    }

    pub fn insert_voxel_volume(
        &mut self,
        descriptor: VoxelVolumeDescriptor,
    ) -> Result<VoxelVolumeId> {
        if !descriptor.voxel_size.is_finite()
            || descriptor.voxel_size <= 0.0
            || !descriptor.root_extent.is_finite()
            || descriptor.root_extent <= 0.0
        {
            return Err(super::errors::SceneError::InvalidOperation {
                reason: "voxel volume size must be finite and positive",
            });
        }

        let mode = descriptor.mode.unwrap_or(VoxelMode::Auto);
        let record = VoxelVolumeRecord::new(&descriptor);
        let brick_count = record.brick_count().ok_or(
            super::errors::SceneError::InvalidOperation {
                reason: "voxel volume brick dimensions exceed the u32 address domain",
            },
        )?;
        let entity = self.authority.insert(record);
        let allocation = match self
            .authority
            .subsystem_mut::<VoxelResidency>()
            .expect("voxel residency subsystem is registered at Scene construction")
            .allocate_with_palette(
                entity,
                brick_count,
                &descriptor.material_palette,
            )
        {
            Ok(allocation) => allocation,
            Err(_) => {
                let _ = self.authority.despawn(entity);
                return Err(super::errors::SceneError::InvalidOperation {
                    reason: "SceneDB voxel brick or palette residency capacity exceeded",
                });
            }
        };
        let initial_row = self
            .authority
            .get::<VoxelVolumeRecord>(entity)
            .expect("fresh voxel entity must retain its CPU record")
            .authored_gpu_row(
                descriptor.local_to_world,
                allocation.brick_base,
                allocation.palette_base,
                allocation.palette_count,
            );
        let component = SceneVoxelVolume {
            volume: SceneVoxelVolumeRow(initial_row),
            movability: descriptor
                .movability
                .unwrap_or(libhelio::Movability::Static) as u32,
            mode: mode as u32,
            _pad0: 0,
            _pad1: 0,
        };
        assert!(
            self.authority.replace_gpu(entity, component),
            "fresh voxel entity must remain live",
        );
        let component_row = self
            .authority
            .gpu_row::<SceneVoxelVolume>(entity)
            .expect("inserted voxel component must own a mirror row");

        let id = crate::handles::handle_from_entity(entity);
        if mode == VoxelMode::Dynamic {
            let compact = self.voxel_volume_projection.insert(id, component_row);
            let gpu_slot = self.gpu_scene.voxel_volume_indices.push(component_row);
            debug_assert_eq!(compact, gpu_slot);
        } else if let Err(error) = self
            .voxel_mesh_projection
            .allocate(entity, brick_count, component_row)
        {
            self.authority
                .subsystem_mut::<VoxelResidency>()
                .expect("voxel residency subsystem is registered")
                .release(entity)
                .expect("fresh voxel volume must own residency");
            let _ = self.authority.despawn(entity);
            return match error {
                VoxelMeshProjectionError::CapacityExceeded => {
                    Err(super::errors::SceneError::VoxelMeshCapacityExceeded)
                }
                _ => Err(super::errors::SceneError::InvalidOperation {
                    reason: "Helio rejected fresh voxel mesh projection state",
                }),
            };
        }
        Ok(id)
    }

    pub fn remove_voxel_volume(&mut self, id: VoxelVolumeId) -> Result<()> {
        let entity = entity_from_handle(id);
        if self.authority.get::<VoxelVolumeRecord>(entity).is_none()
            || self.authority.get::<SceneVoxelVolume>(entity).is_none()
        {
            return Err(invalid("voxel volume"));
        }
        let mode = self
            .authority
            .get::<SceneVoxelVolume>(entity)
            .map(|component| component.mode)
            .expect("validated voxel component must remain live");
        if mode == VoxelMode::Dynamic as u32 {
            let compact = self
                .voxel_volume_projection
                .remove(id)
                .expect("dynamic voxel volume must have an active projection");
            debug_assert!(
                self.gpu_scene
                    .voxel_volume_indices
                    .swap_remove(compact)
                    .is_some()
            );
        } else {
            self.voxel_mesh_projection
                .release(entity)
                .expect("Auto voxel volume must own stable output slots");
        }
        self.authority
            .subsystem_mut::<VoxelResidency>()
            .expect("voxel residency subsystem is registered")
            .release(entity)
            .expect("voxel volume must own a residency region");
        if !self.authority.despawn(entity) {
            return Err(invalid("voxel volume"));
        }
        Ok(())
    }

    /// Set the material class and graph hash for an existing material.
    ///
    /// `material_class` selects the surface archetype template (0 = default PBR).
    /// `graph_hash` selects a WGSL snippet from the graph registry (0 = none).
    /// `feature_flags` overrides the material's feature flags (pass `None` to keep existing).
    pub fn set_material_class(
        &mut self,
        material_id: MaterialId,
        material_class: u32,
        graph_hash: u64,
        feature_flags: Option<u32>,
    ) -> Result<()> {
        let entity = entity_from_handle(material_id);
        self.authority
            .gpu_row::<SceneMaterial>(entity)
            .ok_or_else(|| invalid("material"))?;
        let (old_material, old_graph_hash, material) = self
            .authority
            .edit_gpu::<SceneMaterial, _>(entity, |record| {
                let old_material = record.material.0;
                let old_graph_hash = record.graph_hash;
                record.material.0.material_class = material_class;
                record.graph_hash = graph_hash;
                if let Some(flags) = feature_flags {
                    record.material.0.flags = flags;
                }
                (old_material, old_graph_hash, record.material.0)
            })
            .ok_or_else(|| invalid("material"))?;
        self.cache_material_projection(entity, material);
        self.note_vg_material_cull_change(old_material.flags, material.flags);
        self.note_material_batch_change(old_material, old_graph_hash, material, graph_hash);
        Ok(())
    }

    /// Update only the class_params of a material (no texture revalidation).
    pub fn update_material_class_params(
        &mut self,
        material_id: MaterialId,
        params: [f32; 4],
    ) -> Result<()> {
        let entity = entity_from_handle(material_id);
        self.authority
            .gpu_row::<SceneMaterial>(entity)
            .ok_or_else(|| invalid("material"))?;
        let material = self
            .authority
            .edit_gpu::<SceneMaterial, _>(entity, |record| {
                record.material.0.class_params = params;
                record.material.0
            })
            .ok_or_else(|| invalid("material"))?;
        self.cache_material_projection(entity, material);
        Ok(())
    }

    /// Mark the CPU octree dirty for an authored edit.
    ///
    /// No hidden GPU edit queue exists. Apply the same edit to the retained
    /// `VoxelTerrain`, then call `upload_voxel_terrain_range` to update the
    /// SceneDB-owned raw bricks and queue Auto meshing when applicable.
    pub fn edit_voxel_volume(&mut self, id: VoxelVolumeId, edit: VoxelEdit) -> Result<()> {
        let entity = entity_from_handle(id);
        self.authority
            .get::<SceneVoxelVolume>(entity)
            .ok_or_else(|| invalid("voxel volume"))?;
        self.authority
            .edit_cpu::<VoxelVolumeRecord, _>(entity, |record| record.edit(&edit))
            .ok_or_else(|| invalid("voxel volume"))?;
        Ok(())
    }

    /// Bake and upload all raw 8x8x8 bricks through SceneDB residency.
    /// Dynamic volumes become immediately raymarchable; Auto volumes also
    /// queue their stable output slots for surface extraction.
    pub fn upload_voxel_terrain(
        &mut self,
        id: VoxelVolumeId,
        terrain: &VoxelTerrain,
    ) -> Result<()> {
        self.upload_voxel_terrain_range(id, terrain, BrickRange::all())
    }

    /// Re-bake only the bricks touched by a terrain edit.
    pub fn upload_voxel_terrain_range(
        &mut self,
        id: VoxelVolumeId,
        terrain: &VoxelTerrain,
        range: BrickRange,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let component = self
            .authority
            .get::<SceneVoxelVolume>(entity)
            .copied()
            .ok_or_else(|| invalid("voxel volume"))?;
        let residency = self
            .authority
            .subsystem::<VoxelResidency>()
            .expect("voxel residency subsystem is registered");
        if residency.brick_count(entity)
            != Some(helio_voxel_core::DEFAULT_VOLUME_BRICK_COUNT)
        {
            return Err(super::errors::SceneError::InvalidOperation {
                reason: "VoxelTerrain uploads require a 64x64x64 voxel volume",
            });
        }
        let mut touched = Vec::new();
        terrain
            .for_each_canonical_brick(range, |local_brick, occupied, words| {
                residency.write_brick(entity, local_brick, occupied, words)?;
                touched.push((local_brick, occupied));
                Ok::<(), helio_scenedb::VoxelResidencyError>(())
            })
            .map_err(|_| super::errors::SceneError::InvalidOperation {
                reason: "SceneDB rejected voxel terrain upload",
            })?;
        if component.mode == VoxelMode::Auto as u32 {
            for (local_brick, occupied) in touched {
                self.voxel_mesh_projection
                    .mark_uploaded(
                        entity,
                        local_brick,
                        occupied,
                        component.volume.0.brick_grid_dim,
                    )
                    .map_err(|_| super::errors::SceneError::InvalidOperation {
                        reason: "Helio rejected Auto voxel mesh work",
                    })?;
            }
        }
        Ok(())
    }

    /// Replace a volume's authored palette. Its residency offset stays stable
    /// while the reserved power-of-two region fits, and is atomically updated
    /// in the canonical volume row if growth requires relocation.
    pub fn update_voxel_material_palette(
        &mut self,
        id: VoxelVolumeId,
        palette: Vec<helio_voxel_core::GpuVoxelMaterial>,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        self.authority
            .get::<SceneVoxelVolume>(entity)
            .ok_or_else(|| invalid("voxel volume"))?;
        let (palette_offset, palette_count) = self
            .authority
            .subsystem_mut::<VoxelResidency>()
            .expect("voxel residency subsystem is registered")
            .write_palette(entity, &palette)
            .map_err(|_| super::errors::SceneError::InvalidOperation {
                reason: "SceneDB rejected voxel palette update",
            })?;
        self.authority
            .edit_cpu::<VoxelVolumeRecord, _>(entity, |record| {
                record.material_palette = palette
            })
            .ok_or_else(|| invalid("voxel volume"))?;
        self.authority
            .edit_gpu::<SceneVoxelVolume, _>(entity, |component| {
                component.volume.0.palette_offset = palette_offset;
                component.volume.0.palette_count = palette_count;
            })
            .ok_or_else(|| invalid("voxel volume"))?;
        Ok(())
    }

    pub fn voxel_volume(&self, id: VoxelVolumeId) -> Option<&VoxelVolumeRecord> {
        self.authority
            .get::<VoxelVolumeRecord>(entity_from_handle(id))
    }

    /// Returns a reference to the TLAS (Top-Level Acceleration Structure) for
    /// hardware ray tracing, if available. Returns `None` on non-RT hardware
    /// or when the TLAS has not been built yet.
    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.gpu_scene.tlas_manager.tlas()
    }

    /// Store a type-erased template registry on the GpuScene so the GBufferPass
    /// can find it across graph rebuilds (window resize).
    pub fn set_template_registry(&mut self, reg: Box<dyn std::any::Any + Send + Sync>) {
        self.gpu_scene.template_registry = Some(reg);
    }

    /// Store a type-erased TRANSPARENT template registry on the GpuScene so the
    /// TransparentPass can find it across graph rebuilds.
    pub fn set_transparent_template_registry(&mut self, reg: Box<dyn std::any::Any + Send + Sync>) {
        self.gpu_scene.transparent_template_registry = Some(reg);
    }
}
