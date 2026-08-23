//! SceneDB ownership boundary used by Helio's high-level scene facade.
//!
//! This type owns the archetype [`pulsar_scenedb::World`], its GPU mirror,
//! and SceneDB's subsystem registry as one unit.  It intentionally does not
//! expose `&mut World`: GPU-partnered components must be replaced through
//! [`SceneAuthority::edit_gpu`] so SceneDB's mirror dispatch observes every
//! mutation.  Read queries remain typed and allocation-free.

use std::collections::HashMap;
use std::sync::Arc;

use pulsar_scenedb::gpu::{
    DirtyTrackedReallocationPolicy, EngineGpuContext, GpuMirrorHandle, RegionClassConfig,
    SceneGpuConfig, SceneGpuStore, SyncStats,
};
use pulsar_scenedb::{Component, Entity, SceneDb, Subsystem};

/// Marker for components whose GPU fields are all DirtyTracked and may
/// therefore be replaced after insertion. Helio implements this for built-in
/// components; downstream custom-data types may opt in through Helio's public
/// extension facade, which reflects and verifies the all-DirtyTracked policy
/// before the first insertion.
pub trait MutableGpuComponent: Component + Copy {}

/// Marker for components with no GPU partner fields. In-place mutation is
/// permitted because there is no mirror dispatch to bypass.
pub trait CpuOnlyComponent: Component {}

impl MutableGpuComponent for crate::components::SceneObject {}
impl MutableGpuComponent for crate::components::SceneLight {}
impl MutableGpuComponent for crate::components::SceneDecal {}
impl MutableGpuComponent for crate::components::SceneWaterVolume {}
impl MutableGpuComponent for crate::components::SceneWaterHitbox {}
impl MutableGpuComponent for crate::components::ScenePostProcessVolume {}
impl MutableGpuComponent for crate::components::SceneReflectionCapture {}
impl MutableGpuComponent for crate::components::ScenePlanarReflector {}
impl MutableGpuComponent for crate::components::SceneMaterial {}
impl MutableGpuComponent for crate::foliage::SceneFoliageType {}
impl MutableGpuComponent for crate::foliage::SceneFoliageLayer {}
impl MutableGpuComponent for crate::foliage::SceneFoliageInteractor {}
impl MutableGpuComponent for crate::sprite::SceneSprite {}
impl CpuOnlyComponent for crate::components::SceneTexture {}
impl CpuOnlyComponent for crate::components::SceneMaterialTextureRefs {}
impl CpuOnlyComponent for crate::components::SceneWind {}
impl CpuOnlyComponent for crate::components::SceneSky {}
impl CpuOnlyComponent for crate::foliage::SceneFoliageLayerTypes {}
impl CpuOnlyComponent for crate::sprite::SceneSpriteAtlasLayer {}

/// SceneDB-owned secondary indices for application tags.
///
/// Tags are query infrastructure rather than canonical component storage, but
/// keeping the index registered with SceneDB prevents Helio from growing a
/// parallel object/light/decal database. Duplicate tags retain the historical
/// newest-claim behavior.
#[derive(Default)]
pub struct SceneIndices {
    objects_by_tag: HashMap<u64, Entity>,
    lights_by_tag: HashMap<u64, Entity>,
    decals_by_tag: HashMap<u64, Entity>,
    water_volumes_by_tag: HashMap<u64, Entity>,
    water_hitboxes_by_tag: HashMap<u64, Entity>,
    post_process_volumes_by_tag: HashMap<u64, Entity>,
    reflection_captures_by_tag: HashMap<u64, Entity>,
    planar_reflectors_by_tag: HashMap<u64, Entity>,
}

impl SceneIndices {
    fn insert(map: &mut HashMap<u64, Entity>, tag: u64, entity: Entity) {
        if tag != 0 {
            map.insert(tag, entity);
        }
    }

    fn remove(map: &mut HashMap<u64, Entity>, tag: u64, entity: Entity) {
        if tag != 0 && map.get(&tag) == Some(&entity) {
            map.remove(&tag);
        }
    }

    fn lookup(map: &HashMap<u64, Entity>, tag: u64) -> Option<Entity> {
        (tag != 0).then(|| map.get(&tag).copied()).flatten()
    }

    pub fn insert_object(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.objects_by_tag, tag, entity);
    }

    pub fn remove_object(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.objects_by_tag, tag, entity);
    }

    pub fn object_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.objects_by_tag, tag)
    }

    pub fn insert_light(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.lights_by_tag, tag, entity);
    }

    pub fn remove_light(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.lights_by_tag, tag, entity);
    }

    pub fn light_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.lights_by_tag, tag)
    }

    pub fn insert_decal(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.decals_by_tag, tag, entity);
    }

    pub fn remove_decal(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.decals_by_tag, tag, entity);
    }

    pub fn decal_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.decals_by_tag, tag)
    }

    pub fn insert_water_volume(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.water_volumes_by_tag, tag, entity);
    }

    pub fn remove_water_volume(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.water_volumes_by_tag, tag, entity);
    }

    pub fn water_volume_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.water_volumes_by_tag, tag)
    }

    pub fn insert_water_hitbox(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.water_hitboxes_by_tag, tag, entity);
    }

    pub fn remove_water_hitbox(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.water_hitboxes_by_tag, tag, entity);
    }

    pub fn water_hitbox_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.water_hitboxes_by_tag, tag)
    }

    pub fn insert_post_process_volume(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.post_process_volumes_by_tag, tag, entity);
    }

    pub fn remove_post_process_volume(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.post_process_volumes_by_tag, tag, entity);
    }

    pub fn post_process_volume_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.post_process_volumes_by_tag, tag)
    }

    pub fn insert_reflection_capture(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.reflection_captures_by_tag, tag, entity);
    }

    pub fn remove_reflection_capture(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.reflection_captures_by_tag, tag, entity);
    }

    pub fn reflection_capture_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.reflection_captures_by_tag, tag)
    }

    pub fn insert_planar_reflector(&mut self, tag: u64, entity: Entity) {
        Self::insert(&mut self.planar_reflectors_by_tag, tag, entity);
    }

    pub fn remove_planar_reflector(&mut self, tag: u64, entity: Entity) {
        Self::remove(&mut self.planar_reflectors_by_tag, tag, entity);
    }

    pub fn planar_reflector_by_tag(&self, tag: u64) -> Option<Entity> {
        Self::lookup(&self.planar_reflectors_by_tag, tag)
    }
}

impl Subsystem for SceneIndices {
    fn name(&self) -> &'static str {
        "helio.scene.indices"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// SceneDB-owned authored visibility policy for the whole scene.
///
/// Object membership remains on each canonical component row. This singleton
/// owns only the global hidden-group mask; Helio derives compact per-draw
/// visibility values from it and never treats that GPU projection as authored
/// state.
#[derive(Default)]
pub struct SceneVisibilityState {
    hidden_groups: u64,
}

impl SceneVisibilityState {
    /// Raw bitmask whose set bits identify hidden groups.
    #[inline(always)]
    pub const fn hidden_groups(&self) -> u64 {
        self.hidden_groups
    }

    /// Replace the authored hidden-group mask.
    ///
    /// Returns `true` only when the canonical value changed, allowing callers
    /// to preserve the visibility API's allocation-free idempotent fast path.
    #[inline(always)]
    pub fn replace_hidden_groups(&mut self, hidden_groups: u64) -> bool {
        if self.hidden_groups == hidden_groups {
            return false;
        }
        self.hidden_groups = hidden_groups;
        true
    }

    /// Reset all groups to visible for a canonical scene clear.
    #[inline(always)]
    pub fn clear(&mut self) -> bool {
        self.replace_hidden_groups(0)
    }
}

impl Subsystem for SceneVisibilityState {
    fn name(&self) -> &'static str {
        "helio.scene.visibility"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Allocation-stable publication captured after a SceneDB mirror flush.
pub struct PartnerBufferSnapshot {
    pub buffer: wgpu::Buffer,
    pub epoch: u64,
    pub row_stride: u32,
}

/// Allocation policy for Helio's SceneDB authority.
#[derive(Clone, Debug)]
pub struct SceneAuthorityConfig {
    /// Initial CPU entity capacity. Full-scene constructors may also reuse
    /// this as the startup hint for selected high-population component-local
    /// GPU partners; it does not preallocate every partnered component.
    pub initial_entity_capacity: u32,
    /// Cell-storage allocation classes used by streaming/spatial subsystems.
    pub gpu: SceneGpuConfig,
    /// Physical growth policy for CPU-shadowed DirtyTracked partners and
    /// their presence/generation columns. The default retains the normal
    /// GPU-copy path; constrained custom backends may opt one authority into
    /// reconstructing replacement allocations from SceneDB's CPU shadows.
    /// This never changes `Once` semantics.
    pub dirty_tracked_reallocation: DirtyTrackedReallocationPolicy,
    /// Built-in subsystems created with the authority. Pass-local authorities
    /// can opt out of unrelated residency managers; the full-scene default
    /// remains unchanged.
    pub subsystems: SceneAuthoritySubsystemConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneAuthoritySubsystemConfig {
    /// Own canonical material-texture GPU objects, stable slots, and asset-key
    /// allocation for a full 3D Scene authority.
    pub material_texture_residency: bool,
    /// Own persistent compiled Radiant graph source assets. Standalone passes
    /// that never resolve materials can omit this registry.
    pub radiant_graph_registry: bool,
}

impl SceneAuthoritySubsystemConfig {
    pub const FULL_SCENE: Self = Self {
        material_texture_residency: true,
        radiant_graph_registry: true,
    };

    pub const SPRITE_STANDALONE: Self = Self {
        material_texture_residency: false,
        radiant_graph_registry: false,
    };
}

impl Default for SceneAuthoritySubsystemConfig {
    fn default() -> Self {
        Self::FULL_SCENE
    }
}

impl Default for SceneAuthorityConfig {
    fn default() -> Self {
        Self {
            initial_entity_capacity: 1_024,
            gpu: SceneGpuConfig {
                classes: vec![RegionClassConfig {
                    capacity: 1_024,
                    max_resident_cells: 64,
                }],
                tombstone_headroom: SceneGpuConfig::default_headroom(),
                max_cells_metadata: 1_024,
            },
            dirty_tracked_reallocation: DirtyTrackedReallocationPolicy::GpuCopy,
            subsystems: SceneAuthoritySubsystemConfig::default(),
        }
    }
}

/// The single persistent scene-data authority on the Helio side.
///
/// Render-pass outputs (indirect arguments, compacted visibility, temporal
/// history, framebuffers, and similar executor state) do not belong here.
/// Persistent components, GPU-partner fields, and stateful scene subsystems
/// do.
pub struct SceneAuthority {
    pub(crate) db: SceneDb,
    store: Arc<SceneGpuStore>,
    mirror: GpuMirrorHandle,
    entity_capacity_hint: u32,
}

impl SceneAuthority {
    /// Construct an authority and attach its World mirror before any entity
    /// can be inserted.
    ///
    /// `register_columns` is the one startup window for derive-generated GPU
    /// column registration.  Keeping it inside construction prevents the
    /// late-attach/backfill hole where existing components have GPU bytes but
    /// no matching generation/liveness publication.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: SceneAuthorityConfig,
        register_columns: impl FnOnce(&mut SceneGpuStore, &Arc<wgpu::Device>),
    ) -> Self {
        let SceneAuthorityConfig {
            initial_entity_capacity,
            gpu,
            dirty_tracked_reallocation,
            subsystems,
        } = config;
        let ctx = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
        let mut store = SceneGpuStore::new(&ctx, gpu);
        store.set_dirty_tracked_reallocation_policy(dirty_tracked_reallocation);
        register_columns(&mut store, &device);

        let store = Arc::new(store);
        let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(&queue));
        let mut db = SceneDb::new();
        db.world.reserve_entities(initial_entity_capacity);
        db.world.attach_gpu_mirror(mirror.clone());
        if subsystems.material_texture_residency {
            let texture_capacity =
                libhelio::MaterialBindingConfig::for_device(&device).max_textures;
            db.register_subsystem(crate::materials::TextureResidency::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                texture_capacity as u32,
            ));
        }
        if subsystems.radiant_graph_registry {
            db.register_subsystem(crate::radiant::RadiantGraphRegistry::new());
        }

        Self {
            db,
            store,
            mirror,
            entity_capacity_hint: initial_entity_capacity,
        }
    }

    /// Register a SceneDB phase-aware subsystem.
    pub fn register_subsystem<T: Subsystem + 'static>(&mut self, subsystem: T) {
        self.db.register_subsystem(subsystem);
    }

    /// Read a registered subsystem by concrete type.
    pub fn subsystem<T: Subsystem + 'static>(&self) -> Option<&T> {
        self.db.subsystem::<T>()
    }

    /// Mutate a registered subsystem by concrete type.
    pub fn subsystem_mut<T: Subsystem + 'static>(&mut self) -> Option<&mut T> {
        self.db.subsystem_mut::<T>()
    }

    /// Run SceneDB's CPU simulation phases and their registered subsystem
    /// hooks.
    pub fn step_subsystems(&mut self) {
        self.db.step();
    }

    /// Insert one component on a fresh SceneDB entity.
    pub fn insert<T: Component>(&mut self, value: T) -> Entity {
        let entity = self.db.world.spawn();
        self.db.world.insert(entity, value);
        entity
    }

    /// Add or replace a CPU-only companion component on an existing entity.
    ///
    /// This is the relationship path for variable-sized authored data such as foliage
    /// layer membership. GPU-partnered components intentionally cannot use it.
    pub fn insert_cpu<T: CpuOnlyComponent>(&mut self, entity: Entity, value: T) -> bool {
        if !self.db.world.is_alive(entity) {
            return false;
        }
        self.db.world.insert(entity, value);
        true
    }

    /// Generic mirror-aware insertion used by Helio's closed custom-data
    /// facade after it has validated the downstream component's declared CPU,
    /// DirtyTracked, or Once policy. This remains hidden from Helio's public
    /// scene API; callers there never receive `SceneAuthority` or raw Entity.
    #[doc(hidden)]
    pub fn insert_registered<T: Component>(&mut self, entity: Entity, value: T) -> bool {
        if !self.db.world.is_alive(entity) {
            return false;
        }
        self.db.world.insert(entity, value);
        true
    }

    /// Add or replace a mutable GPU component through SceneDB's mirror-aware
    /// insert path. Once partners deliberately do not implement this marker.
    pub fn replace_gpu<T: MutableGpuComponent>(&mut self, entity: Entity, value: T) -> bool {
        if !self.db.world.is_alive(entity) {
            return false;
        }
        self.db.world.insert(entity, value);
        true
    }

    /// Copy-edit-replace a mutable GPU-partner component.
    ///
    /// This is intentionally separate from [`Self::edit_cpu`].  Returning a
    /// raw `&mut T` would bypass World's generated mirror dispatch and leave
    /// stale GPU rows.  `T: Copy` also makes panic behavior transactional: if
    /// `edit` unwinds, the authoritative component was never changed.
    pub fn edit_gpu<T: MutableGpuComponent, R>(
        &mut self,
        entity: Entity,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        let mut value = *self.db.world.get::<T>(entity)?;
        let result = edit(&mut value);
        self.db.world.insert(entity, value);
        Some(result)
    }

    /// Allocation-free bulk copy/edit/reinsert for DirtyTracked components.
    ///
    /// The callback receives the component's stable local GPU row together
    /// with a transactional copy of its canonical value. Returning `true`
    /// commits through World's generated differential mirror dispatch;
    /// returning `false` leaves both CPU and GPU state untouched.
    pub fn edit_gpu_each<T: MutableGpuComponent>(
        &mut self,
        mut edit: impl FnMut(Entity, u32, &mut T) -> bool,
    ) -> usize {
        let mirror = &self.mirror;
        self.db.world.edit_each::<T>(|entity, value| {
            let gpu_row = mirror
                .gpu_row::<T>(entity)
                .expect("queried GPU component must retain its local mirror row");
            edit(entity, gpu_row, value)
        })
    }

    /// Edit a CPU-only component in place.
    ///
    /// Callers must not use this for a type with GPU-partnered fields.  Helio
    /// seals that distinction at its component layer; this lower-level seam
    /// keeps the method explicit so accidental use is visible in review.
    pub fn edit_cpu<T: CpuOnlyComponent, R>(
        &mut self,
        entity: Entity,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        self.db.world.get_mut::<T>(entity).map(edit)
    }

    /// Generic in-place edit used only after Helio's custom-data facade has
    /// reflected and validated a CPU-only downstream component. `World` also
    /// rejects GPU-mirrored `T` here, preserving the lower-level invariant.
    #[doc(hidden)]
    pub fn edit_registered_cpu<T: Component, R>(
        &mut self,
        entity: Entity,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        self.db.world.get_mut::<T>(entity).map(edit)
    }

    /// Read one component by stable entity identity.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.db.world.get::<T>(entity)
    }

    /// Iterate one component type through SceneDB's typed read query.
    pub fn query<T: Component>(&self) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.db.world.query::<&T>()
    }

    /// Remove one component.  SceneDB owns GPU-row invalidation for mirrored
    /// fields; Helio must not manufacture a parallel presence table here.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.db.world.remove::<T>(entity)
    }

    /// Despawn an entity and publish its new generation through the mirror.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.db.world.despawn(entity)
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.db.world.is_alive(entity)
    }

    /// Resolve one live component to its stable component-local GPU row.
    /// This row is deliberately not `Entity::index()`: unrelated World
    /// component types do not inflate one another's partner buffers.
    pub fn gpu_row<T: Component>(&self, entity: Entity) -> Option<u32> {
        self.mirror.gpu_row::<T>(entity)
    }

    /// Addressable high-water span for one component's stable GPU rows. The
    /// span may contain reusable holes and resets to zero when no rows remain.
    pub fn gpu_row_span<T: Component>(&self) -> u32 {
        self.mirror.gpu_row_span::<T>()
    }

    pub fn gpu_live_count<T: Component>(&self) -> u32 {
        self.mirror.gpu_live_count::<T>()
    }

    /// Reflect the GPU columns registered in this concrete authority.
    ///
    /// This deliberately resolves through the attached mirror rather than
    /// depending on link-time inventory, which is not guaranteed to survive
    /// every executable packaging format (including TRUEOS Blueprints).
    #[doc(hidden)]
    pub fn gpu_column_descs_for_component(
        &self,
        id: pulsar_scenedb::ComponentId,
    ) -> Option<Vec<pulsar_scenedb::gpu::GpuColumnDesc>> {
        self.mirror.gpu_column_descs_for_component(id)
    }

    /// Pre-grow only one component's local value and presence buffers.
    /// Entity-generation capacity remains a separate global domain.
    pub fn reserve_gpu_component_capacity<T: Component>(
        &self,
        capacity: u32,
    ) -> Result<(), pulsar_scenedb::CapacityError> {
        self.mirror.reserve_gpu_component_capacity::<T>(capacity)
    }

    /// Shrink one component's local buffers to its current stable row span.
    pub fn shrink_gpu_component_to_fit<T: Component>(&self, slack_factor: f32) -> bool {
        self.mirror
            .shrink_gpu_component_to_fit::<T>(slack_factor)
    }

    /// Reserve only SceneDB's CPU Entity domain. Component-local GPU columns
    /// are separate allocation domains and should be reserved explicitly by
    /// callers that do not intend to grow every mirrored component.
    pub fn reserve_entity_capacity(&mut self, capacity: u32) {
        let additional = capacity.saturating_sub(self.entity_capacity_hint);
        if additional > 0 {
            self.db.world.reserve_entities(additional);
            self.entity_capacity_hint = capacity;
        }
    }

    /// Reserve CPU entity slots and every registered World-mirrored buffer
    /// before a known batch. Returns an error only when a manually registered
    /// partner imposed an explicit maximum below `capacity`; the generated
    /// growable registration path has no such ceiling. A detached mirror is
    /// unreachable for this authority after construction.
    pub fn reserve(&mut self, capacity: u32) -> Result<(), pulsar_scenedb::CapacityError> {
        self.reserve_entity_capacity(capacity);
        self.db
            .world
            .reserve_gpu_mirror_capacity(self.mirror.queue(), capacity)
            .expect("SceneAuthority always keeps its GPU mirror attached")
    }

    /// Commit all queued World partner-field and liveness writes.
    pub fn flush_gpu(&self) -> SyncStats {
        self.db
            .world
            .flush_gpu_mirror(self.mirror.queue())
            .expect("SceneAuthority always keeps its GPU mirror attached")
    }

    pub fn gpu_store(&self) -> &SceneGpuStore {
        &self.store
    }

    pub fn gpu_mirror(&self) -> &GpuMirrorHandle {
        &self.mirror
    }

    /// Snapshot a named partner after [`Self::flush_gpu`]. The stable key is
    /// the renderer contract; generated component/field wrapper ids remain an
    /// implementation detail inside SceneDB.
    pub fn partner_buffer_snapshot(&self, key: &str) -> Option<PartnerBufferSnapshot> {
        let (buffer, epoch, descriptor) = self.store.gpu_buffer_snapshot_for_key(key)?;
        Some(PartnerBufferSnapshot {
            buffer,
            epoch,
            row_stride: descriptor.value_token.desc().size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SceneIndices;
    use pulsar_scenedb::Entity;

    #[test]
    fn scene_indices_preserve_newest_claim_and_guard_removal() {
        let older = Entity::from_bits((3u64 << 32) | 7);
        let newer = Entity::from_bits((4u64 << 32) | 9);
        let mut indices = SceneIndices::default();

        indices.insert_object(42, older);
        indices.insert_object(42, newer);
        indices.remove_object(42, older);
        assert_eq!(indices.object_by_tag(42), Some(newer));

        indices.remove_object(42, newer);
        assert_eq!(indices.object_by_tag(42), None);
    }

    #[test]
    fn scene_indices_ignore_zero_and_keep_domains_separate() {
        let entity = Entity::from_bits(5);
        let mut indices = SceneIndices::default();

        indices.insert_object(0, entity);
        indices.insert_light(17, entity);
        indices.insert_decal(23, entity);
        assert_eq!(indices.object_by_tag(0), None);
        assert_eq!(indices.object_by_tag(17), None);
        assert_eq!(indices.light_by_tag(17), Some(entity));
        assert_eq!(indices.decal_by_tag(23), Some(entity));
    }
}
