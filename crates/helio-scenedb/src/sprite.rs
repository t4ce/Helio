//! SceneDB authority for Helio's 2D sprite pipeline.
//!
//! The canonical sprite row is a regular World component with one named GPU
//! partner.  SceneDB's component-local row allocator and presence column are
//! therefore the slot allocator and alive table; render passes must not keep a
//! second CPU pool.  Atlas image objects remain GPU residency, while stable
//! layer identity and configuration are ordinary SceneDB components.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use pulsar_scenedb::gpu::SceneGpuStore;
use pulsar_scenedb::gpu::CapacityError;
use pulsar_scenedb::page::Pod as SceneDbPod;
use pulsar_scenedb::{component_id, Entity, Subsystem};
use pulsar_scenedb_derive::SceneStore;

pub const SPRITE_BUFFER_KEY: &str = "helio.scene.sprites";
pub const SPRITE_ROW_BYTES: u64 = std::mem::size_of::<SceneSpriteRow>() as u64;
pub const SPRITE_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// A generation-checked sprite identity.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneSpriteId(pub Entity);

impl SceneSpriteId {
    pub const fn entity(self) -> Entity {
        self.0
    }

    pub fn bits(self) -> u64 {
        self.0.bits()
    }

    pub fn from_bits(bits: u64) -> Self {
        Self(Entity::from_bits(bits))
    }
}

/// Stable identity of one resident atlas layer.
///
/// The shader sees a physical array layer resolved from this identity.  That
/// layer is deliberately not exposed as the authored handle: residency slots
/// may be reused after removal, while this Entity generation cannot alias a
/// stale reference.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneSpriteAtlasId(pub Entity);

impl SceneSpriteAtlasId {
    pub const fn entity(self) -> Entity {
        self.0
    }

    pub fn bits(self) -> u64 {
        self.0.bits()
    }

    pub fn from_bits(bits: u64) -> Self {
        Self(Entity::from_bits(bits))
    }
}

/// Shader-exact authored sprite row.
///
/// Two otherwise-required WGSL padding regions carry useful CPU/GPU data at
/// no stride cost: `simulation_velocity` is the initial velocity consumed by
/// the optional simulation pass, and the tail stores the generation-bearing
/// atlas Entity plus an authored epoch.  Shaders that do not need those fields
/// may ignore them, but must retain the exact 80-byte layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneSpriteRow {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub rotation: f32,
    pub depth: f32,
    pub simulation_velocity: [f32; 2],
    pub uv_rect: [f32; 4],
    pub color: [f32; 4],
    /// Renderer-resolved physical layer. This is deliberately private: atlas
    /// Entity identity is authored, while the physical slot is residency.
    atlas_layer: u32,
    atlas_entity_bits: [u32; 2],
    authored_epoch: u32,
}

unsafe impl SceneDbPod for SceneSpriteRow {}

impl SceneSpriteRow {
    pub fn new(position: [f32; 2], size: [f32; 2]) -> Self {
        let dangling = Entity::DANGLING.bits();
        Self {
            position,
            size,
            rotation: 0.0,
            depth: 0.0,
            simulation_velocity: [0.0; 2],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            atlas_layer: 0,
            atlas_entity_bits: [dangling as u32, (dangling >> 32) as u32],
            authored_epoch: 0,
        }
    }

    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    pub fn with_depth(mut self, depth: f32) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_uv_rect(mut self, uv_rect: [f32; 4]) -> Self {
        self.uv_rect = uv_rect;
        self
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_atlas_layer(mut self, layer: SceneSpriteAtlasId) -> Self {
        self.set_atlas(layer);
        self
    }

    /// Opt this sprite into GPU simulation. A zero vector leaves it authored.
    pub fn with_simulation_velocity(mut self, velocity: [f32; 2]) -> Self {
        self.simulation_velocity = velocity;
        self
    }

    pub fn atlas(self) -> Option<SceneSpriteAtlasId> {
        let bits = u64::from(self.atlas_entity_bits[0])
            | (u64::from(self.atlas_entity_bits[1]) << 32);
        let entity = Entity::from_bits(bits);
        (entity != Entity::DANGLING).then_some(SceneSpriteAtlasId(entity))
    }

    pub fn authored_epoch(self) -> u32 {
        self.authored_epoch
    }

    #[doc(hidden)]
    pub fn set_atlas(&mut self, layer: SceneSpriteAtlasId) {
        let bits = layer.bits();
        self.atlas_entity_bits = [bits as u32, (bits >> 32) as u32];
    }

    #[doc(hidden)]
    pub fn resolve_atlas_for_gpu(&mut self, atlas: SceneSpriteAtlasId, physical_layer: u32) {
        self.set_atlas(atlas);
        self.atlas_layer = physical_layer;
    }

    #[doc(hidden)]
    pub fn clear_atlas(&mut self) {
        let bits = Entity::DANGLING.bits();
        self.atlas_entity_bits = [bits as u32, (bits >> 32) as u32];
        self.atlas_layer = 0;
    }

    #[doc(hidden)]
    pub fn set_authored_epoch(&mut self, epoch: u32) {
        self.authored_epoch = epoch;
    }
}

/// Canonical sprite component. The single field is already shader-exact.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneSprite {
    #[gpu(buffer = "helio.scene.sprites")]
    pub sprite: SceneSpriteRow,
}

/// Persistent identity/configuration for one atlas image.
///
/// Raw uncompressed pixels intentionally do not have a second CPU copy here.
/// They are handed to [`SpriteAtlasResidency`] once; `content_hash` is the
/// stable reload/cache key an asset integration can replace with its own key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneSpriteAtlasLayer {
    pub width: u32,
    pub height: u32,
    pub content_hash: [u64; 2],
}

#[derive(Debug)]
pub enum SpriteAuthorityError {
    StaleSprite(SceneSpriteId),
    StaleAtlas(SceneSpriteAtlasId),
    Atlas(SpriteAtlasError),
    Capacity(CapacityError),
    CapacityRequestTooLarge(usize),
    AuthoredEpochExhausted,
    ExternalAuthority,
}

impl fmt::Display for SpriteAuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleSprite(handle) => write!(f, "stale sprite handle {handle:?}"),
            Self::StaleAtlas(handle) => write!(f, "stale sprite-atlas handle {handle:?}"),
            Self::Atlas(error) => error.fmt(f),
            Self::Capacity(error) => error.fmt(f),
            Self::CapacityRequestTooLarge(capacity) => {
                write!(f, "sprite capacity {capacity} exceeds u32 row addressing")
            }
            Self::AuthoredEpochExhausted => write!(f, "sprite authored epoch exhausted"),
            Self::ExternalAuthority => write!(
                f,
                "this sprite pass consumes a shared SceneDB publication; mutate the owning Scene authority"
            ),
        }
    }
}

impl std::error::Error for SpriteAuthorityError {}

impl From<SpriteAtlasError> for SpriteAuthorityError {
    fn from(value: SpriteAtlasError) -> Self {
        Self::Atlas(value)
    }
}

impl From<CapacityError> for SpriteAuthorityError {
    fn from(value: CapacityError) -> Self {
        Self::Capacity(value)
    }
}

/// One immutable snapshot consumed by sprite render/compute passes.
#[derive(Clone)]
pub struct SpriteBufferSnapshot {
    pub instances: wgpu::Buffer,
    pub instances_epoch: u64,
    pub presence: wgpu::Buffer,
    pub presence_epoch: u64,
    pub row_span: u32,
    pub runtime: Option<SpriteRuntimeSnapshot>,
}

/// Optional pass-derived simulation rows. This is deliberately not authored
/// scene storage: each row is valid only while its `authored_epoch` matches
/// the SceneDB row at the same component-local index.
#[derive(Clone)]
pub struct SpriteRuntimeSnapshot {
    pub buffer: wgpu::Buffer,
    pub epoch: u64,
    pub row_capacity: u32,
    token: u64,
}

#[derive(Clone)]
pub struct SpriteAtlasSnapshot {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub epoch: u64,
}

/// Allocation-epoch-aware atlas binding publication. Integrated render
/// passes retain this source instead of retaining or recreating the owning
/// SceneDB authority.
#[derive(Clone)]
pub struct SpriteAtlasSource {
    inner: Arc<RwLock<SpriteAtlasSnapshot>>,
}

impl SpriteAtlasSource {
    fn new(view: wgpu::TextureView, sampler: wgpu::Sampler, epoch: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SpriteAtlasSnapshot {
                view,
                sampler,
                epoch,
            })),
        }
    }

    pub fn snapshot(&self) -> SpriteAtlasSnapshot {
        self.inner
            .read()
            .expect("sprite atlas source lock poisoned")
            .clone()
    }

    fn publish(&self, view: wgpu::TextureView, sampler: wgpu::Sampler, epoch: u64) {
        *self
            .inner
            .write()
            .expect("sprite atlas source lock poisoned") = SpriteAtlasSnapshot {
            view,
            sampler,
            epoch,
        };
    }
}

/// A SceneDB sprite publication may have one authoritative runtime projection.
/// Multiple simulation writers would otherwise race and make the batch/cull
/// consumers observe whichever pass prepared last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpriteRuntimeAlreadyInstalled;

impl fmt::Display for SpriteRuntimeAlreadyInstalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "this sprite source already has a runtime projection owner")
    }
}

impl std::error::Error for SpriteRuntimeAlreadyInstalled {}

struct SpriteBufferState {
    instances: wgpu::Buffer,
    instances_epoch: u64,
    presence: wgpu::Buffer,
    presence_epoch: u64,
    row_span: u32,
    runtime: Option<SpriteRuntimeSnapshot>,
    next_runtime_token: u64,
    runtime_epoch_counter: u64,
}

/// Allocation-epoch-aware publication shared by batch, cull, and simulation.
/// Consumers clone cheap wgpu handles only when an epoch changes.
#[derive(Clone)]
pub struct SpriteBufferSource {
    inner: Arc<RwLock<SpriteBufferState>>,
}

impl SpriteBufferSource {
    pub fn new(
        instances: wgpu::Buffer,
        instances_epoch: u64,
        presence: wgpu::Buffer,
        presence_epoch: u64,
        row_span: u32,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SpriteBufferState {
                instances,
                instances_epoch,
                presence,
                presence_epoch,
                row_span,
                runtime: None,
                next_runtime_token: 1,
                runtime_epoch_counter: 0,
            })),
        }
    }

    pub fn snapshot(&self) -> SpriteBufferSnapshot {
        let state = self.inner.read().expect("sprite buffer source lock poisoned");
        SpriteBufferSnapshot {
            instances: state.instances.clone(),
            instances_epoch: state.instances_epoch,
            presence: state.presence.clone(),
            presence_epoch: state.presence_epoch,
            row_span: state.row_span,
            runtime: state.runtime.clone(),
        }
    }

    pub fn publish_authored(
        &self,
        instances: wgpu::Buffer,
        instances_epoch: u64,
        presence: wgpu::Buffer,
        presence_epoch: u64,
        row_span: u32,
    ) {
        let mut state = self.inner.write().expect("sprite buffer source lock poisoned");
        state.instances = instances;
        state.instances_epoch = instances_epoch;
        state.presence = presence;
        state.presence_epoch = presence_epoch;
        state.row_span = row_span;
    }

    pub fn install_runtime(
        &self,
        buffer: wgpu::Buffer,
        row_capacity: u32,
    ) -> Result<u64, SpriteRuntimeAlreadyInstalled> {
        let mut state = self.inner.write().expect("sprite buffer source lock poisoned");
        if state.runtime.is_some() {
            return Err(SpriteRuntimeAlreadyInstalled);
        }
        let token = state.next_runtime_token;
        state.next_runtime_token = state.next_runtime_token.wrapping_add(1).max(1);
        state.runtime_epoch_counter = state.runtime_epoch_counter.wrapping_add(1).max(1);
        let epoch = state.runtime_epoch_counter;
        state.runtime = Some(SpriteRuntimeSnapshot {
            buffer,
            epoch,
            row_capacity,
            token,
        });
        Ok(token)
    }

    pub fn replace_runtime(
        &self,
        token: u64,
        buffer: wgpu::Buffer,
        row_capacity: u32,
    ) -> bool {
        let mut state = self.inner.write().expect("sprite buffer source lock poisoned");
        if !state.runtime.as_ref().is_some_and(|runtime| runtime.token == token) {
            return false;
        }
        state.runtime_epoch_counter = state.runtime_epoch_counter.wrapping_add(1).max(1);
        let epoch = state.runtime_epoch_counter;
        let runtime = state.runtime.as_mut().expect("runtime token was just validated");
        runtime.buffer = buffer;
        runtime.row_capacity = row_capacity;
        runtime.epoch = epoch;
        true
    }

    pub fn remove_runtime(&self, token: u64) {
        let mut state = self.inner.write().expect("sprite buffer source lock poisoned");
        if state.runtime.as_ref().is_some_and(|runtime| runtime.token == token) {
            state.runtime = None;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpriteAtlasError {
    ZeroExtent,
    ByteLengthOverflow,
    InvalidByteLength { expected: usize, actual: usize },
    DimensionMismatch { expected: [u32; 2], actual: [u32; 2] },
    HardwareCapacityExceeded { maximum_layers: u32 },
    DimensionLimitExceeded { maximum: u32, actual: [u32; 2] },
    AlreadyResident,
    NotResident,
    LayerInUse { references: u32 },
    ReferenceCountOverflow,
    ReferenceCountUnderflow,
}

impl fmt::Display for SpriteAtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroExtent => write!(f, "sprite atlas layers must have non-zero dimensions"),
            Self::ByteLengthOverflow => write!(f, "sprite atlas byte length overflows usize"),
            Self::InvalidByteLength { expected, actual } => write!(
                f,
                "sprite atlas needs {expected} RGBA8 bytes, received {actual}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "sprite atlas layer is {}x{}, but this atlas is {}x{}",
                actual[0], actual[1], expected[0], expected[1]
            ),
            Self::HardwareCapacityExceeded { maximum_layers } => write!(
                f,
                "sprite atlas reached {maximum_layers} imported layers; one device array layer is reserved for the white fallback"
            ),
            Self::DimensionLimitExceeded { maximum, actual } => write!(
                f,
                "sprite atlas layer {}x{} exceeds the device 2D limit {maximum}",
                actual[0], actual[1]
            ),
            Self::AlreadyResident => write!(f, "sprite atlas entity is already resident"),
            Self::NotResident => write!(f, "sprite atlas entity is not resident"),
            Self::LayerInUse { references } => {
                write!(f, "sprite atlas layer still has {references} sprite references")
            }
            Self::ReferenceCountOverflow => write!(f, "sprite atlas reference count overflowed"),
            Self::ReferenceCountUnderflow => write!(f, "sprite atlas reference count underflowed"),
        }
    }
}

impl std::error::Error for SpriteAtlasError {}

#[derive(Debug, Clone, Copy)]
struct AtlasRecord {
    slot: u32,
    references: u32,
}

/// SceneDB-registered physical residency for one RGBA8-sRGB texture array.
/// Stable Entity relationships stay in components; only physical slots,
/// texture allocation, and reference counts live here.
pub struct SpriteAtlasResidency {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    dimensions: Option<[u32; 2]>,
    capacity: u32,
    next_slot: u32,
    free_slots: Vec<u32>,
    records: HashMap<Entity, AtlasRecord>,
    epoch: u64,
    maximum_layers: u32,
    maximum_dimension: u32,
    publication: SpriteAtlasSource,
}

impl SpriteAtlasResidency {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let maximum_layers = device.limits().max_texture_array_layers.max(1);
        let maximum_dimension = device.limits().max_texture_dimension_2d;
        let texture = create_atlas_texture(&device, 1, 1, 1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SceneDB Sprite Atlas Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let publication = SpriteAtlasSource::new(view.clone(), sampler.clone(), 1);
        Self {
            device,
            queue,
            texture,
            view,
            sampler,
            dimensions: None,
            capacity: 1,
            next_slot: 1,
            free_slots: Vec::new(),
            records: HashMap::new(),
            epoch: 1,
            maximum_layers,
            maximum_dimension,
            publication,
        }
    }

    pub fn validate(
        &self,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), SpriteAtlasError> {
        if width == 0 || height == 0 {
            return Err(SpriteAtlasError::ZeroExtent);
        }
        if width > self.maximum_dimension || height > self.maximum_dimension {
            return Err(SpriteAtlasError::DimensionLimitExceeded {
                maximum: self.maximum_dimension,
                actual: [width, height],
            });
        }
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| usize::try_from(height).ok().and_then(|height| width.checked_mul(height)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(SpriteAtlasError::ByteLengthOverflow)?;
        if rgba8.len() != expected {
            return Err(SpriteAtlasError::InvalidByteLength {
                expected,
                actual: rgba8.len(),
            });
        }
        if let Some(expected) = self.dimensions {
            if expected != [width, height] {
                return Err(SpriteAtlasError::DimensionMismatch {
                    expected,
                    actual: [width, height],
                });
            }
        }
        if self.free_slots.is_empty() && self.next_slot >= self.maximum_layers {
            return Err(SpriteAtlasError::HardwareCapacityExceeded {
                maximum_layers: self.maximum_layers.saturating_sub(1),
            });
        }
        Ok(())
    }

    pub fn insert(
        &mut self,
        entity: Entity,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<u32, SpriteAtlasError> {
        self.validate(width, height, rgba8)?;
        if self.records.contains_key(&entity) {
            return Err(SpriteAtlasError::AlreadyResident);
        }
        if self.dimensions.is_none() {
            self.recreate_for_dimensions(width, height)?;
        }
        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            if self.next_slot >= self.capacity {
                self.grow()?;
            }
            let slot = self.next_slot;
            self.next_slot += 1;
            slot
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: slot },
                aspect: wgpu::TextureAspect::All,
            },
            rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.records.insert(
            entity,
            AtlasRecord {
                slot,
                references: 0,
            },
        );
        Ok(slot)
    }

    fn recreate_for_dimensions(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), SpriteAtlasError> {
        let capacity = 4.min(self.maximum_layers).max(1);
        if capacity <= 1 {
            return Err(SpriteAtlasError::HardwareCapacityExceeded {
                maximum_layers: self.maximum_layers.saturating_sub(1),
            });
        }
        let texture = create_atlas_texture(&self.device, width, height, capacity);
        clear_fallback_layer_white(&self.device, &self.queue, &texture);
        self.texture = texture;
        self.view = self.texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.dimensions = Some([width, height]);
        self.capacity = capacity;
        self.next_slot = 1;
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.publication
            .publish(self.view.clone(), self.sampler.clone(), self.epoch);
        Ok(())
    }

    fn grow(&mut self) -> Result<(), SpriteAtlasError> {
        let new_capacity = self
            .capacity
            .saturating_mul(2)
            .min(self.maximum_layers);
        if new_capacity <= self.capacity {
            return Err(SpriteAtlasError::HardwareCapacityExceeded {
                maximum_layers: self.maximum_layers.saturating_sub(1),
            });
        }
        let [width, height] = self
            .dimensions
            .expect("a real atlas is dimensioned before it can grow");
        let texture = create_atlas_texture(&self.device, width, height, new_capacity);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SceneDB Sprite Atlas Grow"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: self.next_slot,
            },
        );
        self.queue.submit([encoder.finish()]);
        self.texture = texture;
        self.view = self.texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.capacity = new_capacity;
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.publication
            .publish(self.view.clone(), self.sampler.clone(), self.epoch);
        Ok(())
    }

    pub fn resolve(&self, entity: Entity) -> Option<u32> {
        self.records.get(&entity).map(|record| record.slot)
    }

    pub fn retain(&mut self, entity: Entity) -> Result<u32, SpriteAtlasError> {
        let record = self
            .records
            .get_mut(&entity)
            .ok_or(SpriteAtlasError::NotResident)?;
        record.references = record
            .references
            .checked_add(1)
            .ok_or(SpriteAtlasError::ReferenceCountOverflow)?;
        Ok(record.slot)
    }

    pub fn release_reference(&mut self, entity: Entity) -> Result<(), SpriteAtlasError> {
        let record = self
            .records
            .get_mut(&entity)
            .ok_or(SpriteAtlasError::NotResident)?;
        record.references = record
            .references
            .checked_sub(1)
            .ok_or(SpriteAtlasError::ReferenceCountUnderflow)?;
        Ok(())
    }

    /// Preflight the infallible half of an authored sprite removal/update.
    /// Callers can validate before mutating SceneDB, then perform the matching
    /// release knowing that single-threaded authority access cannot fail.
    pub fn validate_release_reference(&self, entity: Entity) -> Result<(), SpriteAtlasError> {
        let record = self
            .records
            .get(&entity)
            .ok_or(SpriteAtlasError::NotResident)?;
        if record.references == 0 {
            return Err(SpriteAtlasError::ReferenceCountUnderflow);
        }
        Ok(())
    }

    pub fn remove(&mut self, entity: Entity) -> Result<u32, SpriteAtlasError> {
        let record = self
            .records
            .get(&entity)
            .copied()
            .ok_or(SpriteAtlasError::NotResident)?;
        if record.references != 0 {
            return Err(SpriteAtlasError::LayerInUse {
                references: record.references,
            });
        }
        self.records.remove(&entity);
        self.free_slots.push(record.slot);
        Ok(record.slot)
    }

    pub fn references(&self, entity: Entity) -> Option<u32> {
        self.records.get(&entity).map(|record| record.references)
    }

    pub fn publication(&self) -> (wgpu::TextureView, wgpu::Sampler, u64) {
        let publication = self.publication.snapshot();
        (publication.view, publication.sampler, publication.epoch)
    }

    pub fn publication_source(&self) -> SpriteAtlasSource {
        self.publication.clone()
    }

    pub fn live_count(&self) -> u32 {
        self.records.len() as u32
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn maximum_imported_layers(&self) -> u32 {
        self.maximum_layers.saturating_sub(1)
    }

    pub fn maximum_dimension(&self) -> u32 {
        self.maximum_dimension
    }
}

impl Subsystem for SpriteAtlasResidency {
    fn name(&self) -> &'static str {
        "helio.scene.sprite_atlas_residency"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn create_atlas_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    layers: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SceneDB Sprite Atlas Array"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SPRITE_ATLAS_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn clear_fallback_layer_white(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) {
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("SceneDB Sprite Atlas White Fallback View"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: 0,
        array_layer_count: Some(1),
        ..Default::default()
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SceneDB Sprite Atlas White Fallback Encoder"),
    });
    {
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })];
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SceneDB Sprite Atlas White Fallback Clear"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    queue.submit([encoder.finish()]);
}

pub fn register_sprite_component_buffer(
    store: &mut SceneGpuStore,
    initial_capacity: u32,
    device: &Arc<wgpu::Device>,
) {
    SceneSprite::register_gpu_columns_growable(store, initial_capacity.max(1), device);
}

/// Snapshot SceneDB's component-local presence buffer for [`SceneSprite`].
pub fn sprite_presence_snapshot(store: &SceneGpuStore) -> Option<(wgpu::Buffer, u64)> {
    store.component_presence_buffer_snapshot_for_id(component_id::<SceneSprite>())
}

/// Create a shared publication from a SceneAuthority that registered the
/// sprite partner during construction.
pub fn sprite_buffer_source_for(
    authority: &crate::storage::SceneAuthority,
) -> Option<SpriteBufferSource> {
    let instances = authority.partner_buffer_snapshot(SPRITE_BUFFER_KEY)?;
    let (presence, presence_epoch) = sprite_presence_snapshot(authority.gpu_store())?;
    Some(SpriteBufferSource::new(
        instances.buffer,
        instances.epoch,
        presence,
        presence_epoch,
        authority.gpu_row_span::<SceneSprite>(),
    ))
}

/// Publish the authority's latest allocation identities and component-local
/// row span into an existing shared source.
pub fn refresh_sprite_buffer_source(
    authority: &crate::storage::SceneAuthority,
    source: &SpriteBufferSource,
) -> bool {
    let Some(instances) = authority.partner_buffer_snapshot(SPRITE_BUFFER_KEY) else {
        return false;
    };
    let Some((presence, presence_epoch)) = sprite_presence_snapshot(authority.gpu_store()) else {
        return false;
    };
    source.publish_authored(
        instances.buffer,
        instances.epoch,
        presence,
        presence_epoch,
        authority.gpu_row_span::<SceneSprite>(),
    );
    true
}

/// Stable, allocation-free content identity used by the typed atlas import
/// API. Asset integrations may replace it with their own stable key.
pub fn sprite_content_hash(bytes: &[u8]) -> [u64; 2] {
    let mut a = 0xcbf2_9ce4_8422_2325u64;
    let mut b = 0x9e37_79b9_7f4a_7c15u64;
    for &byte in bytes {
        a ^= u64::from(byte);
        a = a.wrapping_mul(0x0000_0100_0000_01b3);
        b ^= a.rotate_left(17).wrapping_add(u64::from(byte));
        b = b.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    [a, b]
}

const _: () = {
    assert!(std::mem::size_of::<SceneSpriteRow>() == 80);
    assert!(std::mem::offset_of!(SceneSpriteRow, simulation_velocity) == 24);
    assert!(std::mem::offset_of!(SceneSpriteRow, uv_rect) == 32);
    assert!(std::mem::offset_of!(SceneSpriteRow, color) == 48);
    assert!(std::mem::offset_of!(SceneSpriteRow, atlas_layer) == 64);
    assert!(std::mem::offset_of!(SceneSpriteRow, atlas_entity_bits) == 68);
    assert!(std::mem::offset_of!(SceneSpriteRow, authored_epoch) == 76);
    assert!(std::mem::size_of::<SceneSprite>() == 80);
};
