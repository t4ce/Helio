//! Canonical material/texture lifecycle and SceneDB-owned texture residency.
//!
//! World components keep authored metadata, generation-bearing relations,
//! and reference counts. [`TextureResidency`] wraps SceneDB's existing
//! [`TextureStore`] so there is exactly one owner of each `wgpu::Texture`;
//! the subsystem adds the matching view and sampler at the same stable slot.
//! Renderer bind arrays must preserve this slot order and fill vacant entries
//! with their placeholder resources rather than compacting or reordering.

use std::collections::HashMap;
use std::sync::Arc;

use pulsar_scenedb::gpu::{TextureError, TextureStore};
use pulsar_scenedb::{Entity, Subsystem};

use crate::components::{
    SceneMaterial, SceneMaterialRow, SceneMaterialTextureRef, SceneMaterialTextureRefs,
    SceneMaterialTextureSlotRow, SceneMaterialTexturesRow, SceneTexture, SceneTextureAssetKey,
    SceneTextureSampler,
};
use crate::storage::SceneAuthority;

#[derive(Debug)]
struct ResidentTexture {
    slot: u32,
    asset_key: SceneTextureAssetKey,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Secondary safety index. `SceneTexture::ref_count` is canonical; this
    /// cached count prevents a physical slot from being recycled if a caller
    /// accidentally bypasses the component lifecycle API.
    material_pins: u32,
}

/// SceneDB subsystem owning material texture GPU residency.
///
/// `slot_for(entity)` is stable until that exact generation-bearing entity is
/// unpinned and removed. It is the physical array index consumed by material
/// shader rows; Helio must bind `view_for_slot(i)`/`sampler_for_slot(i)` at
/// index `i` without sorting or dense repacking.
pub struct TextureResidency {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    textures: TextureStore,
    residents: HashMap<Entity, ResidentTexture>,
    entities_by_slot: Vec<Option<Entity>>,
    entities_by_asset: HashMap<SceneTextureAssetKey, Entity>,
    /// High-water mark for SceneDB-minted asset identities only. Explicit
    /// caller-owned keys are unordered (often hashes/UUIDs), so they never
    /// advance or exhaust this counter. This state intentionally survives
    /// scene-content clears and removals: an automatic key is never recycled.
    next_asset_key: Option<u128>,
    binding_epoch: u64,
}

fn allocate_monotonic_asset_key(
    next_asset_key: &mut Option<u128>,
    live_assets: &HashMap<SceneTextureAssetKey, Entity>,
) -> Result<SceneTextureAssetKey, SceneAssetError> {
    loop {
        let raw = next_asset_key.ok_or(SceneAssetError::TextureAssetKeyExhausted)?;
        *next_asset_key = raw.checked_add(1);
        let candidate = SceneTextureAssetKey(raw);
        if !live_assets.contains_key(&candidate) {
            return Ok(candidate);
        }
    }
}

#[cfg(test)]
mod asset_key_tests {
    use super::{allocate_monotonic_asset_key, SceneAssetError, SceneTextureAssetKey};
    use std::collections::HashMap;

    #[test]
    fn automatic_domain_exhaustion_never_wraps() {
        let mut next = Some(u128::MAX);
        let live_assets = HashMap::new();
        assert_eq!(
            allocate_monotonic_asset_key(&mut next, &live_assets),
            Ok(SceneTextureAssetKey(u128::MAX))
        );
        assert_eq!(
            allocate_monotonic_asset_key(&mut next, &live_assets),
            Err(SceneAssetError::TextureAssetKeyExhausted)
        );
    }
}

impl TextureResidency {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, max_slots: u32) -> Self {
        Self {
            device,
            queue,
            textures: TextureStore::new(max_slots),
            residents: HashMap::new(),
            entities_by_slot: Vec::new(),
            entities_by_asset: HashMap::new(),
            next_asset_key: Some(1),
            binding_epoch: 0,
        }
    }

    /// Mint a monotonic, non-recycling texture asset identity.
    ///
    /// Live explicit keys are skipped collision-safely without changing the
    /// counter's ordering policy. A key is consumed as soon as it is returned;
    /// a later upload failure does not make it reusable.
    pub fn allocate_asset_key(&mut self) -> Result<SceneTextureAssetKey, SceneAssetError> {
        allocate_monotonic_asset_key(&mut self.next_asset_key, &self.entities_by_asset)
    }

    pub fn binding_epoch(&self) -> u64 {
        self.binding_epoch
    }

    /// One past the highest SceneDB residency slot ever allocated.
    /// Vacant entries inside this extent must remain placeholders.
    pub fn slot_count(&self) -> u32 {
        self.textures.slot_count()
    }

    pub fn slot_for(&self, entity: Entity) -> Option<u32> {
        self.residents.get(&entity).map(|entry| entry.slot)
    }

    pub fn entity_for_slot(&self, slot: u32) -> Option<Entity> {
        self.entities_by_slot.get(slot as usize).copied().flatten()
    }

    pub fn entity_for_asset(&self, asset_key: SceneTextureAssetKey) -> Option<Entity> {
        self.entities_by_asset.get(&asset_key).copied()
    }

    pub fn texture(&self, entity: Entity) -> Option<&wgpu::Texture> {
        let slot = self.slot_for(entity)?;
        self.textures.texture(slot)
    }

    pub fn view(&self, entity: Entity) -> Option<&wgpu::TextureView> {
        self.residents.get(&entity).map(|entry| &entry.view)
    }

    pub fn sampler(&self, entity: Entity) -> Option<&wgpu::Sampler> {
        self.residents.get(&entity).map(|entry| &entry.sampler)
    }

    pub fn view_for_slot(&self, slot: u32) -> Option<&wgpu::TextureView> {
        self.entity_for_slot(slot)
            .and_then(|entity| self.view(entity))
    }

    pub fn sampler_for_slot(&self, slot: u32) -> Option<&wgpu::Sampler> {
        self.entity_for_slot(slot)
            .and_then(|entity| self.sampler(entity))
    }

    pub fn material_pin_count(&self, entity: Entity) -> Option<u32> {
        self.residents.get(&entity).map(|entry| entry.material_pins)
    }

    /// Resolve canonical texture entities into the two shader rows consumed
    /// by Helio. This is a pure projection: lifecycle methods apply the
    /// matching reference-count pins before publishing the rows.
    pub fn project_material_rows(
        &self,
        mut authored: libhelio::GpuMaterial,
        refs: &SceneMaterialTextureRefs,
    ) -> Result<(SceneMaterialRow, SceneMaterialTexturesRow), SceneAssetError> {
        let base_color = self.project_slot(refs.base_color)?;
        let normal = self.project_slot(refs.normal)?;
        let roughness_metallic = self.project_slot(refs.roughness_metallic)?;
        let emissive = self.project_slot(refs.emissive)?;
        let occlusion = self.project_slot(refs.occlusion)?;
        let specular_color = self.project_slot(refs.specular_color)?;
        let specular_weight = self.project_slot(refs.specular_weight)?;

        // Any caller-provided indices are renderer coordinates from the old
        // API, not authored identity. Always overwrite them with the exact
        // non-compacting SceneDB residency indices resolved above.
        authored.tex_base_color = base_color.texture_index;
        authored.tex_normal = normal.texture_index;
        authored.tex_roughness = roughness_metallic.texture_index;
        authored.tex_emissive = emissive.texture_index;
        authored.tex_occlusion = occlusion.texture_index;

        Ok((
            SceneMaterialRow(authored),
            SceneMaterialTexturesRow {
                base_color,
                normal,
                roughness_metallic,
                emissive,
                occlusion,
                specular_color,
                specular_weight,
                params: [
                    refs.normal_scale,
                    refs.occlusion_strength,
                    refs.alpha_cutoff,
                    0.0,
                ],
            },
        ))
    }

    fn project_slot(
        &self,
        reference: Option<SceneMaterialTextureRef>,
    ) -> Result<SceneMaterialTextureSlotRow, SceneAssetError> {
        let Some(reference) = reference else {
            return Ok(SceneMaterialTextureSlotRow::missing());
        };
        let slot = self
            .slot_for(reference.texture)
            .ok_or(SceneAssetError::TextureNotResident(reference.texture))?;
        let rotation = reference.transform.rotation_radians;
        Ok(SceneMaterialTextureSlotRow {
            texture_index: slot,
            // Keep high bits intact: Helio material shaders use them for
            // slot-local sampling modifiers (for example, repeating a tiled
            // UV before applying an atlas transform). The low bits still
            // select the authored UV channel.
            uv_channel: reference.uv_channel,
            _pad: [0; 2],
            offset_scale: [
                reference.transform.offset[0],
                reference.transform.offset[1],
                reference.transform.scale[0],
                reference.transform.scale[1],
            ],
            rotation: [rotation.sin(), rotation.cos(), 0.0, 0.0],
        })
    }

    fn register(
        &mut self,
        entity: Entity,
        metadata: &SceneTexture,
        label: Option<&str>,
        data: &[u8],
        mip_data: &[Vec<u8>],
    ) -> Result<u32, SceneAssetError> {
        if !metadata.asset_key.is_valid() {
            return Err(SceneAssetError::InvalidTextureAssetKey);
        }
        if self.residents.contains_key(&entity) {
            return Err(SceneAssetError::TextureAlreadyResident(entity));
        }
        if let Some(existing) = self.entity_for_asset(metadata.asset_key) {
            return Err(SceneAssetError::DuplicateTextureAsset {
                asset_key: metadata.asset_key,
                existing,
            });
        }
        validate_texture_metadata(metadata, data.len())?;

        let descriptor = wgpu::TextureDescriptor {
            label,
            size: metadata.size,
            mip_level_count: metadata.mip_level_count,
            sample_count: metadata.sample_count,
            dimension: metadata.dimension,
            format: metadata.format,
            usage: metadata.usage,
            view_formats: &[],
        };
        let slot = self
            .textures
            .register_mips(
                &self.device,
                &self.queue,
                &descriptor,
                &std::iter::once(data)
                    .chain(mip_data.iter().map(Vec::as_slice))
                    .collect::<Vec<_>>(),
            )
            .map_err(SceneAssetError::TextureStore)?;
        let view = self
            .textures
            .texture(slot)
            .expect("TextureStore returned a vacant slot after register")
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label,
            address_mode_u: metadata.sampler.address_mode_u,
            address_mode_v: metadata.sampler.address_mode_v,
            address_mode_w: metadata.sampler.address_mode_w,
            mag_filter: metadata.sampler.mag_filter,
            min_filter: metadata.sampler.min_filter,
            mipmap_filter: metadata.sampler.mipmap_filter,
            ..Default::default()
        });

        let slot_index = slot as usize;
        if self.entities_by_slot.len() <= slot_index {
            self.entities_by_slot.resize(slot_index + 1, None);
        }
        debug_assert!(self.entities_by_slot[slot_index].is_none());
        self.entities_by_slot[slot_index] = Some(entity);
        self.entities_by_asset.insert(metadata.asset_key, entity);
        self.residents.insert(
            entity,
            ResidentTexture {
                slot,
                asset_key: metadata.asset_key,
                view,
                sampler,
                material_pins: 0,
            },
        );
        self.binding_epoch = self.binding_epoch.wrapping_add(1);
        Ok(slot)
    }

    fn unregister(&mut self, entity: Entity) -> Result<u32, SceneAssetError> {
        let entry = self
            .residents
            .get(&entity)
            .ok_or(SceneAssetError::TextureNotResident(entity))?;
        if entry.material_pins != 0 {
            return Err(SceneAssetError::TextureInUse {
                entity,
                ref_count: entry.material_pins,
            });
        }
        let slot = entry.slot;
        let asset_key = entry.asset_key;
        self.textures
            .unregister(slot)
            .map_err(SceneAssetError::TextureStore)?;
        self.residents.remove(&entity);
        self.entities_by_asset.remove(&asset_key);
        self.entities_by_slot[slot as usize] = None;
        self.binding_epoch = self.binding_epoch.wrapping_add(1);
        Ok(slot)
    }

    fn replace_sampler(
        &mut self,
        entity: Entity,
        sampler: SceneTextureSampler,
        label: Option<&str>,
    ) -> Result<(), SceneAssetError> {
        let entry = self
            .residents
            .get_mut(&entity)
            .ok_or(SceneAssetError::TextureNotResident(entity))?;
        entry.sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label,
            address_mode_u: sampler.address_mode_u,
            address_mode_v: sampler.address_mode_v,
            address_mode_w: sampler.address_mode_w,
            mag_filter: sampler.mag_filter,
            min_filter: sampler.min_filter,
            mipmap_filter: sampler.mipmap_filter,
            ..Default::default()
        });
        self.binding_epoch = self.binding_epoch.wrapping_add(1);
        Ok(())
    }

    fn adjust_material_pins(&mut self, entity: Entity, delta: i64) {
        let entry = self
            .residents
            .get_mut(&entity)
            .expect("validated texture residency disappeared during material mutation");
        entry.material_pins = if delta >= 0 {
            entry
                .material_pins
                .checked_add(delta as u32)
                .expect("validated texture pin addition overflowed")
        } else {
            entry
                .material_pins
                .checked_sub((-delta) as u32)
                .expect("validated texture pin subtraction underflowed")
        };
    }
}

impl Subsystem for TextureResidency {
    fn name(&self) -> &'static str {
        "helio.scene.texture_residency"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Material/texture authority errors. Every removal and slot-reuse failure is
/// explicit; no stale material row can silently begin sampling a replacement
/// texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneAssetError {
    InvalidTextureAssetKey,
    TextureAssetKeyExhausted,
    DuplicateTextureAsset {
        asset_key: SceneTextureAssetKey,
        existing: Entity,
    },
    TextureAlreadyResident(Entity),
    TextureNotResident(Entity),
    TextureMissingComponent(Entity),
    MaterialMissingComponent(Entity),
    MaterialTextureRefsMissing(Entity),
    TextureRefCountMustStartAtZero(u32),
    TextureInUse {
        entity: Entity,
        ref_count: u32,
    },
    MaterialInUse {
        entity: Entity,
        ref_count: u32,
    },
    TextureRefCountOverflow(Entity),
    TextureRefCountUnderflow(Entity),
    MaterialRefCountOverflow(Entity),
    MaterialRefCountUnderflow(Entity),
    TextureRefCountDiverged {
        entity: Entity,
        component: u32,
        residency: u32,
    },
    UnsupportedTexture(&'static str),
    TextureDataLength {
        expected: usize,
        actual: usize,
    },
    TextureStore(TextureError),
}

impl std::fmt::Display for SceneAssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTextureAssetKey => write!(f, "texture asset key zero is reserved"),
            Self::TextureAssetKeyExhausted => {
                write!(f, "texture asset key space is exhausted")
            }
            Self::DuplicateTextureAsset { asset_key, existing } => write!(
                f,
                "texture asset {:?} is already resident as {existing}",
                asset_key
            ),
            Self::TextureAlreadyResident(entity) => {
                write!(f, "texture {entity} is already resident")
            }
            Self::TextureNotResident(entity) => write!(f, "texture {entity} is not resident"),
            Self::TextureMissingComponent(entity) => {
                write!(f, "entity {entity} has no SceneTexture component")
            }
            Self::MaterialMissingComponent(entity) => {
                write!(f, "entity {entity} has no SceneMaterial component")
            }
            Self::MaterialTextureRefsMissing(entity) => write!(
                f,
                "entity {entity} has no SceneMaterialTextureRefs component"
            ),
            Self::TextureRefCountMustStartAtZero(count) => write!(
                f,
                "new texture metadata must start with ref_count 0, got {count}"
            ),
            Self::TextureInUse { entity, ref_count } => {
                write!(f, "texture {entity} still has {ref_count} material references")
            }
            Self::MaterialInUse { entity, ref_count } => {
                write!(f, "material {entity} still has {ref_count} object references")
            }
            Self::TextureRefCountOverflow(entity) => {
                write!(f, "texture {entity} reference count overflow")
            }
            Self::TextureRefCountUnderflow(entity) => {
                write!(f, "texture {entity} reference count underflow")
            }
            Self::MaterialRefCountOverflow(entity) => {
                write!(f, "material {entity} reference count overflow")
            }
            Self::MaterialRefCountUnderflow(entity) => {
                write!(f, "material {entity} reference count underflow")
            }
            Self::TextureRefCountDiverged {
                entity,
                component,
                residency,
            } => write!(
                f,
                "texture {entity} component ref_count {component} disagrees with residency pins {residency}"
            ),
            Self::UnsupportedTexture(reason) => write!(f, "unsupported material texture: {reason}"),
            Self::TextureDataLength { expected, actual } => write!(
                f,
                "texture upload byte length is {actual}, expected exactly {expected}"
            ),
            Self::TextureStore(error) => write!(f, "SceneDB TextureStore error: {error:?}"),
        }
    }
}

impl std::error::Error for SceneAssetError {}

fn validate_texture_metadata(
    metadata: &SceneTexture,
    data_len: usize,
) -> Result<(), SceneAssetError> {
    if metadata.size.width == 0 || metadata.size.height == 0 {
        return Err(SceneAssetError::UnsupportedTexture("zero-sized extent"));
    }
    if metadata.dimension != wgpu::TextureDimension::D2 || metadata.size.depth_or_array_layers != 1
    {
        return Err(SceneAssetError::UnsupportedTexture(
            "material residency currently accepts one 2D layer",
        ));
    }
    if metadata.sample_count != 1 {
        return Err(SceneAssetError::UnsupportedTexture(
            "SceneDB TextureStore currently accepts one sample",
        ));
    }
    let required_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if !metadata.usage.contains(required_usage) {
        return Err(SceneAssetError::UnsupportedTexture(
            "usage must contain TEXTURE_BINDING | COPY_DST",
        ));
    }
    let block_size =
        metadata
            .format
            .block_copy_size(None)
            .ok_or(SceneAssetError::UnsupportedTexture(
                "depth, stencil, and multi-planar formats are not material textures",
            ))? as usize;
    let (block_width, block_height) = metadata.format.block_dimensions();
    let physical_size = metadata.size.physical_size(metadata.format);
    let width_blocks = physical_size.width / block_width;
    let height_blocks = physical_size.height / block_height;
    let expected = block_size
        .checked_mul(width_blocks as usize)
        .and_then(|bytes| bytes.checked_mul(height_blocks as usize))
        .and_then(|bytes| bytes.checked_mul(physical_size.depth_or_array_layers as usize))
        .ok_or(SceneAssetError::UnsupportedTexture(
            "texture byte size overflow",
        ))?;
    if data_len != expected {
        return Err(SceneAssetError::TextureDataLength {
            expected,
            actual: data_len,
        });
    }
    Ok(())
}

fn reference_counts(refs: &SceneMaterialTextureRefs) -> Vec<(Entity, u32)> {
    let mut counts = Vec::with_capacity(7);
    for reference in refs.references() {
        if let Some((_, count)) = counts
            .iter_mut()
            .find(|(entity, _)| *entity == reference.texture)
        {
            *count += 1;
        } else {
            counts.push((reference.texture, 1));
        }
    }
    counts
}

fn combined_deltas(
    old: &SceneMaterialTextureRefs,
    new: &SceneMaterialTextureRefs,
) -> Vec<(Entity, i64)> {
    let mut deltas: Vec<(Entity, i64)> = reference_counts(new)
        .into_iter()
        .map(|(entity, count)| (entity, count as i64))
        .collect();
    for (entity, count) in reference_counts(old) {
        if let Some((_, delta)) = deltas
            .iter_mut()
            .find(|(candidate, _)| *candidate == entity)
        {
            *delta -= count as i64;
        } else {
            deltas.push((entity, -(count as i64)));
        }
    }
    deltas.retain(|(_, delta)| *delta != 0);
    deltas
}

impl SceneAuthority {
    /// Insert canonical texture metadata, upload exactly one SceneDB-owned
    /// texture, and create its view/sampler in the registered residency
    /// subsystem. The returned Entity is the only persistent texture handle.
    pub fn insert_texture_asset(
        &mut self,
        metadata: SceneTexture,
        label: Option<&str>,
        data: &[u8],
    ) -> Result<Entity, SceneAssetError> {
        self.insert_texture_asset_with_mips(metadata, label, data, &[])
    }

    /// Insert a texture with complete mip levels. `data` is level zero and
    /// `mip_data` is ordered from level one onwards.
    pub fn insert_texture_asset_with_mips(
        &mut self,
        metadata: SceneTexture,
        label: Option<&str>,
        data: &[u8],
        mip_data: &[Vec<u8>],
    ) -> Result<Entity, SceneAssetError> {
        if metadata.ref_count != 0 {
            return Err(SceneAssetError::TextureRefCountMustStartAtZero(
                metadata.ref_count,
            ));
        }
        let entity = self.db.world.spawn();
        self.db.world.insert(entity, metadata);
        let result = self
            .db
            .subsystem_mut::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new")
            .register(entity, &metadata, label, data, mip_data);
        if let Err(error) = result {
            self.db.world.despawn(entity);
            return Err(error);
        }
        Ok(entity)
    }

    /// Remove an unreferenced texture and release its physical slot. A slot
    /// cannot enter SceneDB's reuse free-list while either the canonical
    /// component count or the residency safety pins are non-zero.
    pub fn remove_texture_asset(
        &mut self,
        entity: Entity,
    ) -> Result<SceneTexture, SceneAssetError> {
        let metadata = *self
            .db
            .world
            .get::<SceneTexture>(entity)
            .ok_or(SceneAssetError::TextureMissingComponent(entity))?;
        let residency_count = self
            .db
            .subsystem::<TextureResidency>()
            .and_then(|residency| residency.material_pin_count(entity))
            .ok_or(SceneAssetError::TextureNotResident(entity))?;
        if metadata.ref_count != residency_count {
            return Err(SceneAssetError::TextureRefCountDiverged {
                entity,
                component: metadata.ref_count,
                residency: residency_count,
            });
        }
        if metadata.ref_count != 0 {
            return Err(SceneAssetError::TextureInUse {
                entity,
                ref_count: metadata.ref_count,
            });
        }
        self.db
            .subsystem_mut::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new")
            .unregister(entity)?;
        self.db.world.despawn(entity);
        Ok(metadata)
    }

    /// Recreate only a texture's sampler. The physical texture slot remains
    /// unchanged; `binding_epoch` advances so renderer bind groups rebind.
    pub fn update_texture_sampler(
        &mut self,
        entity: Entity,
        sampler: SceneTextureSampler,
        label: Option<&str>,
    ) -> Result<(), SceneAssetError> {
        if self.db.world.get::<SceneTexture>(entity).is_none() {
            return Err(SceneAssetError::TextureMissingComponent(entity));
        }
        self.db
            .subsystem_mut::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new")
            .replace_sampler(entity, sampler, label)?;
        self.db
            .world
            .get_mut::<SceneTexture>(entity)
            .expect("validated SceneTexture disappeared")
            .sampler = sampler;
        Ok(())
    }

    /// Insert one material and pin all generation-bearing texture relations.
    /// Caller-provided texture indices in `authored` are ignored and replaced
    /// with exact SceneDB residency slots.
    pub fn insert_material_asset(
        &mut self,
        authored: libhelio::GpuMaterial,
        refs: SceneMaterialTextureRefs,
        graph_hash: u64,
    ) -> Result<Entity, SceneAssetError> {
        let (material, textures) = self
            .db
            .subsystem::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new")
            .project_material_rows(authored, &refs)?;
        let deltas: Vec<_> = reference_counts(&refs)
            .into_iter()
            .map(|(entity, count)| (entity, count as i64))
            .collect();
        self.validate_texture_ref_deltas(&deltas)?;
        self.apply_texture_ref_deltas(&deltas);

        let entity = self.db.world.spawn();
        self.db.world.insert(entity, refs);
        self.db.world.insert(
            entity,
            SceneMaterial {
                graph_hash,
                material,
                textures,
                ref_count: 0,
                _pad: 0,
            },
        );
        Ok(entity)
    }

    /// Replace authored material parameters and/or texture relations. Only
    /// changed shader rows are dirtied by SceneDB's differential dispatch.
    pub fn update_material_asset(
        &mut self,
        entity: Entity,
        authored: libhelio::GpuMaterial,
        refs: SceneMaterialTextureRefs,
        graph_hash: u64,
    ) -> Result<(), SceneAssetError> {
        let current = *self
            .db
            .world
            .get::<SceneMaterial>(entity)
            .ok_or(SceneAssetError::MaterialMissingComponent(entity))?;
        let old_refs = *self
            .db
            .world
            .get::<SceneMaterialTextureRefs>(entity)
            .ok_or(SceneAssetError::MaterialTextureRefsMissing(entity))?;
        let (material, textures) = self
            .db
            .subsystem::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new")
            .project_material_rows(authored, &refs)?;
        let deltas = combined_deltas(&old_refs, &refs);
        self.validate_texture_ref_deltas(&deltas)?;
        self.apply_texture_ref_deltas(&deltas);
        self.db.world.insert(entity, refs);
        self.db.world.insert(
            entity,
            SceneMaterial {
                graph_hash,
                material,
                textures,
                ref_count: current.ref_count,
                _pad: 0,
            },
        );
        Ok(())
    }

    /// Remove an unused material and release its texture pins. Textures are
    /// not cascade-deleted: their stable asset identity remains queryable
    /// until an explicit `remove_texture_asset` call.
    pub fn remove_material_asset(
        &mut self,
        entity: Entity,
    ) -> Result<(SceneMaterial, SceneMaterialTextureRefs), SceneAssetError> {
        let material = *self
            .db
            .world
            .get::<SceneMaterial>(entity)
            .ok_or(SceneAssetError::MaterialMissingComponent(entity))?;
        if material.ref_count != 0 {
            return Err(SceneAssetError::MaterialInUse {
                entity,
                ref_count: material.ref_count,
            });
        }
        let refs = *self
            .db
            .world
            .get::<SceneMaterialTextureRefs>(entity)
            .ok_or(SceneAssetError::MaterialTextureRefsMissing(entity))?;
        let deltas: Vec<_> = reference_counts(&refs)
            .into_iter()
            .map(|(texture, count)| (texture, -(count as i64)))
            .collect();
        self.validate_texture_ref_deltas(&deltas)?;
        self.apply_texture_ref_deltas(&deltas);
        self.db.world.despawn(entity);
        Ok((material, refs))
    }

    /// Increment the object/use count guarding material removal.
    pub fn retain_material(&mut self, entity: Entity) -> Result<(), SceneAssetError> {
        let mut material = *self
            .db
            .world
            .get::<SceneMaterial>(entity)
            .ok_or(SceneAssetError::MaterialMissingComponent(entity))?;
        material.ref_count = material
            .ref_count
            .checked_add(1)
            .ok_or(SceneAssetError::MaterialRefCountOverflow(entity))?;
        self.db.world.insert(entity, material);
        Ok(())
    }

    /// Decrement the object/use count guarding material removal.
    pub fn release_material(&mut self, entity: Entity) -> Result<(), SceneAssetError> {
        let mut material = *self
            .db
            .world
            .get::<SceneMaterial>(entity)
            .ok_or(SceneAssetError::MaterialMissingComponent(entity))?;
        material.ref_count = material
            .ref_count
            .checked_sub(1)
            .ok_or(SceneAssetError::MaterialRefCountUnderflow(entity))?;
        self.db.world.insert(entity, material);
        Ok(())
    }

    fn validate_texture_ref_deltas(&self, deltas: &[(Entity, i64)]) -> Result<(), SceneAssetError> {
        let residency = self
            .db
            .subsystem::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new");
        for &(entity, delta) in deltas {
            let texture = self
                .db
                .world
                .get::<SceneTexture>(entity)
                .ok_or(SceneAssetError::TextureMissingComponent(entity))?;
            let pins = residency
                .material_pin_count(entity)
                .ok_or(SceneAssetError::TextureNotResident(entity))?;
            if texture.ref_count != pins {
                return Err(SceneAssetError::TextureRefCountDiverged {
                    entity,
                    component: texture.ref_count,
                    residency: pins,
                });
            }
            if delta >= 0 {
                texture
                    .ref_count
                    .checked_add(delta as u32)
                    .ok_or(SceneAssetError::TextureRefCountOverflow(entity))?;
            } else {
                texture
                    .ref_count
                    .checked_sub((-delta) as u32)
                    .ok_or(SceneAssetError::TextureRefCountUnderflow(entity))?;
            }
        }
        Ok(())
    }

    fn apply_texture_ref_deltas(&mut self, deltas: &[(Entity, i64)]) {
        for &(entity, delta) in deltas {
            let texture = self
                .db
                .world
                .get_mut::<SceneTexture>(entity)
                .expect("validated SceneTexture disappeared during material mutation");
            texture.ref_count = if delta >= 0 {
                texture.ref_count + delta as u32
            } else {
                texture.ref_count - (-delta) as u32
            };
        }
        let residency = self
            .db
            .subsystem_mut::<TextureResidency>()
            .expect("TextureResidency is registered by SceneAuthority::new");
        for &(entity, delta) in deltas {
            residency.adjust_material_pins(entity, delta);
        }
    }
}
