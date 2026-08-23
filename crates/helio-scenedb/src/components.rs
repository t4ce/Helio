//! Canonical Helio scene components stored by SceneDB.
//!
//! A component is the semantic CPU record; only fields tagged `#[gpu]` have
//! partner residency. Each GPU field is already one shader-exact row, so a
//! named destination can be shared across compatible component paths without
//! relying on Rust field names or generated wrapper identities.

use std::sync::Arc;

use pulsar_scenedb::gpu::SceneGpuStore;
use pulsar_scenedb::page::Pod as SceneDbPod;
use pulsar_scenedb::Entity;
use pulsar_scenedb_derive::SceneStore;

pub const OBJECT_SPATIAL_BUFFER_KEY: &str = "helio.scene.object.spatial";
pub const OBJECT_RENDER_BUFFER_KEY: &str = "helio.scene.object.render";
pub const LIGHT_BUFFER_KEY: &str = "helio.scene.lights";
pub const DECAL_BUFFER_KEY: &str = "helio.scene.decals";
pub const MATERIAL_BUFFER_KEY: &str = "helio.scene.materials";
pub const MATERIAL_TEXTURES_BUFFER_KEY: &str = "helio.scene.material_textures";
pub const WATER_VOLUME_BUFFER_KEY: &str = "helio.scene.water_volumes";
pub const WATER_HITBOX_BUFFER_KEY: &str = "helio.scene.water_hitboxes";
pub const POST_PROCESS_VOLUME_BUFFER_KEY: &str = "helio.scene.post_process_volumes";
pub const REFLECTION_CAPTURE_BUFFER_KEY: &str = "helio.scene.reflection_captures";
pub const PLANAR_REFLECTOR_BUFFER_KEY: &str = "helio.scene.planar_reflectors";

macro_rules! shader_row {
    ($name:ident, $inner:ty) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name(pub $inner);

        // SAFETY: the transparent wrapper has exactly the byte layout of the
        // bytemuck::Pod shader row and therefore contains no uninitialised
        // padding, invalid bit patterns, or drop glue.
        unsafe impl SceneDbPod for $name {}

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        const _: () = {
            assert!(std::mem::size_of::<$name>() == std::mem::size_of::<$inner>());
            assert!(std::mem::align_of::<$name>() == std::mem::align_of::<$inner>());
        };
    };
}

shader_row!(SceneDecalRow, libhelio::GpuDecal);
shader_row!(SceneMaterialRow, libhelio::GpuMaterial);
shader_row!(SceneWaterVolumeRow, libhelio::GpuWaterVolume);
shader_row!(SceneWaterHitboxRow, libhelio::GpuWaterHitbox);
shader_row!(ScenePostProcessVolumeRow, libhelio::GpuPostProcessVolume);

/// Shader-layout authored reflection-capture row. The cube-array layer is
/// deliberately fixed to -1 here: layer residency and influence ordering are
/// renderer-derived and travel in Helio's compact capture projection.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneReflectionCaptureRow {
    pub position_radius: [f32; 4],
    pub extents_transition: [f32; 4],
    pub world_to_local: [[f32; 4]; 4],
    pub cubemap_residency_sentinel: i32,
    pub shape: u32,
    pub mobility: u32,
    pub brightness: f32,
}

impl SceneReflectionCaptureRow {
    pub fn as_authored_gpu_capture(&self) -> libhelio::GpuReflectionCapture {
        libhelio::GpuReflectionCapture {
            position_radius: self.position_radius,
            extents_transition: self.extents_transition,
            world_to_local: self.world_to_local,
            cubemap_index: -1,
            shape: self.shape,
            mobility: self.mobility,
            brightness: self.brightness,
        }
    }

    pub fn influence_size(&self) -> f32 {
        self.as_authored_gpu_capture().influence_size()
    }
}

impl From<libhelio::GpuReflectionCapture> for SceneReflectionCaptureRow {
    fn from(value: libhelio::GpuReflectionCapture) -> Self {
        Self {
            position_radius: value.position_radius,
            extents_transition: value.extents_transition,
            world_to_local: value.world_to_local,
            cubemap_residency_sentinel: -1,
            shape: value.shape,
            mobility: value.mobility,
            brightness: value.brightness,
        }
    }
}

unsafe impl SceneDbPod for SceneReflectionCaptureRow {}

const _: () = {
    assert!(std::mem::size_of::<SceneReflectionCaptureRow>() == 112);
    assert!(std::mem::size_of::<SceneReflectionCaptureRow>()
        == std::mem::size_of::<libhelio::GpuReflectionCapture>());
    assert!(std::mem::offset_of!(SceneReflectionCaptureRow, cubemap_residency_sentinel) == 96);
};

/// Shader-exact authored planar-reflector row.
///
/// `position_tolerance.w` is the maximum absolute point-to-plane distance
/// used to associate a G-buffer surface with this reflector. The normal and
/// tangent are stored as an orthonormal basis by Helio's validated CRUD seam;
/// the shader derives the bitangent with `cross(normal, tangent)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScenePlanarReflectorRow {
    pub position_tolerance: [f32; 4],
    pub normal_cos_threshold: [f32; 4],
    pub tangent_priority: [f32; 4],
    pub half_extents_reserved: [f32; 4],
}

// SAFETY: four vec4-compatible fields form one padding-free 64-byte shader
// row, and the type derives bytemuck::Pod.
unsafe impl SceneDbPod for ScenePlanarReflectorRow {}

const _: () = {
    assert!(std::mem::size_of::<ScenePlanarReflectorRow>() == 64);
    assert!(std::mem::align_of::<ScenePlanarReflectorRow>() == 4);
};

/// One material texture selection in the legacy shader ABI.
///
/// `texture_index` is deliberately a SceneDB [`TextureStore`](pulsar_scenedb::gpu::TextureStore)
/// residency slot, not a Helio sparse-pool or bind-group reorder index. The
/// residency subsystem keeps that slot fixed until the referenced texture
/// entity is unpinned and removed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneMaterialTextureSlotRow {
    pub texture_index: u32,
    pub uv_channel: u32,
    pub _pad: [u32; 2],
    pub offset_scale: [f32; 4],
    pub rotation: [f32; 4],
}

impl SceneMaterialTextureSlotRow {
    pub const fn missing() -> Self {
        Self {
            texture_index: libhelio::GpuMaterial::NO_TEXTURE,
            uv_channel: 0,
            _pad: [0; 2],
            offset_scale: [0.0, 0.0, 1.0, 1.0],
            rotation: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

// SAFETY: the row derives bytemuck::Pod and its explicit 48-byte shader
// layout is asserted below.
unsafe impl SceneDbPod for SceneMaterialTextureSlotRow {}

/// Shader-exact material texture selections and UV transforms.
///
/// This is the layout-equivalent replacement for Helio's legacy
/// `GpuMaterialTextures`. It is a cached GPU projection of
/// [`SceneMaterialTextureRefs`]; generation-bearing texture entities remain
/// the canonical references.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneMaterialTexturesRow {
    pub base_color: SceneMaterialTextureSlotRow,
    pub normal: SceneMaterialTextureSlotRow,
    pub roughness_metallic: SceneMaterialTextureSlotRow,
    pub emissive: SceneMaterialTextureSlotRow,
    pub occlusion: SceneMaterialTextureSlotRow,
    pub specular_color: SceneMaterialTextureSlotRow,
    pub specular_weight: SceneMaterialTextureSlotRow,
    pub params: [f32; 4],
}

impl SceneMaterialTexturesRow {
    pub const fn missing() -> Self {
        Self {
            base_color: SceneMaterialTextureSlotRow::missing(),
            normal: SceneMaterialTextureSlotRow::missing(),
            roughness_metallic: SceneMaterialTextureSlotRow::missing(),
            emissive: SceneMaterialTextureSlotRow::missing(),
            occlusion: SceneMaterialTextureSlotRow::missing(),
            specular_color: SceneMaterialTextureSlotRow::missing(),
            specular_weight: SceneMaterialTextureSlotRow::missing(),
            params: [1.0, 1.0, 0.5, 0.0],
        }
    }
}

// SAFETY: every field is Pod and the asserted 352-byte layout has no hidden
// padding.
unsafe impl SceneDbPod for SceneMaterialTexturesRow {}

/// Persistent asset identity for one texture.
///
/// The key is supplied by the asset layer (normally a content hash or stable
/// asset UUID encoded as `u128`) or minted by SceneDB's non-recycling
/// [`TextureResidency`](crate::TextureResidency) allocator. It survives
/// renderer recreation and is independent of the physical texture-array slot.
/// Zero is reserved so a missing integration key cannot silently enter the
/// canonical index.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneTextureAssetKey(pub u128);

impl SceneTextureAssetKey {
    pub const INVALID: Self = Self(0);

    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// Authored sampler state retained independently from the resident wgpu
/// sampler object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneTextureSampler {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,
}

impl Default for SceneTextureSampler {
    fn default() -> Self {
        Self {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
        }
    }
}

/// Canonical CPU metadata for one SceneDB-owned material texture.
///
/// The texture, view, sampler, and non-compacting residency slot live in the
/// registered `TextureResidency` subsystem. Raw upload bytes remain an asset
/// loading concern and can be recovered through `asset_key`; retaining a
/// second uncompressed image copy here would multiply scene memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneTexture {
    pub asset_key: SceneTextureAssetKey,
    pub size: wgpu::Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: wgpu::TextureDimension,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub sampler: SceneTextureSampler,
    /// Number of live material texture fields referencing this entity.
    pub ref_count: u32,
}

impl SceneTexture {
    pub fn sampled_2d(
        asset_key: SceneTextureAssetKey,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sampler: SceneTextureSampler,
    ) -> Self {
        Self {
            asset_key,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            sampler,
            ref_count: 0,
        }
    }
}

/// Authored UV transform associated with a material-to-texture relation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneTextureTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation_radians: f32,
}

impl Default for SceneTextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation_radians: 0.0,
        }
    }
}

/// One generation-checked canonical material-to-texture reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneMaterialTextureRef {
    pub texture: Entity,
    pub uv_channel: u32,
    pub transform: SceneTextureTransform,
}

impl SceneMaterialTextureRef {
    pub fn new(texture: Entity) -> Self {
        Self {
            texture,
            uv_channel: 0,
            transform: SceneTextureTransform::default(),
        }
    }
}

/// Canonical CPU texture relationships for one material.
///
/// Physical texture slots are intentionally absent. They are resolved by
/// `TextureResidency` only when this component or a referenced texture
/// changes, then cached in [`SceneMaterialTexturesRow`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneMaterialTextureRefs {
    pub base_color: Option<SceneMaterialTextureRef>,
    pub normal: Option<SceneMaterialTextureRef>,
    pub roughness_metallic: Option<SceneMaterialTextureRef>,
    pub emissive: Option<SceneMaterialTextureRef>,
    pub occlusion: Option<SceneMaterialTextureRef>,
    pub specular_color: Option<SceneMaterialTextureRef>,
    pub specular_weight: Option<SceneMaterialTextureRef>,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
}

impl SceneMaterialTextureRefs {
    pub fn references(self) -> impl Iterator<Item = SceneMaterialTextureRef> {
        [
            self.base_color,
            self.normal,
            self.roughness_metallic,
            self.emissive,
            self.occlusion,
            self.specular_color,
            self.specular_weight,
        ]
        .into_iter()
        .flatten()
    }
}

impl Default for SceneMaterialTextureRefs {
    fn default() -> Self {
        Self {
            base_color: None,
            normal: None,
            roughness_metallic: None,
            emissive: None,
            occlusion: None,
            specular_color: None,
            specular_weight: None,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_cutoff: 0.5,
        }
    }
}

const _: () = {
    assert!(std::mem::size_of::<SceneMaterialTextureSlotRow>() == 48);
    assert!(std::mem::offset_of!(SceneMaterialTextureSlotRow, offset_scale) == 16);
    assert!(std::mem::offset_of!(SceneMaterialTextureSlotRow, rotation) == 32);
    assert!(std::mem::size_of::<SceneMaterialTexturesRow>() == 352);
    assert!(std::mem::offset_of!(SceneMaterialTexturesRow, params) == 336);
};

/// High-frequency authored object state. Flags live beside spatial data so
/// cull/occlusion/shadow shaders do not spend another storage-buffer binding
/// just to read culling policy or the primary coordinate-space bits.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneObjectSpatialRow {
    pub model: [f32; 16],
    pub normal_mat: [f32; 12],
    pub sphere: [f32; 4],
    pub flags: u32,
    pub _pad: [u32; 3],
}

// SAFETY: every field is Pod and the asserted 144-byte layout has no padding.
unsafe impl SceneDbPod for SceneObjectSpatialRow {}

/// Lower-frequency shader metadata. `mesh_row` and `material_row` are stable
/// SceneDB GPU table rows derived from the CPU handles stored on `SceneObject`.
/// Lightmap assignment is authored by the bake integration, while draw and
/// atlas execution slots stay in Helio.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneObjectRenderRow {
    pub mesh_row: u32,
    pub material_row: u32,
    pub lightmap_index: u32,
    pub reserved: u32,
}

// SAFETY: four u32 fields form one padding-free 16-byte shader row.
unsafe impl SceneDbPod for SceneObjectRenderRow {}

const _: () = {
    assert!(std::mem::size_of::<SceneObjectSpatialRow>() == 144);
    assert!(std::mem::offset_of!(SceneObjectSpatialRow, normal_mat) == 64);
    assert!(std::mem::offset_of!(SceneObjectSpatialRow, sphere) == 112);
    assert!(std::mem::offset_of!(SceneObjectSpatialRow, flags) == 128);
    assert!(std::mem::size_of::<SceneObjectRenderRow>() == 16);
    assert!(std::mem::offset_of!(SceneObjectRenderRow, lightmap_index) == 8);
};

/// Shader-layout authored light row.
///
/// Offset 48 deliberately stores only the authored request sentinel (`0` =
/// requested, `u32::MAX` = disabled), never Helio's assigned atlas slice. The
/// compact render projection overlays its own `shadow_index` after loading this
/// row, keeping atlas policy out of persistent scene data while retaining the
/// established 128-byte shader stride and public `GpuLight` query ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneLightRow {
    pub position_range: [f32; 4],
    pub direction_outer: [f32; 4],
    pub color_intensity: [f32; 4],
    pub shadow_requested: u32,
    pub light_type: u32,
    pub inner_angle: f32,
    pub _pad: u32,
    pub god_rays_enabled: u32,
    pub god_rays_density: f32,
    pub god_rays_weight: f32,
    pub god_rays_decay: f32,
    pub god_rays_exposure: f32,
    pub flare_enabled: u32,
    pub flare_type: u32,
    pub flare_intensity: f32,
    pub flare_scale: f32,
    pub flare_tint_r: f32,
    pub flare_tint_g: f32,
    pub flare_tint_b: f32,
    pub ies_profile_index: i32,
    pub light_function_index: i32,
    pub ies_angle_scale: f32,
    pub ies_angle_offset: f32,
}

impl SceneLightRow {
    /// Borrow the shader-layout payload through Helio's public light type.
    /// `shadow_index` in that view is the authored 0/MAX request sentinel; assigned
    /// atlas slices exist only in the compact render projection.
    pub fn as_authored_gpu_light(&self) -> &libhelio::GpuLight {
        // SAFETY: size, alignment, field order, and offset 48 are asserted
        // below. Both types are Pod and therefore admit every bit pattern.
        unsafe { &*(self as *const Self).cast::<libhelio::GpuLight>() }
    }

    #[inline]
    pub fn requests_shadow(&self) -> bool {
        self.shadow_requested != u32::MAX
    }
}

// SAFETY: `SceneLightRow` is bytemuck Pod, has no implicit padding, and its
// explicit layout is asserted against the shader-facing source type below.
unsafe impl SceneDbPod for SceneLightRow {}

impl From<libhelio::GpuLight> for SceneLightRow {
    fn from(value: libhelio::GpuLight) -> Self {
        Self {
            position_range: value.position_range,
            direction_outer: value.direction_outer,
            color_intensity: value.color_intensity,
            shadow_requested: if value.shadow_index == u32::MAX { u32::MAX } else { 0 },
            light_type: value.light_type,
            inner_angle: value.inner_angle,
            _pad: value._pad,
            god_rays_enabled: value.god_rays_enabled,
            god_rays_density: value.god_rays_density,
            god_rays_weight: value.god_rays_weight,
            god_rays_decay: value.god_rays_decay,
            god_rays_exposure: value.god_rays_exposure,
            flare_enabled: value.flare_enabled,
            flare_type: value.flare_type,
            flare_intensity: value.flare_intensity,
            flare_scale: value.flare_scale,
            flare_tint_r: value.flare_tint_r,
            flare_tint_g: value.flare_tint_g,
            flare_tint_b: value.flare_tint_b,
            ies_profile_index: value.ies_profile_index,
            light_function_index: value.light_function_index,
            ies_angle_scale: value.ies_angle_scale,
            ies_angle_offset: value.ies_angle_offset,
        }
    }
}

impl From<SceneLightRow> for libhelio::GpuLight {
    fn from(value: SceneLightRow) -> Self {
        Self {
            position_range: value.position_range,
            direction_outer: value.direction_outer,
            color_intensity: value.color_intensity,
            shadow_index: value.shadow_requested,
            light_type: value.light_type,
            inner_angle: value.inner_angle,
            _pad: value._pad,
            god_rays_enabled: value.god_rays_enabled,
            god_rays_density: value.god_rays_density,
            god_rays_weight: value.god_rays_weight,
            god_rays_decay: value.god_rays_decay,
            god_rays_exposure: value.god_rays_exposure,
            flare_enabled: value.flare_enabled,
            flare_type: value.flare_type,
            flare_intensity: value.flare_intensity,
            flare_scale: value.flare_scale,
            flare_tint_r: value.flare_tint_r,
            flare_tint_g: value.flare_tint_g,
            flare_tint_b: value.flare_tint_b,
            ies_profile_index: value.ies_profile_index,
            light_function_index: value.light_function_index,
            ies_angle_scale: value.ies_angle_scale,
            ies_angle_offset: value.ies_angle_offset,
        }
    }
}

const _: () = {
    assert!(std::mem::size_of::<SceneLightRow>() == std::mem::size_of::<libhelio::GpuLight>());
    assert!(std::mem::align_of::<SceneLightRow>() == std::mem::align_of::<libhelio::GpuLight>());
    assert!(std::mem::offset_of!(SceneLightRow, shadow_requested) == 48);
};

/// One renderable object. CPU metadata and GPU-partner fields intentionally
/// remain one semantic component: World currently has no bundle insertion,
/// so splitting these would force an extra archetype migration on every spawn.
/// The two `#[gpu]` fields still publish as independent shader buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneObject {
    pub mesh_handle_bits: u64,
    pub material_handle_bits: u64,
    pub groups: u64,
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.object.spatial")]
    pub spatial: SceneObjectSpatialRow,
    #[gpu(buffer = "helio.scene.object.render")]
    pub render: SceneObjectRenderRow,
    pub movability: u32,
    pub _pad: u32,
}

/// Authored light metadata plus the shader-exact partner row. Atlas assignment
/// remains exclusively in Helio's compact render projection.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneLight {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.lights")]
    pub light: SceneLightRow,
    pub movability: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneDecal {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.decals")]
    pub decal: SceneDecalRow,
    pub movability: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneWaterVolume {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.water_volumes")]
    pub volume: SceneWaterVolumeRow,
    pub _reserved: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneWaterHitbox {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.water_hitboxes")]
    pub hitbox: SceneWaterHitboxRow,
    pub _reserved: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct ScenePostProcessVolume {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.post_process_volumes")]
    pub volume: ScenePostProcessVolumeRow,
    pub _reserved: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneReflectionCapture {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.reflection_captures")]
    pub capture: SceneReflectionCaptureRow,
    pub _reserved: u64,
}

/// One authored finite planar reflector. Membership/order in the per-frame
/// trace set is a Helio render projection; this component and its GPU field
/// remain the only persistent description.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct ScenePlanarReflector {
    pub user_tag: u64,
    #[gpu(buffer = "helio.scene.planar_reflectors")]
    pub reflector: ScenePlanarReflectorRow,
    pub _reserved: u64,
}

/// Lossless, Pod-compatible CPU storage for global wind authority. Direction
/// remains authored (not normalized); normalization belongs to `Wind::to_gpu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneWindRow {
    pub direction_speed: [f32; 4],
    pub gust: [f32; 4],
    pub time_prev_time: [f32; 2],
    pub _pad: [f32; 2],
}

impl From<libhelio::Wind> for SceneWindRow {
    fn from(value: libhelio::Wind) -> Self {
        Self {
            direction_speed: [
                value.direction.x,
                value.direction.y,
                value.direction.z,
                value.speed,
            ],
            gust: [
                value.gust_amplitude,
                value.gust_frequency,
                value.gust_phase,
                value.turbulence_scale,
            ],
            time_prev_time: [value.time, value.prev_time],
            _pad: [0.0; 2],
        }
    }
}

impl From<SceneWindRow> for libhelio::Wind {
    fn from(value: SceneWindRow) -> Self {
        Self {
            direction: [
                value.direction_speed[0],
                value.direction_speed[1],
                value.direction_speed[2],
            ]
            .into(),
            speed: value.direction_speed[3],
            gust_amplitude: value.gust[0],
            gust_frequency: value.gust[1],
            gust_phase: value.gust[2],
            turbulence_scale: value.gust[3],
            time: value.time_prev_time[0],
            prev_time: value.time_prev_time[1],
        }
    }
}

unsafe impl SceneDbPod for SceneWindRow {}

const _: () = assert!(std::mem::size_of::<SceneWindRow>() == 48);

/// Global wind authority. The authored parameters and the continuous clock
/// live in SceneDB; the normalized `GpuWind` uniform remains a per-frame Helio
/// projection so it is evaluated exactly once per frame.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneWind {
    pub wind: SceneWindRow,
}

/// Canonical CPU-only sky payload for one Helio scene.
///
/// `None` means no actor has claimed the scene sky yet. Keeping that state
/// distinct from [`libhelio::sky::SkyContext::default`] preserves the public
/// first-sky-wins insertion rule while allowing the winning actor to author
/// an indoor/no-atmosphere context.
#[derive(Debug, Clone, Copy, Default)]
pub struct SceneSky {
    pub context: Option<libhelio::sky::SkyContext>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneMaterial {
    pub graph_hash: u64,
    #[gpu(buffer = "helio.scene.materials")]
    pub material: SceneMaterialRow,
    #[gpu(buffer = "helio.scene.material_textures")]
    pub textures: SceneMaterialTexturesRow,
    pub ref_count: u32,
    pub _pad: u32,
}

/// Register every World-mirrored canonical column before the authority is
/// attached. Explicit names are physical identities: compatible rows alias;
/// type/layout/mirror-mode collisions are rejected by SceneDB.
pub fn register_scene_component_buffers(
    store: &mut SceneGpuStore,
    initial_object_capacity: u32,
    device: &Arc<wgpu::Device>,
) {
    // The configured entity hint is useful for the dominant object
    // population, but it is not a sound proxy for every component's count.
    // Component-local SceneDB rows let sparse partners start at one physical
    // row and grow independently instead of preallocating the global 1K
    // entity hint for lights, decals, volumes, captures, and materials.
    let object_capacity = initial_object_capacity.max(1);
    const SPARSE_COMPONENT_CAPACITY: u32 = 1;
    SceneObject::register_gpu_columns_growable(store, object_capacity, device);
    SceneLight::register_gpu_columns_growable(store, SPARSE_COMPONENT_CAPACITY, device);
    SceneDecal::register_gpu_columns_growable(store, SPARSE_COMPONENT_CAPACITY, device);
    SceneWaterVolume::register_gpu_columns_growable(store, SPARSE_COMPONENT_CAPACITY, device);
    SceneWaterHitbox::register_gpu_columns_growable(store, SPARSE_COMPONENT_CAPACITY, device);
    ScenePostProcessVolume::register_gpu_columns_growable(store, SPARSE_COMPONENT_CAPACITY, device);
    SceneReflectionCapture::register_gpu_columns_growable(
        store,
        SPARSE_COMPONENT_CAPACITY,
        device,
    );
    ScenePlanarReflector::register_gpu_columns_growable(
        store,
        SPARSE_COMPONENT_CAPACITY,
        device,
    );
    SceneMaterial::register_gpu_columns_growable(store, SPARSE_COMPONENT_CAPACITY, device);
}

const _: () = {
    // No implicit padding is permitted: SceneDB's page storage and GPU upload
    // paths treat these records as bytes. Keep these asserts next to the
    // semantic component definitions rather than trusting derive output.
    assert!(std::mem::size_of::<SceneObject>() == 200);
    assert!(std::mem::offset_of!(SceneObject, spatial) == 32);
    assert!(std::mem::offset_of!(SceneObject, render) == 176);
    assert!(std::mem::size_of::<SceneLight>() == 144);
    assert!(std::mem::size_of::<SceneDecal>() == 144);
    assert!(std::mem::size_of::<SceneWaterVolume>() == 272);
    assert!(std::mem::size_of::<SceneWaterHitbox>() == 96);
    assert!(std::mem::size_of::<ScenePostProcessVolume>() == 544);
    assert!(std::mem::size_of::<SceneReflectionCapture>() == 128);
    assert!(std::mem::size_of::<ScenePlanarReflector>() == 80);
    assert!(std::mem::offset_of!(SceneMaterial, material) == 8);
    assert!(std::mem::offset_of!(SceneMaterial, textures) == 104);
    assert!(std::mem::offset_of!(SceneMaterial, ref_count) == 456);
    assert!(std::mem::size_of::<SceneMaterial>() == 464);
};

#[cfg(test)]
mod tests {
    use super::SceneLightRow;

    #[test]
    fn borrowed_and_owned_light_queries_preserve_the_same_request_sentinel() {
        for input_shadow_index in [0, 37, u32::MAX] {
            let input = libhelio::GpuLight {
                shadow_index: input_shadow_index,
                ..Default::default()
            };
            let row = SceneLightRow::from(input);
            let owned = libhelio::GpuLight::from(row);
            let borrowed = row.as_authored_gpu_light();
            let expected = if input_shadow_index == u32::MAX {
                u32::MAX
            } else {
                0
            };

            assert_eq!(row.shadow_requested, expected);
            assert_eq!(borrowed.shadow_index, expected);
            assert_eq!(owned.shadow_index, expected);
            assert_eq!(bytemuck::bytes_of(borrowed), bytemuck::bytes_of(&owned));
        }
    }
}
