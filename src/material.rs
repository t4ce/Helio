use crate::{GpuMaterial, TextureId};

/// Maximum bindless textures per shader stage.
/// WebGPU baseline guarantees only 16; native Vulkan/D3D12 supports 256.
/// Mobile GPUs (Apple, Android/Adreno) get the same conservative cap as wasm —
/// their shader compilers/descriptor limits choke on a 256-wide binding array.
pub const MAX_TEXTURES: usize = libhelio::MAX_MATERIAL_TEXTURES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSamplerDesc {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,
}

impl Default for TextureSamplerDesc {
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

#[derive(Debug, Clone)]
pub struct TextureUpload {
    pub label: Option<String>,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub data: Vec<u8>,
    /// Complete downscaled mip levels, ordered from level 1 onward.
    pub mip_data: Vec<Vec<u8>>,
    pub sampler: TextureSamplerDesc,
}

impl TextureUpload {
    pub fn rgba8(
        label: impl Into<String>,
        width: u32,
        height: u32,
        srgb: bool,
        data: Vec<u8>,
        sampler: TextureSamplerDesc,
    ) -> Self {
        Self {
            label: Some(label.into()),
            width,
            height,
            format: if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            data,
            mip_data: Vec::new(),
            sampler,
        }
    }

    /// Attach complete mip levels for this texture. Level zero remains
    /// [`Self::data`]; `mip_data[0]` is level one.
    pub fn with_mip_data(mut self, mip_data: Vec<Vec<u8>>) -> Self {
        self.mip_data = mip_data;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation_radians: f32,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation_radians: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialTextureRef {
    pub texture: TextureId,
    pub uv_channel: u32,
    pub transform: TextureTransform,
}

impl MaterialTextureRef {
    pub fn new(texture: TextureId) -> Self {
        Self {
            texture,
            uv_channel: 0,
            transform: TextureTransform::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaterialTextures {
    pub base_color: Option<MaterialTextureRef>,
    pub normal: Option<MaterialTextureRef>,
    pub roughness_metallic: Option<MaterialTextureRef>,
    pub emissive: Option<MaterialTextureRef>,
    pub occlusion: Option<MaterialTextureRef>,
    pub specular_color: Option<MaterialTextureRef>,
    pub specular_weight: Option<MaterialTextureRef>,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
}

impl Default for MaterialTextures {
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

#[derive(Debug, Clone)]
pub struct MaterialAsset {
    pub gpu: GpuMaterial,
    pub textures: MaterialTextures,
}

impl MaterialAsset {
    pub fn new(gpu: GpuMaterial) -> Self {
        Self {
            gpu,
            textures: MaterialTextures::default(),
        }
    }
}

impl From<GpuMaterial> for MaterialAsset {
    fn from(value: GpuMaterial) -> Self {
        Self::new(value)
    }
}
