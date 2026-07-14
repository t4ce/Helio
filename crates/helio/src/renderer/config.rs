use crate::material::MAX_TEXTURES;
use helio_pass_perf_overlay::PerfOverlayMode;

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let timestamps =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    let optional = if adapter_features.contains(timestamps) {
        timestamps
    } else {
        wgpu::Features::empty()
    };
    wgpu::Features::INDIRECT_FIRST_INSTANCE | optional
}

pub fn required_wgpu_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: (MAX_TEXTURES as u32)
            .min(adapter_limits.max_sampled_textures_per_shader_stage),
        max_samplers_per_shader_stage: (MAX_TEXTURES as u32)
            .min(adapter_limits.max_samplers_per_shader_stage),
        ..adapter_limits
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    pub shadow_quality: libhelio::ShadowQuality,
    pub debug_mode: u32,
    pub render_scale: f32,
    pub perf_overlay_mode: PerfOverlayMode,
    /// Resolution of each shadow atlas face (width × height). Default 256.
    /// Higher values improve shadow quality at the cost of VRAM (N² scaling).
    pub shadow_atlas_size: u32,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            surface_format,
            shadow_quality: libhelio::ShadowQuality::Medium,
            debug_mode: 0,
            render_scale: 0.75,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            shadow_atlas_size: 256,
        }
    }

    pub fn with_shadow_quality(mut self, quality: libhelio::ShadowQuality) -> Self {
        self.shadow_quality = quality;
        self
    }

    pub fn with_render_scale(mut self, scale: f32) -> Self {
        self.render_scale = scale.clamp(0.25, 1.0);
        self
    }

    pub fn with_perf_overlay_mode(mut self, mode: PerfOverlayMode) -> Self {
        self.perf_overlay_mode = mode;
        self
    }

    pub fn internal_width(&self) -> u32 {
        (((self.width as f32) * self.render_scale).ceil() as u32).max(1)
    }

    pub fn internal_height(&self) -> u32 {
        (((self.height as f32) * self.render_scale).ceil() as u32).max(1)
    }
}
