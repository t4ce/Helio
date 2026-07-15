//! Helio's browser-WebGPU vocabulary.
//!
//! These are intentionally plain data types. They describe the subset of the
//! WebGPU standard used by Helio and are converted to JavaScript dictionaries
//! by `types.rs`.

use std::num::{NonZeroU32, NonZeroU64};

use bitflags::bitflags;

pub type BufferAddress = u64;
pub type DynamicOffset = u32;
pub type BufferSize = NonZeroU64;
pub type Label<'a> = Option<&'a str>;

pub const COPY_BUFFER_ALIGNMENT: u64 = 4;
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
pub const MAP_ALIGNMENT: u64 = 8;
pub const QUERY_RESOLVE_BUFFER_ALIGNMENT: u64 = 256;
pub const VERTEX_ALIGNMENT: u64 = 4;

bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct Backends: u32 {
        const NOOP = 1 << 0;
        const VULKAN = 1 << 1;
        const METAL = 1 << 2;
        const DX12 = 1 << 3;
        const GL = 1 << 4;
        const BROWSER_WEBGPU = 1 << 5;
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::BrowserWebGpu
    }
}

impl Default for Backends {
    fn default() -> Self {
        Self::BROWSER_WEBGPU
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Backend {
    Noop,
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
}

#[derive(Clone, Debug, Default)]
pub struct BackendOptions;

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct InstanceFlags: u32 {
        const DEBUG = 1 << 0;
        const VALIDATION = 1 << 1;
        const DISCARD_HAL_LABELS = 1 << 2;
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemoryBudgetThresholds;

#[derive(Clone, Debug)]
pub struct InstanceDescriptor {
    pub backends: Backends,
    pub flags: InstanceFlags,
    pub memory_budget_thresholds: MemoryBudgetThresholds,
    pub backend_options: BackendOptions,
    pub display: Option<()>,
}

impl Default for InstanceDescriptor {
    fn default() -> Self {
        Self::new_without_display_handle()
    }
}

impl InstanceDescriptor {
    pub fn new_without_display_handle() -> Self {
        Self {
            backends: Backends::default(),
            flags: InstanceFlags::default(),
            memory_budget_thresholds: MemoryBudgetThresholds,
            backend_options: BackendOptions,
            display: None,
        }
    }

    pub fn new_with_display_handle<T>(_display: Box<T>) -> Self
    where
        T: ?Sized,
    {
        Self::new_without_display_handle()
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct Features: u128 {
        const DEPTH_CLIP_CONTROL = 1 << 0;
        const DEPTH32FLOAT_STENCIL8 = 1 << 1;
        const TEXTURE_COMPRESSION_BC = 1 << 2;
        const TEXTURE_COMPRESSION_BC_SLICED_3D = 1 << 3;
        const TEXTURE_COMPRESSION_ETC2 = 1 << 4;
        const TEXTURE_COMPRESSION_ASTC = 1 << 5;
        const TEXTURE_COMPRESSION_ASTC_SLICED_3D = 1 << 6;
        const TIMESTAMP_QUERY = 1 << 7;
        const INDIRECT_FIRST_INSTANCE = 1 << 8;
        const SHADER_F16 = 1 << 9;
        const RG11B10UFLOAT_RENDERABLE = 1 << 10;
        const BGRA8UNORM_STORAGE = 1 << 11;
        const FLOAT32_FILTERABLE = 1 << 12;
        const FLOAT32_BLENDABLE = 1 << 13;
        const DUAL_SOURCE_BLENDING = 1 << 14;
        const CLIP_DISTANCES = 1 << 15;
        const MULTI_DRAW_INDIRECT_COUNT = 1 << 16;
        const TIMESTAMP_QUERY_INSIDE_ENCODERS = 1 << 17;
        const TIMESTAMP_QUERY_INSIDE_PASSES = 1 << 18;
        const VERTEX_WRITABLE_STORAGE = 1 << 19;
        const BUFFER_BINDING_ARRAY = 1 << 20;
        const TEXTURE_BINDING_ARRAY = 1 << 21;
        const STORAGE_RESOURCE_BINDING_ARRAY = 1 << 22;
        const SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING = 1 << 23;
        const EXPERIMENTAL_RAY_QUERY = 1 << 24;
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct ExperimentalFeatures: u32 { const ENABLED = 1; }
}

#[derive(Clone, Debug, Default)]
pub enum MemoryHints {
    #[default]
    Performance,
    MemoryUsage,
}

#[derive(Clone, Debug, Default)]
pub enum Trace {
    #[default]
    Off,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceLostReason {
    Unknown,
    Destroyed,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: DeviceType,
    pub device_pci_bus_id: String,
    pub driver: String,
    pub driver_info: String,
    pub backend: Backend,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
    pub transient_saves_memory: Option<bool>,
    pub limit_bucket: Option<()>,
}

impl AdapterInfo {
    pub fn new(device_type: DeviceType, backend: Backend) -> Self {
        Self {
            name: String::new(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: None,
            limit_bucket: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub max_bind_groups: u32,
    pub max_bind_groups_plus_vertex_buffers: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
    pub max_binding_array_elements_per_shader_stage: u32,
    pub max_binding_array_acceleration_structure_elements_per_shader_stage: u32,
    pub max_binding_array_sampler_elements_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_vertex_buffers: u32,
    pub max_buffer_size: u64,
    pub max_vertex_attributes: u32,
    pub max_vertex_buffer_array_stride: u32,
    pub max_inter_stage_shader_variables: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_color_attachments: u32,
    pub max_color_attachment_bytes_per_sample: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_bind_groups: 4,
            max_bind_groups_plus_vertex_buffers: 24,
            max_bindings_per_bind_group: 1000,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_binding_array_elements_per_shader_stage: 0,
            max_binding_array_acceleration_structure_elements_per_shader_stage: 0,
            max_binding_array_sampler_elements_per_shader_stage: 0,
            max_uniform_buffer_binding_size: 65_536,
            max_storage_buffer_binding_size: 134_217_728,
            max_vertex_buffers: 8,
            max_buffer_size: 268_435_456,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_inter_stage_shader_variables: 16,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_color_attachments: 8,
            max_color_attachment_bytes_per_sample: 32,
            max_compute_workgroup_storage_size: 16_384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65_535,
        }
    }
}

impl Limits {
    /// Conservative limits suitable for browser and downlevel adapters.
    #[must_use]
    pub fn downlevel_defaults() -> Self {
        Self {
            max_texture_dimension_1d: 2048,
            max_texture_dimension_2d: 2048,
            max_texture_dimension_3d: 256,
            max_storage_buffers_per_shader_stage: 4,
            max_uniform_buffer_binding_size: 16 << 10,
            max_inter_stage_shader_variables: 15,
            max_color_attachments: 4,
            max_compute_workgroup_storage_size: 16_352,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeviceDescriptor<'a> {
    pub label: Label<'a>,
    pub required_features: Features,
    pub required_limits: Limits,
    pub experimental_features: ExperimentalFeatures,
    pub memory_hints: MemoryHints,
    pub trace: Trace,
}

#[derive(Clone, Debug)]
pub enum PollType<T> {
    Poll,
    Wait { submission_index: Option<T> },
}

impl<T> PollType<T> {
    pub fn wait_indefinitely() -> Self {
        Self::Wait {
            submission_index: None,
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct BufferUsages: u32 {
        const MAP_READ = 1 << 0;
        const MAP_WRITE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const INDEX = 1 << 4;
        const VERTEX = 1 << 5;
        const UNIFORM = 1 << 6;
        const STORAGE = 1 << 7;
        const INDIRECT = 1 << 8;
        const QUERY_RESOLVE = 1 << 9;
    }
}

#[derive(Clone, Debug)]
pub struct BufferDescriptor<'a> {
    pub label: Label<'a>,
    pub size: BufferAddress,
    pub usage: BufferUsages,
    pub mapped_at_creation: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TextureDimension {
    D1,
    #[default]
    D2,
    D3,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TextureViewDimension {
    D1,
    #[default]
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TextureAspect {
    #[default]
    All,
    StencilOnly,
    DepthOnly,
    Plane0,
    Plane1,
    Plane2,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AstcBlock {
    B4x4,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AstcChannel {
    Unorm,
    UnormSrgb,
    Hdr,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TextureFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    R32Uint,
    R32Sint,
    R32Float,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Stencil8,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
}

impl TextureFormat {
    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    pub fn block_copy_size(self, _aspect: Option<TextureAspect>) -> Option<u32> {
        Some(match self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint | Self::Stencil8 => 1,
            Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::Depth16Unorm => 2,
            Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float => 8,
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => 16,
            _ => 4,
        })
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct TextureUsages: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        const TEXTURE_BINDING = 1 << 2;
        const STORAGE_BINDING = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
        const TRANSIENT_ATTACHMENT = 1 << 5;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Origin2d {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Origin3d {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

#[derive(Clone, Debug)]
pub struct TextureDescriptor<'a> {
    pub label: Label<'a>,
    pub size: Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsages,
    pub view_formats: &'a [TextureFormat],
}

#[derive(Clone, Debug)]
pub struct TextureViewDescriptor<'a> {
    pub label: Label<'a>,
    pub format: Option<TextureFormat>,
    pub dimension: Option<TextureViewDimension>,
    pub usage: Option<TextureUsages>,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
}

impl Default for TextureViewDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum FilterMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MipmapFilterMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Debug)]
pub struct SamplerDescriptor<'a> {
    pub label: Label<'a>,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: MipmapFilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color: Option<SamplerBorderColor>,
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
    Zero,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct TexelCopyBufferLayout {
    pub offset: BufferAddress,
    pub bytes_per_row: Option<u32>,
    pub rows_per_image: Option<u32>,
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct ShaderStages: u32 {
        const VERTEX = 1 << 0;
        const FRAGMENT = 1 << 1;
        const COMPUTE = 1 << 2;
        const VERTEX_FRAGMENT = Self::VERTEX.bits() | Self::FRAGMENT.bits();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum BufferBindingType {
    #[default]
    Uniform,
    Storage {
        read_only: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Depth,
    Sint,
    Uint,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
    Atomic,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BindingType {
    Buffer {
        ty: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: Option<BufferSize>,
    },
    Sampler(SamplerBindingType),
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
    AccelerationStructure {
        vertex_return: bool,
    },
    ExternalTexture,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStages,
    pub ty: BindingType,
    pub count: Option<NonZeroU32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum FrontFace {
    Cw,
    #[default]
    Ccw,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Face {
    Front,
    Back,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: Option<Face>,
    pub unclipped_depth: bool,
    pub polygon_mode: PolygonMode,
    pub conservative: bool,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
    Src1,
    OneMinusSrc1,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum BlendOperation {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendComponent {
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self::REPLACE
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::OVER,
    };
}

impl Default for BlendState {
    fn default() -> Self {
        Self::REPLACE
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct ColorWrites: u32 {
        const RED = 1 << 0;
        const GREEN = 1 << 1;
        const BLUE = 1 << 2;
        const ALPHA = 1 << 3;
        const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
        const ALL = Self::COLOR.bits() | Self::ALPHA.bits();
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum StencilOperation {
    #[default]
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StencilState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            front: StencilFaceState::default(),
            back: StencilFaceState::default(),
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthBiasState {
    pub constant: i32,
    pub slope_scale: f32,
    pub clamp: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: Option<bool>,
    pub depth_compare: Option<CompareFunction>,
    pub stencil: StencilState,
    pub bias: DepthBiasState,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum VertexStepMode {
    #[default]
    Vertex,
    Instance,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexFormat {
    Uint8,
    Uint8x2,
    Uint8x4,
    Sint8,
    Sint8x2,
    Sint8x4,
    Unorm8,
    Unorm8x2,
    Unorm8x4,
    Snorm8,
    Snorm8x2,
    Snorm8x4,
    Uint16,
    Uint16x2,
    Uint16x4,
    Sint16,
    Sint16x2,
    Sint16x4,
    Unorm16,
    Unorm16x2,
    Unorm16x4,
    Snorm16,
    Snorm16x2,
    Snorm16x4,
    Float16,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: BufferAddress,
    pub shader_location: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum PowerPreference {
    #[default]
    None,
    LowPower,
    HighPerformance,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum PresentMode {
    #[default]
    AutoVsync,
    AutoNoVsync,
    Fifo,
    FifoRelaxed,
    Immediate,
    Mailbox,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CompositeAlphaMode {
    #[default]
    Auto,
    Opaque,
    PreMultiplied,
    PostMultiplied,
    Inherit,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SurfaceColorSpace {
    #[default]
    Auto,
    Srgb,
    DisplayP3,
    ExtendedSrgb,
    ExtendedDisplayP3,
    ExtendedSrgbLinear,
    Bt2100Pq,
    Bt2100Hlg,
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct SurfaceColorSpaces: u32 {
        const SRGB = 1 << 0;
        const DISPLAY_P3 = 1 << 1;
        const EXTENDED_SRGB = 1 << 2;
        const EXTENDED_DISPLAY_P3 = 1 << 3;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfaceFormatCapabilities {
    pub format: TextureFormat,
    pub color_spaces: SurfaceColorSpaces,
}

#[derive(Clone, Debug)]
pub struct SurfaceCapabilities {
    pub formats: Vec<TextureFormat>,
    pub format_capabilities: Vec<SurfaceFormatCapabilities>,
    pub present_modes: Vec<PresentMode>,
    pub alpha_modes: Vec<CompositeAlphaMode>,
    pub usages: TextureUsages,
}

impl Default for SurfaceCapabilities {
    fn default() -> Self {
        Self {
            formats: Vec::new(),
            format_capabilities: Vec::new(),
            present_modes: Vec::new(),
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceConfiguration {
    pub usage: TextureUsages,
    pub format: TextureFormat,
    pub color_space: SurfaceColorSpace,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
    pub desired_maximum_frame_latency: u32,
    pub alpha_mode: CompositeAlphaMode,
    pub view_formats: Vec<TextureFormat>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QueryType {
    Occlusion,
    Timestamp,
    PipelineStatistics(PipelineStatisticsTypes),
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct PipelineStatisticsTypes: u32 {
        const VERTEX_SHADER_INVOCATIONS = 1 << 0;
        const CLIPPER_INVOCATIONS = 1 << 1;
        const CLIPPER_PRIMITIVES_OUT = 1 << 2;
        const FRAGMENT_SHADER_INVOCATIONS = 1 << 3;
        const COMPUTE_SHADER_INVOCATIONS = 1 << 4;
    }
}

#[derive(Clone, Debug)]
pub struct QuerySetDescriptor<'a> {
    pub label: Label<'a>,
    pub ty: QueryType,
    pub count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CommandEncoderDescriptor<'a> {
    pub label: Label<'a>,
}

#[derive(Clone, Debug, Default)]
pub struct CommandBufferDescriptor<'a> {
    pub label: Label<'a>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LoadOp<V> {
    Load,
    Clear(V),
    DontCare(LoadOpDontCare),
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LoadOpDontCare;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum StoreOp {
    #[default]
    Store,
    Discard,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Operations<V> {
    pub load: LoadOp<V>,
    pub store: StoreOp,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RenderBundleDepthStencil {
    pub format: TextureFormat,
    pub depth_read_only: bool,
    pub stencil_read_only: bool,
}
