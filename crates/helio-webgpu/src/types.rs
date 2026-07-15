use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    fmt,
    marker::PhantomData,
    ops::{Bound, Deref, DerefMut, Range, RangeBounds},
    rc::Rc,
    sync::Arc,
};

use js_sys::{Array, ArrayBuffer, Uint8Array};
use wasm_bindgen::{closure::Closure, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use crate::{
    js, AdapterInfo, Backend, BindGroupLayoutEntry, BindingType, BlendComponent, BlendOperation,
    BufferAddress, BufferBindingType, BufferDescriptor, BufferSize, BufferUsages, Color,
    ColorTargetState, CommandBufferDescriptor, CommandEncoderDescriptor, CompositeAlphaMode,
    DepthStencilState, DeviceDescriptor, DeviceType, DynamicOffset, Extent3d, Features,
    IndexFormat, InstanceDescriptor, Limits, LoadOp, MultisampleState, Operations, Origin3d,
    PolygonMode, PowerPreference, PresentMode, PrimitiveState, QuerySetDescriptor,
    SamplerBindingType, SamplerDescriptor, StencilFaceState, StorageTextureAccess, StoreOp,
    SurfaceCapabilities, SurfaceColorSpace, SurfaceConfiguration, TexelCopyBufferLayout,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexStepMode,
};

// Descriptor types deliberately mirror Helio's former public API. Their
// implementation below is a thin conversion into WebGPU JavaScript objects.

#[derive(Clone, Debug)]
pub struct RequestAdapterOptions<'a, 'surface> {
    pub power_preference: PowerPreference,
    pub force_fallback_adapter: bool,
    pub compatible_surface: Option<&'a Surface<'surface>>,
    pub apply_limit_buckets: bool,
}

impl Default for RequestAdapterOptions<'_, '_> {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipelineCompilationOptions<'a> {
    pub constants: &'a [(&'a str, f64)],
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for PipelineCompilationOptions<'_> {
    fn default() -> Self {
        Self {
            constants: &[],
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ShaderSource<'a> {
    Wgsl(Cow<'a, str>),
}

#[derive(Clone, Debug)]
pub struct ShaderModuleDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub source: ShaderSource<'a>,
}

#[derive(Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub bind_group_layouts: &'a [Option<&'a BindGroupLayout>],
    pub immediate_size: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: BufferAddress,
    pub size: Option<BufferSize>,
}

#[derive(Clone, Copy, Debug)]
pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    BufferArray(&'a [BufferBinding<'a>]),
    Sampler(&'a Sampler),
    SamplerArray(&'a [&'a Sampler]),
    TextureView(&'a TextureView),
    TextureViewArray(&'a [&'a TextureView]),
    ExternalTexture(&'a ExternalTexture),
    AccelerationStructure(&'a Tlas),
}

#[derive(Clone, Copy, Debug)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

#[derive(Clone, Debug)]
pub struct BindGroupDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub layout: &'a BindGroupLayout,
    pub entries: &'a [BindGroupEntry<'a>],
}

#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub entries: &'a [BindGroupLayoutEntry],
}

#[derive(Clone, Debug)]
pub struct VertexBufferLayout<'a> {
    pub array_stride: BufferAddress,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

#[derive(Clone, Debug)]
pub struct VertexState<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: PipelineCompilationOptions<'a>,
    pub buffers: &'a [Option<VertexBufferLayout<'a>>],
}

#[derive(Clone, Debug)]
pub struct FragmentState<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: PipelineCompilationOptions<'a>,
    pub targets: &'a [Option<ColorTargetState>],
}

#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub layout: Option<&'a PipelineLayout>,
    pub vertex: VertexState<'a>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<FragmentState<'a>>,
    pub multiview_mask: Option<std::num::NonZeroU32>,
    pub cache: Option<&'a PipelineCache>,
}

#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub layout: Option<&'a PipelineLayout>,
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: PipelineCompilationOptions<'a>,
    pub cache: Option<&'a PipelineCache>,
}

#[derive(Clone, Copy, Debug)]
pub struct ComputePassTimestampWrites<'a> {
    pub query_set: &'a QuerySet,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct ComputePassDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderPassTimestampWrites<'a> {
    pub query_set: &'a QuerySet,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderPassColorAttachment<'a> {
    pub view: &'a TextureView,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<&'a TextureView>,
    pub ops: Operations<Color>,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderPassDepthStencilAttachment<'a> {
    pub view: &'a TextureView,
    pub depth_ops: Option<Operations<f32>>,
    pub stencil_ops: Option<Operations<u32>>,
}

#[derive(Clone, Debug)]
pub struct RenderPassDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a>>],
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
    pub timestamp_writes: Option<RenderPassTimestampWrites<'a>>,
    pub occlusion_query_set: Option<&'a QuerySet>,
    pub multiview_mask: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TexelCopyTextureInfo<'a> {
    pub texture: &'a Texture,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

#[derive(Clone, Copy, Debug)]
pub struct TexelCopyBufferInfo<'a> {
    pub buffer: &'a Buffer,
    pub layout: TexelCopyBufferLayout,
}

#[derive(Clone, Debug)]
pub struct RenderBundleEncoderDescriptor<'a> {
    pub label: crate::Label<'a>,
    pub color_formats: &'a [Option<TextureFormat>],
    pub depth_stencil: Option<crate::RenderBundleDepthStencil>,
    pub sample_count: u32,
    pub multiview: Option<std::num::NonZeroU32>,
}

pub type RenderBundleDescriptor<'a> = CommandBufferDescriptor<'a>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MapMode {
    Read,
    #[default]
    Write,
}

impl MapMode {
    fn bits(self) -> u32 {
        match self {
            Self::Read => 1,
            Self::Write => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubmissionIndex;

#[derive(Clone, Debug)]
pub struct RequestAdapterError(String);

#[derive(Clone, Debug)]
pub struct RequestDeviceError(String);

#[derive(Clone, Debug)]
pub struct CreateSurfaceError(String);

#[derive(Clone, Debug)]
pub struct BufferAsyncError(String);

macro_rules! display_error {
    ($ty:ty) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
        impl std::error::Error for $ty {}
    };
}

display_error!(RequestAdapterError);
display_error!(RequestDeviceError);
display_error!(CreateSurfaceError);
display_error!(BufferAsyncError);

#[derive(Clone, Debug)]
pub enum Error {
    OutOfMemory { source: String },
    Validation { source: String, description: String },
    Internal { source: String, description: String },
}

#[derive(Clone, Debug)]
pub struct PipelineCache;

#[derive(Clone, Debug)]
pub struct Tlas;

#[derive(Clone, Debug)]
pub struct ExternalTexture {
    raw: JsValue,
}

macro_rules! js_handle {
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            raw: JsValue,
        }
    };
}

js_handle!(ShaderModule);
js_handle!(BindGroupLayout);
js_handle!(BindGroup);
js_handle!(PipelineLayout);
js_handle!(Sampler);
js_handle!(TextureView);
js_handle!(RenderPipeline);
js_handle!(ComputePipeline);
js_handle!(QuerySet);
js_handle!(RenderBundle);

#[derive(Clone, Debug)]
pub struct Buffer {
    raw: JsValue,
    size: u64,
    usage: BufferUsages,
}

#[derive(Clone, Debug)]
pub struct Texture {
    raw: JsValue,
    size: Extent3d,
    format: TextureFormat,
    usage: TextureUsages,
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: JsValue,
}

#[derive(Debug)]
pub struct CommandEncoder {
    raw: JsValue,
}

#[derive(Clone, Debug)]
pub struct Queue {
    raw: JsValue,
}

#[derive(Clone, Debug)]
pub struct Device {
    raw: JsValue,
    queue: Queue,
    features: Features,
    limits: Limits,
}

#[derive(Clone, Debug)]
pub struct Adapter {
    raw: JsValue,
    features: Features,
    limits: Limits,
}

#[derive(Clone, Debug)]
pub struct Instance {
    gpu: JsValue,
}

#[derive(Clone, Debug)]
pub struct Surface<'window> {
    canvas: web_sys::HtmlCanvasElement,
    context: JsValue,
    gpu: JsValue,
    config: Rc<RefCell<Option<SurfaceConfiguration>>>,
    configure_failed: Rc<Cell<bool>>,
    _window: PhantomData<&'window ()>,
}

#[derive(Clone, Debug)]
pub struct SurfaceTexture {
    pub texture: Texture,
}

#[derive(Clone, Debug)]
pub enum CurrentSurfaceTexture {
    Success(SurfaceTexture),
    Suboptimal(SurfaceTexture),
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

#[derive(Clone, Debug)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    range: Range<u64>,
}

#[derive(Clone, Debug)]
pub struct BufferView {
    data: Vec<u8>,
}

impl Deref for BufferView {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[derive(Debug)]
pub struct BufferViewMut {
    data: Vec<u8>,
    mapped: Uint8Array,
}

impl Deref for BufferViewMut {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for BufferViewMut {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Drop for BufferViewMut {
    fn drop(&mut self) {
        self.mapped.copy_from(&self.data);
    }
}

#[derive(Debug)]
pub struct RenderPass<'a> {
    raw: JsValue,
    ended: bool,
    _encoder: PhantomData<&'a mut CommandEncoder>,
}

#[derive(Debug)]
pub struct ComputePass<'a> {
    raw: JsValue,
    ended: bool,
    _encoder: PhantomData<&'a mut CommandEncoder>,
}

#[derive(Debug)]
pub struct RenderBundleEncoder {
    raw: JsValue,
}

fn label(target: &js_sys::Object, value: Option<&str>) {
    js::set_opt_str(target, "label", value);
}

fn kebab(debug: impl fmt::Debug) -> String {
    let input = format!("{debug:?}");
    let mut output = String::new();
    for (index, ch) in input.chars().enumerate() {
        if ch.is_ascii_uppercase() && index != 0 {
            output.push('-');
        }
        output.push(ch.to_ascii_lowercase());
    }
    output
}

fn texture_format(value: TextureFormat) -> String {
    use TextureFormat::*;
    match value {
        R8Unorm => "r8unorm".into(),
        R8Snorm => "r8snorm".into(),
        R8Uint => "r8uint".into(),
        R8Sint => "r8sint".into(),
        R16Unorm => "r16unorm".into(),
        R16Snorm => "r16snorm".into(),
        R16Uint => "r16uint".into(),
        R16Sint => "r16sint".into(),
        R16Float => "r16float".into(),
        Rg8Unorm => "rg8unorm".into(),
        Rg8Snorm => "rg8snorm".into(),
        Rg8Uint => "rg8uint".into(),
        Rg8Sint => "rg8sint".into(),
        R32Uint => "r32uint".into(),
        R32Sint => "r32sint".into(),
        R32Float => "r32float".into(),
        Rg16Unorm => "rg16unorm".into(),
        Rg16Snorm => "rg16snorm".into(),
        Rg16Uint => "rg16uint".into(),
        Rg16Sint => "rg16sint".into(),
        Rg16Float => "rg16float".into(),
        Rgba8Unorm => "rgba8unorm".into(),
        Rgba8UnormSrgb => "rgba8unorm-srgb".into(),
        Rgba8Snorm => "rgba8snorm".into(),
        Rgba8Uint => "rgba8uint".into(),
        Rgba8Sint => "rgba8sint".into(),
        Bgra8Unorm => "bgra8unorm".into(),
        Bgra8UnormSrgb => "bgra8unorm-srgb".into(),
        Rgb10a2Uint => "rgb10a2uint".into(),
        Rgb10a2Unorm => "rgb10a2unorm".into(),
        Rg11b10Ufloat => "rg11b10ufloat".into(),
        Rg32Uint => "rg32uint".into(),
        Rg32Sint => "rg32sint".into(),
        Rg32Float => "rg32float".into(),
        Rgba16Unorm => "rgba16unorm".into(),
        Rgba16Snorm => "rgba16snorm".into(),
        Rgba16Uint => "rgba16uint".into(),
        Rgba16Sint => "rgba16sint".into(),
        Rgba16Float => "rgba16float".into(),
        Rgba32Uint => "rgba32uint".into(),
        Rgba32Sint => "rgba32sint".into(),
        Rgba32Float => "rgba32float".into(),
        Stencil8 => "stencil8".into(),
        Depth16Unorm => "depth16unorm".into(),
        Depth24Plus => "depth24plus".into(),
        Depth24PlusStencil8 => "depth24plus-stencil8".into(),
        Depth32Float => "depth32float".into(),
        Depth32FloatStencil8 => "depth32float-stencil8".into(),
    }
}

fn parse_texture_format(value: &str) -> Option<TextureFormat> {
    Some(match value {
        "bgra8unorm" => TextureFormat::Bgra8Unorm,
        "rgba8unorm" => TextureFormat::Rgba8Unorm,
        "rgba8unorm-srgb" => TextureFormat::Rgba8UnormSrgb,
        "bgra8unorm-srgb" => TextureFormat::Bgra8UnormSrgb,
        _ => return None,
    })
}

fn extent(value: Extent3d) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "width", js::number(value.width));
    js::set(&result, "height", js::number(value.height));
    js::set(
        &result,
        "depthOrArrayLayers",
        js::number(value.depth_or_array_layers),
    );
    result
}

fn origin(value: Origin3d) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "x", js::number(value.x));
    js::set(&result, "y", js::number(value.y));
    js::set(&result, "z", js::number(value.z));
    result
}

fn load_store<V: Copy>(ops: Operations<V>, value: impl Fn(V) -> JsValue) -> js_sys::Object {
    let result = js::object();
    match ops.load {
        LoadOp::Load => js::set(&result, "loadOp", js::string("load")),
        LoadOp::Clear(clear) => {
            js::set(&result, "loadOp", js::string("clear"));
            js::set(&result, "clearValue", value(clear));
        }
        LoadOp::DontCare(_) => js::set(&result, "loadOp", js::string("load")),
    }
    js::set(
        &result,
        "storeOp",
        js::string(match ops.store {
            StoreOp::Store => "store",
            StoreOp::Discard => "discard",
        }),
    );
    result
}

fn color(value: Color) -> JsValue {
    let result = js::object();
    js::set(&result, "r", js::number(value.r));
    js::set(&result, "g", js::number(value.g));
    js::set(&result, "b", js::number(value.b));
    js::set(&result, "a", js::number(value.a));
    result.into()
}

fn buffer_binding(value: BufferBinding<'_>) -> JsValue {
    let result = js::object();
    js::set(&result, "buffer", &value.buffer.raw);
    js::set(&result, "offset", JsValue::from_f64(value.offset as f64));
    if let Some(size) = value.size {
        js::set(&result, "size", JsValue::from_f64(size.get() as f64));
    }
    result.into()
}

fn binding_resource(value: BindingResource<'_>) -> JsValue {
    match value {
        BindingResource::Buffer(value) => buffer_binding(value),
        BindingResource::BufferArray(values) => {
            js::array(values.iter().copied().map(buffer_binding)).into()
        }
        BindingResource::Sampler(value) => value.raw.clone(),
        BindingResource::SamplerArray(values) => {
            js::array(values.iter().map(|v| v.raw.clone())).into()
        }
        BindingResource::TextureView(value) => value.raw.clone(),
        BindingResource::TextureViewArray(values) => {
            js::array(values.iter().map(|v| v.raw.clone())).into()
        }
        BindingResource::ExternalTexture(value) => value.raw.clone(),
        BindingResource::AccelerationStructure(_) => {
            panic!("browser WebGPU does not expose Helio ray-tracing bindings")
        }
    }
}

fn range_of<R: RangeBounds<u64>>(range: R, size: u64) -> Range<u64> {
    let start = match range.start_bound() {
        Bound::Included(value) => *value,
        Bound::Excluded(value) => value.saturating_add(1),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(value) => value.saturating_add(1),
        Bound::Excluded(value) => *value,
        Bound::Unbounded => size,
    };
    assert!(start <= end && end <= size, "buffer slice is out of bounds");
    start..end
}

fn browser_gpu() -> Result<JsValue, String> {
    if let Some(window) = web_sys::window() {
        return js::get_opt(window.navigator().as_ref(), "gpu")
            .ok_or_else(|| "navigator.gpu is unavailable".to_string());
    }

    let global = js_sys::global();
    let navigator = js::get_opt(&global, "navigator")
        .ok_or_else(|| "the browser global has no navigator".to_string())?;
    js::get_opt(&navigator, "gpu").ok_or_else(|| "navigator.gpu is unavailable".to_string())
}

fn feature_name(feature: Features) -> Option<&'static str> {
    Some(if feature == Features::TEXTURE_COMPRESSION_BC {
        "texture-compression-bc"
    } else if feature == Features::TEXTURE_COMPRESSION_ETC2 {
        "texture-compression-etc2"
    } else if feature == Features::TEXTURE_COMPRESSION_ASTC {
        "texture-compression-astc"
    } else if feature == Features::TIMESTAMP_QUERY {
        "timestamp-query"
    } else if feature == Features::INDIRECT_FIRST_INSTANCE {
        "indirect-first-instance"
    } else if feature == Features::SHADER_F16 {
        "shader-f16"
    } else if feature == Features::RG11B10UFLOAT_RENDERABLE {
        "rg11b10ufloat-renderable"
    } else if feature == Features::BGRA8UNORM_STORAGE {
        "bgra8unorm-storage"
    } else if feature == Features::FLOAT32_FILTERABLE {
        "float32-filterable"
    } else {
        return None;
    })
}

fn features_from(raw: &JsValue) -> Features {
    let Some(iterable) = js::get_opt(raw, "features") else {
        return Features::empty();
    };
    let Ok(Some(iterator)) = js_sys::try_iter(&iterable) else {
        return Features::empty();
    };
    let mut result = Features::empty();
    for value in iterator.flatten().filter_map(|value| value.as_string()) {
        let feature = match value.as_str() {
            "texture-compression-bc" => Features::TEXTURE_COMPRESSION_BC,
            "texture-compression-etc2" => Features::TEXTURE_COMPRESSION_ETC2,
            "texture-compression-astc" => Features::TEXTURE_COMPRESSION_ASTC,
            "timestamp-query" => Features::TIMESTAMP_QUERY,
            "indirect-first-instance" => Features::INDIRECT_FIRST_INSTANCE,
            "shader-f16" => Features::SHADER_F16,
            "rg11b10ufloat-renderable" => Features::RG11B10UFLOAT_RENDERABLE,
            "bgra8unorm-storage" => Features::BGRA8UNORM_STORAGE,
            "float32-filterable" => Features::FLOAT32_FILTERABLE,
            _ => continue,
        };
        result |= feature;
    }
    result
}

fn requested_features(features: Features) -> Array {
    let result = Array::new();
    let known = [
        Features::TEXTURE_COMPRESSION_BC,
        Features::TEXTURE_COMPRESSION_ETC2,
        Features::TEXTURE_COMPRESSION_ASTC,
        Features::TIMESTAMP_QUERY,
        Features::INDIRECT_FIRST_INSTANCE,
        Features::SHADER_F16,
        Features::RG11B10UFLOAT_RENDERABLE,
        Features::BGRA8UNORM_STORAGE,
        Features::FLOAT32_FILTERABLE,
    ];
    let mut consumed = Features::empty();
    for feature in known {
        if features.contains(feature) {
            consumed |= feature;
            let name = feature_name(feature).expect("known browser feature must have a name");
            result.push(&js::string(name));
        }
    }
    let unknown = features - consumed;
    assert!(
        unknown.is_empty(),
        "requested features {unknown:?} have no browser WebGPU equivalent"
    );
    result
}

fn limit_u32(raw: &JsValue, name: &str, fallback: u32) -> u32 {
    js::get_opt(raw, name)
        .and_then(|value| value.as_f64())
        .map(|value| value as u32)
        .unwrap_or(fallback)
}

fn limit_u64(raw: &JsValue, name: &str, fallback: u64) -> u64 {
    js::get_opt(raw, name)
        .and_then(|value| value.as_f64())
        .map(|value| value as u64)
        .unwrap_or(fallback)
}

fn limits_from(handle: &JsValue) -> Limits {
    let Some(raw) = js::get_opt(handle, "limits") else {
        return Limits::default();
    };
    let mut limits = Limits::default();
    macro_rules! u32_limits {
        ($($field:ident => $name:literal),+ $(,)?) => {$({
            limits.$field = limit_u32(&raw, $name, limits.$field);
        })+};
    }
    macro_rules! u64_limits {
        ($($field:ident => $name:literal),+ $(,)?) => {$({
            limits.$field = limit_u64(&raw, $name, limits.$field);
        })+};
    }
    u32_limits!(
        max_texture_dimension_1d => "maxTextureDimension1D",
        max_texture_dimension_2d => "maxTextureDimension2D",
        max_texture_dimension_3d => "maxTextureDimension3D",
        max_texture_array_layers => "maxTextureArrayLayers",
        max_bind_groups => "maxBindGroups",
        max_bind_groups_plus_vertex_buffers => "maxBindGroupsPlusVertexBuffers",
        max_bindings_per_bind_group => "maxBindingsPerBindGroup",
        max_dynamic_uniform_buffers_per_pipeline_layout => "maxDynamicUniformBuffersPerPipelineLayout",
        max_dynamic_storage_buffers_per_pipeline_layout => "maxDynamicStorageBuffersPerPipelineLayout",
        max_sampled_textures_per_shader_stage => "maxSampledTexturesPerShaderStage",
        max_samplers_per_shader_stage => "maxSamplersPerShaderStage",
        max_storage_buffers_per_shader_stage => "maxStorageBuffersPerShaderStage",
        max_storage_textures_per_shader_stage => "maxStorageTexturesPerShaderStage",
        max_uniform_buffers_per_shader_stage => "maxUniformBuffersPerShaderStage",
        max_vertex_buffers => "maxVertexBuffers",
        max_vertex_attributes => "maxVertexAttributes",
        max_vertex_buffer_array_stride => "maxVertexBufferArrayStride",
        max_inter_stage_shader_variables => "maxInterStageShaderVariables",
        min_uniform_buffer_offset_alignment => "minUniformBufferOffsetAlignment",
        min_storage_buffer_offset_alignment => "minStorageBufferOffsetAlignment",
        max_color_attachments => "maxColorAttachments",
        max_color_attachment_bytes_per_sample => "maxColorAttachmentBytesPerSample",
        max_compute_workgroup_storage_size => "maxComputeWorkgroupStorageSize",
        max_compute_invocations_per_workgroup => "maxComputeInvocationsPerWorkgroup",
        max_compute_workgroup_size_x => "maxComputeWorkgroupSizeX",
        max_compute_workgroup_size_y => "maxComputeWorkgroupSizeY",
        max_compute_workgroup_size_z => "maxComputeWorkgroupSizeZ",
        max_compute_workgroups_per_dimension => "maxComputeWorkgroupsPerDimension",
    );
    u64_limits!(
        max_uniform_buffer_binding_size => "maxUniformBufferBindingSize",
        max_storage_buffer_binding_size => "maxStorageBufferBindingSize",
        max_buffer_size => "maxBufferSize",
    );
    limits
}

fn required_limits(limits: &Limits) -> js_sys::Object {
    let result = js::object();
    macro_rules! set_limits {
        ($($field:ident => $name:literal),+ $(,)?) => {$({
            js::set(&result, $name, JsValue::from_f64(limits.$field as f64));
        })+};
    }
    set_limits!(
        max_texture_dimension_1d => "maxTextureDimension1D",
        max_texture_dimension_2d => "maxTextureDimension2D",
        max_texture_dimension_3d => "maxTextureDimension3D",
        max_texture_array_layers => "maxTextureArrayLayers",
        max_bind_groups => "maxBindGroups",
        max_bind_groups_plus_vertex_buffers => "maxBindGroupsPlusVertexBuffers",
        max_bindings_per_bind_group => "maxBindingsPerBindGroup",
        max_dynamic_uniform_buffers_per_pipeline_layout => "maxDynamicUniformBuffersPerPipelineLayout",
        max_dynamic_storage_buffers_per_pipeline_layout => "maxDynamicStorageBuffersPerPipelineLayout",
        max_sampled_textures_per_shader_stage => "maxSampledTexturesPerShaderStage",
        max_samplers_per_shader_stage => "maxSamplersPerShaderStage",
        max_storage_buffers_per_shader_stage => "maxStorageBuffersPerShaderStage",
        max_storage_textures_per_shader_stage => "maxStorageTexturesPerShaderStage",
        max_uniform_buffers_per_shader_stage => "maxUniformBuffersPerShaderStage",
        max_uniform_buffer_binding_size => "maxUniformBufferBindingSize",
        max_storage_buffer_binding_size => "maxStorageBufferBindingSize",
        max_vertex_buffers => "maxVertexBuffers",
        max_buffer_size => "maxBufferSize",
        max_vertex_attributes => "maxVertexAttributes",
        max_vertex_buffer_array_stride => "maxVertexBufferArrayStride",
        max_inter_stage_shader_variables => "maxInterStageShaderVariables",
        min_uniform_buffer_offset_alignment => "minUniformBufferOffsetAlignment",
        min_storage_buffer_offset_alignment => "minStorageBufferOffsetAlignment",
        max_color_attachments => "maxColorAttachments",
        max_color_attachment_bytes_per_sample => "maxColorAttachmentBytesPerSample",
        max_compute_workgroup_storage_size => "maxComputeWorkgroupStorageSize",
        max_compute_invocations_per_workgroup => "maxComputeInvocationsPerWorkgroup",
        max_compute_workgroup_size_x => "maxComputeWorkgroupSizeX",
        max_compute_workgroup_size_y => "maxComputeWorkgroupSizeY",
        max_compute_workgroup_size_z => "maxComputeWorkgroupSizeZ",
        max_compute_workgroups_per_dimension => "maxComputeWorkgroupsPerDimension",
    );
    result
}

impl Instance {
    pub fn new(_descriptor: InstanceDescriptor) -> Self {
        Self {
            gpu: browser_gpu().unwrap_or_else(|error| panic!("{error}")),
        }
    }

    pub async fn request_adapter(
        &self,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> Result<Adapter, RequestAdapterError> {
        let descriptor = js::object();
        match options.power_preference {
            PowerPreference::LowPower => {
                js::set(&descriptor, "powerPreference", js::string("low-power"));
            }
            PowerPreference::HighPerformance => {
                js::set(
                    &descriptor,
                    "powerPreference",
                    js::string("high-performance"),
                );
            }
            PowerPreference::None => {}
        }
        js::set(
            &descriptor,
            "forceFallbackAdapter",
            js::bool_value(options.force_fallback_adapter),
        );
        let promise = js::call_promise(&self.gpu, "requestAdapter", &[descriptor.into()]);
        let raw = JsFuture::from(promise)
            .await
            .map_err(|error| RequestAdapterError(js::error_string(&error)))?;
        if raw.is_null() || raw.is_undefined() {
            return Err(RequestAdapterError(
                "the browser returned no compatible WebGPU adapter".into(),
            ));
        }
        Ok(Adapter {
            features: features_from(&raw),
            limits: limits_from(&raw),
            raw,
        })
    }

    pub fn create_surface(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<Surface<'static>, CreateSurfaceError> {
        let context = canvas
            .get_context("webgpu")
            .map_err(|error| CreateSurfaceError(js::error_string(&error)))?
            .ok_or_else(|| {
                CreateSurfaceError("canvas.getContext('webgpu') returned null".into())
            })?;
        Ok(Surface {
            canvas,
            context: context.into(),
            gpu: self.gpu.clone(),
            config: Rc::new(RefCell::new(None)),
            configure_failed: Rc::new(Cell::new(false)),
            _window: PhantomData,
        })
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self::new(InstanceDescriptor::new_without_display_handle())
    }
}

impl Adapter {
    pub fn features(&self) -> Features {
        self.features
    }

    pub fn limits(&self) -> Limits {
        self.limits.clone()
    }

    pub fn get_info(&self) -> AdapterInfo {
        let mut info = AdapterInfo::new(DeviceType::Other, Backend::BrowserWebGpu);
        if let Some(raw) = js::get_opt(&self.raw, "info") {
            info.name = js::get_opt(&raw, "description")
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "Browser WebGPU adapter".into());
            info.vendor = js::get_opt(&raw, "vendor")
                .and_then(|value| value.as_string())
                .and_then(|value| u32::from_str_radix(value.trim_start_matches("0x"), 16).ok())
                .unwrap_or(0);
        }
        info
    }

    pub async fn request_device(
        &self,
        descriptor: &DeviceDescriptor<'_>,
    ) -> Result<(Device, Queue), RequestDeviceError> {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "requiredFeatures",
            requested_features(descriptor.required_features),
        );
        js::set(
            &raw_descriptor,
            "requiredLimits",
            required_limits(&descriptor.required_limits),
        );
        let promise = js::call_promise(&self.raw, "requestDevice", &[raw_descriptor.into()]);
        let raw = JsFuture::from(promise)
            .await
            .map_err(|error| RequestDeviceError(js::error_string(&error)))?;
        let queue = Queue {
            raw: js::get(&raw, "queue"),
        };
        let device = Device {
            raw,
            queue: queue.clone(),
            features: descriptor.required_features,
            limits: descriptor.required_limits.clone(),
        };
        Ok((device, queue))
    }
}

impl Surface<'_> {
    pub fn get_capabilities(&self, _adapter: &Adapter) -> SurfaceCapabilities {
        let preferred = js::call(&self.gpu, "getPreferredCanvasFormat", &[])
            .as_string()
            .and_then(|value| parse_texture_format(&value))
            .unwrap_or(TextureFormat::Bgra8Unorm);
        let mut formats = vec![preferred];
        for format in [TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm] {
            if !formats.contains(&format) {
                formats.push(format);
            }
        }
        SurfaceCapabilities {
            formats,
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![
                CompositeAlphaMode::Opaque,
                CompositeAlphaMode::PreMultiplied,
            ],
            usages: TextureUsages::RENDER_ATTACHMENT,
            ..Default::default()
        }
    }

    pub fn configure(&self, device: &Device, config: &SurfaceConfiguration) {
        assert!(
            config.width > 0 && config.height > 0,
            "surface extent must be nonzero"
        );
        self.canvas.set_width(config.width);
        self.canvas.set_height(config.height);
        let descriptor = js::object();
        js::set(&descriptor, "device", &device.raw);
        js::set(
            &descriptor,
            "format",
            js::string(&texture_format(config.format)),
        );
        js::set(&descriptor, "usage", js::number(config.usage.bits()));
        js::set(
            &descriptor,
            "alphaMode",
            js::string(match config.alpha_mode {
                CompositeAlphaMode::PreMultiplied => "premultiplied",
                _ => "opaque",
            }),
        );
        if !config.view_formats.is_empty() {
            js::set(
                &descriptor,
                "viewFormats",
                js::array(
                    config
                        .view_formats
                        .iter()
                        .map(|format| js::string(&texture_format(*format))),
                ),
            );
        }
        if !matches!(config.color_space, SurfaceColorSpace::Auto) {
            js::set(&descriptor, "colorSpace", js::string("srgb"));
        }
        self.configure_failed
            .set(js::call_result(&self.context, "configure", &[descriptor.into()]).is_err());
        *self.config.borrow_mut() = Some(config.clone());
    }

    pub fn get_configuration(&self) -> Option<SurfaceConfiguration> {
        self.config.borrow().clone()
    }

    pub fn get_current_texture(&self) -> CurrentSurfaceTexture {
        if self.configure_failed.get() {
            return CurrentSurfaceTexture::Lost;
        }
        let Some(config) = self.config.borrow().clone() else {
            return CurrentSurfaceTexture::Validation;
        };
        match js::call_result(&self.context, "getCurrentTexture", &[]) {
            Ok(raw) => CurrentSurfaceTexture::Success(SurfaceTexture {
                texture: Texture {
                    raw,
                    size: Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    format: config.format,
                    usage: config.usage,
                },
            }),
            Err(_) => CurrentSurfaceTexture::Lost,
        }
    }
}

fn texture_dimension(value: TextureDimension) -> &'static str {
    match value {
        TextureDimension::D1 => "1d",
        TextureDimension::D2 => "2d",
        TextureDimension::D3 => "3d",
    }
}

fn texture_view_dimension(value: TextureViewDimension) -> &'static str {
    match value {
        TextureViewDimension::D1 => "1d",
        TextureViewDimension::D2 => "2d",
        TextureViewDimension::D2Array => "2d-array",
        TextureViewDimension::Cube => "cube",
        TextureViewDimension::CubeArray => "cube-array",
        TextureViewDimension::D3 => "3d",
    }
}

fn texture_aspect(value: TextureAspect) -> &'static str {
    match value {
        TextureAspect::All => "all",
        TextureAspect::StencilOnly => "stencil-only",
        TextureAspect::DepthOnly => "depth-only",
        _ => panic!("unsupported WebGPU texture aspect {value:?}"),
    }
}

fn texture_descriptor(value: &TextureDescriptor<'_>) -> js_sys::Object {
    let result = js::object();
    label(&result, value.label);
    js::set(&result, "size", extent(value.size));
    js::set(&result, "mipLevelCount", js::number(value.mip_level_count));
    js::set(&result, "sampleCount", js::number(value.sample_count));
    js::set(
        &result,
        "dimension",
        js::string(texture_dimension(value.dimension)),
    );
    js::set(&result, "format", js::string(&texture_format(value.format)));
    js::set(&result, "usage", js::number(value.usage.bits()));
    if !value.view_formats.is_empty() {
        js::set(
            &result,
            "viewFormats",
            js::array(
                value
                    .view_formats
                    .iter()
                    .map(|value| js::string(&texture_format(*value))),
            ),
        );
    }
    result
}

fn texture_view_descriptor(value: &TextureViewDescriptor<'_>) -> js_sys::Object {
    let result = js::object();
    label(&result, value.label);
    if let Some(format) = value.format {
        js::set(&result, "format", js::string(&texture_format(format)));
    }
    if let Some(dimension) = value.dimension {
        js::set(
            &result,
            "dimension",
            js::string(texture_view_dimension(dimension)),
        );
    }
    js::set(&result, "aspect", js::string(texture_aspect(value.aspect)));
    js::set(&result, "baseMipLevel", js::number(value.base_mip_level));
    if let Some(count) = value.mip_level_count {
        js::set(&result, "mipLevelCount", js::number(count));
    }
    js::set(
        &result,
        "baseArrayLayer",
        js::number(value.base_array_layer),
    );
    if let Some(count) = value.array_layer_count {
        js::set(&result, "arrayLayerCount", js::number(count));
    }
    if let Some(usage) = value.usage {
        js::set(&result, "usage", js::number(usage.bits()));
    }
    result
}

fn buffer_binding_layout(
    ty: BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<BufferSize>,
) -> js_sys::Object {
    let result = js::object();
    js::set(
        &result,
        "type",
        js::string(match ty {
            BufferBindingType::Uniform => "uniform",
            BufferBindingType::Storage { read_only: true } => "read-only-storage",
            BufferBindingType::Storage { read_only: false } => "storage",
        }),
    );
    js::set(
        &result,
        "hasDynamicOffset",
        js::bool_value(has_dynamic_offset),
    );
    if let Some(size) = min_binding_size {
        js::set(
            &result,
            "minBindingSize",
            JsValue::from_f64(size.get() as f64),
        );
    }
    result
}

fn bind_group_layout_entry(value: &BindGroupLayoutEntry) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "binding", js::number(value.binding));
    js::set(&result, "visibility", js::number(value.visibility.bits()));
    match value.ty {
        BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => js::set(
            &result,
            "buffer",
            buffer_binding_layout(ty, has_dynamic_offset, min_binding_size),
        ),
        BindingType::Sampler(ty) => {
            let layout = js::object();
            js::set(
                &layout,
                "type",
                js::string(match ty {
                    SamplerBindingType::Filtering => "filtering",
                    SamplerBindingType::NonFiltering => "non-filtering",
                    SamplerBindingType::Comparison => "comparison",
                }),
            );
            js::set(&result, "sampler", layout);
        }
        BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => {
            let layout = js::object();
            js::set(
                &layout,
                "sampleType",
                js::string(match sample_type {
                    TextureSampleType::Float { filterable: true } => "float",
                    TextureSampleType::Float { filterable: false } => "unfilterable-float",
                    TextureSampleType::Depth => "depth",
                    TextureSampleType::Sint => "sint",
                    TextureSampleType::Uint => "uint",
                }),
            );
            js::set(
                &layout,
                "viewDimension",
                js::string(texture_view_dimension(view_dimension)),
            );
            js::set(&layout, "multisampled", js::bool_value(multisampled));
            js::set(&result, "texture", layout);
        }
        BindingType::StorageTexture {
            access,
            format,
            view_dimension,
        } => {
            let layout = js::object();
            js::set(
                &layout,
                "access",
                js::string(match access {
                    StorageTextureAccess::WriteOnly => "write-only",
                    StorageTextureAccess::ReadOnly => "read-only",
                    StorageTextureAccess::ReadWrite => "read-write",
                    StorageTextureAccess::Atomic => {
                        panic!("atomic storage textures are not a browser WebGPU binding type")
                    }
                }),
            );
            js::set(&layout, "format", js::string(&texture_format(format)));
            js::set(
                &layout,
                "viewDimension",
                js::string(texture_view_dimension(view_dimension)),
            );
            js::set(&result, "storageTexture", layout);
        }
        BindingType::ExternalTexture => {
            js::set(&result, "externalTexture", js::object());
        }
        BindingType::AccelerationStructure { .. } => {
            panic!("browser WebGPU does not expose Helio ray-tracing bindings")
        }
    }
    if let Some(count) = value.count {
        // This member is part of the binding-array WebGPU extension. Browsers
        // that do not implement it report a normal validation error.
        js::set(&result, "count", js::number(count.get()));
    }
    result
}

fn compilation_options(value: &PipelineCompilationOptions<'_>) -> js_sys::Object {
    let result = js::object();
    if !value.constants.is_empty() {
        let constants = js::object();
        for (name, value) in value.constants {
            js::set(&constants, name, JsValue::from_f64(*value));
        }
        js::set(&result, "constants", constants);
    }
    result
}

fn blend_component(value: BlendComponent) -> js_sys::Object {
    let result = js::object();
    js::set(
        &result,
        "operation",
        js::string(match value.operation {
            BlendOperation::Add => "add",
            BlendOperation::Subtract => "subtract",
            BlendOperation::ReverseSubtract => "reverse-subtract",
            BlendOperation::Min => "min",
            BlendOperation::Max => "max",
        }),
    );
    js::set(&result, "srcFactor", js::string(&kebab(value.src_factor)));
    js::set(&result, "dstFactor", js::string(&kebab(value.dst_factor)));
    result
}

fn color_target(value: &ColorTargetState) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "format", js::string(&texture_format(value.format)));
    js::set(&result, "writeMask", js::number(value.write_mask.bits()));
    if let Some(blend) = value.blend {
        let descriptor = js::object();
        js::set(&descriptor, "color", blend_component(blend.color));
        js::set(&descriptor, "alpha", blend_component(blend.alpha));
        js::set(&result, "blend", descriptor);
    }
    result
}

fn stencil_face(value: StencilFaceState) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "compare", js::string(&kebab(value.compare)));
    js::set(&result, "failOp", js::string(&kebab(value.fail_op)));
    js::set(
        &result,
        "depthFailOp",
        js::string(&kebab(value.depth_fail_op)),
    );
    js::set(&result, "passOp", js::string(&kebab(value.pass_op)));
    result
}

fn depth_stencil(value: &DepthStencilState) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "format", js::string(&texture_format(value.format)));
    if let Some(enabled) = value.depth_write_enabled {
        js::set(&result, "depthWriteEnabled", js::bool_value(enabled));
    }
    if let Some(compare) = value.depth_compare {
        js::set(&result, "depthCompare", js::string(&kebab(compare)));
    }
    js::set(&result, "stencilFront", stencil_face(value.stencil.front));
    js::set(&result, "stencilBack", stencil_face(value.stencil.back));
    js::set(
        &result,
        "stencilReadMask",
        js::number(value.stencil.read_mask),
    );
    js::set(
        &result,
        "stencilWriteMask",
        js::number(value.stencil.write_mask),
    );
    js::set(
        &result,
        "depthBias",
        JsValue::from_f64(value.bias.constant as f64),
    );
    js::set(
        &result,
        "depthBiasSlopeScale",
        js::number(value.bias.slope_scale),
    );
    js::set(&result, "depthBiasClamp", js::number(value.bias.clamp));
    result
}

fn vertex_state(value: &VertexState<'_>) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "module", &value.module.raw);
    if let Some(entry_point) = value.entry_point {
        js::set(&result, "entryPoint", js::string(entry_point));
    }
    let buffers = Array::new();
    for buffer in value.buffers {
        let Some(buffer) = buffer else {
            buffers.push(&JsValue::NULL);
            continue;
        };
        let descriptor = js::object();
        js::set(
            &descriptor,
            "arrayStride",
            JsValue::from_f64(buffer.array_stride as f64),
        );
        js::set(
            &descriptor,
            "stepMode",
            js::string(match buffer.step_mode {
                VertexStepMode::Vertex => "vertex",
                VertexStepMode::Instance => "instance",
            }),
        );
        let attributes = Array::new();
        for attribute in buffer.attributes {
            let raw = js::object();
            js::set(&raw, "format", js::string(&kebab(attribute.format)));
            js::set(&raw, "offset", JsValue::from_f64(attribute.offset as f64));
            js::set(
                &raw,
                "shaderLocation",
                js::number(attribute.shader_location),
            );
            attributes.push(&raw);
        }
        js::set(&descriptor, "attributes", attributes);
        buffers.push(&descriptor);
    }
    js::set(&result, "buffers", buffers);
    let options = compilation_options(&value.compilation_options);
    for key in ["constants"] {
        if let Some(raw) = js::get_opt(options.as_ref(), key) {
            js::set(&result, key, raw);
        }
    }
    result
}

fn fragment_state(value: &FragmentState<'_>) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "module", &value.module.raw);
    if let Some(entry_point) = value.entry_point {
        js::set(&result, "entryPoint", js::string(entry_point));
    }
    let targets = Array::new();
    for target in value.targets {
        match target {
            Some(target) => targets.push(&color_target(target)),
            None => targets.push(&JsValue::NULL),
        };
    }
    js::set(&result, "targets", targets);
    let options = compilation_options(&value.compilation_options);
    if let Some(raw) = js::get_opt(options.as_ref(), "constants") {
        js::set(&result, "constants", raw);
    }
    result
}

impl Device {
    pub fn features(&self) -> Features {
        self.features
    }

    pub fn limits(&self) -> Limits {
        self.limits.clone()
    }

    pub(crate) fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn poll(&self, _poll_type: crate::PollType) -> Result<(), String> {
        // Browser mapping completion is driven by the JavaScript event loop.
        Ok(())
    }

    pub fn on_uncaptured_error(&self, callback: Arc<dyn Fn(Error) + Send + Sync + 'static>) {
        let listener = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
            let raw = js::get_opt(&event, "error").unwrap_or(event);
            let description = js::error_string(&raw);
            callback(Error::Validation {
                source: description.clone(),
                description,
            });
        });
        let _ = js::call_result(
            &self.raw,
            "addEventListener",
            &[js::string("uncapturederror"), listener.as_ref().clone()],
        );
        listener.forget();
    }

    pub fn create_shader_module(&self, descriptor: ShaderModuleDescriptor<'_>) -> ShaderModule {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        match descriptor.source {
            ShaderSource::Wgsl(source) => js::set(&raw_descriptor, "code", js::string(&source)),
        }
        ShaderModule {
            raw: js::call(&self.raw, "createShaderModule", &[raw_descriptor.into()]),
        }
    }

    pub fn create_buffer(&self, descriptor: &BufferDescriptor<'_>) -> Buffer {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "size",
            JsValue::from_f64(descriptor.size as f64),
        );
        js::set(
            &raw_descriptor,
            "usage",
            js::number(descriptor.usage.bits()),
        );
        js::set(
            &raw_descriptor,
            "mappedAtCreation",
            js::bool_value(descriptor.mapped_at_creation),
        );
        Buffer {
            raw: js::call(&self.raw, "createBuffer", &[raw_descriptor.into()]),
            size: descriptor.size,
            usage: descriptor.usage,
        }
    }

    pub fn create_texture(&self, descriptor: &TextureDescriptor<'_>) -> Texture {
        Texture {
            raw: js::call(
                &self.raw,
                "createTexture",
                &[texture_descriptor(descriptor).into()],
            ),
            size: descriptor.size,
            format: descriptor.format,
            usage: descriptor.usage,
        }
    }

    pub fn create_sampler(&self, descriptor: &SamplerDescriptor<'_>) -> Sampler {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "addressModeU",
            js::string(&kebab(descriptor.address_mode_u)),
        );
        js::set(
            &raw_descriptor,
            "addressModeV",
            js::string(&kebab(descriptor.address_mode_v)),
        );
        js::set(
            &raw_descriptor,
            "addressModeW",
            js::string(&kebab(descriptor.address_mode_w)),
        );
        js::set(
            &raw_descriptor,
            "magFilter",
            js::string(&kebab(descriptor.mag_filter)),
        );
        js::set(
            &raw_descriptor,
            "minFilter",
            js::string(&kebab(descriptor.min_filter)),
        );
        js::set(
            &raw_descriptor,
            "mipmapFilter",
            js::string(&kebab(descriptor.mipmap_filter)),
        );
        js::set(
            &raw_descriptor,
            "lodMinClamp",
            js::number(descriptor.lod_min_clamp),
        );
        js::set(
            &raw_descriptor,
            "lodMaxClamp",
            js::number(descriptor.lod_max_clamp),
        );
        if let Some(compare) = descriptor.compare {
            js::set(&raw_descriptor, "compare", js::string(&kebab(compare)));
        }
        js::set(
            &raw_descriptor,
            "maxAnisotropy",
            js::number(descriptor.anisotropy_clamp),
        );
        Sampler {
            raw: js::call(&self.raw, "createSampler", &[raw_descriptor.into()]),
        }
    }

    pub fn create_bind_group_layout(
        &self,
        descriptor: &BindGroupLayoutDescriptor<'_>,
    ) -> BindGroupLayout {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "entries",
            js::array(
                descriptor
                    .entries
                    .iter()
                    .map(|value| bind_group_layout_entry(value).into()),
            ),
        );
        BindGroupLayout {
            raw: js::call(&self.raw, "createBindGroupLayout", &[raw_descriptor.into()]),
        }
    }

    pub fn create_bind_group(&self, descriptor: &BindGroupDescriptor<'_>) -> BindGroup {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(&raw_descriptor, "layout", &descriptor.layout.raw);
        let entries = Array::new();
        for entry in descriptor.entries {
            let raw = js::object();
            js::set(&raw, "binding", js::number(entry.binding));
            js::set(&raw, "resource", binding_resource(entry.resource));
            entries.push(&raw);
        }
        js::set(&raw_descriptor, "entries", entries);
        BindGroup {
            raw: js::call(&self.raw, "createBindGroup", &[raw_descriptor.into()]),
        }
    }

    pub fn create_pipeline_layout(
        &self,
        descriptor: &PipelineLayoutDescriptor<'_>,
    ) -> PipelineLayout {
        assert_eq!(
            descriptor.immediate_size, 0,
            "browser WebGPU has no immediate/push-constant API"
        );
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "bindGroupLayouts",
            js::array(descriptor.bind_group_layouts.iter().map(|layout| {
                layout
                    .map(|layout| layout.raw.clone())
                    .unwrap_or(JsValue::NULL)
            })),
        );
        PipelineLayout {
            raw: js::call(&self.raw, "createPipelineLayout", &[raw_descriptor.into()]),
        }
    }

    pub fn create_render_pipeline(
        &self,
        descriptor: &RenderPipelineDescriptor<'_>,
    ) -> RenderPipeline {
        assert!(
            descriptor.cache.is_none(),
            "browser WebGPU has no pipeline cache objects"
        );
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "layout",
            descriptor
                .layout
                .map(|layout| layout.raw.clone())
                .unwrap_or_else(|| js::string("auto")),
        );
        js::set(&raw_descriptor, "vertex", vertex_state(&descriptor.vertex));
        let primitive = js::object();
        js::set(
            &primitive,
            "topology",
            js::string(&kebab(descriptor.primitive.topology)),
        );
        if let Some(strip_index_format) = descriptor.primitive.strip_index_format {
            js::set(
                &primitive,
                "stripIndexFormat",
                js::string(&kebab(strip_index_format)),
            );
        }
        js::set(
            &primitive,
            "frontFace",
            js::string(&kebab(descriptor.primitive.front_face)),
        );
        if let Some(cull_mode) = descriptor.primitive.cull_mode {
            js::set(&primitive, "cullMode", js::string(&kebab(cull_mode)));
        } else {
            js::set(&primitive, "cullMode", js::string("none"));
        }
        if descriptor.primitive.polygon_mode != PolygonMode::Fill
            || descriptor.primitive.unclipped_depth
            || descriptor.primitive.conservative
        {
            panic!("requested primitive state is not supported by browser WebGPU");
        }
        js::set(&raw_descriptor, "primitive", primitive);
        if let Some(depth) = descriptor.depth_stencil.as_ref() {
            js::set(&raw_descriptor, "depthStencil", depth_stencil(depth));
        }
        let multisample = js::object();
        js::set(
            &multisample,
            "count",
            js::number(descriptor.multisample.count),
        );
        js::set(
            &multisample,
            "mask",
            js::number(descriptor.multisample.mask as u32),
        );
        js::set(
            &multisample,
            "alphaToCoverageEnabled",
            js::bool_value(descriptor.multisample.alpha_to_coverage_enabled),
        );
        js::set(&raw_descriptor, "multisample", multisample);
        if let Some(fragment) = descriptor.fragment.as_ref() {
            js::set(&raw_descriptor, "fragment", fragment_state(fragment));
        }
        if descriptor.multiview_mask.is_some() {
            panic!("browser WebGPU does not expose multiview pipeline masks");
        }
        RenderPipeline {
            raw: js::call(&self.raw, "createRenderPipeline", &[raw_descriptor.into()]),
        }
    }

    pub fn create_compute_pipeline(
        &self,
        descriptor: &ComputePipelineDescriptor<'_>,
    ) -> ComputePipeline {
        assert!(
            descriptor.cache.is_none(),
            "browser WebGPU has no pipeline cache objects"
        );
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "layout",
            descriptor
                .layout
                .map(|layout| layout.raw.clone())
                .unwrap_or_else(|| js::string("auto")),
        );
        let compute = js::object();
        js::set(&compute, "module", &descriptor.module.raw);
        if let Some(entry_point) = descriptor.entry_point {
            js::set(&compute, "entryPoint", js::string(entry_point));
        }
        let options = compilation_options(&descriptor.compilation_options);
        if let Some(constants) = js::get_opt(options.as_ref(), "constants") {
            js::set(&compute, "constants", constants);
        }
        js::set(&raw_descriptor, "compute", compute);
        ComputePipeline {
            raw: js::call(&self.raw, "createComputePipeline", &[raw_descriptor.into()]),
        }
    }

    pub fn create_command_encoder(
        &self,
        descriptor: &CommandEncoderDescriptor<'_>,
    ) -> CommandEncoder {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        CommandEncoder {
            raw: js::call(&self.raw, "createCommandEncoder", &[raw_descriptor.into()]),
        }
    }

    pub fn create_query_set(&self, descriptor: &QuerySetDescriptor<'_>) -> QuerySet {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "type",
            js::string(match descriptor.ty {
                crate::QueryType::Occlusion => "occlusion",
                crate::QueryType::Timestamp => "timestamp",
                crate::QueryType::PipelineStatistics(_) => {
                    panic!("pipeline statistics queries are not part of browser WebGPU")
                }
            }),
        );
        js::set(&raw_descriptor, "count", js::number(descriptor.count));
        QuerySet {
            raw: js::call(&self.raw, "createQuerySet", &[raw_descriptor.into()]),
        }
    }

    pub fn create_render_bundle_encoder(
        &self,
        descriptor: &RenderBundleEncoderDescriptor<'_>,
    ) -> RenderBundleEncoder {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        js::set(
            &raw_descriptor,
            "colorFormats",
            js::array(descriptor.color_formats.iter().map(|format| {
                format
                    .map(|format| js::string(&texture_format(format)))
                    .unwrap_or(JsValue::NULL)
            })),
        );
        if let Some(depth) = descriptor.depth_stencil.as_ref() {
            js::set(
                &raw_descriptor,
                "depthStencilFormat",
                js::string(&texture_format(depth.format)),
            );
            js::set(
                &raw_descriptor,
                "depthReadOnly",
                js::bool_value(depth.depth_read_only),
            );
            js::set(
                &raw_descriptor,
                "stencilReadOnly",
                js::bool_value(depth.stencil_read_only),
            );
        }
        js::set(
            &raw_descriptor,
            "sampleCount",
            js::number(descriptor.sample_count),
        );
        RenderBundleEncoder {
            raw: js::call(
                &self.raw,
                "createRenderBundleEncoder",
                &[raw_descriptor.into()],
            ),
        }
    }
}

impl Queue {
    pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]) {
        let bytes = Uint8Array::from(data);
        js::call(
            &self.raw,
            "writeBuffer",
            &[
                buffer.raw.clone(),
                JsValue::from_f64(offset as f64),
                bytes.into(),
            ],
        );
    }

    pub fn write_texture(
        &self,
        destination: TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: TexelCopyBufferLayout,
        size: Extent3d,
    ) {
        let bytes = Uint8Array::from(data);
        let destination = texel_copy_texture(destination);
        let layout = texel_copy_layout(layout);
        js::call(
            &self.raw,
            "writeTexture",
            &[
                destination.into(),
                bytes.into(),
                layout.into(),
                extent(size).into(),
            ],
        );
    }

    pub fn submit<I>(&self, command_buffers: I) -> SubmissionIndex
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        let buffers = js::array(command_buffers.into_iter().map(|buffer| buffer.raw));
        js::call(&self.raw, "submit", &[buffers.into()]);
        SubmissionIndex
    }

    pub fn present(&self, _texture: SurfaceTexture) {
        // Browser WebGPU presents the current canvas texture automatically at
        // the end of the JavaScript task.
    }

    pub fn get_timestamp_period(&self) -> f32 {
        // WebGPU timestamps are specified in nanoseconds.
        1.0
    }
}

impl SurfaceTexture {
    pub fn present(self) {
        // Browser WebGPU presents canvas textures automatically.
    }
}

impl Buffer {
    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn usage(&self) -> BufferUsages {
        self.usage
    }

    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice<'_> {
        BufferSlice {
            buffer: self,
            range: range_of(bounds, self.size),
        }
    }

    pub fn as_entire_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        })
    }

    pub fn unmap(&self) {
        js::call(&self.raw, "unmap", &[]);
    }

    pub fn destroy(&self) {
        js::call(&self.raw, "destroy", &[]);
    }
}

impl BufferSlice<'_> {
    pub fn map_async(
        &self,
        mode: MapMode,
        callback: impl FnOnce(Result<(), BufferAsyncError>) + 'static,
    ) {
        let raw = self.buffer.raw.clone();
        let offset = self.range.start;
        let size = self.range.end - self.range.start;
        wasm_bindgen_futures::spawn_local(async move {
            let promise = js::call_promise(
                &raw,
                "mapAsync",
                &[
                    js::number(mode.bits()),
                    JsValue::from_f64(offset as f64),
                    JsValue::from_f64(size as f64),
                ],
            );
            let result = JsFuture::from(promise)
                .await
                .map(|_| ())
                .map_err(|error| BufferAsyncError(js::error_string(&error)));
            callback(result);
        });
    }

    fn mapped_bytes(&self) -> Result<Uint8Array, BufferAsyncError> {
        let raw = js::call_result(
            &self.buffer.raw,
            "getMappedRange",
            &[
                JsValue::from_f64(self.range.start as f64),
                JsValue::from_f64((self.range.end - self.range.start) as f64),
            ],
        )
        .map_err(|error| BufferAsyncError(js::error_string(&error)))?;
        let array_buffer: ArrayBuffer = raw
            .dyn_into()
            .map_err(|_| BufferAsyncError("getMappedRange did not return an ArrayBuffer".into()))?;
        Ok(Uint8Array::new(&array_buffer))
    }

    pub fn get_mapped_range(&self) -> Result<BufferView, BufferAsyncError> {
        Ok(BufferView {
            data: self.mapped_bytes()?.to_vec(),
        })
    }

    pub fn get_mapped_range_mut(&self) -> Result<BufferViewMut, BufferAsyncError> {
        let mapped = self.mapped_bytes()?;
        Ok(BufferViewMut {
            data: mapped.to_vec(),
            mapped,
        })
    }
}

impl Texture {
    pub fn size(&self) -> Extent3d {
        self.size
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn depth_or_array_layers(&self) -> u32 {
        self.size.depth_or_array_layers
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn usage(&self) -> TextureUsages {
        self.usage
    }

    pub fn create_view(&self, descriptor: &TextureViewDescriptor<'_>) -> TextureView {
        TextureView {
            raw: js::call(
                &self.raw,
                "createView",
                &[texture_view_descriptor(descriptor).into()],
            ),
        }
    }

    pub fn as_image_copy(&self) -> TexelCopyTextureInfo<'_> {
        TexelCopyTextureInfo {
            texture: self,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        }
    }

    pub fn destroy(&self) {
        js::call(&self.raw, "destroy", &[]);
    }
}

impl RenderPipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        BindGroupLayout {
            raw: js::call(&self.raw, "getBindGroupLayout", &[js::number(index)]),
        }
    }
}

impl ComputePipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        BindGroupLayout {
            raw: js::call(&self.raw, "getBindGroupLayout", &[js::number(index)]),
        }
    }
}

impl QuerySet {
    pub fn destroy(&self) {
        js::call(&self.raw, "destroy", &[]);
    }
}

fn texel_copy_texture(value: TexelCopyTextureInfo<'_>) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "texture", &value.texture.raw);
    js::set(&result, "mipLevel", js::number(value.mip_level));
    js::set(&result, "origin", origin(value.origin));
    js::set(&result, "aspect", js::string(texture_aspect(value.aspect)));
    result
}

fn texel_copy_buffer(value: TexelCopyBufferInfo<'_>) -> js_sys::Object {
    let result = texel_copy_layout(value.layout);
    js::set(&result, "buffer", &value.buffer.raw);
    result
}

fn texel_copy_layout(value: TexelCopyBufferLayout) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "offset", JsValue::from_f64(value.offset as f64));
    if let Some(bytes_per_row) = value.bytes_per_row {
        js::set(&result, "bytesPerRow", js::number(bytes_per_row));
    }
    if let Some(rows_per_image) = value.rows_per_image {
        js::set(&result, "rowsPerImage", js::number(rows_per_image));
    }
    result
}

fn render_pass_timestamp_writes(value: RenderPassTimestampWrites<'_>) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "querySet", &value.query_set.raw);
    if let Some(index) = value.beginning_of_pass_write_index {
        js::set(&result, "beginningOfPassWriteIndex", js::number(index));
    }
    if let Some(index) = value.end_of_pass_write_index {
        js::set(&result, "endOfPassWriteIndex", js::number(index));
    }
    result
}

fn compute_pass_timestamp_writes(value: ComputePassTimestampWrites<'_>) -> js_sys::Object {
    let result = js::object();
    js::set(&result, "querySet", &value.query_set.raw);
    if let Some(index) = value.beginning_of_pass_write_index {
        js::set(&result, "beginningOfPassWriteIndex", js::number(index));
    }
    if let Some(index) = value.end_of_pass_write_index {
        js::set(&result, "endOfPassWriteIndex", js::number(index));
    }
    result
}

impl CommandEncoder {
    pub fn begin_render_pass(&mut self, descriptor: &RenderPassDescriptor<'_>) -> RenderPass<'_> {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        let color_attachments = Array::new();
        for attachment in descriptor.color_attachments {
            let Some(attachment) = attachment else {
                color_attachments.push(&JsValue::NULL);
                continue;
            };
            let raw = load_store(attachment.ops, color);
            js::set(&raw, "view", &attachment.view.raw);
            if let Some(depth_slice) = attachment.depth_slice {
                js::set(&raw, "depthSlice", js::number(depth_slice));
            }
            if let Some(resolve_target) = attachment.resolve_target {
                js::set(&raw, "resolveTarget", &resolve_target.raw);
            }
            color_attachments.push(&raw);
        }
        js::set(&raw_descriptor, "colorAttachments", color_attachments);
        if let Some(attachment) = descriptor.depth_stencil_attachment {
            let raw = js::object();
            js::set(&raw, "view", &attachment.view.raw);
            if let Some(ops) = attachment.depth_ops {
                let operations = load_store(ops, |value| js::number(value));
                for key in ["loadOp", "clearValue", "storeOp"] {
                    if let Some(value) = js::get_opt(operations.as_ref(), key) {
                        let target = match key {
                            "loadOp" => "depthLoadOp",
                            "clearValue" => "depthClearValue",
                            "storeOp" => "depthStoreOp",
                            _ => unreachable!(),
                        };
                        js::set(&raw, target, value);
                    }
                }
            } else {
                js::set(&raw, "depthReadOnly", js::bool_value(true));
            }
            if let Some(ops) = attachment.stencil_ops {
                let operations = load_store(ops, |value| js::number(value));
                for key in ["loadOp", "clearValue", "storeOp"] {
                    if let Some(value) = js::get_opt(operations.as_ref(), key) {
                        let target = match key {
                            "loadOp" => "stencilLoadOp",
                            "clearValue" => "stencilClearValue",
                            "storeOp" => "stencilStoreOp",
                            _ => unreachable!(),
                        };
                        js::set(&raw, target, value);
                    }
                }
            } else {
                js::set(&raw, "stencilReadOnly", js::bool_value(true));
            }
            js::set(&raw_descriptor, "depthStencilAttachment", raw);
        }
        if let Some(writes) = descriptor.timestamp_writes {
            js::set(
                &raw_descriptor,
                "timestampWrites",
                render_pass_timestamp_writes(writes),
            );
        }
        if let Some(query_set) = descriptor.occlusion_query_set {
            js::set(&raw_descriptor, "occlusionQuerySet", &query_set.raw);
        }
        if descriptor.multiview_mask.is_some() {
            panic!("browser WebGPU does not expose multiview masks");
        }
        RenderPass {
            raw: js::call(&self.raw, "beginRenderPass", &[raw_descriptor.into()]),
            ended: false,
            _encoder: PhantomData,
        }
    }

    pub fn begin_compute_pass(
        &mut self,
        descriptor: &ComputePassDescriptor<'_>,
    ) -> ComputePass<'_> {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        if let Some(writes) = descriptor.timestamp_writes {
            js::set(
                &raw_descriptor,
                "timestampWrites",
                compute_pass_timestamp_writes(writes),
            );
        }
        ComputePass {
            raw: js::call(&self.raw, "beginComputePass", &[raw_descriptor.into()]),
            ended: false,
            _encoder: PhantomData,
        }
    }

    pub fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) {
        let mut args = vec![buffer.raw.clone(), JsValue::from_f64(offset as f64)];
        if let Some(size) = size {
            args.push(JsValue::from_f64(size as f64));
        }
        js::call(&self.raw, "clearBuffer", &args);
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: BufferAddress,
        destination: &Buffer,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        js::call(
            &self.raw,
            "copyBufferToBuffer",
            &[
                source.raw.clone(),
                JsValue::from_f64(source_offset as f64),
                destination.raw.clone(),
                JsValue::from_f64(destination_offset as f64),
                JsValue::from_f64(copy_size as f64),
            ],
        );
    }

    pub fn copy_texture_to_texture(
        &mut self,
        source: TexelCopyTextureInfo<'_>,
        destination: TexelCopyTextureInfo<'_>,
        copy_size: Extent3d,
    ) {
        js::call(
            &self.raw,
            "copyTextureToTexture",
            &[
                texel_copy_texture(source).into(),
                texel_copy_texture(destination).into(),
                extent(copy_size).into(),
            ],
        );
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: TexelCopyTextureInfo<'_>,
        destination: TexelCopyBufferInfo<'_>,
        copy_size: Extent3d,
    ) {
        js::call(
            &self.raw,
            "copyTextureToBuffer",
            &[
                texel_copy_texture(source).into(),
                texel_copy_buffer(destination).into(),
                extent(copy_size).into(),
            ],
        );
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        source: TexelCopyBufferInfo<'_>,
        destination: TexelCopyTextureInfo<'_>,
        copy_size: Extent3d,
    ) {
        js::call(
            &self.raw,
            "copyBufferToTexture",
            &[
                texel_copy_buffer(source).into(),
                texel_copy_texture(destination).into(),
                extent(copy_size).into(),
            ],
        );
    }

    pub fn resolve_query_set(
        &mut self,
        query_set: &QuerySet,
        query_range: Range<u32>,
        destination: &Buffer,
        destination_offset: BufferAddress,
    ) {
        js::call(
            &self.raw,
            "resolveQuerySet",
            &[
                query_set.raw.clone(),
                js::number(query_range.start),
                js::number(query_range.end - query_range.start),
                destination.raw.clone(),
                JsValue::from_f64(destination_offset as f64),
            ],
        );
    }

    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        js::call(
            &self.raw,
            "writeTimestamp",
            &[query_set.raw.clone(), js::number(query_index)],
        );
    }

    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            raw: js::call(&self.raw, "finish", &[]),
        }
    }
}

fn dynamic_offsets(values: &[DynamicOffset]) -> Array {
    js::array(values.iter().map(|value| js::number(*value)))
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        js::call(&self.raw, "setPipeline", &[pipeline.raw.clone()]);
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &BindGroup,
        offsets: &[DynamicOffset],
    ) {
        js::call(
            &self.raw,
            "setBindGroup",
            &[
                js::number(index),
                bind_group.raw.clone(),
                dynamic_offsets(offsets).into(),
            ],
        );
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, slice: BufferSlice<'_>) {
        js::call(
            &self.raw,
            "setVertexBuffer",
            &[
                js::number(slot),
                slice.buffer.raw.clone(),
                JsValue::from_f64(slice.range.start as f64),
                JsValue::from_f64((slice.range.end - slice.range.start) as f64),
            ],
        );
    }

    pub fn set_index_buffer(&mut self, slice: BufferSlice<'_>, format: IndexFormat) {
        js::call(
            &self.raw,
            "setIndexBuffer",
            &[
                slice.buffer.raw.clone(),
                js::string(&kebab(format)),
                JsValue::from_f64(slice.range.start as f64),
                JsValue::from_f64((slice.range.end - slice.range.start) as f64),
            ],
        );
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        js::call(
            &self.raw,
            "draw",
            &[
                js::number(vertices.end - vertices.start),
                js::number(instances.end - instances.start),
                js::number(vertices.start),
                js::number(instances.start),
            ],
        );
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        js::call(
            &self.raw,
            "drawIndexed",
            &[
                js::number(indices.end - indices.start),
                js::number(instances.end - instances.start),
                js::number(indices.start),
                JsValue::from_f64(base_vertex as f64),
                js::number(instances.start),
            ],
        );
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: BufferAddress) {
        js::call(
            &self.raw,
            "drawIndirect",
            &[buffer.raw.clone(), JsValue::from_f64(offset as f64)],
        );
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: BufferAddress) {
        js::call(
            &self.raw,
            "drawIndexedIndirect",
            &[buffer.raw.clone(), JsValue::from_f64(offset as f64)],
        );
    }

    pub fn multi_draw_indirect(&mut self, buffer: &Buffer, offset: BufferAddress, count: u32) {
        for index in 0..count {
            self.draw_indirect(buffer, offset + u64::from(index) * 16);
        }
    }

    pub fn multi_draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: BufferAddress,
        count: u32,
    ) {
        for index in 0..count {
            self.draw_indexed_indirect(buffer, offset + u64::from(index) * 20);
        }
    }

    pub fn multi_draw_indirect_count(
        &mut self,
        _buffer: &Buffer,
        _offset: BufferAddress,
        _count_buffer: &Buffer,
        _count_buffer_offset: BufferAddress,
        _max_count: u32,
    ) {
        panic!("multi-draw indirect count is not part of browser WebGPU");
    }

    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        _buffer: &Buffer,
        _offset: BufferAddress,
        _count_buffer: &Buffer,
        _count_buffer_offset: BufferAddress,
        _max_count: u32,
    ) {
        panic!("multi-draw indexed indirect count is not part of browser WebGPU");
    }

    pub fn execute_bundles<'bundle>(
        &mut self,
        bundles: impl IntoIterator<Item = &'bundle RenderBundle>,
    ) {
        js::call(
            &self.raw,
            "executeBundles",
            &[js::array(bundles.into_iter().map(|bundle| bundle.raw.clone())).into()],
        );
    }

    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        js::call(
            &self.raw,
            "setViewport",
            &[
                js::number(x),
                js::number(y),
                js::number(width),
                js::number(height),
                js::number(min_depth),
                js::number(max_depth),
            ],
        );
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        js::call(
            &self.raw,
            "setScissorRect",
            &[
                js::number(x),
                js::number(y),
                js::number(width),
                js::number(height),
            ],
        );
    }

    pub fn set_blend_constant(&mut self, value: Color) {
        js::call(&self.raw, "setBlendConstant", &[color(value)]);
    }

    pub fn set_stencil_reference(&mut self, value: u32) {
        js::call(&self.raw, "setStencilReference", &[js::number(value)]);
    }

    pub fn end(mut self) {
        if !self.ended {
            js::call(&self.raw, "end", &[]);
            self.ended = true;
        }
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        if !self.ended {
            js::call(&self.raw, "end", &[]);
            self.ended = true;
        }
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        js::call(&self.raw, "setPipeline", &[pipeline.raw.clone()]);
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &BindGroup,
        offsets: &[DynamicOffset],
    ) {
        js::call(
            &self.raw,
            "setBindGroup",
            &[
                js::number(index),
                bind_group.raw.clone(),
                dynamic_offsets(offsets).into(),
            ],
        );
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        js::call(
            &self.raw,
            "dispatchWorkgroups",
            &[js::number(x), js::number(y), js::number(z)],
        );
    }

    pub fn dispatch_workgroups_indirect(&mut self, buffer: &Buffer, offset: BufferAddress) {
        js::call(
            &self.raw,
            "dispatchWorkgroupsIndirect",
            &[buffer.raw.clone(), JsValue::from_f64(offset as f64)],
        );
    }

    pub fn end(mut self) {
        if !self.ended {
            js::call(&self.raw, "end", &[]);
            self.ended = true;
        }
    }
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        if !self.ended {
            js::call(&self.raw, "end", &[]);
            self.ended = true;
        }
    }
}

impl RenderBundleEncoder {
    pub fn finish(self, descriptor: &RenderBundleDescriptor<'_>) -> RenderBundle {
        let raw_descriptor = js::object();
        label(&raw_descriptor, descriptor.label);
        RenderBundle {
            raw: js::call(&self.raw, "finish", &[raw_descriptor.into()]),
        }
    }
}
