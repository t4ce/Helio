//! GPU-instanced 2D sprite batcher with texture atlasing.
//!
//! Sprites live in SceneDB's persistent, handle-addressed World
//! ([`SpriteBatchPass::insert_sprite`] / [`update_sprite`](SpriteBatchPass::update_sprite) /
//! [`remove_sprite`](SpriteBatchPass::remove_sprite)) — not a per-frame push
//! list. This matters for what gets uploaded to the GPU each frame, matching
//! the dirty-range convention the rest of the engine uses for its scene
//! buffers: the named partner column and its component-local presence column
//! are flushed by SceneDB's dirty tracker. The pass owns no parallel slot,
//! alive, or free-list registry.
//!
//! Culling and depth-sorting the component rows into a draw order is **not** done
//! here, and not on the CPU at all — pair this pass with
//! `helio-pass-sprite-cull`'s `SpriteCullPass`, added to the graph *before*
//! this one, and wire its outputs in via [`SpriteBatchPass::use_gpu_culling`].
//! That pass culls + radix-sorts the current component-local row span on the GPU
//! every frame without per-row CPU work, and this pass's `execute()` issues one
//! `draw_indexed_indirect` reading the GPU-computed instance count, never
//! learning the visible count on the CPU at all. See that crate's module doc
//! comment for the full design (and why it's a separate crate: no Cargo
//! dependency either way, just `Arc<wgpu::Buffer>` handles passed between
//! them, matching how `helio-pass-shadow-cull`/`helio-pass-shadow` are wired).
//!
//! The pass renders via vertex-pulling (a `var<storage, read> instances` array
//! indexed through a separate `draw_order` array), not
//! `VertexStepMode::Instance` — see the shader's module doc comment for why.
//!
//! The contained authority is SceneDB itself, so the pass remains usable in a
//! standalone 2D graph without manufacturing a second Helio scene registry.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result};
use helio_scenedb::{
    register_sprite_component_buffer, sprite_content_hash, sprite_presence_snapshot,
    SceneAuthority, SceneAuthorityConfig, SceneSprite, SceneSpriteAtlasLayer, SpriteAtlasError,
    SceneAuthoritySubsystemConfig, SpriteAtlasResidency, SpriteAtlasSource, SpriteBufferSource,
    SPRITE_BUFFER_KEY,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub use helio_scenedb::{
    SceneSpriteAtlasId as SpriteAtlasHandle, SceneSpriteId as SpriteHandle,
    SceneSpriteRow as SpriteInstance, SpriteAuthorityError as SpriteError,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

// Y-up world space; v=0 at the top of the quad to match texture-space UVs.
const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { pos: [-0.5, -0.5], uv: [0.0, 1.0] },
    QuadVertex { pos: [0.5, -0.5], uv: [1.0, 1.0] },
    QuadVertex { pos: [-0.5, 0.5], uv: [0.0, 0.0] },
    QuadVertex { pos: [0.5, 0.5], uv: [1.0, 0.0] },
];
const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    runtime_capacity: u32,
    _pad: [u32; 3],
}

/// GPU-instanced 2D sprite batch pass.
///
/// Owns its own orthographic camera (recomputed from the render target size
/// every frame — call [`SpriteBatchPass::set_camera`] to override
/// framing/zoom/pan) and starts with a 1×1 white fallback atlas layer, so it
/// renders solid-colored quads out of the box; call
/// [`SpriteBatchPass::add_atlas_layer`] to load real sprite sheets.
pub struct SpriteBatchPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bound_instance_epoch: u64,
    bound_runtime_epoch: u64,
    bound_atlas_epoch: u64,
    fallback_runtime_buf: wgpu::Buffer,

    authority: Option<SceneAuthority>,
    buffer_source: SpriteBufferSource,
    atlas_source: SpriteAtlasSource,
    next_authored_epoch: u32,

    quad_vertex_buf: wgpu::Buffer,
    quad_index_buf: wgpu::Buffer,

    // ── GPU cull/sort wiring (provided by `helio-pass-sprite-cull`) ───────
    gpu_culling: Option<GpuCulling>,

    camera_buf: wgpu::Buffer,
    camera_dirty: bool,
    last_width: u32,
    last_height: u32,
    /// Half-extent of the orthographic view, in world units. `None` means
    /// "derive from the render target's pixel size" (1 world unit = 1 pixel).
    camera_half_extent: Option<[f32; 2]>,
    camera_center: [f32; 2],
    clear_color: Option<wgpu::Color>,
}

/// Outputs of a paired `helio-pass-sprite-cull` `SpriteCullPass`, wired in
/// via [`SpriteBatchPass::use_gpu_culling`]. `draw_order_buf` is the GPU-sorted
/// list of component-local sprite rows to draw; `indirect_buf` holds
/// `DrawIndexedIndirectArgs` whose `instance_count` the cull pass writes every
/// frame — the CPU never learns the visible count.
struct GpuCulling {
    draw_order_buf: Arc<wgpu::Buffer>,
    indirect_buf: Arc<wgpu::Buffer>,
}

impl SpriteBatchPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) -> Self {
        Self::new_internal(device, queue, surface_format, None)
    }

    /// Render a sprite component owned by an existing Helio/SceneDB scene.
    /// This mode creates no second authority; authored CRUD belongs to the
    /// scene that publishes these sources.
    pub fn from_publications(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        buffer_source: SpriteBufferSource,
        atlas_source: SpriteAtlasSource,
    ) -> Self {
        Self::new_internal(
            device,
            queue,
            surface_format,
            Some((buffer_source, atlas_source)),
        )
    }

    fn new_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        publications: Option<(SpriteBufferSource, SpriteAtlasSource)>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Batch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Batch BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Batch PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sprite Batch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Straight-alpha atlas texels over an opaque or transparent
                    // target; premultiplied atlases should premultiply at import
                    // time and use `BlendState::REPLACE` via a future variant.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Camera Uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let quad_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Quad Vertices"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sprite Quad Indices"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        const INITIAL_CAPACITY: u32 = 256;
        let device = Arc::new(device.clone());
        let queue = Arc::new(queue.clone());
        let (authority, buffer_source, atlas_source) = if let Some((buffer_source, atlas_source)) = publications {
            (None, buffer_source, atlas_source)
        } else {
            let mut config = SceneAuthorityConfig::default();
            config.initial_entity_capacity = INITIAL_CAPACITY;
            config.subsystems = SceneAuthoritySubsystemConfig::SPRITE_STANDALONE;
            let mut authority = SceneAuthority::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                config,
                |store, device| register_sprite_component_buffer(store, INITIAL_CAPACITY, device),
            );
            authority.register_subsystem(SpriteAtlasResidency::new(
                Arc::clone(&device),
                Arc::clone(&queue),
            ));
            let instance = authority
                .partner_buffer_snapshot(SPRITE_BUFFER_KEY)
                .expect("SceneSprite partner is registered during construction");
            let (presence, presence_epoch) = sprite_presence_snapshot(authority.gpu_store())
                .expect("SceneSprite presence is registered during construction");
            let buffer_source = SpriteBufferSource::new(
                instance.buffer,
                instance.epoch,
                presence,
                presence_epoch,
                0,
            );
            let atlas_source = authority
                .subsystem::<SpriteAtlasResidency>()
                .expect("sprite atlas residency is registered")
                .publication_source();
            (Some(authority), buffer_source, atlas_source)
        };
        let fallback_runtime_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Runtime Fallback"),
            size: 24,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            bind_group: None,
            bound_instance_epoch: u64::MAX,
            bound_runtime_epoch: u64::MAX,
            bound_atlas_epoch: u64::MAX,
            fallback_runtime_buf,
            authority,
            buffer_source,
            atlas_source,
            next_authored_epoch: 1,
            quad_vertex_buf,
            quad_index_buf,
            gpu_culling: None,
            camera_buf,
            camera_dirty: true,
            last_width: 0,
            last_height: 0,
            camera_half_extent: None,
            camera_center: [0.0, 0.0],
            clear_color: Some(wgpu::Color::BLACK),
        }
    }

    fn allocate_authored_epoch(&mut self) -> std::result::Result<u32, SpriteError> {
        let epoch = self.next_authored_epoch;
        if epoch == 0 {
            return Err(SpriteError::AuthoredEpochExhausted);
        }
        self.next_authored_epoch = self.next_authored_epoch.checked_add(1).unwrap_or(0);
        Ok(epoch)
    }

    fn standalone_authority(&self) -> std::result::Result<&SceneAuthority, SpriteError> {
        self.authority.as_ref().ok_or(SpriteError::ExternalAuthority)
    }

    fn standalone_authority_mut(
        &mut self,
    ) -> std::result::Result<&mut SceneAuthority, SpriteError> {
        self.authority.as_mut().ok_or(SpriteError::ExternalAuthority)
    }

    fn refresh_buffer_source(&self) {
        let Some(authority) = self.authority.as_ref() else {
            return;
        };
        let instance = authority
            .partner_buffer_snapshot(SPRITE_BUFFER_KEY)
            .expect("SceneSprite partner remains registered");
        let (presence, presence_epoch) = sprite_presence_snapshot(authority.gpu_store())
            .expect("SceneSprite presence remains registered");
        self.buffer_source.publish_authored(
            instance.buffer,
            instance.epoch,
            presence,
            presence_epoch,
            authority.gpu_row_span::<SceneSprite>(),
        );
    }

    fn resolve_atlas_for_row(
        &mut self,
        row: &mut SpriteInstance,
        retain: bool,
    ) -> std::result::Result<Option<SpriteAtlasHandle>, SpriteError> {
        let Some(atlas) = row.atlas() else {
            row.clear_atlas();
            return Ok(None);
        };
        let authority = self.standalone_authority_mut()?;
        if authority
            .get::<SceneSpriteAtlasLayer>(atlas.entity())
            .is_none()
        {
            return Err(SpriteError::StaleAtlas(atlas));
        }
        let residency = authority
            .subsystem_mut::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered");
        let physical_layer = if retain {
            residency.retain(atlas.entity())?
        } else {
            residency
                .resolve(atlas.entity())
                .ok_or(SpriteError::StaleAtlas(atlas))?
        };
        row.resolve_atlas_for_gpu(atlas, physical_layer);
        Ok(Some(atlas))
    }

    pub fn try_insert_sprite(
        &mut self,
        mut instance: SpriteInstance,
    ) -> std::result::Result<SpriteHandle, SpriteError> {
        let authority = self.standalone_authority()?;
        let row_span = authority.gpu_row_span::<SceneSprite>();
        let live_count = authority.gpu_live_count::<SceneSprite>();
        let required = if live_count < row_span {
            row_span.max(1)
        } else {
            row_span
                .checked_add(1)
                .ok_or(SpriteError::CapacityRequestTooLarge(usize::MAX))?
        };
        self.standalone_authority_mut()?
            .reserve_gpu_component_capacity::<SceneSprite>(required)?;
        let authored_epoch = self.allocate_authored_epoch()?;
        self.resolve_atlas_for_row(&mut instance, true)?;
        instance.set_authored_epoch(authored_epoch);
        let entity = self
            .standalone_authority_mut()?
            .insert(SceneSprite { sprite: instance });
        debug_assert!(self
            .standalone_authority()?
            .gpu_row::<SceneSprite>(entity)
            .is_some());
        self.refresh_buffer_source();
        Ok(SpriteHandle(entity))
    }

    pub fn insert_sprite(&mut self, instance: SpriteInstance) -> SpriteHandle {
        self.try_insert_sprite(instance)
            .expect("SpriteBatchPass::insert_sprite failed")
    }

    pub fn try_update_sprite(
        &mut self,
        handle: SpriteHandle,
        mut instance: SpriteInstance,
    ) -> std::result::Result<(), SpriteError> {
        let old = self
            .standalone_authority()?
            .get::<SceneSprite>(handle.entity())
            .copied()
            .ok_or(SpriteError::StaleSprite(handle))?;
        let old_atlas = old.sprite.atlas();
        let new_atlas = instance.atlas();
        let authored_epoch = self.allocate_authored_epoch()?;
        if old_atlas != new_atlas {
            if let Some(old_atlas) = old_atlas {
                self.standalone_authority()?
                    .subsystem::<SpriteAtlasResidency>()
                    .expect("sprite atlas residency is registered")
                    .validate_release_reference(old_atlas.entity())?;
            }
        }
        self.resolve_atlas_for_row(&mut instance, old_atlas != new_atlas)?;
        instance.set_authored_epoch(authored_epoch);
        if !self
            .standalone_authority_mut()?
            .replace_gpu(handle.entity(), SceneSprite { sprite: instance })
        {
            if old_atlas != new_atlas {
                if let Some(new_atlas) = new_atlas {
                    let _ = self
                        .standalone_authority_mut()?
                        .subsystem_mut::<SpriteAtlasResidency>()
                        .expect("sprite atlas residency is registered")
                        .release_reference(new_atlas.entity());
                }
            }
            return Err(SpriteError::StaleSprite(handle));
        }
        if old_atlas != new_atlas {
            if let Some(old_atlas) = old_atlas {
                self.standalone_authority_mut()?
                    .subsystem_mut::<SpriteAtlasResidency>()
                    .expect("sprite atlas residency is registered")
                    .release_reference(old_atlas.entity())?;
            }
        }
        self.refresh_buffer_source();
        Ok(())
    }

    pub fn update_sprite(&mut self, handle: SpriteHandle, instance: SpriteInstance) {
        self.try_update_sprite(handle, instance)
            .expect("SpriteBatchPass::update_sprite failed");
    }

    pub fn try_remove_sprite(
        &mut self,
        handle: SpriteHandle,
    ) -> std::result::Result<SpriteInstance, SpriteError> {
        let sprite = self
            .standalone_authority()?
            .get::<SceneSprite>(handle.entity())
            .copied()
            .ok_or(SpriteError::StaleSprite(handle))?;
        if let Some(atlas) = sprite.sprite.atlas() {
            self.standalone_authority()?
                .subsystem::<SpriteAtlasResidency>()
                .expect("sprite atlas residency is registered")
                .validate_release_reference(atlas.entity())?;
            self.standalone_authority_mut()?
                .subsystem_mut::<SpriteAtlasResidency>()
                .expect("sprite atlas residency is registered")
                .release_reference(atlas.entity())?;
        }
        if !self.standalone_authority_mut()?.despawn(handle.entity()) {
            if let Some(atlas) = sprite.sprite.atlas() {
                self.standalone_authority_mut()?
                    .subsystem_mut::<SpriteAtlasResidency>()
                    .expect("sprite atlas residency is registered")
                    .retain(atlas.entity())
                    .expect("sprite removal rollback restores the released reference");
            }
            return Err(SpriteError::StaleSprite(handle));
        }
        self.refresh_buffer_source();
        Ok(sprite.sprite)
    }

    pub fn remove_sprite(&mut self, handle: SpriteHandle) {
        self.try_remove_sprite(handle)
            .expect("SpriteBatchPass::remove_sprite failed");
    }

    pub fn sprite(&self, handle: SpriteHandle) -> Option<SpriteInstance> {
        self.authority
            .as_ref()?
            .get::<SceneSprite>(handle.entity())
            .map(|sprite| sprite.sprite)
    }

    pub fn try_clear_sprites(&mut self) -> std::result::Result<(), SpriteError> {
        let handles: Vec<_> = self
            .standalone_authority()?
            .query::<SceneSprite>()
            .map(|(entity, _)| SpriteHandle(entity))
            .collect();
        for handle in handles {
            self.try_remove_sprite(handle)?;
        }
        Ok(())
    }

    pub fn clear_sprites(&mut self) {
        self.try_clear_sprites()
            .expect("SpriteBatchPass::clear_sprites failed");
    }

    pub fn try_add_atlas_layer(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> std::result::Result<SpriteAtlasHandle, SpriteError> {
        self.standalone_authority()?
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .validate(width, height, rgba8)?;
        let entity = self.standalone_authority_mut()?.insert(SceneSpriteAtlasLayer {
            width,
            height,
            content_hash: sprite_content_hash(rgba8),
        });
        let result = self
            .standalone_authority_mut()?
            .subsystem_mut::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .insert(entity, width, height, rgba8);
        if let Err(error) = result {
            let _ = self.standalone_authority_mut()?.despawn(entity);
            return Err(error.into());
        }
        self.bind_group = None;
        Ok(SpriteAtlasHandle(entity))
    }

    pub fn add_atlas_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> SpriteAtlasHandle {
        self.try_add_atlas_layer(device, queue, width, height, rgba8)
            .expect("SpriteBatchPass::add_atlas_layer failed")
    }

    pub fn try_remove_atlas_layer(
        &mut self,
        handle: SpriteAtlasHandle,
    ) -> std::result::Result<(), SpriteError> {
        if self
            .standalone_authority()?
            .get::<SceneSpriteAtlasLayer>(handle.entity())
            .is_none()
        {
            return Err(SpriteError::StaleAtlas(handle));
        }
        self.standalone_authority_mut()?
            .subsystem_mut::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .remove(handle.entity())?;
        if !self.standalone_authority_mut()?.despawn(handle.entity()) {
            return Err(SpriteError::StaleAtlas(handle));
        }
        Ok(())
    }

    pub fn remove_atlas_layer(&mut self, handle: SpriteAtlasHandle) {
        self.try_remove_atlas_layer(handle)
            .expect("SpriteBatchPass::remove_atlas_layer failed");
    }

    pub fn try_clear_atlas_layers(&mut self) -> std::result::Result<(), SpriteError> {
        let layers: Vec<_> = self
            .standalone_authority()?
            .query::<SceneSpriteAtlasLayer>()
            .map(|(entity, _)| SpriteAtlasHandle(entity))
            .collect();
        let residency = self
            .standalone_authority()?
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered");
        for layer in &layers {
            let references = residency
                .references(layer.entity())
                .ok_or(SpriteError::StaleAtlas(*layer))?;
            if references != 0 {
                return Err(SpriteAtlasError::LayerInUse { references }.into());
            }
        }
        for layer in layers {
            self.try_remove_atlas_layer(layer)?;
        }
        Ok(())
    }

    pub fn clear_atlas_layers(&mut self) {
        self.try_clear_atlas_layers()
            .expect("SpriteBatchPass::clear_atlas_layers failed");
    }

    /// Overrides the orthographic view: `center` and `half_extent` in world
    /// units. `half_extent = None` re-derives it from the render target's
    /// pixel dimensions each frame (1 world unit = 1 pixel, origin centered).
    pub fn set_camera(&mut self, center: [f32; 2], half_extent: Option<[f32; 2]>) {
        self.camera_center = center;
        self.camera_half_extent = half_extent;
        self.camera_dirty = true;
    }

    /// `None` disables the clear (loads the existing target contents);
    /// `Some(color)` clears to that color every frame. Defaults to opaque black.
    pub fn set_clear_color(&mut self, color: Option<wgpu::Color>) {
        self.clear_color = color;
    }

    /// Pre-grow SceneDB's component-local value and presence columns. Existing
    /// consumers follow allocation epochs, so this is a performance hint, not
    /// a correctness requirement.
    pub fn try_reserve(
        &mut self,
        _device: &wgpu::Device,
        capacity: usize,
    ) -> std::result::Result<(), SpriteError> {
        let capacity = u32::try_from(capacity)
            .map_err(|_| SpriteError::CapacityRequestTooLarge(capacity))?;
        self.standalone_authority_mut()?
            .reserve_entity_capacity(capacity);
        self.standalone_authority_mut()?
            .reserve_gpu_component_capacity::<SceneSprite>(capacity)?;
        self.refresh_buffer_source();
        Ok(())
    }

    pub fn reserve(&mut self, device: &wgpu::Device, capacity: usize) {
        self.try_reserve(device, capacity)
            .expect("SpriteBatchPass::reserve failed");
    }

    /// A compatibility snapshot of the current instance-data partner buffer.
    /// It cannot follow later SceneDB growth; integrated consumers should use
    /// [`buffer_source`](Self::buffer_source).
    pub fn instances_buffer(&self) -> Arc<wgpu::Buffer> {
        Arc::new(self.buffer_source.snapshot().instances)
    }

    /// A compatibility snapshot of SceneDB's component-local presence buffer.
    /// It cannot follow later SceneDB growth; integrated consumers should use
    /// [`buffer_source`](Self::buffer_source).
    pub fn alive_buffer(&self) -> Arc<wgpu::Buffer> {
        Arc::new(self.buffer_source.snapshot().presence)
    }

    /// Epoch-aware source used by integrated cull/simulation passes.
    pub fn buffer_source(&self) -> SpriteBufferSource {
        self.buffer_source.clone()
    }

    pub fn owns_scene_authority(&self) -> bool {
        self.authority.is_some()
    }

    /// Flush authored SceneDB rows immediately. Normal graph execution does
    /// this in `prepare`; this hook is useful for explicit hand-off/testing.
    pub fn flush_scene_gpu(&self) {
        if let Some(authority) = self.authority.as_ref() {
            authority.flush_gpu();
        }
        self.refresh_buffer_source();
    }

    /// Hands the pass the cull/sort pass's outputs: a `draw_order_buf`
    /// (GPU-written, radix-sorted component-local rows) and an `indirect_buf`
    /// (`DrawIndexedIndirectArgs` whose `instance_count` the cull pass writes
    /// each frame). After this, `prepare()` no longer does any CPU culling or
    /// sorting and `execute()` issues a single `draw_indexed_indirect` — the
    /// CPU never learns the visible count.
    pub fn use_gpu_culling(&mut self, draw_order_buf: Arc<wgpu::Buffer>, indirect_buf: Arc<wgpu::Buffer>) {
        self.gpu_culling = Some(GpuCulling { draw_order_buf, indirect_buf });
        self.bind_group = None;
    }

    /// Sprites currently present in the SceneDB component (inserted, not removed).
    pub fn sprite_count(&self) -> usize {
        self.standalone_authority()
            .expect("shared-publication batches query counts through their owning Scene")
            .gpu_live_count::<SceneSprite>() as usize
    }

    pub fn atlas_layer_count(&self) -> usize {
        self.standalone_authority()
            .expect("shared-publication batches query atlas counts through their owning Scene")
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .live_count() as usize
    }

    pub fn atlas_capacity(&self) -> u32 {
        self.standalone_authority()
            .expect("shared-publication batches query atlas capacity through their owning Scene")
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered")
            .capacity()
    }

    pub fn atlas_limits(&self) -> (u32, u32) {
        let residency = self
            .standalone_authority()
            .expect("shared-publication batches query atlas limits through their owning Scene")
            .subsystem::<SpriteAtlasResidency>()
            .expect("sprite atlas residency is registered");
        (
            residency.maximum_imported_layers(),
            residency.maximum_dimension(),
        )
    }
}

impl RenderPass for SpriteBatchPass {
    fn name(&self) -> &'static str {
        "SpriteBatch"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // 2D sprites are alpha-blended and GPU-sorted (see `SpriteInstance::depth`)
        // — no depth attachment. `Box::leak` here matches the convention used by
        // every other executor-managed pass: the descriptor only needs to live
        // for this frame's `execute()` call, and the executor drops it before
        // the next `render_pass_descriptor()`.
        let load = match self.clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };
        let attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Sprite Batch Pass"),
            color_attachments: attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        if ctx.width != self.last_width || ctx.height != self.last_height {
            self.last_width = ctx.width;
            self.last_height = ctx.height;
            self.camera_dirty = true;
        }

        if let Some(authority) = self.authority.as_ref() {
            authority.flush_gpu();
        }
        self.refresh_buffer_source();

        // This tiny write also carries the optional runtime projection span.
        // It remains O(1) regardless of sprite population.
        self.camera_dirty = false;
        let half_extent = self
            .camera_half_extent
            .unwrap_or([ctx.width as f32 * 0.5, ctx.height as f32 * 0.5]);
        let [cx, cy] = self.camera_center;
        let [hx, hy] = half_extent;
        let view_proj =
            glam::Mat4::orthographic_rh(cx - hx, cx + hx, cy - hy, cy + hy, -1.0, 1.0);
        let runtime_capacity = self
            .buffer_source
            .snapshot()
            .runtime
            .map_or(0, |runtime| runtime.row_capacity);
        let uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            runtime_capacity,
            _pad: [0; 3],
        };
        ctx.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let Some(gpu) = self.gpu_culling.as_ref() else {
            return Ok(());
        };

        let source = self.buffer_source.snapshot();
        let atlas = self.atlas_source.snapshot();
        let runtime_epoch = source.runtime.as_ref().map_or(0, |runtime| runtime.epoch);
        if self.bind_group.is_none()
            || self.bound_instance_epoch != source.instances_epoch
            || self.bound_runtime_epoch != runtime_epoch
            || self.bound_atlas_epoch != atlas.epoch
        {
            let runtime_buffer = source
                .runtime
                .as_ref()
                .map_or(&self.fallback_runtime_buf, |runtime| &runtime.buffer);
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Batch BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atlas.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&atlas.sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: source.instances.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gpu.draw_order_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: runtime_buffer.as_entire_binding() },
                ],
            }));
            self.bound_instance_epoch = source.instances_epoch;
            self.bound_runtime_epoch = runtime_epoch;
            self.bound_atlas_epoch = atlas.epoch;
        }

        let Some(rp_ptr) = ctx.active_render_pass_ptr() else {
            return Ok(());
        };
        let rp = unsafe { &mut *rp_ptr };
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        rp.set_vertex_buffer(0, self.quad_vertex_buf.slice(..));
        rp.set_index_buffer(self.quad_index_buf.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed_indirect(&gpu.indirect_buf, 0);

        Ok(())
    }
}
