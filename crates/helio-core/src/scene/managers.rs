//! Helio-owned render-projection buffers with dirty tracking.
//!
//! These managers hold compact draw data, history, uniforms, and other derived
//! renderer state. They are not persistent authored scene storage; canonical
//! component values and their GPU partner rows belong to SceneDB. Each manager
//! wraps a `wgpu::Buffer` with a CPU-side `Vec` and clean managers issue no
//! queue writes during `flush()`.

use crate::upload;
use bytemuck::Zeroable;
use libhelio::{DrawIndexedIndirectArgs, GpuCameraUniforms, GpuDrawCall, GpuShadowMatrix};
use std::{collections::HashMap, sync::Arc};

/// A grow-only GPU storage buffer with dirty-tracked CPU mirror.
///
/// - `flush()` is O(1) when clean (no-op)
/// - Automatically reallocates with 2× growth when capacity is exceeded
/// - Buffer usage includes `STORAGE | COPY_DST` (+ optionally `INDIRECT`)
pub struct GrowableBuffer<T: bytemuck::Pod> {
    // Heap-stable per allocation, but replaced with a fresh Box on growth.
    // Pass bind-group caches currently fingerprint `&wgpu::Buffer` by
    // address; keeping the handle inline made that address stay constant
    // when `self.buf` was overwritten, leaving caches bound to the retired
    // allocation.  Boxing makes the existing fingerprint truthful while
    // `buffer_version` remains available to explicit epoch-aware users.
    buf: Box<wgpu::Buffer>,
    data: Vec<T>,
    dirty_range: Option<(usize, usize)>,
    capacity: usize,
    usage: wgpu::BufferUsages,
    label: &'static str,
    device: Arc<wgpu::Device>,
    buffer_version: u64,
}

impl<T: bytemuck::Pod> GrowableBuffer<T> {
    pub fn new(
        device: Arc<wgpu::Device>,
        initial_capacity: usize,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Self {
        let byte_size = (initial_capacity * std::mem::size_of::<T>()).max(64) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf: Box::new(buf),
            data: Vec::with_capacity(initial_capacity),
            dirty_range: None,
            capacity: initial_capacity,
            usage,
            label,
            device,
            buffer_version: 0,
        }
    }

    /// Returns a reference to the underlying GPU buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buf.as_ref()
    }

    /// Returns the buffer version, incremented each time the buffer is reallocated.
    ///
    /// Passes can use this to detect when bind groups need to be recreated.
    pub fn buffer_version(&self) -> u64 {
        self.buffer_version
    }

    /// Returns the number of elements currently stored.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a read-only view of the CPU-side data (mirrors the GPU buffer).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn mark_dirty_range(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }
        match &mut self.dirty_range {
            Some((dirty_start, dirty_end)) => {
                *dirty_start = (*dirty_start).min(start);
                *dirty_end = (*dirty_end).max(end);
            }
            None => {
                self.dirty_range = Some((start, end));
            }
        }
    }

    /// Returns a mutable reference to the CPU-side data Vec.
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Replaces the entire contents. Marks dirty.
    pub fn set_data(&mut self, data: Vec<T>) {
        self.data = data;
        self.dirty_range = (!self.data.is_empty()).then_some((0, self.data.len()));
    }

    /// Pushes one element and returns its index.
    pub fn push(&mut self, item: T) -> usize {
        let index = self.data.len();
        self.data.push(item);
        self.mark_dirty_range(index, index + 1);
        index
    }

    /// Appends a slice of elements and returns the written index range.
    pub fn extend_from_slice(&mut self, items: &[T]) -> std::ops::Range<usize>
    where
        T: Copy,
    {
        let start = self.data.len();
        self.data.extend_from_slice(items);
        let end = self.data.len();
        self.mark_dirty_range(start, end);
        start..end
    }

    /// Updates one element in-place. Returns `false` if the index is out of bounds.
    pub fn update(&mut self, index: usize, item: T) -> bool {
        let Some(slot) = self.data.get_mut(index) else {
            return false;
        };
        *slot = item;
        self.mark_dirty_range(index, index + 1);
        true
    }

    /// Overwrites a contiguous range in-place. Panics if out of bounds.
    ///
    /// This is the write path for dynamic mesh geometry: call it each frame with
    /// updated vertex data, then `flush()` will upload only the dirty range.
    pub fn update_range(&mut self, start: usize, data: &[T])
    where
        T: Copy,
    {
        let end = start + data.len();
        self.data[start..end].copy_from_slice(data);
        self.mark_dirty_range(start, end);
    }

    /// Removes one element in O(1) by swap-removing it. Returns the removed item.
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.data.len() {
            return None;
        }
        let last_index = self.data.len() - 1;
        let removed = self.data.swap_remove(index);
        if index < self.data.len() {
            self.mark_dirty_range(index, index + 1);
        } else if index < last_index {
            self.mark_dirty_range(index, index);
        }
        Some(removed)
    }

    /// Shrink the CPU mirror to `new_len` elements, clamping the dirty range.
    ///
    /// The GPU buffer is not immediately reallocated (it is oversized); it will
    /// be compacted the next time `flush()` triggers a 2× reallocation.  Draw
    /// calls that reference ranges beyond `new_len` must already have been
    /// removed before calling this, or they will read garbage.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.data.len() {
            return;
        }
        self.data.truncate(new_len);
        if let Some((start, end)) = &mut self.dirty_range {
            if *start >= new_len {
                self.dirty_range = None;
            } else {
                *end = (*end).min(new_len);
            }
        }
    }

    /// Reset the CPU-side data to empty without resizing the GPU buffer.
    ///
    /// The GPU buffer retains stale bytes beyond offset 0 until new data is
    /// pushed and flushed, but since no draw call references those ranges after
    /// the reset, this is safe. The next push/extend marks `dirty_range = [0, n)`
    /// and flush() re-uploads the new data starting at byte 0.
    ///
    /// Use this when an entire logical pool (mesh vertex buffer, material buffer,
    /// etc.) is known to be empty so its address space can be reused from the
    /// beginning rather than growing indefinitely.
    pub fn reset(&mut self) {
        self.data.clear();
        self.dirty_range = None;
    }

    /// Returns the number of live elements in the CPU-side mirror.
    pub fn live_len(&self) -> usize {
        self.data.len()
    }

    /// Flushes dirty data to GPU. O(1) if clean.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        let Some((start, end)) = self.dirty_range else {
            return;
        };
        if self.data.is_empty() {
            self.dirty_range = None;
            return;
        }

        // Grow buffer if needed
        if self.data.len() > self.capacity {
            self.capacity = self.data.len() * 2;
            let new_size = (self.capacity * std::mem::size_of::<T>()).max(64) as u64;
            self.buffer_version += 1;
            self.buf = Box::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: new_size,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            upload::write_buffer(
                queue,
                self.buf.as_ref(),
                0,
                bytemuck::cast_slice(&self.data),
            );
            self.dirty_range = None;
            return;
        }
        let end = end.min(self.data.len());
        if start >= end {
            self.dirty_range = None;
            return;
        }
        let byte_offset = (start * std::mem::size_of::<T>()) as u64;
        upload::write_buffer(
            queue,
            self.buf.as_ref(),
            byte_offset,
            bytemuck::cast_slice(&self.data[start..end]),
        );
        self.dirty_range = None;
    }

    /// Marks clean without flushing (use when buffer was written by GPU).
    pub fn mark_clean(&mut self) {
        self.dirty_range = None;
    }
}

// ─── Camera buffer ────────────────────────────────────────────────────────────

/// Storage buffer for up to two cameras (stereo / XR).
///
/// The buffer is sized for two `GpuCameraUniforms` elements. In mono mode
/// only the first element is written; the shader always indexes `cameras[0]`.
pub struct GpuCameraBuffer {
    buf: wgpu::Buffer,
    data: GpuCameraUniforms,
    dirty: bool,
}

impl GpuCameraBuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Storage"),
            size: (std::mem::size_of::<GpuCameraUniforms>() * 2) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf,
            data: GpuCameraUniforms::zeroed(),
            dirty: true,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    /// Returns the camera world-space position as `[x, y, z]`.
    pub fn position(&self) -> [f32; 3] {
        let p = self.data.position_near;
        [p[0], p[1], p[2]]
    }

    /// Returns the camera forward direction as `[x, y, z]`.
    pub fn forward(&self) -> [f32; 3] {
        let f = self.data.forward_far;
        [f[0], f[1], f[2]]
    }

    /// Returns a reference to the raw GPU camera uniform data.
    pub fn data(&self) -> &GpuCameraUniforms {
        &self.data
    }

    pub fn update(&mut self, camera: GpuCameraUniforms) {
        self.data = camera;
        self.dirty = true;
    }

    /// Write both eye cameras straight to GPU (XR multiview path).
    ///
    /// Unlike [`GpuCameraBuffer::update`] this uploads *both* uniforms in one
    /// `write_buffer` (the shader array is `array<Camera, 2>`); `dirty` is left
    /// untouched so a later `flush()` cannot clobber the right eye with a
    /// single-element upload. The left eye is cached as `data` for the
    /// CPU-side consumers (`position()`, `forward()`, ...).
    pub fn update_stereo(
        &mut self,
        queue: &wgpu::Queue,
        left: &GpuCameraUniforms,
        right: &GpuCameraUniforms,
    ) {
        self.data = *left;
        GpuCameraUniforms::upload_stereo(queue, &self.buf, left, right);
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if !self.dirty {
            return;
        }
        upload::write_buffer(queue, &self.buf, 0, bytemuck::bytes_of(&self.data));
        self.dirty = false;
    }
}

// ─── Coordinate space buffer ────────────────────────────────────────────────

/// Column-major 4x4 identity matrix, matching SceneDB's authored model layout.
const COORDINATE_SPACE_IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Renderer-owned previous-frame projection for SceneDB coordinate spaces.
///
/// Current authored transforms live only in SceneDB's
/// `SceneCoordinateSpace` partner buffer. Motion-vector passes additionally
/// need the value rendered one frame ago, while CPU picking/bake composition
/// needs a cheap row-indexed view of current values. This type therefore owns
/// one previous-frame GPU buffer plus two fixed 2 KiB CPU tables; it never
/// allocates a second current-frame GPU buffer.
///
/// [`stage_new`](Self::stage_new) initializes both temporal sides when a
/// SceneDB row starts a new component lifetime, preventing a recycled row's
/// old transform from producing first-frame velocity. Ordinary edits use
/// [`stage_current`](Self::stage_current). At the rendered frame boundary,
/// [`cycle_current`](Self::cycle_current) advances the rendered value into
/// the next frame's history; upload-only flushes do not change time.
pub struct CoordinateSpaceHistory {
    prev_buf: wgpu::Buffer,
    current: [[f32; 16]; libhelio::MAX_COORDINATE_SPACES as usize],
    prev: [[f32; 16]; libhelio::MAX_COORDINATE_SPACES as usize],
    prev_dirty: bool,
}

impl CoordinateSpaceHistory {
    pub fn new(device: &wgpu::Device) -> Self {
        let size =
            (libhelio::MAX_COORDINATE_SPACES as usize * std::mem::size_of::<[f32; 16]>()) as u64;
        Self {
            prev_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Coordinate Space History Buffer (Prev)"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            current: [COORDINATE_SPACE_IDENTITY; libhelio::MAX_COORDINATE_SPACES as usize],
            prev: [COORDINATE_SPACE_IDENTITY; libhelio::MAX_COORDINATE_SPACES as usize],
            prev_dirty: true,
        }
    }

    /// Previous-frame transforms, bound where shaders compute per-space velocity.
    pub fn prev_buffer(&self) -> &wgpu::Buffer {
        &self.prev_buf
    }

    /// Stage a current authored value after SceneDB has inserted or edited it.
    pub fn stage_current(&mut self, slot: u32, matrix: [f32; 16]) {
        let idx = slot as usize;
        debug_assert!(
            idx < libhelio::MAX_COORDINATE_SPACES as usize,
            "coordinate space slot {idx} out of range"
        );
        let Some(dst) = self.current.get_mut(idx) else {
            return;
        };
        *dst = matrix;
    }

    /// Begin a new component lifetime at `slot` without inheriting temporal
    /// state from the row's prior owner.
    pub fn stage_new(&mut self, slot: u32, matrix: [f32; 16]) {
        let idx = slot as usize;
        debug_assert!(
            idx < libhelio::MAX_COORDINATE_SPACES as usize,
            "coordinate space slot {idx} out of range"
        );
        let (Some(current), Some(prev)) = (self.current.get_mut(idx), self.prev.get_mut(idx)) else {
            return;
        };
        *current = matrix;
        if *prev != matrix {
            *prev = matrix;
            self.prev_dirty = true;
        }
    }

    /// Reads a slot's current transform (identity for any slot never set).
    pub fn slot(&self, slot: u32) -> [f32; 16] {
        self.current
            .get(slot as usize)
            .copied()
            .unwrap_or(COORDINATE_SPACE_IDENTITY)
    }

    /// Reads the transform committed at the last rendered frame boundary.
    pub fn previous_slot(&self, slot: u32) -> [f32; 16] {
        self.prev
            .get(slot as usize)
            .copied()
            .unwrap_or(COORDINATE_SPACE_IDENTITY)
    }

    /// Upload the previous-frame projection when it changed. O(1) when clean.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if self.prev_dirty {
            upload::write_buffer(queue, &self.prev_buf, 0, bytemuck::cast_slice(&self.prev));
            self.prev_dirty = false;
        }
    }

    /// Copies current → previous in the CPU mirror. Call once per rendered
    /// frame, so next frame's `flush()` uploads today's transforms as
    /// "previous" before any pass reads them — the same ordering as
    /// [`GpuObjectHistoryBuffer::cycle_current`].
    pub fn cycle_current(&mut self) {
        if self.prev != self.current {
            self.prev = self.current;
            self.prev_dirty = true;
        }
    }
}

// ─── Typed manager aliases ────────────────────────────────────────────────────

/// Helio-owned temporal history keyed by the component-local SceneObject GPU
/// row resolved through SceneAuthority.
///
/// Current authored models live only in SceneDB's spatial partner column.
/// Motion-vector passes still need the model that was rendered one frame ago,
/// which is renderer history rather than persistent scene data. This manager
/// keeps exactly the 96-byte `{ model, sphere, flags }` history row and uploads only
/// changed, coalesced row runs.
/// It deliberately does not use [`GrowableBuffer`]'s single dirty span: two
/// far-apart moving entities must not upload every dormant row between them.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuObjectHistory {
    pub model: [f32; 16],
    pub sphere: [f32; 4],
    pub flags: u32,
    pub _pad: [u32; 3],
}

const _: () = {
    assert!(std::mem::size_of::<GpuObjectHistory>() == 96);
    assert!(std::mem::offset_of!(GpuObjectHistory, sphere) == 64);
    assert!(std::mem::offset_of!(GpuObjectHistory, flags) == 80);
};

pub struct GpuObjectHistoryBuffer {
    buf: Box<wgpu::Buffer>,
    data: Vec<GpuObjectHistory>,
    dirty_rows: Vec<u32>,
    /// Latest model authored during the current frame. It becomes history
    /// only at the rendered frame boundary, never from an upload-only flush.
    pending_current: HashMap<u32, GpuObjectHistory>,
    capacity: usize,
    device: Arc<wgpu::Device>,
    buffer_version: u64,
}

impl GpuObjectHistoryBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let capacity = 4096;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Object Temporal History"),
            size: (capacity * std::mem::size_of::<GpuObjectHistory>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf: Box::new(buf),
            data: Vec::new(),
            dirty_rows: Vec::new(),
            pending_current: HashMap::new(),
            capacity,
            device,
            buffer_version: 0,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buf.as_ref()
    }

    pub fn buffer_version(&self) -> u64 {
        self.buffer_version
    }

    fn ensure_row(&mut self, row: u32) {
        let required = row as usize + 1;
        if self.data.len() < required {
            self.data.resize(
                required,
                GpuObjectHistory {
                    model: COORDINATE_SPACE_IDENTITY,
                    sphere: [0.0; 4],
                    flags: 0,
                    _pad: [0; 3],
                },
            );
        }
    }

    /// Initialize history for a newly inserted/recycled SceneObject row. Its
    /// first rendered frame has zero object-local velocity
    /// (`previous == current`).
    pub fn insert(&mut self, row: u32, model: [f32; 16], sphere: [f32; 4], flags: u32) {
        self.ensure_row(row);
        self.data[row as usize] = GpuObjectHistory {
            model,
            sphere,
            flags,
            _pad: [0; 3],
        };
        self.dirty_rows.push(row);
        self.pending_current.remove(&row);
    }

    /// Record the latest authored model without replacing this frame's prior
    /// history. Multiple edits before one flush collapse to the final model.
    pub fn stage_current(
        &mut self,
        row: u32,
        model: [f32; 16],
        sphere: [f32; 4],
        flags: u32,
    ) {
        self.ensure_row(row);
        self.pending_current.insert(
            row,
            GpuObjectHistory {
                model,
                sphere,
                flags,
                _pad: [0; 3],
            },
        );
    }

    /// Stop a removed row from being promoted after the frame boundary. Stale
    /// bytes are unreachable because active draw projections are rebuilt from
    /// current component membership; reuse calls [`Self::insert`] first.
    pub fn remove(&mut self, row: u32) {
        self.pending_current.remove(&row);
    }

    /// Upload history that belongs to the frame about to render.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if self.data.len() > self.capacity {
            self.capacity = self.data.len().next_power_of_two();
            self.buffer_version = self.buffer_version.wrapping_add(1);
            self.buf = Box::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Object Temporal History"),
                size: (self.capacity * std::mem::size_of::<GpuObjectHistory>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if !self.data.is_empty() {
                upload::write_buffer(
                    queue,
                    self.buf.as_ref(),
                    0,
                    bytemuck::cast_slice(&self.data),
                );
            }
            self.dirty_rows.clear();
            return;
        }

        if self.dirty_rows.is_empty() {
            return;
        }
        self.dirty_rows.sort_unstable();
        self.dirty_rows.dedup();

        let row_size = std::mem::size_of::<GpuObjectHistory>();
        let mut start = 0;
        while start < self.dirty_rows.len() {
            let first = self.dirty_rows[start] as usize;
            let mut end = start + 1;
            while end < self.dirty_rows.len()
                && self.dirty_rows[end] == self.dirty_rows[end - 1] + 1
            {
                end += 1;
            }
            let last_exclusive = self.dirty_rows[end - 1] as usize + 1;
            upload::write_buffer(
                queue,
                self.buf.as_ref(),
                (first * row_size) as u64,
                bytemuck::cast_slice(&self.data[first..last_exclusive]),
            );
            start = end;
        }
        self.dirty_rows.clear();
    }

    /// Promote current models to next frame's history after the current frame
    /// has rendered. This performs no GPU write until the next `flush()`.
    pub fn cycle_current(&mut self) {
        for (row, current) in self.pending_current.drain() {
            let slot = &mut self.data[row as usize];
            if *slot != current {
                *slot = current;
                self.dirty_rows.push(row);
            }
        }
    }
}

/// Storage buffer for draw call templates (source for indirect dispatch).
pub struct GpuDrawCallBuffer(pub GrowableBuffer<GpuDrawCall>);
/// Storage buffer for shadow matrices.
pub struct GpuShadowMatrixBuffer(pub GrowableBuffer<GpuShadowMatrix>);
/// Indirect draw command buffer (written by GPU compute, read by render passes).
pub struct GpuIndirectBuffer(pub GrowableBuffer<DrawIndexedIndirectArgs>);
/// Storage buffer for per-instance visibility bitmask (u32 per instance, 1=visible).
pub struct GpuVisibilityBuffer(pub GrowableBuffer<u32>);
/// Helio-owned projection from compact draw-group slots to canonical SceneDB
/// rows.  This is render-derived topology, not a second copy of scene data.
/// Culling reads canonical instance/AABB columns through these row indices and
/// writes canonical rows into the compacted survivor buffers.
pub struct GpuSourceIndicesBuffer(pub GrowableBuffer<u32>);
/// GPU-written scratch buffer: for each draw-call group, the original instance
/// slots that survived per-instance frustum culling, packed contiguously
/// starting at that group's `first_instance` offset. Written by
/// `IndirectDispatchPass`, read by `GBufferPass` (and any other pass drawing
/// through the same grouped indirect buffer) in place of `instances[instance_index]`.
pub struct GpuCompactedIndicesBuffer(pub GrowableBuffer<u32>);
/// Second-stage compacted indices: surviving instance slots after BOTH
/// frustum culling (IndirectDispatchPass) and Hi-Z occlusion culling
/// (OcclusionCullPass). This is the buffer draw-consuming passes should
/// actually read — it reflects the final, fully-culled instance set.
pub struct GpuCompactedIndices2Buffer(pub GrowableBuffer<u32>);

impl GpuDrawCallBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "DrawCall Buffer",
        ))
    }
}

impl std::ops::Deref for GpuDrawCallBuffer {
    type Target = GrowableBuffer<GpuDrawCall>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuDrawCallBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuShadowMatrixBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            256,
            wgpu::BufferUsages::STORAGE,
            "Shadow Matrix Buffer",
        ))
    }
}

impl std::ops::Deref for GpuShadowMatrixBuffer {
    type Target = GrowableBuffer<GpuShadowMatrix>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuShadowMatrixBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuIndirectBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            "Indirect Draw Buffer",
        ))
    }
}

impl std::ops::Deref for GpuIndirectBuffer {
    type Target = GrowableBuffer<DrawIndexedIndirectArgs>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuIndirectBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuVisibilityBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "Visibility Buffer",
        ))
    }
}

impl std::ops::Deref for GpuVisibilityBuffer {
    type Target = GrowableBuffer<u32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuVisibilityBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuSourceIndicesBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "SceneDB Row Source Indices Buffer",
        ))
    }
}

impl std::ops::Deref for GpuSourceIndicesBuffer {
    type Target = GrowableBuffer<u32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuSourceIndicesBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuCompactedIndicesBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            // COPY_SRC: OcclusionCullPass copies this straight through to
            // compacted_indices_2 on frame 0, when no Hi-Z pyramid exists yet.
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "Compacted Instance Indices Buffer",
        ))
    }
}

impl std::ops::Deref for GpuCompactedIndicesBuffer {
    type Target = GrowableBuffer<u32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuCompactedIndicesBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuCompactedIndices2Buffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "Compacted Instance Indices Buffer 2",
        ))
    }
}

impl std::ops::Deref for GpuCompactedIndices2Buffer {
    type Target = GrowableBuffer<u32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuCompactedIndices2Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
