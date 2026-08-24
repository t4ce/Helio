use bytemuck::{Pod, Zeroable};
use helio_core::GrowableBuffer;
use helio_scenedb::{EngineGpuContext, GeometryArena, Subsystem};
use std::sync::Arc;

use crate::arena::SparsePool;
use crate::handles::MeshId;

/// Stable SceneDB asset-buffer identities for Helio's exceptional general
/// geometry residency path. These are intentionally not component-column
/// names: variable-length vertex/index streams use `GeometryArena`'s
/// explicit one-time handoff API.
pub const GENERAL_MESH_VERTEX_BUFFER_KEY: &str = "general_mesh_buf";
pub const GENERAL_MESH_INDEX_BUFFER_KEY: &str = "general_mesh_index_buf";

/// Determines the lifetime and update policy of mesh geometry on the GPU.
///
/// | Kind    | Can update geometry? | CPU mirror retained? | Use case |
/// |---------|---------------------|----------------------|----------|
/// | Static  | No (upload-once)    | Yes (baking)         | Buildings, terrain, props |
/// | Dynamic | Yes (per-frame OK)  | Yes (dirty tracking) | Skinned characters, morphs, procedural |
///
/// Objects that **move** but keep their shape (rigid bodies) use `MeshKind::Static`
/// geometry combined with `Movability::Movable` on the object. Transform updates
/// go through `update_object_transform()` which is O(1) and never touches mesh data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    /// Geometry is uploaded once and never changed.
    Static,
    /// Geometry can be replaced per-frame via [`MeshPool::update_dynamic_vertices`].
    Dynamic,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords0: [f32; 2],
    pub tex_coords1: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

impl PackedVertex {
    pub fn from_components(
        position: [f32; 3],
        normal: [f32; 3],
        tex_coords: [f32; 2],
        tangent: [f32; 3],
        bitangent_sign: f32,
    ) -> Self {
        Self {
            position,
            bitangent_sign,
            tex_coords0: tex_coords,
            tex_coords1: [0.0, 0.0],
            normal: pack_snorm4x8([normal[0], normal[1], normal[2], 0.0]),
            tangent: pack_snorm4x8([tangent[0], tangent[1], tangent[2], 0.0]),
        }
    }
}

fn pack_snorm4x8(v: [f32; 4]) -> u32 {
    let to_i8 = |x: f32| -> u32 {
        let clamped = x.clamp(-1.0, 1.0);
        let scaled = (clamped * 127.0).round() as i8;
        scaled as u8 as u32
    };

    to_i8(v[0]) | (to_i8(v[1]) << 8) | (to_i8(v[2]) << 16) | (to_i8(v[3]) << 24)
}

#[derive(Debug, Clone)]
pub struct MeshUpload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

/// Upload descriptor for a multi-material (sectioned) mesh.
///
/// All sections share one vertex buffer. Each element of `sections` is an independent
/// index list referencing `vertices`, rendered with its own material per draw call.
/// This mirrors Unreal Engine's Static Mesh section model: one VB/IB, N draw calls.
#[derive(Debug, Clone)]
pub struct SectionedMeshUpload {
    /// The full shared vertex array. All sections index into this.
    pub vertices: Vec<PackedVertex>,
    /// Per-section index lists. `sections[i]` is drawn with the i-th material.
    pub sections: Vec<Vec<u32>>,
}

/// Internal record for a stored multi-material mesh.
/// Sections share the same vertex buffer region but have distinct index ranges.
pub(crate) struct MultiMeshRecord {
    /// One `MeshId` per section (all share the same vertex range in the pool).
    pub section_mesh_ids: Vec<crate::handles::MeshId>,
    /// Number of live [`SectionedObjectId`] instances placed from this mesh.
    pub ref_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshSlice {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct MeshRecord {
    pub slice: MeshSlice,
    pub ref_count: u32,
    pub kind: MeshKind,
    /// Whether this record owns the shared vertex range. Sectioned meshes
    /// have one record per index section but only their first record owns
    /// the single shared vertex allocation.
    owns_vertices: bool,
}

pub struct MeshBuffers<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
}

// ── Free-list range allocator ─────────────────────────────────────────────────

/// First-fit range allocator with coalescing and tail-trimming.
///
/// Tracks free `(start, len)` ranges inside a logically contiguous buffer.
/// On each `free()` call adjacent ranges are merged (O(free_ranges) but
/// typically very small).  When the newly-freed range butts up against the
/// end of the buffer, the tail is trimmed so callers can `truncate()` the
/// physical buffer back to the new high-water mark, actually returning memory.
#[derive(Default)]
struct FreeListAllocator {
    /// Sorted by start offset, coalesced, no overlaps.
    free: Vec<(usize, usize)>,
}

impl FreeListAllocator {
    /// Try to satisfy an allocation of `count` elements using the free list.
    ///
    /// Uses first-fit: picks the first range that is large enough.  Splits
    /// oversized ranges, leaving the remainder on the list.
    ///
    /// Returns `Some(start)` on success or `None` when the caller must append.
    fn alloc(&mut self, count: usize) -> Option<usize> {
        if count == 0 {
            return Some(0);
        }
        let idx = self.free.iter().position(|&(_, len)| len >= count)?;
        let (start, len) = self.free[idx];
        if len == count {
            self.free.remove(idx);
        } else {
            self.free[idx] = (start + count, len - count);
        }
        Some(start)
    }

    /// Mark `[start, start + count)` as free.
    ///
    /// Adjacent ranges are coalesced.  If the resulting free range extends to
    /// `buf_len` (the current logical end of the buffer), the tail is trimmed:
    /// the free entry is removed and the new logical buffer end is returned so
    /// the caller can `truncate()` the physical buffer.
    ///
    /// Returns `Some(new_buf_len)` when a tail-trim occurred, `None` otherwise.
    fn free(&mut self, start: usize, count: usize, buf_len: usize) -> Option<usize> {
        if count == 0 {
            return None;
        }

        // Insert in sorted order.
        let pos = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(pos, (start, count));

        // Coalesce with successor.
        if pos + 1 < self.free.len() {
            let (s, l) = self.free[pos];
            let (ns, nl) = self.free[pos + 1];
            if s + l == ns {
                self.free[pos] = (s, l + nl);
                self.free.remove(pos + 1);
            }
        }
        // Coalesce with predecessor.
        if pos > 0 {
            let prev = pos - 1;
            let (ps, pl) = self.free[prev];
            let (s, l) = self.free[pos.min(self.free.len() - 1)];
            if ps + pl == s {
                self.free[prev] = (ps, pl + l);
                // The coalesced entry is now at `prev`.
                if prev + 1 < self.free.len() {
                    self.free.remove(prev + 1);
                }
            }
        }

        // Tail-trim: if the last free range reaches the buffer end, remove it
        // and report a new (smaller) logical end so the caller can truncate.
        if let Some(&(tail_start, tail_len)) = self.free.last() {
            if tail_start + tail_len == buf_len {
                self.free.pop();
                return Some(tail_start);
            }
        }

        None
    }

}

// ── Sub-pool (vertex + index + their allocators) ──────────────────────────────

struct MeshSubPool {
    vertices: GrowableBuffer<PackedVertex>,
    indices: GrowableBuffer<u32>,
    vertex_alloc: FreeListAllocator,
    index_alloc: FreeListAllocator,
}

impl MeshSubPool {
    fn new(device: std::sync::Arc<wgpu::Device>, kind: MeshKind) -> Self {
        let (v_label, i_label, v_cap, i_cap) = match kind {
            MeshKind::Static => (
                "Helio Static Vertex Buffer",
                "Helio Static Index Buffer",
                4096,
                8192,
            ),
            MeshKind::Dynamic => (
                "Helio Dynamic Vertex Buffer",
                "Helio Dynamic Index Buffer",
                512,
                1024,
            ),
        };
        Self {
            vertices: GrowableBuffer::new(
                device.clone(),
                v_cap,
                wgpu::BufferUsages::VERTEX,
                v_label,
            ),
            indices: GrowableBuffer::new(
                device,
                i_cap,
                wgpu::BufferUsages::INDEX,
                i_label,
            ),
            vertex_alloc: FreeListAllocator::default(),
            index_alloc: FreeListAllocator::default(),
        }
    }

    /// Allocate space for `vcount` vertices and `icount` indices.
    ///
    /// Tries free ranges first; falls back to appending.  Returns the
    /// `(first_vertex, first_index)` slot start.
    fn alloc_and_write(
        &mut self,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> (usize, usize) {
        let vcount = vertices.len();
        let icount = indices.len();

        let vstart = if let Some(s) = self.vertex_alloc.alloc(vcount) {
            self.vertices.update_range(s, vertices);
            s
        } else {
            self.vertices.extend_from_slice(vertices).start
        };

        let istart = if let Some(s) = self.index_alloc.alloc(icount) {
            self.indices.update_range(s, indices);
            s
        } else {
            self.indices.extend_from_slice(indices).start
        };

        (vstart, istart)
    }

    /// Return the vertex and index ranges of `slice` to the free list.
    ///
    /// Performs tail-trimming: if the freed range reaches the current logical
    /// end of the buffer, the buffer is truncated immediately, actually
    /// releasing CPU and (on next flush) GPU memory.
    fn free_slice(&mut self, slice: &MeshSlice, free_vertices: bool) {
        let vstart = slice.first_vertex as usize;
        let vcount = slice.vertex_count as usize;
        let istart = slice.first_index as usize;
        let icount = slice.index_count as usize;

        if free_vertices {
            if let Some(new_vlen) = self.vertex_alloc.free(vstart, vcount, self.vertices.live_len()) {
                self.vertices.truncate(new_vlen);
            }
        }
        if let Some(new_ilen) = self.index_alloc.free(istart, icount, self.indices.live_len()) {
            self.indices.truncate(new_ilen);
        }
    }

    fn buffers(&self) -> MeshBuffers<'_> {
        MeshBuffers {
            vertices: self.vertices.buffer(),
            indices: self.indices.buffer(),
        }
    }

    fn flush(&mut self, queue: &wgpu::Queue) {
        self.vertices.flush(queue);
        self.indices.flush(queue);
    }
}

// ── Static write-once geometry ───────────────────────────────────────────────

/// Static geometry residency backed by SceneDB's explicit one-time asset
/// handoff. The compact CPU arrays are retained solely because Helio's bake,
/// picking, and BLAS build paths genuinely consume source geometry; they are
/// never used as a dirty-tracking shadow for the static GPU buffers.
struct StaticMeshSubPool {
    arena: GeometryArena,
    vertices: Vec<PackedVertex>,
    indices: Vec<u32>,
    queue: Arc<wgpu::Queue>,
}

impl StaticMeshSubPool {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let ctx = EngineGpuContext::new(device, Arc::clone(&queue));
        Self {
            arena: GeometryArena::new_growable_named(
                &ctx,
                (4096 * std::mem::size_of::<PackedVertex>()) as u64,
                (8192 * std::mem::size_of::<u32>()) as u64,
                GENERAL_MESH_VERTEX_BUFFER_KEY,
                GENERAL_MESH_INDEX_BUFFER_KEY,
            ),
            vertices: Vec::with_capacity(4096),
            indices: Vec::with_capacity(8192),
            queue,
        }
    }

    fn upload_vertices(&mut self, vertices: &[PackedVertex]) -> usize {
        let bytes = bytemuck::cast_slice(vertices);
        let byte_offset = self
            .arena
            .upload_vertices(&self.queue, bytes)
            .expect("general mesh vertex residency exceeded the device buffer limit");
        helio_core::upload::record_upload_bytes(bytes.len() as u64);
        let start = byte_offset as usize / std::mem::size_of::<PackedVertex>();
        let end = start + vertices.len();
        if self.vertices.len() < end {
            self.vertices.resize(end, PackedVertex::zeroed());
        }
        self.vertices[start..end].copy_from_slice(vertices);
        start
    }

    fn upload_indices(&mut self, indices: &[u32]) -> usize {
        let bytes = bytemuck::cast_slice(indices);
        let byte_offset = self
            .arena
            .upload_indices(&self.queue, bytes)
            .expect("general mesh index residency exceeded the device buffer limit");
        helio_core::upload::record_upload_bytes(bytes.len() as u64);
        let start = byte_offset as usize / std::mem::size_of::<u32>();
        let end = start + indices.len();
        if self.indices.len() < end {
            self.indices.resize(end, 0);
        }
        self.indices[start..end].copy_from_slice(indices);
        start
    }

    fn alloc_and_write(
        &mut self,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> (usize, usize) {
        (self.upload_vertices(vertices), self.upload_indices(indices))
    }

    fn free_slice(&mut self, slice: &MeshSlice, free_vertices: bool) {
        if free_vertices {
            self.arena.free_vertices(
                slice.first_vertex * std::mem::size_of::<PackedVertex>() as u32,
                slice.vertex_count * std::mem::size_of::<PackedVertex>() as u32,
            );
            self.vertices.truncate(
                self.arena.vertex_high_water_bytes() as usize
                    / std::mem::size_of::<PackedVertex>(),
            );
        }
        self.arena.free_indices(
            slice.first_index * std::mem::size_of::<u32>() as u32,
            slice.index_count * std::mem::size_of::<u32>() as u32,
        );
        self.indices.truncate(
            self.arena.index_high_water_bytes() as usize / std::mem::size_of::<u32>(),
        );
    }

    fn buffers(&self) -> MeshBuffers<'_> {
        MeshBuffers {
            vertices: self.arena.vertex_buffer(),
            indices: self.arena.index_buffer(),
        }
    }
}

// ── Public MeshPool ───────────────────────────────────────────────────────────

pub struct MeshPool {
    static_sub: StaticMeshSubPool,
    dynamic_sub: MeshSubPool,
    meshes: SparsePool<MeshRecord, MeshId>,
}

impl MeshPool {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            static_sub: StaticMeshSubPool::new(device.clone(), queue),
            dynamic_sub: MeshSubPool::new(device, MeshKind::Dynamic),
            meshes: SparsePool::new(),
        }
    }

    pub fn insert(&mut self, mesh: MeshUpload) -> MeshId {
        self.insert_with_kind(mesh, MeshKind::Static)
    }

    pub fn insert_dynamic(&mut self, mesh: MeshUpload) -> MeshId {
        self.insert_with_kind(mesh, MeshKind::Dynamic)
    }

    fn insert_with_kind(&mut self, mesh: MeshUpload, kind: MeshKind) -> MeshId {
        let (first_vertex, first_index) = match kind {
            MeshKind::Static => self
                .static_sub
                .alloc_and_write(&mesh.vertices, &mesh.indices),
            MeshKind::Dynamic => self
                .dynamic_sub
                .alloc_and_write(&mesh.vertices, &mesh.indices),
        };

        let slice = MeshSlice {
            first_vertex: first_vertex as u32,
            vertex_count: mesh.vertices.len() as u32,
            first_index: first_index as u32,
            index_count: mesh.indices.len() as u32,
        };

        let (id, _, _) = self.meshes.insert(MeshRecord {
            slice,
            ref_count: 0,
            kind,
            owns_vertices: true,
        });
        id
    }

    pub fn insert_sectioned(&mut self, upload: SectionedMeshUpload) -> MultiMeshRecord {
        let sub = &mut self.static_sub;

        // Vertices are shared across all sections — allocate once.
        let first_vertex = if upload.sections.is_empty() {
            // An empty section table owns no drawable geometry. Avoid
            // allocating vertices which no MeshRecord could later release.
            0
        } else {
            sub.upload_vertices(&upload.vertices)
        };
        let vertex_count = upload.vertices.len() as u32;

        let section_mesh_ids = upload
            .sections
            .iter()
            .enumerate()
            .map(|(section_index, sec_indices)| {
                let first_index = sub.upload_indices(sec_indices);

                let (id, _, _) = self.meshes.insert(MeshRecord {
                    slice: MeshSlice {
                        first_vertex: first_vertex as u32,
                        vertex_count,
                        first_index: first_index as u32,
                        index_count: sec_indices.len() as u32,
                    },
                    // The sectioned-mesh asset itself owns one reference so
                    // removing its last placed instance cannot silently
                    // destroy geometry behind the still-live MultiMeshId.
                    ref_count: 1,
                    kind: MeshKind::Static,
                    owns_vertices: section_index == 0,
                });
                id
            })
            .collect();

        MultiMeshRecord { section_mesh_ids, ref_count: 0 }
    }

    pub fn update_dynamic_vertices(
        &mut self,
        id: MeshId,
        new_vertices: &[PackedVertex],
    ) -> Result<(), &'static str> {
        let Some(record) = self.meshes.get(id) else {
            return Err("invalid mesh id");
        };
        if record.kind != MeshKind::Dynamic {
            return Err("cannot update static mesh vertices");
        }
        if new_vertices.len() != record.slice.vertex_count as usize {
            return Err("vertex count mismatch: new_vertices.len() must equal the original upload");
        }
        let start = record.slice.first_vertex as usize;
        self.dynamic_sub.vertices.update_range(start, new_vertices);
        Ok(())
    }

    pub fn get(&self, id: MeshId) -> Option<&MeshRecord> {
        self.meshes.get(id)
    }

    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut MeshRecord> {
        self.meshes.get_mut_with_slot(id).map(|(_, record)| record)
    }

    /// Iterate live mesh records with generation-valid IDs.
    pub fn iter(&self) -> impl Iterator<Item = (MeshId, &MeshRecord)> + '_ {
        self.meshes.iter()
    }

    /// Remove a mesh and immediately free its vertex/index ranges back into the
    /// allocator.  If the freed ranges are at the tail of their buffer, the
    /// buffer is truncated on the spot — no separate "compact" call needed.
    pub fn remove(&mut self, id: MeshId) -> Option<MeshRecord> {
        let (_, record) = self.meshes.remove(id)?;
        match record.kind {
            MeshKind::Static => self
                .static_sub
                .free_slice(&record.slice, record.owns_vertices),
            MeshKind::Dynamic => self
                .dynamic_sub
                .free_slice(&record.slice, record.owns_vertices),
        }
        Some(record)
    }

    pub fn buffers(&self) -> MeshBuffers<'_> {
        self.static_sub.buffers()
    }

    pub fn dynamic_buffers(&self) -> MeshBuffers<'_> {
        self.dynamic_sub.buffers()
    }

    pub fn total_vertex_count(&self) -> usize {
        self.static_sub.vertices.len()
    }

    pub fn total_index_count(&self) -> usize {
        self.static_sub.indices.len()
    }

    pub fn unique_mesh_count(&self) -> usize {
        self.meshes.live_len()
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        self.dynamic_sub.flush(queue);
    }

    #[cfg(any(feature = "bake", test))]
    pub(crate) fn extract_mesh_data(&self, id: MeshId) -> Option<MeshUpload> {
        let record = self.meshes.get(id)?;
        let slice = &record.slice;
        let sub = match record.kind {
            MeshKind::Static => {
                let vertex_start = slice.first_vertex as usize;
                let vertex_end = vertex_start + slice.vertex_count as usize;
                let index_start = slice.first_index as usize;
                let index_end = index_start + slice.index_count as usize;
                return Some(MeshUpload {
                    vertices: self.static_sub.vertices.get(vertex_start..vertex_end)?.to_vec(),
                    indices: self.static_sub.indices.get(index_start..index_end)?.to_vec(),
                });
            }
            MeshKind::Dynamic => &self.dynamic_sub,
        };

        let vertex_start = slice.first_vertex as usize;
        let vertex_end = vertex_start + slice.vertex_count as usize;
        let index_start = slice.first_index as usize;
        let index_end = index_start + slice.index_count as usize;

        let vertices = sub.vertices.as_slice().get(vertex_start..vertex_end)?.to_vec();
        let indices = sub.indices.as_slice().get(index_start..index_end)?.to_vec();

        Some(MeshUpload { vertices, indices })
    }
}

impl Subsystem for MeshPool {
    fn name(&self) -> &'static str {
        "helio.geometry.general"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        crate::test_support::test_gpu().expect("mesh residency test requires an adapter")
    }

    fn readback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        bytes: u64,
    ) -> Vec<u8> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("general-mesh-test-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("map geometry readback"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll geometry readback");
        let bytes = slice.get_mapped_range().expect("mapped geometry range").to_vec();
        staging.unmap();
        bytes
    }

    fn vertex(x: f32) -> PackedVertex {
        PackedVertex::from_components(
            [x, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0],
            1.0,
        )
    }

    #[test]
    fn static_geometry_uses_named_once_handoff_and_survives_growth() {
        let (device, queue) = test_device();
        let mut pool = MeshPool::new(Arc::clone(&device), Arc::clone(&queue));
        let original = MeshUpload {
            vertices: vec![vertex(1.0), vertex(2.0), vertex(3.0)],
            indices: vec![0, 1, 2],
        };
        let original_id = pool.insert(original.clone());

        assert_eq!(pool.static_sub.arena.vertex_buffer_key(), GENERAL_MESH_VERTEX_BUFFER_KEY);
        assert_eq!(pool.static_sub.arena.index_buffer_key(), GENERAL_MESH_INDEX_BUFFER_KEY);
        assert_eq!(pool.static_sub.arena.upload_count(), 2);
        pool.flush(&queue);
        assert_eq!(
            pool.static_sub.arena.upload_count(),
            2,
            "frame flush must not re-upload static geometry",
        );

        let large_id = pool.insert(MeshUpload {
            vertices: vec![vertex(7.0); 4097],
            indices: vec![0, 1, 2],
        });
        assert_eq!(pool.static_sub.arena.vertex_epoch(), 1);
        assert_eq!(pool.get(original_id).unwrap().slice.first_vertex, 0);
        assert_eq!(pool.extract_mesh_data(original_id).unwrap().vertices[0].position[0], 1.0);

        let expected: Vec<u8> = bytemuck::cast_slice(&original.vertices).to_vec();
        let gpu = readback(
            &device,
            &queue,
            pool.static_sub.arena.vertex_buffer(),
            expected.len() as u64,
        );
        assert_eq!(gpu, expected, "GPU-to-GPU growth preserves prior handoffs");

        pool.remove(large_id).unwrap();
        pool.remove(original_id).unwrap();
        assert_eq!(pool.static_sub.arena.vertex_high_water_bytes(), 0);
        assert_eq!(pool.static_sub.arena.index_high_water_bytes(), 0);
    }
}
