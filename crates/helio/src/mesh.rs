use bytemuck::{Pod, Zeroable};
use helio_v3::GrowableBuffer;

use crate::arena::SparsePool;
use crate::handles::MeshId;

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
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
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
}

pub struct MeshBuffers<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
}

/// Shared vertex + index storage for one class of mesh geometry (static or dynamic).
struct MeshSubPool {
    vertices: GrowableBuffer<PackedVertex>,
    indices: GrowableBuffer<u32>,
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

pub struct MeshPool {
    /// Upload-once geometry (terrain, buildings, props).
    static_sub: MeshSubPool,
    /// Per-frame-updatable geometry (skinned, morphed, procedural).
    dynamic_sub: MeshSubPool,
    meshes: SparsePool<MeshRecord, MeshId>,
}

impl MeshPool {
    pub fn new(device: std::sync::Arc<wgpu::Device>) -> Self {
        Self {
            static_sub: MeshSubPool::new(device.clone(), MeshKind::Static),
            dynamic_sub: MeshSubPool::new(device, MeshKind::Dynamic),
            meshes: SparsePool::new(),
        }
    }

    /// Insert static (upload-once) mesh geometry. The geometry cannot be changed later.
    ///
    /// Use for terrain, buildings, props — any geometry that never deforms.
    /// Objects that *move* but keep their shape (rigid bodies) should still use
    /// `insert()` here; only the per-object transform changes via
    /// [`update_object_transform`](crate::Scene::update_object_transform).
    pub fn insert(&mut self, mesh: MeshUpload) -> MeshId {
        self.insert_with_kind(mesh, MeshKind::Static)
    }

    /// Insert dynamic mesh geometry that can be updated every frame.
    ///
    /// Use for skinned characters, morphed geometry, or any mesh whose vertex
    /// positions/normals change at runtime. After inserting, call
    /// [`update_dynamic_vertices`](Self::update_dynamic_vertices) each frame.
    pub fn insert_dynamic(&mut self, mesh: MeshUpload) -> MeshId {
        self.insert_with_kind(mesh, MeshKind::Dynamic)
    }

    fn insert_with_kind(&mut self, mesh: MeshUpload, kind: MeshKind) -> MeshId {
        let sub = match kind {
            MeshKind::Static => &mut self.static_sub,
            MeshKind::Dynamic => &mut self.dynamic_sub,
        };
        let vertex_range = sub.vertices.extend_from_slice(&mesh.vertices);
        let index_range = sub.indices.extend_from_slice(&mesh.indices);
        let slice = MeshSlice {
            first_vertex: vertex_range.start as u32,
            vertex_count: (vertex_range.end - vertex_range.start) as u32,
            first_index: index_range.start as u32,
            index_count: (index_range.end - index_range.start) as u32,
        };
        let (id, _, _) = self.meshes.insert(MeshRecord {
            slice,
            ref_count: 0,
            kind,
        });
        id
    }

    /// Upload a sectioned mesh: vertices are pushed ONCE into the shared vertex buffer;
    /// each section's index list gets its own contiguous range in the index buffer.
    /// Returns one `MeshId` per section — all share the same `first_vertex`.
    ///
    /// This is the GPU-native implementation of Unreal's Static Mesh sections.
    pub fn insert_sectioned(&mut self, upload: SectionedMeshUpload) -> MultiMeshRecord {
        let sub = &mut self.static_sub;
        let vertex_range = sub.vertices.extend_from_slice(&upload.vertices);
        let first_vertex = vertex_range.start as u32;
        let vertex_count = (vertex_range.end - vertex_range.start) as u32;

        let section_mesh_ids = upload
            .sections
            .iter()
            .map(|sec_indices| {
                let index_range = sub.indices.extend_from_slice(sec_indices);
                let (id, _, _) = self.meshes.insert(MeshRecord {
                    slice: MeshSlice {
                        first_vertex,
                        vertex_count,
                        first_index: index_range.start as u32,
                        index_count: (index_range.end - index_range.start) as u32,
                    },
                    ref_count: 0,
                    kind: MeshKind::Static,
                });
                id
            })
            .collect();

        MultiMeshRecord {
            section_mesh_ids,
            ref_count: 0,
        }
    }

    /// Replace the vertex data of a **dynamic** mesh in-place.
    ///
    /// The new slice must have the same length as the original upload.
    /// Returns an error string if `id` is invalid, is a static mesh, or the
    /// vertex count doesn't match.
    ///
    /// On the next [`flush`](Self::flush), only the dirty byte range is re-uploaded.
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

    pub fn remove(&mut self, id: MeshId) -> Option<MeshRecord> {
        self.meshes.remove(id).map(|(_, record)| record)
    }

    pub fn buffers(&self) -> MeshBuffers<'_> {
        self.static_sub.buffers()
    }

    pub fn dynamic_buffers(&self) -> MeshBuffers<'_> {
        self.dynamic_sub.buffers()
    }

    /// Total vertices in the static vertex mega-buffer.
    pub fn total_vertex_count(&self) -> usize {
        self.static_sub.vertices.len()
    }

    /// Total indices in the static index mega-buffer.
    pub fn total_index_count(&self) -> usize {
        self.static_sub.indices.len()
    }

    /// Number of unique mesh records currently live (sections each count as one).
    pub fn unique_mesh_count(&self) -> usize {
        self.meshes.live_len()
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        self.static_sub.flush(queue);
        self.dynamic_sub.flush(queue);
    }

    /// Extracts a mesh's vertex and index data from the pool.
    ///
    /// Returns None if the mesh ID is invalid. Used internally for baking.
    pub(crate) fn extract_mesh_data(&self, id: MeshId) -> Option<MeshUpload> {
        let record = self.meshes.get(id)?;
        let slice = &record.slice;
        let sub = match record.kind {
            MeshKind::Static => &self.static_sub,
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

