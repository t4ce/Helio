//! CPU-side virtual geometry: meshlet decomposition for GPU-driven rendering.
//!
//! Uses [meshopt](https://crates.io/crates/meshopt) (meshoptimizer FFI) for
//! simplification, meshlet building, and bounds computation — the same library
//! used by Unreal Engine 5's Nanite pipeline.

use std::mem;

use libhelio::{GpuMeshletEntry, MESHLET_MAX_TRIANGLES};
use meshopt::DecodePosition;

use crate::mesh::PackedVertex;

// ─── Handle types ───────────────────────────────────────────────────────────

/// Opaque handle to a virtual mesh uploaded to the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualMeshId(pub(crate) u32);

// ─── Upload / descriptor types ──────────────────────────────────────────────

/// High-resolution mesh for virtual geometry upload.
///
/// The scene splits this into meshlets automatically when you call
/// `Scene::insert_virtual_mesh()`.  Keep the CPU-side data alive only until
/// after `Scene::flush()` returns.
#[derive(Debug, Clone)]
pub struct VirtualMeshUpload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

/// Descriptor for a virtual object (one instance of a `VirtualMeshId`).
#[derive(Debug, Clone, Copy)]
pub struct VirtualObjectDescriptor {
    pub virtual_mesh: VirtualMeshId,
    pub material_id: u32,
    pub transform: glam::Mat4,
    /// World-space bounding sphere `[cx, cy, cz, radius]`.
    pub bounds: [f32; 4],
    pub flags: u32,
    /// Group membership bitmask.  Use `GroupMask::NONE` for ungrouped objects.
    pub groups: crate::groups::GroupMask,
    /// Movability mode. Defaults to Static when None.
    pub movability: Option<libhelio::Movability>,
}

// ─── meshopt DecodePosition impl ──────────────────────────────────────────

impl DecodePosition for PackedVertex {
    fn decode_position(&self) -> [f32; 3] {
        self.position
    }
}

// ─── LOD generation via meshopt simplify ──────────────────────────────────

/// Generate exactly 8 LOD levels using meshoptimizer's edge-collapse simplifier.
///
/// LOD 0 is the original mesh. Each successive level targets ~50% of the
/// previous level's triangle count. If simplification bottoms out, the last
/// level is duplicated to fill remaining slots so every LOD 0–7 has meshlets.
pub fn generate_lod_meshes(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> Vec<(Vec<PackedVertex>, Vec<u32>)> {
    let base_tri_count = indices.len() / 3;
    let mut levels: Vec<(Vec<PackedVertex>, Vec<u32>)> = Vec::with_capacity(8);
    levels.push((vertices.to_vec(), indices.to_vec()));

    // Weld vertices by position so meshopt can find shared edges to collapse.
    // Many importers (FBX especially) produce unwelded meshes where every
    // triangle has 3 unique vertices even when positions overlap.
    let (welded_verts, welded_indices) = weld_by_position(vertices, indices);
    let welded_tri_count = welded_indices.len() / 3;

    eprintln!(
        "[vg] weld: {} verts → {} verts, {} tris → {} tris",
        vertices.len(), welded_verts.len(),
        base_tri_count, welded_tri_count,
    );

    let lod_params: [(f32, f32); 7] = [
        (0.50,  0.05),
        (0.25,  0.1),
        (0.125, 0.2),
        (0.06,  0.4),
        (0.03,  0.7),
        (0.015, 1.0),
        (0.008, 2.0),
    ];
    for &(ratio, max_error) in &lod_params {
        let target_indices = (((welded_tri_count as f32 * ratio) as usize).max(1)) * 3;

        let simplified_indices = meshopt::simplify_decoder(
            &welded_indices,
            &welded_verts,
            target_indices,
            max_error,
            meshopt::SimplifyOptions::None,
            None,
        );

        if simplified_indices.is_empty() {
            let last = levels.last().unwrap().clone();
            levels.push(last);
        } else {
            let (compact_verts, compact_indices) =
                compact_mesh(&welded_verts, &simplified_indices);
            eprintln!(
                "[vg] LOD {}: {}/{} tris (target {}, error {})",
                levels.len(),
                compact_indices.len() / 3,
                welded_tri_count,
                target_indices / 3,
                max_error,
            );
            levels.push((compact_verts, compact_indices));
        }
    }

    while levels.len() < 8 {
        let last = levels.last().unwrap().clone();
        levels.push(last);
    }

    levels
}

/// Weld vertices that share the same position, merging duplicates.
/// Keeps the first vertex's attributes (normal, UV, etc.) for each unique position.
fn weld_by_position(vertices: &[PackedVertex], indices: &[u32]) -> (Vec<PackedVertex>, Vec<u32>) {
    use std::collections::HashMap;

    // Quantize positions to avoid floating-point mismatch (snap to ~1μm).
    fn pos_key(p: &[f32; 3]) -> (i64, i64, i64) {
        (
            (p[0] as f64 * 1_000_000.0).round() as i64,
            (p[1] as f64 * 1_000_000.0).round() as i64,
            (p[2] as f64 * 1_000_000.0).round() as i64,
        )
    }

    let mut pos_to_new: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; vertices.len()];
    let mut welded_verts: Vec<PackedVertex> = Vec::new();

    for (old_idx, v) in vertices.iter().enumerate() {
        let key = pos_key(&v.position);
        let new_idx = *pos_to_new.entry(key).or_insert_with(|| {
            let idx = welded_verts.len() as u32;
            welded_verts.push(*v);
            idx
        });
        remap[old_idx] = new_idx;
    }

    let welded_indices: Vec<u32> = indices.iter().map(|&i| {
        if (i as usize) < remap.len() { remap[i as usize] } else { 0 }
    }).collect();

    (welded_verts, welded_indices)
}

/// Remove unreferenced vertices and remap indices.
fn compact_mesh(vertices: &[PackedVertex], indices: &[u32]) -> (Vec<PackedVertex>, Vec<u32>) {
    let mut used = vec![u32::MAX; vertices.len()];
    let mut out_verts: Vec<PackedVertex> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::with_capacity(indices.len());

    for &idx in indices {
        let i = idx as usize;
        if i >= vertices.len() {
            out_indices.push(0);
            continue;
        }
        if used[i] == u32::MAX {
            used[i] = out_verts.len() as u32;
            out_verts.push(vertices[i]);
        }
        out_indices.push(used[i]);
    }

    (out_verts, out_indices)
}

// ─── Meshlet building via meshopt ─────────────────────────────────────────

/// Build meshlets using meshoptimizer AND return the reordered index buffer.
///
/// This is the preferred entry point: it returns both the meshlet descriptors
/// and the flat index buffer that those descriptors reference, ready for upload
/// to the mega-buffer.
pub fn meshletize_with_indices(
    vertices: &[PackedVertex],
    indices: &[u32],
    mesh_first_index: u32,
    mesh_first_vertex: u32,
) -> (Vec<GpuMeshletEntry>, Vec<u32>) {
    let tri_count = indices.len() / 3;
    if tri_count == 0 || vertices.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let max_verts = 64usize;
    let max_tris = MESHLET_MAX_TRIANGLES as usize;

    let vertex_adapter = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(vertices),
        mem::size_of::<PackedVertex>(),
        0,
    )
    .expect("valid vertex layout");

    let meshlets = meshopt::clusterize::build_meshlets(
        indices,
        &vertex_adapter,
        max_verts,
        max_tris,
        0.5,
    );

    let mut entries = Vec::with_capacity(meshlets.len());
    let mut flat_indices: Vec<u32> = Vec::new();

    for i in 0..meshlets.len() {
        let m = meshlets.get(i);

        let first_index_offset = flat_indices.len() as u32;

        // Expand meshlet local tri indices → global vertex indices.
        let mut meshlet_global_indices: Vec<u32> = Vec::with_capacity(m.triangles.len());
        for &local_tri_idx in m.triangles {
            let vertex_slot = m.vertices[local_tri_idx as usize];
            meshlet_global_indices.push(vertex_slot);
            flat_indices.push(vertex_slot);
        }

        let bounds = meshopt::clusterize::compute_cluster_bounds_decoder(
            &meshlet_global_indices,
            vertices,
        );

        entries.push(GpuMeshletEntry {
            center: bounds.center,
            radius: bounds.radius,
            cone_apex: bounds.cone_apex,
            cone_cutoff: bounds.cone_cutoff,
            cone_axis: bounds.cone_axis,
            lod_error: 0.0,
            first_index: mesh_first_index + first_index_offset,
            index_count: meshlet_global_indices.len() as u32,
            vertex_offset: mesh_first_vertex as i32,
            instance_index: 0,
        });
    }

    (entries, flat_indices)
}

