//! Virtual geometry (GPU-driven meshlet rendering) for large-scale geometry.
//!
//! Virtual geometry uses GPU-driven meshlet rendering with automatic LOD selection.
//! This system allows rendering millions of triangles efficiently by:
//! - Decomposing meshes into small clusters (meshlets) of ~64 triangles
//! - Generating multiple LOD levels via vertex clustering
//! - GPU-driven frustum culling and LOD selection per-meshlet
//!
//! # Architecture
//!
//! Unlike regular objects (which use indirect rendering with CPU-side scene management),
//! virtual geometry is **fully GPU-driven**:
//! - Meshlets are stored in a flat GPU buffer
//! - Instances reference ranges in the meshlet buffer
//! - GPU stage one performs conservative object culling and selects one LOD per object
//! - GPU stage two culls the selected LOD through fixed 64-meshlet work spans
//! - No CPU readback or per-frame iteration
//!
//! # Module Organization
//!
//! - [`meshes`]: Virtual mesh upload and removal (meshletization, LOD generation)
//! - [`objects`]: Virtual object instancing and transform updates
//! - [`rebuild`]: CPU-side buffer rebuild and frame data packaging
//!
//! # Example
//!
//! ```ignore
//! use helio::{VirtualMeshUpload, VirtualObjectDescriptor};
//!
//! // Upload high-res mesh (auto-generates LODs and meshlets)
//! let vg_mesh_id = scene.insert_virtual_mesh(VirtualMeshUpload {
//!     vertices: high_res_vertices,
//!     indices: high_res_indices,
//! });
//!
//! // Instance it multiple times
//! for transform in transforms {
//!     scene.insert_virtual_object(VirtualObjectDescriptor {
//!         virtual_mesh: vg_mesh_id,
//!         transform,
//!         bounds: [0.0, 0.0, 0.0, 10.0],
//!         material_id,
//!         flags: 0,
//!         groups: GroupMask::NONE,
//!     })?;
//! }
//! ```

use std::collections::HashMap;

use helio_scenedb::Subsystem;

use crate::arena::DenseArena;
use crate::handles::VirtualObjectId;
use crate::vg::VirtualMeshId;

use super::types::{VirtualMeshRecord, VirtualObjectRecord};

/// Canonical CPU identity/metadata for virtual geometry.
///
/// GPU meshlet/object/work arrays remain Helio-owned derived render data, but
/// the assets and placed instances from which they are rebuilt live in
/// SceneDB's subsystem registry rather than beside the renderer executor.
pub(in crate::scene) struct VirtualGeometryStorage {
    pub meshes: HashMap<VirtualMeshId, VirtualMeshRecord>,
    pub next_mesh_id: u32,
    pub objects: DenseArena<VirtualObjectRecord, VirtualObjectId>,
}

impl Default for VirtualGeometryStorage {
    fn default() -> Self {
        Self {
            meshes: HashMap::new(),
            next_mesh_id: 0,
            objects: DenseArena::new(),
        }
    }
}

impl Subsystem for VirtualGeometryStorage {
    fn name(&self) -> &'static str {
        "helio.geometry.virtual"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

mod meshes;
mod objects;
mod rebuild;
