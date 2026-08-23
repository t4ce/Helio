//! Internal helper functions for scene object management.

use glam::{Mat3, Mat4};
use helio_scenedb::{
    SceneMaterialTextureRef as SceneDbMaterialTextureRef, SceneMaterialTextureRefs, SceneObject,
    SceneObjectRenderRow, SceneObjectSpatialRow, SceneTextureTransform as SceneDbTextureTransform,
};

use crate::groups::GroupMask;
use crate::handles::{
    bits_from_handle, entity_from_handle, handle_from_bits, MaterialId, MeshId,
};
use crate::material::{MaterialTextureRef, MaterialTextures};
use super::ObjectDescriptor;

/// Compute the normal matrix from a model transform matrix.
///
/// The normal matrix is the inverse-transpose of the model matrix's upper-left 3×3 block.
/// This transformation correctly handles non-uniform scaling when transforming normal vectors
/// from object space to world space.
///
/// # Why precompute this?
///
/// Computing a 3×3 matrix inverse is expensive (requires adjugate calculation). By computing
/// it once on the CPU when the transform changes, we avoid doing this operation per-vertex
/// in the vertex shader.
///
/// # Performance
/// - CPU cost: O(1) per transform update
/// - GPU savings: Eliminates O(vertices) inverse operations per frame
///
/// # Returns
/// A 3×4 padded normal matrix (12 floats) ready for GPU upload.
pub(super) fn normal_matrix(transform: Mat4) -> [f32; 12] {
    let mat3 = Mat3::from_mat4(transform).inverse().transpose();
    let cols = mat3.to_cols_array();
    [
        cols[0], cols[1], cols[2], 0.0, cols[3], cols[4], cols[5], 0.0, cols[6], cols[7], cols[8],
        0.0,
    ]
}

/// Test if an object is visible based on group membership and hidden groups.
///
/// # Visibility Semantics
/// - An object is **hidden** if **any** of its groups are currently hidden
/// - Ungrouped objects (`groups == GroupMask::NONE`) are **always visible**
///
/// # Parameters
/// - `groups`: The object's group membership bitmask
/// - `group_hidden`: Bitmask of currently hidden groups
///
/// # Returns
/// `true` if the object should be rendered, `false` if hidden
///
/// # Example
/// ```ignore
/// let obj_groups = GroupMask::from_id(GroupId(0)); // Object in group 0
/// let hidden = GroupMask::from_id(GroupId(0));     // Group 0 is hidden
/// assert!(!object_is_visible(obj_groups, hidden)); // Object is hidden
/// ```
#[inline(always)]
pub(super) fn object_is_visible(groups: GroupMask, group_hidden: GroupMask) -> bool {
    groups.is_empty() || !groups.intersects(group_hidden)
}

/// Construct SceneDB's canonical object component from an object descriptor.
///
/// Only authored spatial and render fields live here. Temporal transforms,
/// visibility ordering, AABBs, and draw arguments are Helio-derived data.
///
/// # Parameters
/// - `mesh`: Mesh handle (for slot lookup)
/// - `desc`: User-provided object descriptor
pub(super) fn object_gpu_data(
    mesh: MeshId,
    material_row: u32,
    desc: ObjectDescriptor,
) -> SceneObject {
    SceneObject {
        mesh_handle_bits: bits_from_handle(mesh),
        material_handle_bits: bits_from_handle(desc.material),
        groups: desc.groups.0,
        user_tag: desc.user_tag,
        spatial: SceneObjectSpatialRow {
            model: desc.transform.to_cols_array(),
            normal_mat: normal_matrix(desc.transform),
            sphere: desc.bounds,
            flags: desc.flags,
            _pad: [0; 3],
        },
        render: SceneObjectRenderRow {
            mesh_row: mesh.slot(),
            material_row,
            lightmap_index: 0xFFFFFFFF,
            reserved: 0,
        },
        movability: desc.movability.unwrap_or_default() as u32,
        _pad: 0,
    }
}

#[inline]
pub(super) fn object_mesh(record: &SceneObject) -> MeshId {
    handle_from_bits(record.mesh_handle_bits)
}

#[inline]
pub(super) fn object_material(record: &SceneObject) -> MaterialId {
    handle_from_bits(record.material_handle_bits)
}

#[inline]
pub(super) fn object_groups(record: &SceneObject) -> GroupMask {
    GroupMask(record.groups)
}

#[inline]
pub(super) fn object_movability(record: &SceneObject) -> libhelio::Movability {
    match record.movability {
        0 => libhelio::Movability::Static,
        1 => libhelio::Movability::Stationary,
        2 => libhelio::Movability::Movable,
        value => panic!("invalid SceneObject movability discriminant {value}"),
    }
}

fn scene_texture_ref(reference: MaterialTextureRef) -> SceneDbMaterialTextureRef {
    SceneDbMaterialTextureRef {
        texture: entity_from_handle(reference.texture),
        uv_channel: reference.uv_channel,
        transform: SceneDbTextureTransform {
            offset: reference.transform.offset,
            scale: reference.transform.scale,
            rotation_radians: reference.transform.rotation_radians,
        },
    }
}

/// Convert public Helio handles into generation-bearing canonical SceneDB
/// relations. No physical texture-array slot is carried across this boundary.
pub(super) fn scene_material_texture_refs(textures: &MaterialTextures) -> SceneMaterialTextureRefs {
    SceneMaterialTextureRefs {
        base_color: textures.base_color.map(scene_texture_ref),
        normal: textures.normal.map(scene_texture_ref),
        roughness_metallic: textures.roughness_metallic.map(scene_texture_ref),
        emissive: textures.emissive.map(scene_texture_ref),
        occlusion: textures.occlusion.map(scene_texture_ref),
        specular_color: textures.specular_color.map(scene_texture_ref),
        specular_weight: textures.specular_weight.map(scene_texture_ref),
        normal_scale: textures.normal_scale,
        occlusion_strength: textures.occlusion_strength,
        alpha_cutoff: textures.alpha_cutoff,
    }
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use super::object_gpu_data;
    use crate::{GroupMask, MaterialId, MeshId, ObjectDescriptor};

    #[test]
    fn object_render_row_uses_resolved_material_gpu_row() {
        let material = MaterialId::from_raw(41, 7);
        let record = object_gpu_data(
            MeshId::from_raw(3, 2),
            5,
            ObjectDescriptor {
                mesh: MeshId::from_raw(3, 2),
                material,
                transform: Mat4::IDENTITY,
                bounds: [0.0; 4],
                flags: 0,
                groups: GroupMask::NONE,
                movability: None,
                user_tag: 0,
            },
        );

        assert_eq!(record.render.material_row, 5);
        assert_ne!(record.render.material_row, material.slot());
    }
}
