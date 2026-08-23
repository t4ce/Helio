//! Depth-only prepass shader.
//!
//! Transforms vertices through the camera view-projection and writes to the depth buffer.
//! No fragment output — depth writes are implicit.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct SceneObjectSpatial {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    sphere:        vec4<f32>,
    flags:         u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> object_spatial:    array<SceneObjectSpatial>;
// Per-draw-call-group compacted original instance slots (see IndirectDispatchPass).
@group(0) @binding(2) var<storage, read> compacted_indices: array<u32>;
@group(0) @binding(3) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;

@vertex
fn vs_main(
    @location(0)             position:    vec3<f32>,
    @location(2)             _tex_coords: vec2<f32>,  // kept for vertex layout compatibility
    @builtin(instance_index) slot:        u32,
) -> @invariant @builtin(position) vec4<f32> {
    let inst      = object_spatial[compacted_indices[slot]];
    let space_id  = (inst.flags >> 8u) & 0xFFu;
    let world_pos = coordinate_spaces[space_id] * inst.transform * vec4<f32>(position, 1.0);
    return cameras[0].view_proj * world_pos;
}
