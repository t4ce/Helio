// GPU per-face shadow frustum culling.
// Each thread tests every instance in one mesh-batched draw against all active
// dirty shadow faces. If any member intersects a face, the complete batch is
// conservatively appended. This preserves instancing without treating the
// first (spatially arbitrary) batch member as a proxy for all of them.

const MAX_FACES: u32 = 256u;

struct GpuShadowMatrix {
    mat: mat4x4<f32>,
}

struct SceneObjectSpatial {
    model_0:      vec4<f32>,  //   0
    model_1:      vec4<f32>,  //  16
    model_2:      vec4<f32>,  //  32
    model_3:      vec4<f32>,  //  48
    normal_0:     vec4<f32>,  //  64
    normal_1:     vec4<f32>,  //  80
    normal_2:     vec4<f32>,  //  96
    sphere:       vec4<f32>,  // 112
    flags:        u32,        // 128
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

struct CullUniforms {
    instance_count: u32,
    max_draws_per_face: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             uniforms:         CullUniforms;
@group(0) @binding(1) var<storage, read>       shadow_matrices:  array<GpuShadowMatrix>;
@group(0) @binding(2) var<storage, read>       object_spatial:   array<SceneObjectSpatial>;
@group(0) @binding(3) var<storage, read>       src_indirect:     array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write> dst_indirect:     array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read_write> face_counts:      array<atomic<u32>>;
@group(0) @binding(6) var<storage, read>       face_dirty:       array<u32>;
// Coordinate-space transforms (current frame). Slot 0 = identity. See
// indirect_dispatch.wgsl for the full mechanism.
@group(0) @binding(7) var<storage, read>       coordinate_spaces: array<mat4x4<f32>>;
@group(0) @binding(8) var<storage, read>       source_indices:   array<u32>;

fn normalize_plane(p: vec4<f32>) -> vec4<f32> {
    let len = length(p.xyz);
    if len > 1e-10 {
        return vec4<f32>(p.xyz / len, p.w / len);
    }
    return p;
}

/// Mirrors `libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE`.
const INSTANCE_FLAG_ALWAYS_VISIBLE: u32 = 4u;

fn sphere_in_frustum(vp: mat4x4<f32>, center: vec3<f32>, radius: f32) -> bool {
    // WGSL indexes matrices as m[column][row]. Gribb-Hartmann combines rows,
    // so construct them explicitly; combining columns only works accidentally
    // for symmetric matrices such as identity.
    let r0 = vec4f(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    let r1 = vec4f(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    let r2 = vec4f(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    let r3 = vec4f(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);

    let p0 = normalize_plane(r3 + r0);
    if dot(p0.xyz, center) + p0.w + radius < 0.0 { return false; }
    let p1 = normalize_plane(r3 - r0);
    if dot(p1.xyz, center) + p1.w + radius < 0.0 { return false; }
    let p2 = normalize_plane(r3 + r1);
    if dot(p2.xyz, center) + p2.w + radius < 0.0 { return false; }
    let p3 = normalize_plane(r3 - r1);
    if dot(p3.xyz, center) + p3.w + radius < 0.0 { return false; }
    let p4 = normalize_plane(r2);
    if dot(p4.xyz, center) + p4.w + radius < 0.0 { return false; }
    let p5 = normalize_plane(r3 - r2);
    if dot(p5.xyz, center) + p5.w + radius < 0.0 { return false; }
    return true;
}

fn affine_frobenius_scale(m: mat4x4f) -> f32 {
    // ||A||₂ <= ||A||F. Unlike max basis-column length, this remains a
    // conservative sphere-radius multiplier for arbitrary affine shear.
    return sqrt(
        dot(m[0].xyz, m[0].xyz)
        + dot(m[1].xyz, m[1].xyz)
        + dot(m[2].xyz, m[2].xyz)
    );
}

fn instance_in_frustum(vp: mat4x4<f32>, source_slot: u32) -> bool {
    let inst = object_spatial[source_indices[source_slot]];
    // ALWAYS_VISIBLE is the explicit escape hatch for geometry whose bounds
    // are absent or intentionally unusable. Honour it before rejecting an
    // empty sphere, otherwise the camera pass can draw an object whose shadow
    // silently disappears.
    if (inst.flags & INSTANCE_FLAG_ALWAYS_VISIBLE) != 0u {
        return true;
    }
    let space_id = (inst.flags >> 8u) & 0xFFu;
    let space = coordinate_spaces[space_id];
    let center = select(
        inst.sphere.xyz,
        (space * vec4<f32>(inst.sphere.xyz, 1.0)).xyz,
        space_id != 0u,
    );
    let radius_scale = select(
        1.0,
        affine_frobenius_scale(space),
        space_id != 0u,
    );
    let radius = abs(inst.sphere.w) * radius_scale;
    if radius <= 0.0 { return false; }

    return sphere_in_frustum(vp, center, radius);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_idx = gid.x;
    if draw_idx >= uniforms.instance_count { return; }

    let draw = src_indirect[draw_idx];

    for (var face = 0u; face < MAX_FACES; face++) {
        if face_dirty[face] == 0u { continue; }

        let vp = shadow_matrices[face].mat;
        var batch_visible = false;
        for (var instance = 0u; instance < draw.instance_count; instance++) {
            if instance_in_frustum(vp, draw.first_instance + instance) {
                batch_visible = true;
                break;
            }
        }
        if batch_visible {
            let slot = atomicAdd(&face_counts[face], 1u);
            if slot < uniforms.max_draws_per_face {
                let base = face * uniforms.max_draws_per_face;
                dst_indirect[base + slot] = draw;
            }
        }
    }
}
