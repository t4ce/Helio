/// Per-face shadow dirty detection compute shader.
///
/// Runs once per frame AFTER ShadowMatrixPass has updated shadow VP matrices.
/// One thread per movable shadow-caster source row.
///
/// Algorithm:
///   1. Thread i maps the movable source slot to its canonical SceneDB object row.
///   2. Reads current spatial data and Helio-owned previous-frame history for that row.
///   3. If the object moved more than EPSILON:
///      For each active shadow face, extract 6 frustum planes from the VP matrix
///      (Gribb-Hartmann), and sphere-test both the old and new bounding spheres.
///      Any face that sees the moved object gets `face_dirty[face] = 1` (atomicOr).
///
/// The zeroing of `face_dirty` each frame is done by a command-encoder buffer
/// clear before this dispatch. This is intentionally not done by invocation 0:
/// WGSL has no device-wide workgroup barrier.

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_FACES: u32 = 256u;

/// Minimum world-space displacement (metres) that counts as a "move".
/// Set to ~0.1 mm — below floating point noise threshold at scene scale.
const MOVE_EPSILON: f32 = 0.0001;

/// Mirrors `libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE`.
const INSTANCE_FLAG_ALWAYS_VISIBLE: u32 = 4u;

// ── Structs ───────────────────────────────────────────────────────────────────

struct SceneObjectSpatial {
    transform:    mat4x4f,
    normal_mat_0: vec4f,
    normal_mat_1: vec4f,
    normal_mat_2: vec4f,
    sphere:       vec4f,
    flags:        u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

/// Helio-owned previous-frame state, keyed by the same component-local
/// SceneObject row as object_spatial.
struct ObjectHistory {
    transform: mat4x4f,
    sphere:    vec4f,
    flags:     u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

/// Must match GpuShadowMatrix in shadow_matrices.wgsl / libhelio (64 bytes).
struct GpuShadowMatrix {
    mat: mat4x4f,
}

struct ShadowDirtyUniforms {
    /// Number of active movable source rows, not mesh-batched draw calls.
    movable_object_count: u32,
    /// Number of active shadow faces (= shadow_count from SceneResources).
    face_count: u32,
    /// Set to 1 on the frame when movable_draw_count changes — dirties all faces.
    force_dirty_all: u32,
    _pad: u32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>         object_spatial: array<SceneObjectSpatial>;
@group(0) @binding(1) var<storage, read>         source_indices: array<u32>;
@group(0) @binding(2) var<storage, read>         object_history: array<ObjectHistory>;
@group(0) @binding(3) var<storage, read>          shadow_mats:    array<GpuShadowMatrix>;
/// Per-face dirty flag (0 = clean, 1 = dirty). Also used as clear-draw count by ShadowPass.
@group(0) @binding(4) var<storage, read_write>    face_dirty:     array<atomic<u32>>;
@group(0) @binding(5) var<storage, read>          coordinate_spaces: array<mat4x4f>;
@group(0) @binding(6) var<uniform>                uniforms:       ShadowDirtyUniforms;
/// Per-caster flags written by ShadowMatrixPass when a light matrix changes.
@group(0) @binding(7) var<storage, read_write>     light_dirty:    array<atomic<u32>>;
@group(0) @binding(8) var<storage, read>          coordinate_spaces_prev: array<mat4x4f>;

// ── Frustum helpers (Gribb-Hartmann) ─────────────────────────────────────────

/// Extract 6 view-frustum half-space planes from a VP matrix (column-major WGSL mat4x4f).
///
/// Each plane is vec4f(nx, ny, nz, d) where the signed distance of a point P from the
/// plane is:  dot(normal, P) + d.  A positive value means "inside" the frustum.
/// The direction convention is: planes[i].xyz points INWARD into the frustum.
fn normalize_plane(p: vec4f) -> vec4f {
    let normal_length = length(p.xyz);
    if normal_length > 1e-10 {
        return p / normal_length;
    }
    return p;
}

fn extract_frustum_planes(m: mat4x4f) -> array<vec4f, 6> {
    // WGSL mat4x4f: m[col][row], i.e. m[c].r accesses column c, row r.
    // We need the rows of the matrix for Gribb-Hartmann:
    let r0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
    let r1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
    let r2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
    let r3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);

    var planes: array<vec4f, 6>;
    // Sphere distances are meaningful only for unit normals. Omitting this
    // normalization makes the effective radius depend on projection scale.
    planes[0] = normalize_plane(r3 + r0); // Left
    planes[1] = normalize_plane(r3 - r0); // Right
    planes[2] = normalize_plane(r3 + r1); // Bottom
    planes[3] = normalize_plane(r3 - r1); // Top
    planes[4] = normalize_plane(r2);      // Near (depth [0,1])
    planes[5] = normalize_plane(r3 - r2); // Far
    return planes;
}

/// Returns true if the sphere (center, radius) intersects or is inside all 6 planes.
fn sphere_vs_frustum(center: vec3f, radius: f32, planes: array<vec4f, 6>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let p = planes[i];
        // Signed distance from center to plane (positive = inside half-space).
        let dist = dot(p.xyz, center) + p.w;
        if dist < -radius {
            return false;  // entirely outside this plane → not in frustum
        }
    }
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

fn matrix_moved(current: mat4x4f, previous: mat4x4f) -> bool {
    for (var column = 0u; column < 4u; column++) {
        if any(abs(current[column] - previous[column]) > vec4f(MOVE_EPSILON)) {
            return true;
        }
    }
    return false;
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;

    let movable_count = uniforms.movable_object_count;
    let face_count    = min(uniforms.face_count, MAX_FACES);
    let force_all     = uniforms.force_dirty_all;

    // A moving light changes every face frustum for that caster. Consume the
    // matrix pass's per-caster flag and dirty all six allocated face slots so
    // ShadowCullPass rebuilds its compacted indirect lists before rendering.
    if tid == 0u {
        let caster_count = (face_count + 5u) / 6u;
        for (var caster = 0u; caster < caster_count; caster++) {
            if atomicExchange(&light_dirty[caster], 0u) != 0u {
                let first_face = caster * 6u;
                let last_face = min(first_face + 6u, face_count);
                for (var face = first_face; face < last_face; face++) {
                    atomicStore(&face_dirty[face], 1u);
                }
            }
        }
    }

    // force_dirty_all: topology changed (movable count changed) — dirty every face.
    if force_all != 0u {
        if tid == 0u {
            for (var f = 0u; f < face_count; f++) {
                atomicStore(&face_dirty[f], 1u);
            }
        }
        return;
    }

    // Per-draw-call dirty detection.
    if tid >= movable_count {
        return;
    }

    let entity_row = source_indices[tid];
    let current = object_spatial[entity_row];
    let previous = object_history[entity_row];

    let current_space_id = (current.flags >> 8u) & 0xFFu;
    let previous_space_id = (previous.flags >> 8u) & 0xFFu;
    let current_space = coordinate_spaces[current_space_id];
    let previous_space = coordinate_spaces_prev[previous_space_id];
    let current_model = current_space * current.transform;
    let previous_model = previous_space * previous.transform;

    let current_center = (current_space * vec4f(current.sphere.xyz, 1.0)).xyz;
    let previous_center = (previous_space * vec4f(previous.sphere.xyz, 1.0)).xyz;
    let current_radius_scale = select(
        1.0,
        affine_frobenius_scale(current_space),
        current_space_id != 0u,
    );
    let previous_radius_scale = select(
        1.0,
        affine_frobenius_scale(previous_space),
        previous_space_id != 0u,
    );
    let current_radius = abs(current.sphere.w) * current_radius_scale;
    let previous_radius = abs(previous.sphere.w) * previous_radius_scale;
    let sphere_delta = abs(current_center - previous_center);
    let moved = matrix_moved(current_model, previous_model)
        || any(sphere_delta > vec3f(MOVE_EPSILON))
        || abs(current_radius - previous_radius) > MOVE_EPSILON;

    if !moved {
        return;
    }

    // The explicit no-cull contract also applies to dirty-face selection. A
    // row can use ALWAYS_VISIBLE precisely because its sphere is absent or too
    // poor for conservative testing, so movement must invalidate every face.
    if ((current.flags | previous.flags) & INSTANCE_FLAG_ALWAYS_VISIBLE) != 0u {
        for (var face = 0u; face < face_count; face++) {
            atomicOr(&face_dirty[face], 1u);
        }
        return;
    }

    // Object moved → dirty every face touched either before or after the move.
    // Testing only the current sphere leaves the caster's old depth cached in
    // faces it just exited, producing a persistent ghost shadow.
    for (var face = 0u; face < face_count; face++) {
        let planes = extract_frustum_planes(shadow_mats[face].mat);
        if sphere_vs_frustum(current_center, current_radius, planes)
            || sphere_vs_frustum(previous_center, previous_radius, planes) {
            atomicOr(&face_dirty[face], 1u);
        }
    }
}
