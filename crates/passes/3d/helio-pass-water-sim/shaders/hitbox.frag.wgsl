// hitbox.frag.wgsl — AABB-based water displacement (replaces sphere.frag).
//
// For each hitbox we compute how much water the AABB *was* displacing (old bounds)
// and how much it *now* displaces (new bounds).  The difference drives a height
// change:  rise where the box vacated, fall where it now sits.
//
// Texture layout (Rgba16Float):
//   R = height  (read-write)
//   G = velocity (read-only, pass through)
//   B = normal.x (read-only, pass through)
//   A = normal.z (read-only, pass through)

@group(0) @binding(0) var water_texture: texture_2d<f32>;
@group(0) @binding(1) var water_sampler: sampler;

/// One AABB hitbox (80 bytes = 5 × vec4<f32>)
struct GpuWaterHitbox {
    old_min:  vec4<f32>,   // xyz = old AABB min
    old_max:  vec4<f32>,   // xyz = old AABB max
    new_min:  vec4<f32>,   // xyz = new AABB min
    new_max:  vec4<f32>,   // xyz = new AABB max
    params:   vec4<f32>,   // x = edge_softness, y = strength
}

struct HitboxUniforms {
    /// Number of active hitboxes
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> u: HitboxUniforms;

@group(0) @binding(3) var<storage, read> hitboxes: array<GpuWaterHitbox>;
@group(0) @binding(4) var<storage, read> hitbox_indices: array<u32>;

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    sim_dynamics:          vec4f,
    wind_params:           vec4f,
    _pad:                  vec4f,
}

struct WaterVolumeProjection {
    entity_row: u32,
    sim_slot: u32,
}

@group(0) @binding(5) var<storage, read> water_volumes: array<WaterVolume>;
@group(0) @binding(6) var<storage, read> water_volume_projections: array<WaterVolumeProjection>;

const CASCADE_0_PATCH_SIZE: f32 = 30.0;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Smooth 3-D Gaussian falloff inside an AABB.
///
/// Returns a value in [0, 1]:
///   1.0 at the box centre, smoothly tapering to 0 at and beyond the edges.
/// `softness` scales how quickly the falloff extends outside the box interior.
fn volume_in_box(
    box_min: vec3<f32>,
    box_max: vec3<f32>,
    uv: vec2<f32>,
    softness: f32,
    water: WaterVolume,
) -> f32 {
    let box_center  = (box_min + box_max) * 0.5;
    let box_half    = (box_max - box_min) * 0.5;

    // Reject a hitbox outside this target volume before periodic mapping. The
    // simulation tile repeats globally, but authored bounds remain the volume
    // ownership boundary.
    if any(box_max.xz < water.bounds_min.xz) || any(box_min.xz > water.bounds_max.xz) {
        return 0.0;
    }

    // Surface rendering samples cascade 0 with fract(world / 30m). Measure the
    // nearest periodic distance to the box centre in that same tile. WGSL
    // `fract` handles negative world coordinates consistently for both paths.
    let tile_point = uv * CASCADE_0_PATCH_SIZE;
    let tiled_center = fract(box_center.xz / CASCADE_0_PATCH_SIZE) * CASCADE_0_PATCH_SIZE;
    let direct_delta = abs(tile_point - tiled_center);
    let periodic_delta = min(
        direct_delta,
        vec2<f32>(CASCADE_0_PATCH_SIZE) - direct_delta,
    );
    let world_y  = water.bounds_max.w;

    // Per-axis distance from box surface (negative inside, positive outside)
    let d = vec3<f32>(
        periodic_delta.x - box_half.x,
        abs(world_y - box_center.y) - box_half.y,
        periodic_delta.y - box_half.z,
    );

    // Smooth falloff: exp(-clamp(d/softness, 0, 4)^2) per axis, multiplied together
    let soft = max(d, vec3<f32>(0.0)) / max(softness, 0.001);
    let weight = exp(-dot(soft * soft, vec3<f32>(1.0)));

    // Only the vertical overlap below the authored water surface displaces
    // water. A box entirely above the surface contributes zero; a fully
    // submerged box contributes its complete height.
    let box_height = max(box_max.y - box_min.y, 0.0);
    let submerged_depth = clamp(min(box_max.y, world_y) - box_min.y, 0.0, box_height);

    return weight * submerged_depth * 0.1;
}

// ── Entry point ──────────────────────────────────────────────────────────────

@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) volume_projection_index: u32,
) -> @location(0) vec4<f32> {
    var info = textureSample(water_texture, water_sampler, uv);
    let projection = water_volume_projections[volume_projection_index];
    let water = water_volumes[projection.entity_row];

    for (var i: u32 = 0u; i < u.count; i = i + 1u) {
        let hb = hitboxes[hitbox_indices[i]];
        let softness = hb.params.x;
        let strength = hb.params.y;

        // Water rises where the box *was* (old position)
        info.r += volume_in_box(hb.old_min.xyz, hb.old_max.xyz, uv, softness, water) * strength;
        // Water falls where the box *is* (new position)
        info.r -= volume_in_box(hb.new_min.xyz, hb.new_max.xyz, uv, softness, water) * strength;
    }

    return info;
}
