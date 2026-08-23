// drop.frag.wgsl — adds a cosine-falloff ripple to the water heightfield.
//
// Texture layout (Rgba16Float):
//   R = height
//   G = velocity
//   B = normal.x
//   A = normal.z

@group(0) @binding(0) var water_texture: texture_2d<f32>;
@group(0) @binding(1) var water_sampler: sampler;

struct DropUniforms {
    /// Drop centre in world XZ space.
    world_center: vec2<f32>,
    /// Drop radius in world metres.
    radius: f32,
    /// Height increment to add at the drop center (can be negative)
    strength: f32,
    /// Canonical SceneDB row selected for the targeted stable simulation slot.
    volume_row: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> u: DropUniforms;

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
    _pad6:                 vec4f,
}
@group(0) @binding(3) var<storage, read> water_volumes: array<WaterVolume>;

const CASCADE_0_PATCH_SIZE: f32 = 30.0;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var info = textureSample(water_texture, water_sampler, uv);
    let volume = water_volumes[u.volume_row];
    let inside = all(u.world_center >= volume.bounds_min.xz)
        && all(u.world_center <= volume.bounds_max.xz);
    if !inside {
        return info;
    }

    // Surface rendering samples cascade 0 with fract(world / 30m). Mirror that
    // exact periodic contract here. `fract` intentionally maps negative world
    // coordinates into [0, 1) with the same rule used by the surface shader.
    let center_uv = fract(u.world_center / CASCADE_0_PATCH_SIZE);
    let direct_delta = abs(center_uv - uv);
    let wrapped_delta = min(direct_delta, vec2<f32>(1.0) - direct_delta);
    let world_distance = length(wrapped_delta * CASCADE_0_PATCH_SIZE);
    let drop = max(0.0, 1.0 - world_distance / u.radius);
    let drop_val = 0.5 - cos(drop * 3.14159265) * 0.5;

    info.r += drop_val * u.strength;
    return info;
}
