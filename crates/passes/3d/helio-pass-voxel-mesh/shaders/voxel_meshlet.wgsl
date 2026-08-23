// Rasterizes VoxelMeshPass's interleaved local-space output. The canonical
// SceneDB volume row is work-derived into the interleaved vertex's otherwise
// padded normal.w lane, so no pass-owned descriptor table or optional
// INDIRECT_FIRST_INSTANCE device feature is required.

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct GpuVoxelVolume {
    local_to_world: mat4x4<f32>,
    world_to_local: mat4x4<f32>,
    dimensions: vec3<u32>,
    brick_grid_dim: u32,
    voxel_size: f32,
    palette_offset: u32,
    brick_offset: u32,
    palette_count: u32,
    _pad: vec2<u32>,
}

struct GpuVoxelMaterial {
    color: vec3<f32>,
    roughness: f32,
    metalness: f32,
    emissive: f32,
    _pad: vec2<u32>,
}

struct GpuLight {
    position_range: vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index: u32,
    light_type: u32,
    inner_angle: f32,
    _pad: u32,
    god_rays_enabled: u32,
    god_rays_density: f32,
    god_rays_weight: f32,
    god_rays_decay: f32,
    god_rays_exposure: f32,
    flare_enabled: u32,
    flare_type: u32,
    flare_intensity: f32,
    flare_scale: f32,
    flare_tint_r: f32,
    flare_tint_g: f32,
    flare_tint_b: f32,
    ies_profile_index: i32,
    light_function_index: i32,
    ies_angle_scale: f32,
    ies_angle_offset: f32,
}

struct MeshletParams {
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct LightProjection {
    entity_row: u32,
    shadow_index: u32,
}

struct VertexInput {
    @location(0) position_material: vec4<f32>,
    @location(1) normal_aux: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) @interpolate(flat) volume_row: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<uniform> params: MeshletParams;
@group(0) @binding(3) var<storage, read> light_projections: array<LightProjection>;
@group(0) @binding(4) var<storage, read> volumes: array<GpuVoxelVolume>;
@group(0) @binding(5) var<storage, read> palettes: array<GpuVoxelMaterial>;

fn projected_light(compact_index: u32) -> GpuLight {
    let projection = light_projections[compact_index];
    var light = lights[projection.entity_row];
    light.shadow_index = projection.shadow_index;
    return light;
}

fn light_contribution(light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var direction: vec3<f32>;
    var radiance: vec3<f32>;
    if light.light_type == 0u {
        direction = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        let to_light = light.position_range.xyz - world_pos;
        let distance = length(to_light);
        if distance > light.position_range.w {
            return vec3<f32>(0.0);
        }
        direction = to_light / max(distance, 0.0001);
        var attenuation = 1.0 / (distance * distance + 0.0001);
        let normalized_distance = distance / light.position_range.w;
        attenuation *= max(
            0.0,
            1.0 - normalized_distance * normalized_distance
                * normalized_distance * normalized_distance,
        );
        if light.light_type == 2u {
            let cos_angle = dot(-direction, light.direction_outer.xyz);
            attenuation *= smoothstep(light.direction_outer.w, light.inner_angle, cos_angle);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * attenuation;
    }
    return radiance * max(dot(normal, direction), 0.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let volume_row = bitcast<u32>(input.normal_aux.w);
    let volume = volumes[volume_row];
    let world_position = volume.local_to_world * vec4<f32>(input.position_material.xyz, 1.0);
    let world_normal = normalize(
        (transpose(volume.world_to_local) * vec4<f32>(input.normal_aux.xyz, 0.0)).xyz
    );
    var output: VertexOutput;
    output.clip_pos = cameras[0].view_proj * world_position;
    output.material = u32(input.position_material.w);
    output.world_pos = world_position.xyz;
    output.world_normal = world_normal;
    output.volume_row = volume_row;
    return output;
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let volume = volumes[input.volume_row];
    var material = GpuVoxelMaterial(
        vec3<f32>(1.0, 0.0, 1.0),
        1.0,
        0.0,
        0.0,
        vec2<u32>(0u),
    );
    if input.material < volume.palette_count {
        material = palettes[volume.palette_offset + input.material];
    }

    let normal = normalize(input.world_normal) * select(-1.0, 1.0, front);
    var direct = vec3<f32>(0.0);
    for (var light_index = 0u; light_index < params.light_count; light_index++) {
        direct += light_contribution(projected_light(light_index), input.world_pos, normal);
    }
    let diffuse_weight = 1.0 - clamp(material.metalness, 0.0, 1.0);
    let ambient = 0.2 * mix(1.0, 0.5, clamp(material.roughness, 0.0, 1.0));
    let emissive = material.color * max(material.emissive, 0.0);
    let lit = material.color * diffuse_weight * (ambient + direct) + emissive;
    return vec4<f32>(lit, 1.0);
}
