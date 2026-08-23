struct Camera {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
    time:      f32,
}

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    rc_world_min:      vec4<f32>,
    rc_world_max:      vec4<f32>,
    csm_splits:        vec4<f32>,
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

struct SceneObjectRender {
    mesh_row:       u32,
    material_row:   u32,
    lightmap_index: u32,
    _reserved:      u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> object_spatial: array<SceneObjectSpatial>;
@group(0) @binding(3) var<storage, read> compacted_indices: array<u32>;
@group(0) @binding(4) var<storage, read> object_render: array<SceneObjectRender>;
@group(0) @binding(5) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let entity_row = compacted_indices[slot];
    let inst       = object_spatial[entity_row];
    let space_id   = (inst.flags >> 8u) & 0xFFu;
    let space      = coordinate_spaces[space_id];
    let space_rot  = mat3x3<f32>(space[0].xyz, space[1].xyz, space[2].xyz);
    let world_pos  = space * (inst.transform * vec4<f32>(vertex.position, 1.0));
    let normal_mat = space_rot * mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    var out: VertexOutput;
    out.clip_position  = cameras[0].view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * decode_snorm8x4(vertex.normal));
    out.tex_coords     = vertex.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ambient = globals.ambient_color.rgb * globals.ambient_intensity;
    let normal_shade = in.world_normal * 0.5 + 0.5;
    let color = ambient + normal_shade * 0.3;
    return vec4<f32>(color, 0.3);
}
