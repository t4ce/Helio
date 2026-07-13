// ── Vertex shader for voxel meshlet rendering ──────────────────────────────
// Reads per-vertex data from storage buffer (vec4: xyz=position, w=material)
// Outputs to G-buffer: albedo @ loc0, normal @ loc1, orm @ loc2, emissive @ loc3

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}

struct VertexInput {
    @location(0) data: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4(v.data.xyz, 1.0);
    out.material = u32(v.data.w);
    out.world_pos = v.data.xyz;
    out.world_normal = normalize(v.normal.xyz);
    return out;
}

// ── Fragment shader: G-buffer output ──────────────────────────────────────

fn golden_ratio_hue(index: u32) -> f32 {
    return f32(index) * 0.6180339887;
}

fn material_color(index: u32) -> vec3<f32> {
    let h = golden_ratio_hue(index);
    let r = cos(h * 6.28318 + 0.0) * 0.5 + 0.5;
    let g = cos(h * 6.28318 + 2.09439) * 0.5 + 0.5;
    let b = cos(h * 6.28318 + 4.18879) * 0.5 + 0.5;
    return vec3<f32>(r, g, b);
}

fn material_roughness(index: u32) -> f32 {
    return 0.4 + (f32(index % 16u) * 0.04);
}

fn material_metalness(index: u32) -> f32 {
    return select(0.0, 0.9, (index / 8u) % 2u == 1u);
}

fn material_emissive(index: u32) -> vec3<f32> {
    return select(vec3<f32>(0.0), material_color(index) * 2.0, index == 0u);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let col = material_color(in.material);
    let rough = material_roughness(in.material);
    let metal = material_metalness(in.material);
    let emissive = material_emissive(in.material);

    // Simple directional lighting for preview
    let sun_dir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let diff = max(dot(n, sun_dir), 0.0);
    let ambient = 0.3;
    let lit = col * (ambient + diff * 0.7) + emissive * 0.1;

    return vec4(lit, 1.0);
}
