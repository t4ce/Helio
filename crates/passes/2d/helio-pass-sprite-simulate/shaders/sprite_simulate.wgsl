// Pass-derived sprite bounce simulation.
//
// Authored SceneDB rows are read-only. Runtime position/depth/velocity live in
// a separate Helio-owned projection and are accepted by cull/render only when
// `authored_epoch` matches, so an authored update transactionally resets the
// simulation from its new CPU/GPU partner row.

struct SimUniforms {
    bounds_min: vec2<f32>,
    bounds_max: vec2<f32>,
    dt: f32,
    slot_count: u32,
    _reserved: u32,
    dispatched_threads: u32,
}

struct SpriteInstance {
    position: vec2<f32>,
    size: vec2<f32>,
    rotation: f32,
    depth: f32,
    simulation_velocity: vec2<f32>,
    uv_rect: vec4<f32>,
    color: vec4<f32>,
    atlas_layer: u32,
    atlas_entity_low: u32,
    atlas_entity_high: u32,
    authored_epoch: u32,
}

struct SpriteRuntime {
    position: vec2<f32>,
    depth: f32,
    authored_epoch: u32,
    velocity: vec2<f32>,
}

@group(0) @binding(0) var<uniform> su: SimUniforms;
@group(0) @binding(1) var<storage, read> slot_alive: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<SpriteInstance>;
@group(0) @binding(3) var<storage, read_write> runtime: array<SpriteRuntime>;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn cs_simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i = gid.x;
    loop {
        if i >= su.slot_count {
            break;
        }
        simulate_slot(i);
        i += su.dispatched_threads;
    }
}

fn simulate_slot(i: u32) {
    if slot_alive[i] == 0u {
        runtime[i].authored_epoch = 0u;
        return;
    }

    let authored = instances[i];
    let initial_velocity = authored.simulation_velocity;
    if initial_velocity.x == 0.0 && initial_velocity.y == 0.0 {
        runtime[i].authored_epoch = 0u;
        return;
    }

    var pos: vec2<f32>;
    var vel: vec2<f32>;
    if runtime[i].authored_epoch != authored.authored_epoch {
        pos = authored.position;
        vel = initial_velocity;
    } else {
        pos = runtime[i].position;
        vel = runtime[i].velocity;
    }

    pos += vel * su.dt;
    if pos.x < su.bounds_min.x || pos.x > su.bounds_max.x {
        vel.x = -vel.x;
        pos.x = clamp(pos.x, su.bounds_min.x, su.bounds_max.x);
    }
    if pos.y < su.bounds_min.y || pos.y > su.bounds_max.y {
        vel.y = -vel.y;
        pos.y = clamp(pos.y, su.bounds_min.y, su.bounds_max.y);
    }

    runtime[i].position = pos;
    runtime[i].depth = pos.y;
    runtime[i].authored_epoch = authored.authored_epoch;
    runtime[i].velocity = vel;
}
