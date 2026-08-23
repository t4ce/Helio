// ── Sprite GPU Culling ──────────────────────────────────────────────────────
//
// One thread per pool slot. Alive + in-view slots are atomically compacted
// straight into `visible_indices`/`sort_keys`, and the surviving count is
// atomically accumulated directly into `indirect_args[1]` (the
// `DrawIndexedIndirectArgs.instance_count` field — binding it as `atomic<u32>`
// is purely a WGSL-side access annotation; the underlying bytes are identical
// to a plain u32, so `SpriteBatchPass` later issues one `draw_indexed_indirect`
// reading those same bytes with no CPU-known count at all).

struct CullUniforms {
    view_min: vec2<f32>,
    view_max: vec2<f32>,
    slot_count: u32,
    max_visible: u32,
    runtime_capacity: u32,
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

@group(0) @binding(0) var<uniform> uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> instances: array<SpriteInstance>;
@group(0) @binding(2) var<storage, read> slot_alive: array<u32>;
@group(0) @binding(3) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> runtime: array<SpriteRuntime>;

// Saturating reservation: indirect_args[1] can never exceed the allocated
// draw-order/sort capacity. index 5 is overflow telemetry, outside the
// 20-byte DrawIndexedIndirectArgs prefix consumed by draw_indexed_indirect.
fn reserve_visible_slot() -> u32 {
    loop {
        let current = atomicLoad(&indirect_args[1]);
        if current >= uniforms.max_visible {
            atomicAdd(&indirect_args[5], 1u);
            return uniforms.max_visible;
        }
        let claimed = atomicCompareExchangeWeak(
            &indirect_args[1],
            current,
            current + 1u,
        );
        if claimed.exchanged {
            return current;
        }
    }
    return uniforms.max_visible;
}

/// Monotonic ascending-order-preserving `f32 -> u32` transform — mirrors
/// `depth_to_radix_key` in `src/lib.rs` exactly. Flip the sign bit for
/// positives, flip every bit for negatives.
fn depth_to_radix_key(depth: f32) -> u32 {
    let bits = bitcast<u32>(depth);
    if (bits & 0x80000000u) != 0u {
        return ~bits;
    }
    return bits | 0x80000000u;
}

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i = gid.x;
    loop {
        if i >= uniforms.slot_count {
            break;
        }
        cull_slot(i);
        i += uniforms.dispatched_threads;
    }
}

fn cull_slot(i: u32) {
    if slot_alive[i] == 0u {
        return;
    }

    let inst = instances[i];
    var position = inst.position;
    var depth = inst.depth;
    if i < uniforms.runtime_capacity && runtime[i].authored_epoch == inst.authored_epoch {
        position = runtime[i].position;
        depth = runtime[i].depth;
    }
    // Circle-vs-AABB: clamp the sprite's center into the view rect, then
    // compare the distance to that clamped point against the sprite's
    // bounding radius (half the quad's diagonal — conservative at any
    // rotation).
    let clamped = clamp(position, uniforms.view_min, uniforms.view_max);
    let d = position - clamped;
    let radius = 0.5 * length(inst.size);
    if dot(d, d) > radius * radius {
        return;
    }

    let slot = reserve_visible_slot();
    if slot < uniforms.max_visible {
        visible_indices[slot] = i;
        sort_keys[slot] = depth_to_radix_key(depth);
    }
}
