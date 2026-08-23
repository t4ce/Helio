// Per-portal-*chain* GPU frustum culling — selects which instances get a
// duplicate draw through each active portal chain (a sequence of up to
// `MAX_CHAIN_DEPTH` portals, e.g. "through P, then through P again, then
// again" — see `libhelio::GpuPortalChain`'s docs for why chains, not single
// portals, are what makes portals reflect each other automatically).
//
// Two passes, `select` then `finalize`, mirroring why the main scene's own
// GPU cull does the same split:
//
// `select`: one workgroup per (draw-call group, chain) pair, its 64 lanes
// cooperatively test that group's instances — mapped through the chain's
// *composed* transform, built here from `coordinate_spaces` — against the
// main camera frustum. Survivors are appended into that draw group's own
// reserved region of a *shared* compacted-index buffer (capacity is per
// draw-*group*, not per chain — chains vastly outnumber draw groups, and
// realistically only a few chains ever have any given group's content in
// frustum at once, so reserving capacity per chain would be enormous waste
// for essentially no headroom benefit). Each survivor is tagged with which
// chain selected it, since a single draw group can (and typically does) be
// selected by several different chains at once, each wanting its own
// separately-transformed duplicate.
//
// `finalize`: one thread per draw group, turns that group's final survivor
// count into a normal `DrawIndexedIndirect` — split out into its own pass
// because the count isn't known until every chain's `select` workgroup for
// that group has finished (they run concurrently, all bumping the same
// atomic counter).
//
// No occlusion (Hi-Z) test here — frustum-only. Portal-visible content is
// already bounded by the frustum test; occlusion is a pure fill-rate
// optimization this v1 skips (see docs/).

struct CullUniforms {
    frustum_planes:     array<vec4<f32>, 6>,
    draw_count:         u32,
    chain_count:        u32,
    // How many (instance, chain) survivor slots are reserved per draw
    // group — NOT per chain. See module doc.
    group_capacity:     u32,
    _pad:               u32,
}
@group(0) @binding(1) var<uniform> cull: CullUniforms;

struct SceneObjectSpatial {
    transform:    mat4x4<f32>,
    normal_mat_0: vec4<f32>,
    normal_mat_1: vec4<f32>,
    normal_mat_2: vec4<f32>,
    sphere:       vec4<f32>,
    flags:        u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}
@group(0) @binding(2) var<storage, read> object_spatial: array<SceneObjectSpatial>;

struct PortalDraw {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,
    instance_count: u32,
    material_row:   u32,
}
@group(0) @binding(3) var<storage, read> portal_draws: array<PortalDraw>;

// Coordinate-space transforms (current frame). Slot 0 = identity — see
// `libhelio::{coordinate_space, set_coordinate_space}`.
@group(0) @binding(4) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;
@group(0) @binding(11) var<storage, read> source_indices: array<u32>;

// One active portal's render data. Must match libhelio::GpuPortalView (144 bytes).
struct GpuPortalView {
    transform:         mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    half_extent:       vec2<f32>,
    coordinate_space:  u32,
    _pad:              u32,
}
@group(0) @binding(5) var<storage, read> portal_views: array<GpuPortalView>;

// One valid portal chain. Must match libhelio::GpuPortalChain (16 bytes at
// the default MAX_CHAIN_DEPTH=3) — `portals[0]` outermost (nearest the real
// camera) to `portals[depth-1]` innermost (deepest reflection).
struct GpuPortalChain {
    portals: array<u32, 3>,
    depth:   u32,
}
@group(0) @binding(9) var<storage, read> portal_chains: array<GpuPortalChain>;

struct DrawIndexedIndirectAtomic {
    index_count:    u32,
    instance_count: atomic<u32>,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}
// One entry per draw group (NOT per chain) — same shape as the main scene's
// own indirect buffer.
@group(0) @binding(6) var<storage, read_write> portal_indirect: array<DrawIndexedIndirectAtomic>;

struct PortalProjection {
    entity_row:   u32,
    chain_index:  u32,
    material_row: u32,
}
// Shared projection buffer: draw group `g`'s reserved region is
// `[g * cull.group_capacity, (g+1) * cull.group_capacity)`, filled by
// whichever chains' `select` workgroups actually found survivors there.
@group(0) @binding(7) var<storage, read_write> portal_projections: array<PortalProjection>;

/// Mirrors `libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE`.
const INSTANCE_FLAG_ALWAYS_VISIBLE: u32 = 4u;

fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = cull.frustum_planes[i];
        if dot(plane.xyz, center) + plane.w + radius < 0.0 { return false; }
    }
    return true;
}

fn max_basis_scale(m: mat4x4<f32>) -> f32 {
    return max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
}

@compute @workgroup_size(64)
fn select(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let draw_idx = wg_id.x;
    let chain_idx = wg_id.y;
    let is_active = draw_idx < cull.draw_count && chain_idx < cull.chain_count;

    // Every invocation must reach the barrier below regardless of
    // `is_active` — see indirect_dispatch.wgsl for why (FXC
    // barrier-uniformity rejection). No workgroup-local state to init this
    // time (the survivor counter is the indirect command's atomic field), so
    // the barrier is only here to keep that uniform-control-flow shape.
    workgroupBarrier();

    if is_active {
        let chain = portal_chains[chain_idx];
        let dc = portal_draws[draw_idx];
        let group_base = draw_idx * cull.group_capacity;

        for (var i = lid.x; i < dc.instance_count; i += 64u) {
            let slot_idx = dc.first_instance + i;
            let source_idx = source_indices[slot_idx];
            let inst = object_spatial[source_idx];
            let own_space_id = (inst.flags >> 8u) & 0xFFu;

            // Skip an instance already living in one of THIS chain's own
            // portal coordinate spaces — it would duplicate onto itself.
            // Generalizes the single-portal recursion guard the pre-chain
            // cull used.
            var self_referential = false;
            for (var k = 0u; k < chain.depth; k++) {
                if own_space_id == portal_views[chain.portals[k]].coordinate_space {
                    self_referential = true;
                }
            }
            if self_referential {
                continue;
            }

            let own_space = coordinate_spaces[own_space_id];
            var radius = inst.sphere.w * max_basis_scale(own_space);

            // Apply the chain deepest-portal-first, same order as the draw
            // pass, but *also* reject as soon as the mapped bounding sphere
            // clearly misses that stage's own portal window (padded by the
            // sphere's radius) — a coarse, conservative version of the
            // fragment shader's exact per-stage clip. Without this the cull
            // pass is frustum-only: a chain's composed position can still
            // land broadly "somewhere in the camera's frustum" even when
            // nowhere near any of its portals' actual openings, which
            // massively overselects (every wall panel through every chain)
            // and blows straight through the per-group capacity.
            //
            // The *outermost* stage (`s == 1`, portal `chain.portals[0]`)
            // only checks "behind the surface", not the X/Y window — that
            // window is enforced precisely by the screen-space mask in the
            // fragment shader instead (see gbuffer_portal.wgsl's module doc
            // for why), and content behind a portal is allowed to be wider
            // than the opening itself. Rejecting on X/Y here too would
            // coarse-cull away content the fragment shader would have
            // legitimately kept.
            var pos = (own_space * vec4<f32>(inst.sphere.xyz, 1.0)).xyz;
            var rejected = false;
            for (var s = chain.depth; s > 0u; s--) {
                let p = portal_views[chain.portals[s - 1u]];
                let portal_space = coordinate_spaces[p.coordinate_space];
                pos = (portal_space * vec4<f32>(pos, 1.0)).xyz;
                radius *= max_basis_scale(portal_space);
                let local = (p.inverse_transform * vec4<f32>(pos, 1.0)).xyz;
                let is_outermost = s == 1u;
                if local.z > radius || (!is_outermost && (
                    abs(local.x) > p.half_extent.x + radius || abs(local.y) > p.half_extent.y + radius
                )) {
                    rejected = true;
                    break;
                }
            }
            if rejected {
                continue;
            }
            let world_center = pos;

            let visible = (inst.flags & INSTANCE_FLAG_ALWAYS_VISIBLE) != 0u
                || sphere_in_frustum(world_center, radius);
            if visible {
                // Global (cross-workgroup) claim — other chains' workgroups
                // for this same draw group are appending concurrently.
                let slot = atomicAdd(&portal_indirect[draw_idx].instance_count, 1u);
                if slot < cull.group_capacity {
                    portal_projections[group_base + slot] = PortalProjection(
                        source_idx,
                        chain_idx,
                        dc.material_row,
                    );
                }
                // Else: this draw group's reserved region is full for this
                // frame — drop the excess rather than corrupt the next
                // group's region, same accepted degradation shape
                // helio-pass-shadow-cull's per-face draw cap uses.
            }
        }
    }
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_idx = gid.x;
    if draw_idx >= cull.draw_count {
        return;
    }
    let dc = portal_draws[draw_idx];
    let survivor_count = min(
        atomicLoad(&portal_indirect[draw_idx].instance_count),
        cull.group_capacity,
    );
    portal_indirect[draw_idx].index_count = dc.index_count;
    atomicStore(&portal_indirect[draw_idx].instance_count, survivor_count);
    portal_indirect[draw_idx].first_index = dc.first_index;
    portal_indirect[draw_idx].base_vertex = dc.vertex_offset;
    portal_indirect[draw_idx].first_instance = draw_idx * cull.group_capacity;
}
