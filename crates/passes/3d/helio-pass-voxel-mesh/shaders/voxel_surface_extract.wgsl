// Marching-cubes extraction for SceneDB-backed Auto voxel volumes.
// One workgroup per stable Helio mesh-output slot. Raw 8^3 bricks are read
// directly from canonical residency; boundary samples resolve neighbouring
// bricks instead of relying on a second padded upload representation.

const MAX_VERTS: u32 = 2048u;
const MAX_INDICES: u32 = 2048u;
const CELLS_PER_DIM: u32 = 8u;
const TOTAL_CELLS: u32 = 512u;
const WG_SIZE: u32 = 64u;
const CELLS_PER_THREAD: u32 = 8u;
const WORK_ALLOCATED: u32 = 1u;

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

struct GpuVoxelMeshWork {
    volume_row: u32,
    local_brick: u32,
    flags: u32,
    generation: u32,
}

struct GpuVoxelMeshVertex {
    position_material: vec4<f32>,
    normal_aux: vec4<f32>,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct ExtractParams {
    generation: u32,
    bootstrap: u32,
    work_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> brick_meta: array<u32>;
@group(0) @binding(1) var<storage, read> voxel_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertices: array<GpuVoxelMeshVertex>;
@group(0) @binding(3) var<storage, read_write> indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> indirect_draws: array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read> work_rows: array<GpuVoxelMeshWork>;
@group(0) @binding(6) var<storage, read> packed_tri_table: array<u32>;
@group(0) @binding(7) var<storage, read> volumes: array<GpuVoxelVolume>;
@group(0) @binding(8) var<uniform> params: ExtractParams;

var<workgroup> wg_vertex_count: atomic<u32>;
var<workgroup> wg_index_count: atomic<u32>;

fn triangle_edge(words: vec2<u32>, slot: u32) -> u32 {
    let word = select(words.x, words.y, slot >= 8u);
    return (word >> ((slot % 8u) * 4u)) & 0xFu;
}

fn edge_vertex(edge: u32) -> vec3<f32> {
    let edge_mid = array<vec3<f32>, 12>(
        vec3<f32>(0.5, 0.0, 0.0),
        vec3<f32>(1.0, 0.5, 0.0),
        vec3<f32>(0.5, 1.0, 0.0),
        vec3<f32>(0.0, 0.5, 0.0),
        vec3<f32>(0.5, 0.0, 1.0),
        vec3<f32>(1.0, 0.5, 1.0),
        vec3<f32>(0.5, 1.0, 1.0),
        vec3<f32>(0.0, 0.5, 1.0),
        vec3<f32>(0.0, 0.0, 0.5),
        vec3<f32>(1.0, 0.0, 0.5),
        vec3<f32>(1.0, 1.0, 0.5),
        vec3<f32>(0.0, 1.0, 0.5),
    );
    return edge_mid[edge];
}

fn read_volume_voxel(vol: GpuVoxelVolume, coordinate: vec3<i32>) -> u32 {
    if any(coordinate < vec3<i32>(0)) || any(coordinate >= vec3<i32>(vol.dimensions)) {
        return 0u;
    }
    let brick_coordinate = coordinate / 8;
    let grid_dim = i32(vol.brick_grid_dim);
    let local_brick = u32(
        brick_coordinate.z * grid_dim * grid_dim
        + brick_coordinate.y * grid_dim
        + brick_coordinate.x
    );
    let absolute_brick = vol.brick_offset + local_brick;
    let meta_word = brick_meta[absolute_brick * 2u];
    let occupancy = meta_word >> 24u;
    if occupancy == 0u {
        return 0u;
    }
    let data_offset = meta_word & 0x00FFFFFFu;
    let local = coordinate % 8;
    let linear = u32(local.z * 64 + local.y * 8 + local.x);
    let word = voxel_data[data_offset + linear / 4u];
    return (word >> ((linear % 4u) * 8u)) & 0xFFu;
}

fn occupancy_field(vol: GpuVoxelVolume, coordinate: vec3<i32>) -> f32 {
    return select(-1.0, 1.0, read_volume_voxel(vol, coordinate) > 0u);
}

fn compute_normal(vol: GpuVoxelVolume, coordinate: vec3<i32>) -> vec3<f32> {
    let gradient = vec3<f32>(
        occupancy_field(vol, coordinate + vec3<i32>(1, 0, 0))
            - occupancy_field(vol, coordinate - vec3<i32>(1, 0, 0)),
        occupancy_field(vol, coordinate + vec3<i32>(0, 1, 0))
            - occupancy_field(vol, coordinate - vec3<i32>(0, 1, 0)),
        occupancy_field(vol, coordinate + vec3<i32>(0, 0, 1))
            - occupancy_field(vol, coordinate - vec3<i32>(0, 0, 1)),
    );
    let magnitude_squared = dot(gradient, gradient);
    return select(
        vec3<f32>(0.0, 1.0, 0.0),
        gradient * inverseSqrt(max(magnitude_squared, 0.000001)),
        magnitude_squared >= 0.000001,
    );
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if wg_id.x >= params.work_count {
        return;
    }
    let work = work_rows[wg_id.x];
    if params.bootstrap == 0u && work.generation != params.generation {
        return;
    }
    if (work.flags & WORK_ALLOCATED) == 0u {
        if lid.x == 0u {
            indirect_draws[wg_id.x] = DrawIndexedIndirect(0u, 0u, 0u, 0, 0u);
        }
        return;
    }

    let vol = volumes[work.volume_row];
    let plane = vol.brick_grid_dim * vol.brick_grid_dim;
    let brick_coordinate = vec3<u32>(
        work.local_brick % vol.brick_grid_dim,
        (work.local_brick / vol.brick_grid_dim) % vol.brick_grid_dim,
        work.local_brick / plane,
    );
    let brick_voxel_origin = vec3<i32>(brick_coordinate * 8u);

    if lid.x == 0u {
        atomicStore(&wg_vertex_count, 0u);
        atomicStore(&wg_index_count, 0u);
    }
    workgroupBarrier();

    let thread_first = lid.x * CELLS_PER_THREAD;
    let thread_last = min(thread_first + CELLS_PER_THREAD, TOTAL_CELLS);
    for (var cell_linear = thread_first; cell_linear < thread_last; cell_linear++) {
        let cell = vec3<i32>(
            i32(cell_linear % CELLS_PER_DIM),
            i32((cell_linear / CELLS_PER_DIM) % CELLS_PER_DIM),
            i32(cell_linear / (CELLS_PER_DIM * CELLS_PER_DIM)),
        );
        let global_cell = brick_voxel_origin + cell;

        var corner: array<u32, 8>;
        corner[0] = read_volume_voxel(vol, global_cell + vec3<i32>(0, 0, 0));
        corner[1] = read_volume_voxel(vol, global_cell + vec3<i32>(1, 0, 0));
        corner[2] = read_volume_voxel(vol, global_cell + vec3<i32>(1, 1, 0));
        corner[3] = read_volume_voxel(vol, global_cell + vec3<i32>(0, 1, 0));
        corner[4] = read_volume_voxel(vol, global_cell + vec3<i32>(0, 0, 1));
        corner[5] = read_volume_voxel(vol, global_cell + vec3<i32>(1, 0, 1));
        corner[6] = read_volume_voxel(vol, global_cell + vec3<i32>(1, 1, 1));
        corner[7] = read_volume_voxel(vol, global_cell + vec3<i32>(0, 1, 1));

        var cube_index = 0u;
        for (var corner_index = 0u; corner_index < 8u; corner_index++) {
            if corner[corner_index] != 0u {
                cube_index |= 1u << corner_index;
            }
        }
        if cube_index == 0u || cube_index == 0xFFu {
            continue;
        }

        let table_offset = cube_index * 2u;
        let triangle_words = vec2<u32>(
            packed_tri_table[table_offset],
            packed_tri_table[table_offset + 1u],
        );
        var index_count = 0u;
        for (var table_index = 0u; table_index < 15u; table_index++) {
            if triangle_edge(triangle_words, table_index) == 0xFu {
                break;
            }
            index_count++;
        }
        if index_count == 0u {
            continue;
        }

        let vertex_base = atomicAdd(&wg_vertex_count, index_count);
        let index_base = atomicAdd(&wg_index_count, index_count);
        if vertex_base + index_count > MAX_VERTS || index_base + index_count > MAX_INDICES {
            continue;
        }

        var material = 0u;
        for (var corner_index = 0u; corner_index < 8u; corner_index++) {
            if corner[corner_index] != 0u {
                material = corner[corner_index];
                break;
            }
        }
        let volume_half = vec3<f32>(vol.dimensions) * vol.voxel_size * 0.5;
        let output_vertex_base = wg_id.x * MAX_VERTS;
        let output_index_base = wg_id.x * MAX_INDICES;
        for (var table_index = 0u; table_index < index_count; table_index++) {
            let edge = triangle_edge(triangle_words, table_index);
            if edge >= 12u {
                continue;
            }
            let edge_position = edge_vertex(edge);
            let local_position =
                (vec3<f32>(global_cell) + edge_position) * vol.voxel_size - volume_half;
            let normal_coordinate = global_cell + vec3<i32>(round(edge_position));
            let normal = compute_normal(vol, normal_coordinate);
            vertices[output_vertex_base + vertex_base + table_index] = GpuVoxelMeshVertex(
                vec4<f32>(local_position, f32(material)),
                vec4<f32>(normal, bitcast<f32>(work.volume_row)),
            );
            indices[output_index_base + index_base + table_index] = vertex_base + table_index;
        }
    }

    workgroupBarrier();
    if lid.x == 0u {
        let vertex_count = min(atomicLoad(&wg_vertex_count), MAX_VERTS);
        let index_count = min(atomicLoad(&wg_index_count), MAX_INDICES);
        let has_geometry = select(0u, 1u, vertex_count > 0u && index_count > 0u);
        indirect_draws[wg_id.x] = DrawIndexedIndirect(
            index_count,
            has_geometry,
            wg_id.x * MAX_INDICES,
            i32(wg_id.x * MAX_VERTS),
            0u,
        );
    }
}
