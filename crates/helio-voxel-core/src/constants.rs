pub const BRICK_SIZE: u32 = 8;
pub const BRICK_VOXEL_COUNT: usize = (BRICK_SIZE as usize).pow(3);
/// Dense raymarch asset shape used by Helio's current voxel terrain source.
pub const DEFAULT_VOLUME_BRICKS_PER_AXIS: u32 = 8;
pub const DEFAULT_VOLUME_BRICK_COUNT: u32 =
    DEFAULT_VOLUME_BRICKS_PER_AXIS * DEFAULT_VOLUME_BRICKS_PER_AXIS
        * DEFAULT_VOLUME_BRICKS_PER_AXIS;
/// Four packed u8 material values per storage word.
pub const RAYMARCH_WORDS_PER_BRICK: u32 = BRICK_VOXEL_COUNT as u32 / 4;
pub const PADDED_BRICK_SIZE: u32 = BRICK_SIZE + 1;
pub const PADDED_BRICK_VOXEL_COUNT: usize = (PADDED_BRICK_SIZE as usize).pow(3);
pub const MAX_BRICKS_PER_VOLUME: u32 = 262144;
pub const MAX_VOLUMES: u32 = 1024;
pub const EDIT_RING_CAPACITY: u32 = 1024;
pub const MAX_PALETTE_SIZE: u32 = 256;
// voxel_meshlet.wgsl never deduplicates vertices — every generated triangle
// gets 3 fresh vertex_buf slots, so vertex count always equals index count.
// A vert budget lower than the index budget (as this was: 256 vs 768) means
// generation actually stops at the vertex cap, well before the index cap is
// ever reached, silently truncating any brick whose surface needs more than
// ~85 triangles (a heightfield brick with real slope exceeds that easily).
pub const MAX_SURFACE_VERTS_PER_BRICK: u32 = 2048;
pub const MAX_SURFACE_INDICES_PER_BRICK: u32 = 2048;
pub const BRICK_EMPTY: u32 = 0xFFFFFFFF;
pub const BRICK_SOLID: u32 = 0xFFFFFFFE;
