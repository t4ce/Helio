//! CPU-side procedural voxel world + brick baker for `voxel_demo`.
//!
//! Rendering goes through `VoxelMeshPass` (mesh mode): this module owns the
//! voxel material data on the CPU, generates it procedurally, bakes each
//! brick into `VoxelMeshPass`'s `brick_meta_buf`/`voxel_data_buf` layout, and
//! the caller calls `VoxelMeshPass::mark_dirty()` per touched brick so its
//! marching-cubes compute pass re-extracts a real triangle mesh — rendered
//! through the normal rasterization pipeline instead of a per-frame raymarch.
//!
//! The grid is always a dense 64^3 voxel volume (8 bricks per axis of 8
//! voxels each, matching `VOXEL_MESH_BRICK_VOXEL_WORDS`/`BRICK_SIZE`).

use std::sync::Arc;

pub const BRICK_DIM: u32 = 8;
pub const BRICKS_PER_AXIS: u32 = 8;
pub const GRID_DIM: u32 = BRICKS_PER_AXIS * BRICK_DIM; // 64
pub const BRICK_COUNT: usize = (BRICKS_PER_AXIS * BRICKS_PER_AXIS * BRICKS_PER_AXIS) as usize;
pub const VOXELS_PER_BRICK: usize = (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize; // 512
pub const WORDS_PER_BRICK: usize = VOXELS_PER_BRICK / 4; // 128

// ── materials ───────────────────────────────────────────────────────────────

pub const MAT_AIR: u8 = 0;
pub const MAT_GRASS: u8 = 1;
pub const MAT_DIRT: u8 = 2;
pub const MAT_STONE: u8 = 3;
pub const MAT_ORE: u8 = 4;

// ── cheap deterministic value noise (no external crate needed) ─────────────

fn hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((y as u32).wrapping_mul(668265263))
        .wrapping_add((z as u32).wrapping_mul(2654435761))
        .wrapping_add(seed.wrapping_mul(2246822519));
    h = (h ^ (h >> 15)).wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(3266489917);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn value_noise2(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let sx = smoothstep(x - x0 as f32);
    let sz = smoothstep(z - z0 as f32);
    let n00 = hash(x0, 0, z0, seed);
    let n10 = hash(x0 + 1, 0, z0, seed);
    let n01 = hash(x0, 0, z0 + 1, seed);
    let n11 = hash(x0 + 1, 0, z0 + 1, seed);
    lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sz)
}

fn fbm2(x: f32, z: f32, seed: u32, octaves: u32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for i in 0..octaves {
        sum += value_noise2(x * freq, z * freq, seed.wrapping_add(i * 101)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

fn value_noise3(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let sx = smoothstep(x - x0 as f32);
    let sy = smoothstep(y - y0 as f32);
    let sz = smoothstep(z - z0 as f32);
    let c000 = hash(x0, y0, z0, seed);
    let c100 = hash(x0 + 1, y0, z0, seed);
    let c010 = hash(x0, y0 + 1, z0, seed);
    let c110 = hash(x0 + 1, y0 + 1, z0, seed);
    let c001 = hash(x0, y0, z0 + 1, seed);
    let c101 = hash(x0 + 1, y0, z0 + 1, seed);
    let c011 = hash(x0, y0 + 1, z0 + 1, seed);
    let c111 = hash(x0 + 1, y0 + 1, z0 + 1, seed);
    let x00 = lerp(c000, c100, sx);
    let x10 = lerp(c010, c110, sx);
    let x01 = lerp(c001, c101, sx);
    let x11 = lerp(c011, c111, sx);
    let y0v = lerp(x00, x10, sy);
    let y1v = lerp(x01, x11, sy);
    lerp(y0v, y1v, sz)
}

fn fbm3(x: f32, y: f32, z: f32, seed: u32, octaves: u32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for i in 0..octaves {
        sum += value_noise3(x * freq, y * freq, z * freq, seed.wrapping_add(i * 71)) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

// ── world ────────────────────────────────────────────────────────────────────

/// Dense 64^3 voxel material grid, baked into GPU bricks on demand.
pub struct VoxelWorld {
    materials: Vec<u8>,
}

impl VoxelWorld {
    pub fn empty() -> Self {
        Self {
            materials: vec![MAT_AIR; (GRID_DIM * GRID_DIM * GRID_DIM) as usize],
        }
    }

    fn idx(x: u32, y: u32, z: u32) -> usize {
        (x + y * GRID_DIM + z * GRID_DIM * GRID_DIM) as usize
    }

    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0 && y >= 0 && z >= 0 && (x as u32) < GRID_DIM && (y as u32) < GRID_DIM && (z as u32) < GRID_DIM
    }

    /// Fills the grid with procedurally generated hills, dirt/stone layers, caves and ore.
    pub fn generate(&mut self, seed: u32) {
        let base_height = GRID_DIM as f32 * 0.45;
        let amplitude = GRID_DIM as f32 * 0.22;
        let freq = 1.0 / 18.0;

        for x in 0..GRID_DIM {
            for z in 0..GRID_DIM {
                let h = fbm2(x as f32 * freq, z as f32 * freq, seed, 4);
                let terrain_height = base_height + h * amplitude;

                for y in 0..GRID_DIM {
                    let yf = y as f32;
                    if yf > terrain_height {
                        self.materials[Self::idx(x, y, z)] = MAT_AIR;
                        continue;
                    }

                    let depth = terrain_height - yf;
                    let mut mat = if depth < 1.0 {
                        MAT_GRASS
                    } else if depth < 4.0 {
                        MAT_DIRT
                    } else {
                        MAT_STONE
                    };

                    if mat == MAT_STONE {
                        let cave = fbm3(
                            x as f32 * 0.12,
                            y as f32 * 0.12,
                            z as f32 * 0.12,
                            seed ^ 0x9E3779B9,
                            3,
                        );
                        if cave > 0.42 && depth > 5.0 {
                            mat = MAT_AIR;
                        } else if hash(x as i32, y as i32, z as i32, seed ^ 0x1234_5678) > 0.985 {
                            mat = MAT_ORE;
                        }
                    }

                    self.materials[Self::idx(x, y, z)] = mat;
                }
            }
        }
    }

    /// Applies a sphere edit (add fills with `material`, subtract clears to air) in
    /// voxel-grid coordinates. Returns the touched region's brick range for partial rebaking.
    pub fn paint_sphere(&mut self, center: [f32; 3], radius: f32, material: u8, add: bool) -> Option<BrickRange> {
        let r = radius.ceil() as i32;
        let cx = center[0].floor() as i32;
        let cy = center[1].floor() as i32;
        let cz = center[2].floor() as i32;
        let r2 = radius * radius;

        let mut touched = false;
        let mut min = [GRID_DIM as i32; 3];
        let mut max = [-1i32; 3];

        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let d2 = (dx * dx + dy * dy + dz * dz) as f32;
                    if d2 > r2 {
                        continue;
                    }
                    let (x, y, z) = (cx + dx, cy + dy, cz + dz);
                    if !Self::in_bounds(x, y, z) {
                        continue;
                    }
                    self.materials[Self::idx(x as u32, y as u32, z as u32)] =
                        if add { material } else { MAT_AIR };
                    touched = true;
                    min[0] = min[0].min(x);
                    min[1] = min[1].min(y);
                    min[2] = min[2].min(z);
                    max[0] = max[0].max(x);
                    max[1] = max[1].max(y);
                    max[2] = max[2].max(z);
                }
            }
        }

        if !touched {
            return None;
        }
        Some(BrickRange {
            min: [
                (min[0] as u32) / BRICK_DIM,
                (min[1] as u32) / BRICK_DIM,
                (min[2] as u32) / BRICK_DIM,
            ],
            max: [
                (max[0] as u32) / BRICK_DIM,
                (max[1] as u32) / BRICK_DIM,
                (max[2] as u32) / BRICK_DIM,
            ],
        })
    }

    fn bake_brick(&self, bx: u32, by: u32, bz: u32, data_out: &mut [u32; WORDS_PER_BRICK]) -> bool {
        let mut occupied = false;
        for lz in 0..BRICK_DIM {
            for ly in 0..BRICK_DIM {
                for lx in 0..BRICK_DIM {
                    let gx = bx * BRICK_DIM + lx;
                    let gy = by * BRICK_DIM + ly;
                    let gz = bz * BRICK_DIM + lz;
                    let mat = self.materials[Self::idx(gx, gy, gz)];
                    if mat != MAT_AIR {
                        occupied = true;
                    }
                    // Matches voxel_raymarch.wgsl::read_voxel: linear = z*64 + y*8 + x.
                    let linear = (lz * BRICK_DIM * BRICK_DIM + ly * BRICK_DIM + lx) as usize;
                    let word = linear / 4;
                    let byte_in_word = linear % 4;
                    data_out[word] |= (mat as u32) << (byte_in_word * 8);
                }
            }
        }
        occupied
    }

    /// World-space origin of a brick's local (0,0,0) voxel corner — the value
    /// `VoxelMeshPass::mark_dirty` needs so its extract shader can place
    /// generated vertices in world space (see `voxel_surface_extract.wgsl`).
    fn brick_origin(bx: u32, by: u32, bz: u32, voxel_size: f32) -> [f32; 3] {
        let half = GRID_DIM as f32 / 2.0;
        let gx = (bx * BRICK_DIM) as f32 - half;
        let gy = (by * BRICK_DIM) as f32 - half;
        let gz = (bz * BRICK_DIM) as f32 - half;
        [gx * voxel_size, gy * voxel_size, gz * voxel_size]
    }

    /// Re-bakes and uploads the bricks touched by a `BrickRange` into
    /// `VoxelMeshPass`'s buffers. Returns `(brick_idx, origin)` for every
    /// touched brick (occupied or not) — the caller must `mark_dirty()` each
    /// one so the extract pass re-runs (an emptied brick needs to re-extract
    /// to zero triangles too, not just a newly-filled one).
    pub fn upload_range_mesh(
        &self,
        queue: &Arc<wgpu::Queue>,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
        voxel_size: f32,
        range: BrickRange,
    ) -> Vec<(u32, [f32; 3])> {
        let mut touched = Vec::new();
        for bz in range.min[2]..=range.max[2] {
            for by in range.min[1]..=range.max[1] {
                for bx in range.min[0]..=range.max[0] {
                    let brick_idx = bz * BRICKS_PER_AXIS * BRICKS_PER_AXIS + by * BRICKS_PER_AXIS + bx;
                    let mut brick_words = [0u32; WORDS_PER_BRICK];
                    let occupied = self.bake_brick(bx, by, bz, &mut brick_words);

                    let data_offset = brick_idx * WORDS_PER_BRICK as u32;
                    // VoxelMeshPass's GpuBrickMeta is two plain u32 fields
                    // (data_offset, occupancy) — unlike VoxelRayMarchPass's
                    // packed single word, see voxel_surface_extract.wgsl.
                    let meta = [data_offset, occupied as u32];

                    queue.write_buffer(
                        brick_meta_buf,
                        (brick_idx as u64) * 8,
                        bytemuck::cast_slice(&meta),
                    );
                    queue.write_buffer(
                        voxel_data_buf,
                        (data_offset as u64) * 4,
                        bytemuck::cast_slice(&brick_words),
                    );

                    touched.push((brick_idx, Self::brick_origin(bx, by, bz, voxel_size)));
                }
            }
        }
        touched
    }

    /// Bakes and uploads every brick in the volume. See `upload_range_mesh`.
    pub fn upload_all_mesh(
        &self,
        queue: &Arc<wgpu::Queue>,
        brick_meta_buf: &wgpu::Buffer,
        voxel_data_buf: &wgpu::Buffer,
        voxel_size: f32,
    ) -> Vec<(u32, [f32; 3])> {
        self.upload_range_mesh(
            queue,
            brick_meta_buf,
            voxel_data_buf,
            voxel_size,
            BrickRange {
                min: [0, 0, 0],
                max: [BRICKS_PER_AXIS - 1, BRICKS_PER_AXIS - 1, BRICKS_PER_AXIS - 1],
            },
        )
    }
}

#[derive(Clone, Copy)]
pub struct BrickRange {
    min: [u32; 3],
    max: [u32; 3],
}
