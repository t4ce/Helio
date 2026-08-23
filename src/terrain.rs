//! CPU-side voxel terrain component and brick baker.
//!
//! Both Auto mesh extraction and Dynamic ray marching read the same canonical
//! SceneDB `VoxelResidency` representation. A caller uploads through
//! `Scene::upload_voxel_terrain(_range)`, which stores each brick as a raw
//! 8x8x8 block. Mesh extraction samples neighbouring canonical bricks for its
//! halo; there is no padded duplicate or pass-owned input authority.
//!
//! The grid is always a dense 64^3 voxel volume (8 bricks per axis of 8
//! voxels each — fixed GPU-side by the engine's `BRICK_SIZE` constant).

pub const BRICK_DIM: u32 = helio_voxel_core::BRICK_SIZE;
pub const BRICKS_PER_AXIS: u32 = helio_voxel_core::DEFAULT_VOLUME_BRICKS_PER_AXIS;
pub const VOXEL_TERRAIN_GRID_DIM: u32 = BRICKS_PER_AXIS * BRICK_DIM; // 64
pub const RAW_WORDS_PER_BRICK: usize =
    helio_voxel_core::RAYMARCH_WORDS_PER_BRICK as usize;

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

// ── world ────────────────────────────────────────────────────────────────────

/// Dense 64^3 voxel material grid, baked into GPU bricks on demand.
pub struct VoxelTerrain {
    materials: Vec<u8>,
}

impl VoxelTerrain {
    pub fn empty() -> Self {
        Self {
            materials: vec![
                MAT_AIR;
                (VOXEL_TERRAIN_GRID_DIM * VOXEL_TERRAIN_GRID_DIM * VOXEL_TERRAIN_GRID_DIM)
                    as usize
            ],
        }
    }

    fn idx(x: u32, y: u32, z: u32) -> usize {
        (x + y * VOXEL_TERRAIN_GRID_DIM + z * VOXEL_TERRAIN_GRID_DIM * VOXEL_TERRAIN_GRID_DIM)
            as usize
    }

    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < VOXEL_TERRAIN_GRID_DIM
            && (y as u32) < VOXEL_TERRAIN_GRID_DIM
            && (z as u32) < VOXEL_TERRAIN_GRID_DIM
    }

    /// Fills the grid with procedurally generated hills, dirt/stone layers, caves and ore.
    pub fn generate(&mut self, seed: u32) {
        let base_height = VOXEL_TERRAIN_GRID_DIM as f32 * 0.45;
        let amplitude = VOXEL_TERRAIN_GRID_DIM as f32 * 0.22;
        let freq = 1.0 / 18.0;

        for x in 0..VOXEL_TERRAIN_GRID_DIM {
            for z in 0..VOXEL_TERRAIN_GRID_DIM {
                let h = fbm2(x as f32 * freq, z as f32 * freq, seed, 4);
                let terrain_height = base_height + h * amplitude;

                for y in 0..VOXEL_TERRAIN_GRID_DIM {
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

                    // No cave carving here: VoxelMeshPass caps each brick at
                    // MAX_SURFACE_VERTS_PER_BRICK/MAX_SURFACE_INDICES_PER_BRICK
                    // (256/768) — a cave-riddled brick's internal surface area
                    // blows well past that budget and geometry gets silently
                    // truncated mid-brick. A plain heightfield keeps each
                    // brick's surface to roughly one layer of cells.
                    if mat == MAT_STONE
                        && hash(x as i32, y as i32, z as i32, seed ^ 0x1234_5678) > 0.985
                    {
                        mat = MAT_ORE;
                    }

                    self.materials[Self::idx(x, y, z)] = mat;
                }
            }
        }
    }

    /// Applies a sphere edit (add fills with `material`, subtract clears to air) in
    /// voxel-grid coordinates. Returns the touched region's brick range for partial rebaking.
    pub fn paint_sphere(
        &mut self,
        center: [f32; 3],
        radius: f32,
        material: u8,
        add: bool,
    ) -> Option<BrickRange> {
        let r = radius.ceil() as i32;
        let cx = center[0].floor() as i32;
        let cy = center[1].floor() as i32;
        let cz = center[2].floor() as i32;
        let r2 = radius * radius;

        let mut touched = false;
        let mut min = [VOXEL_TERRAIN_GRID_DIM as i32; 3];
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

    /// Bakes a canonical raw 8x8x8 voxel block shared by both voxel passes.
    fn bake_brick(
        &self,
        bx: u32,
        by: u32,
        bz: u32,
        data_out: &mut [u32; RAW_WORDS_PER_BRICK],
    ) -> bool {
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
                    let linear = (lz * BRICK_DIM * BRICK_DIM + ly * BRICK_DIM + lx) as usize;
                    let word = linear / 4;
                    let byte_in_word = linear % 4;
                    data_out[word] |= (mat as u32) << (byte_in_word * 8);
                }
            }
        }
        occupied
    }

    /// Re-bake a range into canonical raw 8x8x8 rows. Only the Scene facade can
    /// invoke this iterator so all writes pass through SceneDB VoxelResidency.
    pub(crate) fn for_each_canonical_brick<E>(
        &self,
        range: BrickRange,
        mut visit: impl FnMut(u32, bool, &[u32]) -> Result<(), E>,
    ) -> Result<(), E> {
        for bz in range.min[2]..=range.max[2] {
            for by in range.min[1]..=range.max[1] {
                for bx in range.min[0]..=range.max[0] {
                    let brick_idx =
                        bz * BRICKS_PER_AXIS * BRICKS_PER_AXIS + by * BRICKS_PER_AXIS + bx;
                    let mut brick_words = [0u32; RAW_WORDS_PER_BRICK];
                    let occupied = self.bake_brick(bx, by, bz, &mut brick_words);

                    visit(brick_idx, occupied, &brick_words)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct BrickRange {
    min: [u32; 3],
    max: [u32; 3],
}

impl BrickRange {
    pub(crate) const fn all() -> Self {
        Self {
            min: [0, 0, 0],
            max: [BRICKS_PER_AXIS - 1; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_terrain_contains_air_and_solid_voxels() {
        let mut terrain = VoxelTerrain::empty();
        terrain.generate(1);

        assert!(terrain.materials.contains(&MAT_AIR));
        assert!(terrain.materials.iter().any(|&material| material != MAT_AIR));
    }

    #[test]
    fn sphere_edits_update_the_dense_material_grid() {
        let mut terrain = VoxelTerrain::empty();
        let center = [32.0, 32.0, 32.0];
        let center_index = VoxelTerrain::idx(32, 32, 32);

        assert!(terrain.paint_sphere(center, 2.0, MAT_ORE, true).is_some());
        assert_eq!(terrain.materials[center_index], MAT_ORE);

        assert!(terrain.paint_sphere(center, 2.0, MAT_ORE, false).is_some());
        assert_eq!(terrain.materials[center_index], MAT_AIR);
    }
}
