//! CPU terrain evaluation matching the SDF WGSL implementation.
//!
//! This lives beside the SceneDB authority so authored SDF queries do not
//! require, or acquire a back-reference from, Helio's renderer-owned pass.

use glam::Vec3;

use crate::{TerrainConfig, TerrainStyle};

fn hash3(px: f32, py: f32, pz: f32) -> f32 {
    let mut qx = (px * 0.3183099 + 0.1).fract();
    let mut qy = (py * 0.3183099 + 0.1).fract();
    let mut qz = (pz * 0.3183099 + 0.1).fract();
    if qx < 0.0 {
        qx += 1.0;
    }
    if qy < 0.0 {
        qy += 1.0;
    }
    if qz < 0.0 {
        qz += 1.0;
    }
    qx *= 17.0;
    qy *= 17.0;
    qz *= 17.0;
    let result = (qx * qy * qz * (qx + qy + qz)).fract();
    if result < 0.0 {
        result + 1.0
    } else {
        result
    }
}

/// Three-dimensional value noise with the shader's quintic interpolation.
pub fn noise3(px: f32, py: f32, pz: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let iz = pz.floor();
    let fx = px - ix;
    let fy = py - iy;
    let fz = pz - iz;
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0);
    let a = hash3(ix, iy, iz);
    let b = hash3(ix + 1.0, iy, iz);
    let c = hash3(ix, iy + 1.0, iz);
    let d = hash3(ix + 1.0, iy + 1.0, iz);
    let e = hash3(ix, iy, iz + 1.0);
    let f = hash3(ix + 1.0, iy, iz + 1.0);
    let g = hash3(ix, iy + 1.0, iz + 1.0);
    let h = hash3(ix + 1.0, iy + 1.0, iz + 1.0);
    let lerp = |left: f32, right: f32, t: f32| left + (right - left) * t;
    let value = lerp(
        lerp(lerp(a, b, ux), lerp(c, d, ux), uy),
        lerp(lerp(e, f, ux), lerp(g, h, ux), uy),
        uz,
    );
    value * 2.0 - 1.0
}

fn fbm_rotate(px: f32, py: f32, pz: f32, lacunarity: f32) -> (f32, f32, f32) {
    (
        lacunarity * (0.80 * py + 0.60 * pz),
        lacunarity * (-0.80 * px + 0.36 * py - 0.48 * pz),
        lacunarity * (-0.60 * px - 0.48 * py + 0.64 * pz),
    )
}

/// Two-dimensional FBM sampled on the value-noise Y=0 plane.
pub fn fbm2(
    x: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut maximum_amplitude = 0.0;
    let mut sample = (x, 0.0, z);
    for _ in 0..octaves {
        value += amplitude * noise3(sample.0, sample.1, sample.2);
        maximum_amplitude += amplitude;
        amplitude *= persistence;
        sample = fbm_rotate(sample.0, sample.1, sample.2, lacunarity);
    }
    value / maximum_amplitude
}

/// Two-layer domain-warped FBM. Returns `(warped_x, warped_z, value)`.
pub fn warped_fbm3(
    x: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    warp_amount: f32,
) -> (f32, f32, f32) {
    let sample = |x, z| fbm2(x, z, octaves, lacunarity, persistence);
    let warp1 = (sample(x, z), sample(x + 5.2, z + 1.3));
    let warped1 = (x + warp_amount * warp1.0, z + warp_amount * warp1.1);
    let warp2 = (
        sample(warped1.0, warped1.1),
        sample(warped1.0 + 1.7, warped1.1 + 9.2),
    );
    let warped2 = (x + warp_amount * warp2.0, z + warp_amount * warp2.1);
    (warped2.0, warped2.1, sample(warped2.0, warped2.1))
}

fn terrain_height_at(x: f32, z: f32, config: &TerrainConfig) -> f32 {
    let fx = x * config.frequency;
    let fz = z * config.frequency;
    let noise_height = match config.style {
        TerrainStyle::Rolling => fbm2(
            fx,
            fz,
            config.octaves,
            config.lacunarity,
            config.persistence,
        ) * config.amplitude,
        TerrainStyle::Mountains => fbm2(
            fx * 1.5,
            fz * 1.5,
            config.octaves,
            config.lacunarity,
            config.persistence,
        ) * config.amplitude
            * 1.3,
        TerrainStyle::Canyons => {
            let base = fbm2(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            let detail = fbm2(fx * 3.0, fz * 3.0, 3, config.lacunarity, 0.4);
            base * config.amplitude + detail * 3.0
        }
        TerrainStyle::Dunes => fbm2(
            fx * 3.0,
            fz,
            config.octaves,
            config.lacunarity,
            config.persistence,
        ) * config.amplitude,
        TerrainStyle::Warped => warped_fbm3(
            fx,
            fz,
            config.octaves,
            config.lacunarity,
            config.persistence,
            config.warp_amount,
        )
        .2 * config.amplitude,
    };
    config.height + noise_height
}

/// Evaluate the configured procedural terrain SDF at a world position.
pub fn terrain_sdf(position: Vec3, config: &TerrainConfig) -> f32 {
    position.y - terrain_height_at(position.x, position.z, config)
}

/// Compatibility name retained for callers that explicitly select a style.
pub fn terrain_sdf_styled(position: Vec3, config: &TerrainConfig) -> f32 {
    terrain_sdf(position, config)
}

/// Conservative sampled terrain-height range over one brick's XZ footprint.
pub fn terrain_height_range(
    brick_min: Vec3,
    brick_max: Vec3,
    config: &TerrainConfig,
) -> (f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for z_index in 0..3u32 {
        let z = brick_min.z + (brick_max.z - brick_min.z) * z_index as f32 * 0.5;
        for x_index in 0..3u32 {
            let x = brick_min.x + (brick_max.x - brick_min.x) * x_index as f32 * 0.5;
            let height = terrain_height_at(x, z, config);
            minimum = minimum.min(height);
            maximum = maximum.max(height);
        }
    }
    let margin = config.amplitude * config.frequency * (brick_max.x - brick_min.x);
    (minimum - margin, maximum + margin)
}
