// update.frag.wgsl — one step of the shallow-water wave propagation.
//
// Texture layout (Rgba16Float):
//   R = height
//   G = velocity
//   B = normal.x  (written by normal pass)
//   A = normal.z  (written by normal pass)

@group(0) @binding(0) var water_texture: texture_2d<f32>;
@group(0) @binding(1) var water_sampler: sampler;

struct UpdateUniforms {
    /// Texel size: (1 / texture_width, 1 / texture_height)
    delta: vec2<f32>,
    /// Pass-owned elapsed time. Authored speed remains in the canonical row.
    time: f32,
    /// Duration of one of the frame's two simulation substeps.
    time_step: f32,
    /// Patch size (metres per tile) for this cascade — scales wind wavenumbers
    /// so smaller patches produce shorter wavelengths (choppy) and larger patches
    /// produce long swells.
    cascade_patch_size: f32,
    /// Component-local SceneDB row assigned to this stable simulation slot.
    volume_row: u32,
    _pad: vec2<u32>,
}
@group(0) @binding(2) var<uniform> u: UpdateUniforms;

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    sim_dynamics:          vec4f,
    wind_params:           vec4f,
    _pad6:                 vec4f,
}

/// Direct SceneDB partner buffer. No pass-global spring/wind copy exists.
@group(0) @binding(3) var<storage, read> water_volumes: array<WaterVolume>;

// ---------------------------------------------------------------------------
// Wind: traveling sinusoidal wave trains, simplified JONSWAP-inspired spectrum.
//
// Each octave is a plane wave W(uv, t) = sin(dot(uv, dir) * k - omega * t).
// We inject the delta  W(t_old) - W(t_new)  directly into info.r, identical to
// the hitbox.frag.wgsl sign convention.  This is the wave's own time-derivative
// ( ~= omega * dt * cos(...) ), which the SWE spring propagates into radiating
// rings.  Spatial mean of each sin() term is 0 -- no DC height drift.
//
// Octave spread: primary swell in wind direction; secondary at +18 deg; cross-
// chop at -30 deg; short ripples at +50 deg.  Amplitudes follow ~1/n^1.5 to
// match the high-frequency roll-off of a real ocean spectrum.
// ---------------------------------------------------------------------------
fn twave(uv: vec2<f32>, t: f32, k: f32, omega: f32, dir: vec2<f32>) -> f32 {
    return sin(dot(uv, dir) * k - omega * t);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var info = textureSample(water_texture, water_sampler, uv);
    let volume = water_volumes[u.volume_row];
    let spring = clamp(volume.sim_dynamics.x, 0.1, 2.0);
    let damping = clamp(volume.sim_dynamics.y, 0.0, 1.0);
    let wave_scale = max(volume.sim_dynamics.z, 0.01);
    let wind_strength = max(volume.wind_params.z, 0.0);
    let wind_len_sq = dot(volume.wind_params.xy, volume.wind_params.xy);
    var wind_dir = vec2<f32>(0.0);
    if wind_len_sq > 1e-12 {
        wind_dir = volume.wind_params.xy * inverseSqrt(wind_len_sq);
    }
    let wave_speed = max(volume.wave_params.z, 0.0);
    let wave_time = u.time * wave_speed;
    let wave_time_step = u.time_step * wave_speed;

    let dx = vec2<f32>(u.delta.x, 0.0);
    let dy = vec2<f32>(0.0, u.delta.y);

    // Average of the four cardinal neighbours' heights
    let avg = (
        textureSample(water_texture, water_sampler, uv - dx).r +
        textureSample(water_texture, water_sampler, uv - dy).r +
        textureSample(water_texture, water_sampler, uv + dx).r +
        textureSample(water_texture, water_sampler, uv + dy).r
    ) * 0.25;

    // Velocity = displacement toward mean (spring) + energy damping
    info.g += (avg - info.r) * spring;
    info.g *= damping;
    // Euler-integrate height
    info.r += info.g;

    // Traveling wave injection -- only when wind is active and normalised.
    if wind_strength > 0.001 && dot(wind_dir, wind_dir) > 0.5 {
        let perp   = vec2<f32>(-wind_dir.y, wind_dir.x);
        // Base wavenumber from cascade patch size — smaller patch → shorter
        // wavelengths (choppy sea), larger patch → long swells.
        // wave_scale acts as a global multiplier on top of the cascade's scale.
        let inv_ws = 1.0 / wave_scale;
        let k_base = 6.2832 / max(u.cascade_patch_size, 0.1);
        let t_old  = wave_time - wave_time_step;

        var dh = 0.0;

        // Octave 0 -- primary swell, strict wind direction (50% of energy)
        let k0 = k_base * 1.5 * inv_ws;
        dh += (twave(uv, t_old, k0, 0.65, wind_dir) -
               twave(uv, wave_time, k0, 0.65, wind_dir)) * 0.50;

        // Octave 1 -- secondary swell +18 deg off wind
        let d1 = normalize(wind_dir + perp * 0.3249);   // tan(18 deg)
        let k1 = k_base * 2.8 * inv_ws;
        dh += (twave(uv, t_old,  k1, 1.10, d1) -
               twave(uv, wave_time, k1, 1.10, d1)) * 0.28;

        // Octave 2 -- cross-chop -30 deg
        let d2 = normalize(wind_dir - perp * 0.5774);   // tan(30 deg)
        let k2 = k_base * 5.3 * inv_ws;
        dh += (twave(uv, t_old,  k2, 2.00, d2) -
               twave(uv, wave_time, k2, 2.00, d2)) * 0.14;

        // Octave 3 -- short ripples +50 deg
        let d3 = normalize(wind_dir + perp * 1.1918);   // tan(50 deg)
        let k3 = k_base * 9.5 * inv_ws;
        dh += (twave(uv, t_old,  k3, 3.60, d3) -
               twave(uv, wave_time, k3, 3.60, d3)) * 0.08;

        info.r += dh * wind_strength * 0.05;
    }

    return info;
}
