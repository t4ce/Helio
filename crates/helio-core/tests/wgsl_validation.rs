//! Parses and validates every `.wgsl` shader in the workspace.
//!
//! wgpu only compiles a shader at `create_shader_module`, i.e. at runtime with a
//! live device. Nothing in `cargo check`, `cargo build`, or `cargo test` looks at
//! WGSL, so a shader that cannot possibly compile ships silently and only fails
//! on the machine that runs it — or never, if the pass is not wired up.
//!
//! That is not hypothetical: `ssr_denoise.wgsl` sat in-tree unable to compile
//! (it negated a `u32`) for as long as it existed, because `SsrPass` never built
//! a pipeline from it.
//!
//! This test walks the repo rather than taking an explicit list, so a new shader
//! is covered the moment it is added.
//!
//! Sources go through `helio_core::shader::resolve` first — the same call the
//! runtime makes — so a prelude-using shader is validated as the GPU will see it,
//! not as the bare file on disk.

use std::path::{Path, PathBuf};

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn workspace_root() -> PathBuf {
    // crates/helio-core -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("helio-core should live at <root>/crates/helio-core")
        .to_path_buf()
}

fn collect_wgsl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target` holds vendored deps' shaders, which are not ours to police.
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_wgsl(&path, out);
        } else if path.extension().is_some_and(|e| e == "wgsl") {
            out.push(path);
        }
    }
}

/// A standalone shader has at least one entry point. A file with none is a
/// fragment that gets string-concatenated into a host shader before compiling
/// (e.g. `vhs_effects.wgsl`, pulled in as `VHS_SHADER_SNIPPET`), so it refers to
/// bindings it does not declare and cannot validate on its own.
fn is_fragment(source: &str) -> bool {
    !(source.contains("@vertex") || source.contains("@fragment") || source.contains("@compute"))
}

/// Maps a line number in resolved source back to the original file, so a
/// diagnostic in a prelude-using shader points at a line that actually exists in
/// the file the reader will open.
fn describe_location(source: &str, rendered: &str) -> String {
    let prepended = helio_core::shader::expanded_lines(source);
    if prepended > 0 {
        format!(" (expanded: subtract {prepended} lines for the original file)")
    } else {
        String::new()
    }
    .to_string()
        + "\n"
        + rendered
}

#[test]
fn every_wgsl_shader_parses_and_validates() {
    let root = workspace_root();
    let mut shaders = Vec::new();
    collect_wgsl(&root.join("crates"), &mut shaders);
    shaders.sort();

    assert!(
        !shaders.is_empty(),
        "found no .wgsl files under {}; the walk is probably broken, and a test \
         that silently checks nothing is worse than no test",
        root.display()
    );

    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut skipped = Vec::new();

    for path in &shaders {
        let rel = path.strip_prefix(&root).unwrap_or(path);
        let source = std::fs::read_to_string(path).expect("shader should be readable");

        if is_fragment(&source) {
            skipped.push(rel.display().to_string());
            continue;
        }
        checked += 1;

        // Exactly what create_shader_module would receive.
        let resolved = helio_core::shader::resolve(&source);

        let module = match naga::front::wgsl::parse_str(&resolved) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!(
                    "{}:{}",
                    rel.display(),
                    describe_location(&source, &e.emit_to_string(&resolved))
                ));
                continue;
            }
        };

        if let Err(e) = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)
        {
            failures.push(format!(
                "{}:{}",
                rel.display(),
                describe_location(&source, &format!("{e:?}"))
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {checked} standalone shaders failed to compile:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );

    // Guard the skip heuristic with an explicit allow-list. A count threshold
    // silently accepts a new entry-point-less file; an exact list forces its
    // composition role to be reviewed.
    let expected_fragments = [
        "crates/examples/vhs_effects.wgsl",
        "crates/helio/templates/anisotropic.wgsl",
        "crates/helio/templates/clear_coat.wgsl",
        "crates/helio/templates/glass.wgsl",
        "crates/helio/templates/glass_transparent.wgsl",
        "crates/helio/templates/opal.wgsl",
        "crates/helio/templates/skin.wgsl",
        "crates/helio/templates/subsurface.wgsl",
        "crates/helio/templates/water.wgsl",
        "crates/helio/templates/water_transparent.wgsl",
        "crates/helio-core/src/shader/foliage_wind.wgsl",
        "crates/helio-core/src/shader/hiz_trace.wgsl",
        "crates/helio-core/src/shader/prelude.wgsl",
        "crates/helio-planet-voxel-core/src/planet_voxel_layout.wgsl",
        "crates/helio-web-demos/examples-wasm/vhs_effects.wgsl",
        "crates/libhelio/shaders/pbr_eval.wgsl",
        "crates/passes/3d/helio-pass-planetary-voxel/src/extraction_layout.wgsl",
    ];
    assert_eq!(
        skipped,
        expected_fragments,
        "entry-point-less WGSL files must be reviewed and explicitly allow-listed",
    );
}

#[test]
fn every_gpu_light_mirror_matches_the_host_layout() {
    let root = workspace_root();
    let mut shaders = Vec::new();
    collect_wgsl(&root.join("crates"), &mut shaders);
    shaders.sort();

    let expected_offsets = [
        ("position_range", std::mem::offset_of!(libhelio::GpuLight, position_range) as u32),
        ("direction_outer", std::mem::offset_of!(libhelio::GpuLight, direction_outer) as u32),
        ("color_intensity", std::mem::offset_of!(libhelio::GpuLight, color_intensity) as u32),
        ("shadow_index", std::mem::offset_of!(libhelio::GpuLight, shadow_index) as u32),
        ("light_type", std::mem::offset_of!(libhelio::GpuLight, light_type) as u32),
        ("inner_angle", std::mem::offset_of!(libhelio::GpuLight, inner_angle) as u32),
        ("_pad", std::mem::offset_of!(libhelio::GpuLight, _pad) as u32),
        ("god_rays_enabled", std::mem::offset_of!(libhelio::GpuLight, god_rays_enabled) as u32),
        ("god_rays_density", std::mem::offset_of!(libhelio::GpuLight, god_rays_density) as u32),
        ("god_rays_weight", std::mem::offset_of!(libhelio::GpuLight, god_rays_weight) as u32),
        ("god_rays_decay", std::mem::offset_of!(libhelio::GpuLight, god_rays_decay) as u32),
        ("god_rays_exposure", std::mem::offset_of!(libhelio::GpuLight, god_rays_exposure) as u32),
        ("flare_enabled", std::mem::offset_of!(libhelio::GpuLight, flare_enabled) as u32),
        ("flare_type", std::mem::offset_of!(libhelio::GpuLight, flare_type) as u32),
        ("flare_intensity", std::mem::offset_of!(libhelio::GpuLight, flare_intensity) as u32),
        ("flare_scale", std::mem::offset_of!(libhelio::GpuLight, flare_scale) as u32),
        ("flare_tint_r", std::mem::offset_of!(libhelio::GpuLight, flare_tint_r) as u32),
        ("flare_tint_g", std::mem::offset_of!(libhelio::GpuLight, flare_tint_g) as u32),
        ("flare_tint_b", std::mem::offset_of!(libhelio::GpuLight, flare_tint_b) as u32),
        ("ies_profile_index", std::mem::offset_of!(libhelio::GpuLight, ies_profile_index) as u32),
        ("light_function_index", std::mem::offset_of!(libhelio::GpuLight, light_function_index) as u32),
        ("ies_angle_scale", std::mem::offset_of!(libhelio::GpuLight, ies_angle_scale) as u32),
        ("ies_angle_offset", std::mem::offset_of!(libhelio::GpuLight, ies_angle_offset) as u32),
    ];

    let mut checked = Vec::new();
    for path in shaders {
        let source = std::fs::read_to_string(&path).expect("shader should be readable");
        if !source.contains("struct GpuLight") {
            continue;
        }
        let resolved = helio_core::shader::resolve(&source);
        let module = naga::front::wgsl::parse_str(&resolved).unwrap_or_else(|error| {
            panic!("{}: {}", path.display(), error.emit_to_string(&resolved))
        });
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).expect("GpuLight shader layout");
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("GpuLight"))
            .expect("source containing `struct GpuLight` should declare that type");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("GpuLight is not a struct in {}", path.display());
        };

        assert_eq!(
            layouter[handle].size as usize,
            std::mem::size_of::<libhelio::GpuLight>(),
            "GpuLight stride drift in {}",
            path.display()
        );
        let actual_offsets: Vec<_> = members
            .iter()
            .map(|member| (member.name.as_deref().unwrap_or(""), member.offset))
            .collect();
        assert_eq!(actual_offsets, expected_offsets, "GpuLight field drift in {}", path.display());
        checked.push(path);
    }

    assert!(!checked.is_empty(), "found no WGSL GpuLight mirrors");
}
