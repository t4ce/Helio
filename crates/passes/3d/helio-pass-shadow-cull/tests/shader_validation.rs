fn validate(label: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{label} must parse as WGSL: {error}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("{label} must validate: {error}"));
}

#[test]
fn shadow_cull_shader_parses_and_validates() {
    validate(
        "shadow cull shader",
        include_str!("../shaders/shadow_cull.wgsl"),
    );
}
