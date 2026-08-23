#[test]
fn shadow_dirty_shader_parses_and_validates() {
    let source = include_str!("../shaders/shadow_dirty.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("shadow dirty shader must parse as WGSL: {error}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("shadow dirty shader must validate: {error}"));
}
