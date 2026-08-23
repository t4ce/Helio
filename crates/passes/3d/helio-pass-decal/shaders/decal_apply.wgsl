//! Decal apply: reads temp textures, writes final values to GBuffer via storage.

@group(0) @binding(3) var temp_albedo: texture_2d<f32>;
@group(0) @binding(4) var temp_normal: texture_2d<f32>;
@group(0) @binding(5) var temp_orm: texture_2d<f32>;
@group(0) @binding(6) var temp_emissive: texture_2d<f32>;

@group(0) @binding(7)  var gbuf_albedo:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8)  var gbuf_normal:  texture_storage_2d<rgba16float, write>;
@group(0) @binding(9)  var gbuf_orm:     texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var gbuf_emissive: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pxl = vec2<i32>(id.xy);
    let sz = textureDimensions(temp_albedo);
    if id.x >= u32(sz.x) || id.y >= u32(sz.y) { return; }

    let a = textureLoad(temp_albedo, pxl, 0);
    let n = textureLoad(temp_normal, pxl, 0);
    let o = textureLoad(temp_orm, pxl, 0);
    let e = textureLoad(temp_emissive, pxl, 0);

    textureStore(gbuf_albedo, pxl, a);
    textureStore(gbuf_normal, pxl, n);
    textureStore(gbuf_orm, pxl, o);
    textureStore(gbuf_emissive, pxl, e);
}
