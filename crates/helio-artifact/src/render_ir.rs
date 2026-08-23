//! Compact normalized render IR used at the Helio/TRUEOS boundary.
//!
//! Version 1 intentionally describes one vertex/index stream, one graphics
//! pipeline, and one indexed draw. It is the minimum honest contract needed
//! by `SimpleCubePass`; all identifiers are artifact-local and no captured
//! wgpu pointer IDs survive lowering.

use alloc::{string::String, vec::Vec};
use core::{fmt, str};

pub const RENDER_IR_SECTION_NAME: &str = "render/ir-v1.bin";
pub const RENDER_IR_MAGIC: [u8; 8] = *b"HELIOIR\0";
pub const RENDER_IR_VERSION: u16 = 1;
pub const RENDER_IR_HEADER_LEN: usize = 256;

const VERTEX_BUFFER_ID: u32 = 1;
const INDEX_BUFFER_ID: u32 = 2;
const CAMERA_BUFFER_ID: u32 = 3;

/// One file emitted by wgpu's trace recorder. Names are relative to its trace
/// directory (for example `trace.ron` or `data3.bin`).
#[derive(Clone, Copy, Debug)]
pub struct TraceFile<'a> {
    pub name: &'a str,
    pub data: &'a [u8],
}

/// Owned, normalized render program. Resource data is embedded when encoded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderIr {
    pub vertex_data: Vec<u8>,
    pub index_data: Vec<u8>,
    pub wgsl: String,
    pub vertex_entry: String,
    pub fragment_entry: String,
    pub camera_dynamic_slot: String,
    pub output_dynamic_slot: String,
    pub pass_label: String,
}

impl RenderIr {
    /// Encodes the fixed little-endian v1 header followed by inline payloads.
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_len = RENDER_IR_HEADER_LEN
            + self.vertex_data.len()
            + self.index_data.len()
            + self.wgsl.len()
            + self.vertex_entry.len()
            + self.fragment_entry.len()
            + self.camera_dynamic_slot.len()
            + self.output_dynamic_slot.len()
            + self.pass_label.len();
        let mut out = alloc::vec![0; total_len];
        out[..8].copy_from_slice(&RENDER_IR_MAGIC);
        put_u16(&mut out, 8, RENDER_IR_VERSION);
        put_u16(&mut out, 10, RENDER_IR_HEADER_LEN as u16);
        put_u32(&mut out, 12, total_len as u32);

        put_u32(&mut out, 20, VERTEX_BUFFER_ID);
        let mut cursor = RENDER_IR_HEADER_LEN;
        put_payload(&mut out, &mut cursor, 24, 28, &self.vertex_data);
        put_u32(&mut out, 32, 36); // array stride

        put_u32(&mut out, 36, INDEX_BUFFER_ID);
        put_payload(&mut out, &mut cursor, 40, 44, &self.index_data);
        put_u32(&mut out, 48, 1); // uint16

        put_u32(&mut out, 52, CAMERA_BUFFER_ID);
        put_u32(&mut out, 56, 192); // WGSL Camera minimum binding size
        put_string16(&mut out, &mut cursor, 60, 64, &self.camera_dynamic_slot);
        put_payload(&mut out, &mut cursor, 68, 72, self.wgsl.as_bytes());
        put_string16(&mut out, &mut cursor, 76, 80, &self.vertex_entry);
        put_string16(&mut out, &mut cursor, 84, 88, &self.fragment_entry);

        put_u32(&mut out, 92, 1); // Bgra8UnormSrgb
        put_u32(&mut out, 96, 1); // Depth32Float
        put_u32(&mut out, 100, 1); // TriangleList
        put_u32(&mut out, 104, 1); // CCW
        put_u32(&mut out, 108, 2); // back-face culling
        put_u32(&mut out, 112, 1); // Less
        put_u32(&mut out, 116, 0b11_1111); // write/store/read-only/clear flags
        put_u32(&mut out, 120, 0xf);
        put_f32(&mut out, 124, 0.01);
        put_f32(&mut out, 128, 0.01);
        put_f32(&mut out, 132, 0.02);
        put_f32(&mut out, 136, 1.0);
        put_f32(&mut out, 140, 1.0);

        put_u32(&mut out, 144, 3);
        for (base, location, offset) in [(148, 0, 0), (160, 1, 12), (172, 2, 24)] {
            put_u32(&mut out, base, location);
            put_u32(&mut out, base + 4, 1); // Float32x3
            put_u32(&mut out, base + 8, offset);
        }

        put_u32(&mut out, 196, 0); // bind group
        put_u32(&mut out, 200, 0); // binding
        put_u32(&mut out, 204, 1); // storage buffer
        put_u32(&mut out, 208, 1); // vertex visibility
        put_u32(&mut out, 212, 36);
        put_u32(&mut out, 216, 1);
        put_u32(&mut out, 220, 0);
        put_i32(&mut out, 224, 0);
        put_u32(&mut out, 228, 0);
        put_string16(&mut out, &mut cursor, 232, 236, &self.output_dynamic_slot);
        put_string16(&mut out, &mut cursor, 240, 244, &self.pass_label);
        debug_assert_eq!(cursor, total_len);
        out
    }
}

/// Bounds-checked borrowed view of a v1 render program.
#[derive(Clone, Copy, Debug)]
pub struct RenderIrRef<'a> {
    bytes: &'a [u8],
}

impl<'a> RenderIrRef<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, RenderIrError> {
        if bytes.len() < RENDER_IR_HEADER_LEN || bytes[..8] != RENDER_IR_MAGIC {
            return Err(RenderIrError::BadMagic);
        }
        if read_u16(bytes, 8)? != RENDER_IR_VERSION {
            return Err(RenderIrError::UnsupportedVersion);
        }
        if read_u16(bytes, 10)? as usize != RENDER_IR_HEADER_LEN
            || read_u32(bytes, 12)? as usize != bytes.len()
        {
            return Err(RenderIrError::Malformed);
        }
        let ir = Self { bytes };
        for (offset_at, len_at, short_len) in [
            (24, 28, false),
            (40, 44, false),
            (68, 72, false),
            (60, 64, true),
            (76, 80, true),
            (84, 88, true),
            (232, 236, true),
            (240, 244, true),
        ] {
            let offset = read_u32(bytes, offset_at)? as usize;
            let len = if short_len {
                read_u16(bytes, len_at)? as usize
            } else {
                read_u32(bytes, len_at)? as usize
            };
            checked_slice(bytes, offset, len)?;
        }
        if ir.vertex_buffer_id() == 0
            || ir.index_buffer_id() == 0
            || ir.camera_buffer_id() == 0
            || read_u32(bytes, 144)? > 3
        {
            return Err(RenderIrError::Malformed);
        }
        // All string fields must be UTF-8.
        ir.wgsl()?;
        ir.vertex_entry()?;
        ir.fragment_entry()?;
        ir.camera_dynamic_slot()?;
        ir.output_dynamic_slot()?;
        ir.pass_label()?;
        Ok(ir)
    }

    pub fn vertex_buffer_id(&self) -> u32 {
        read_u32(self.bytes, 20).unwrap()
    }
    pub fn index_buffer_id(&self) -> u32 {
        read_u32(self.bytes, 36).unwrap()
    }
    pub fn camera_buffer_id(&self) -> u32 {
        read_u32(self.bytes, 52).unwrap()
    }
    pub fn vertex_data(&self) -> &'a [u8] {
        self.long_slice(24, 28).unwrap()
    }
    pub fn index_data(&self) -> &'a [u8] {
        self.long_slice(40, 44).unwrap()
    }
    pub fn wgsl(&self) -> Result<&'a str, RenderIrError> {
        self.long_str(68, 72)
    }
    pub fn vertex_entry(&self) -> Result<&'a str, RenderIrError> {
        self.short_str(76, 80)
    }
    pub fn fragment_entry(&self) -> Result<&'a str, RenderIrError> {
        self.short_str(84, 88)
    }
    pub fn camera_dynamic_slot(&self) -> Result<&'a str, RenderIrError> {
        self.short_str(60, 64)
    }
    pub fn output_dynamic_slot(&self) -> Result<&'a str, RenderIrError> {
        self.short_str(232, 236)
    }
    pub fn pass_label(&self) -> Result<&'a str, RenderIrError> {
        self.short_str(240, 244)
    }

    fn long_slice(&self, o: usize, l: usize) -> Result<&'a [u8], RenderIrError> {
        checked_slice(
            self.bytes,
            read_u32(self.bytes, o)? as usize,
            read_u32(self.bytes, l)? as usize,
        )
    }
    fn long_str(&self, o: usize, l: usize) -> Result<&'a str, RenderIrError> {
        str::from_utf8(self.long_slice(o, l)?).map_err(|_| RenderIrError::InvalidUtf8)
    }
    fn short_str(&self, o: usize, l: usize) -> Result<&'a str, RenderIrError> {
        let bytes = checked_slice(
            self.bytes,
            read_u32(self.bytes, o)? as usize,
            read_u16(self.bytes, l)? as usize,
        )?;
        str::from_utf8(bytes).map_err(|_| RenderIrError::InvalidUtf8)
    }
}

/// Lowers the actual wgpu-v30 `SimpleCubePass` capture. This is intentionally
/// strict: a changed pass or pipeline must fail instead of silently baking a
/// different program under the v1 contract.
pub fn lower_simple_cube_wgpu_trace(files: &[TraceFile<'_>]) -> Result<RenderIr, RenderIrError> {
    let trace =
        str::from_utf8(file(files, "trace.ron")?).map_err(|_| RenderIrError::InvalidUtf8)?;
    for required in [
        "label: Some(\"SimpleCube Shader\")",
        "label: Some(\"SimpleCube Pipeline\")",
        "arrayStride: 36",
        "format: float32x3",
        "shaderLocation: 2",
        "topology: r#triangle-list",
        "frontFace: ccw",
        "cullMode: Some(back)",
        "format: \"depth32float\"",
        "depth_write_enabled: Some(true)",
        "depth_compare: Some(less)",
        "format: \"bgra8unorm-srgb\"",
        "entry_point: Some(\"vs_main\")",
        "entry_point: Some(\"fs_main\")",
        "binding: 0",
        "visibility: \"VERTEX\"",
        "read_only: true",
        "index_format: uint16",
        "index_count: 36",
        "instance_count: 1",
        "first_index: 0",
        "base_vertex: 0",
        "first_instance: 0",
        "r: 0.01",
        "g: 0.01",
        "b: 0.02",
        "a: 1.0",
        "load: clear(1.0)",
    ] {
        if !trace.contains(required) {
            return Err(RenderIrError::MissingTraceState);
        }
    }

    let vertex_file = data_file_for_label(trace, "SimpleCube VB")?;
    let index_file = data_file_for_label(trace, "SimpleCube IB")?;
    let shader_file = shader_file_for_label(trace, "SimpleCube Shader")?;
    // The trace recorder may append alignment bytes to a blob. Preserve the
    // exact ranges named by the recorded WriteBuffer commands, not that
    // recorder-only padding.
    let vertex_blob = file(files, vertex_file)?;
    let index_blob = file(files, index_file)?;
    if vertex_blob.len() < 864 || index_blob.len() < 72 {
        return Err(RenderIrError::UnexpectedResourceSize);
    }
    let vertex_data = vertex_blob[..864].to_vec();
    let index_data = index_blob[..72].to_vec();
    let wgsl = str::from_utf8(file(files, shader_file)?).map_err(|_| RenderIrError::InvalidUtf8)?;
    if !wgsl.contains("@location(2) color")
        || !wgsl.contains("fn vs_main")
        || !wgsl.contains("fn fs_main")
    {
        return Err(RenderIrError::MissingTraceState);
    }
    Ok(RenderIr {
        vertex_data,
        index_data,
        wgsl: wgsl.into(),
        vertex_entry: "vs_main".into(),
        fragment_entry: "fs_main".into(),
        camera_dynamic_slot: "camera.view_proj".into(),
        output_dynamic_slot: "output.surface".into(),
        pass_label: "SimpleCube".into(),
    })
}

fn data_file_for_label<'a>(trace: &'a str, label: &str) -> Result<&'a str, RenderIrError> {
    let label_at = trace
        .find(&alloc::format!("label: Some(\"{label}\")"))
        .ok_or(RenderIrError::MissingTraceState)?;
    let create_at = trace[..label_at]
        .rfind("CreateBuffer(PointerId(")
        .ok_or(RenderIrError::MissingTraceState)?;
    let id_start = create_at + "CreateBuffer(PointerId(".len();
    let id_end = trace[id_start..]
        .find(')')
        .map(|n| id_start + n)
        .ok_or(RenderIrError::Malformed)?;
    let id = &trace[id_start..id_end];
    let write = alloc::format!("id: PointerId({id})");
    let write_at = trace[label_at..]
        .find(&write)
        .map(|n| label_at + n)
        .ok_or(RenderIrError::MissingTraceState)?;
    file_name_after(&trace[write_at..])
}

fn shader_file_for_label<'a>(trace: &'a str, label: &str) -> Result<&'a str, RenderIrError> {
    let at = trace
        .find(&alloc::format!("label: Some(\"{label}\")"))
        .ok_or(RenderIrError::MissingTraceState)?;
    file_name_after(&trace[at..])
}

fn file_name_after(text: &str) -> Result<&str, RenderIrError> {
    let marker = "data: File(\"";
    let start = text
        .find(marker)
        .map(|n| n + marker.len())
        .ok_or(RenderIrError::MissingTraceState)?;
    let end = text[start..]
        .find("\")")
        .map(|n| start + n)
        .ok_or(RenderIrError::Malformed)?;
    Ok(&text[start..end])
}

fn file<'a>(files: &'a [TraceFile<'a>], name: &str) -> Result<&'a [u8], RenderIrError> {
    files
        .iter()
        .find(|file| file.name == name)
        .map(|file| file.data)
        .ok_or(RenderIrError::MissingTraceFile)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderIrError {
    BadMagic,
    UnsupportedVersion,
    Malformed,
    InvalidUtf8,
    MissingTraceFile,
    MissingTraceState,
    UnexpectedResourceSize,
}

impl fmt::Display for RenderIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RenderIrError {}

fn checked_slice(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8], RenderIrError> {
    let end = offset.checked_add(len).ok_or(RenderIrError::Malformed)?;
    if offset < RENDER_IR_HEADER_LEN || end > bytes.len() {
        return Err(RenderIrError::Malformed);
    }
    Ok(&bytes[offset..end])
}
fn read_u16(b: &[u8], o: usize) -> Result<u16, RenderIrError> {
    Ok(u16::from_le_bytes(
        b.get(o..o + 2)
            .ok_or(RenderIrError::Malformed)?
            .try_into()
            .unwrap(),
    ))
}
fn read_u32(b: &[u8], o: usize) -> Result<u32, RenderIrError> {
    Ok(u32::from_le_bytes(
        b.get(o..o + 4)
            .ok_or(RenderIrError::Malformed)?
            .try_into()
            .unwrap(),
    ))
}
fn put_u16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_i32(b: &mut [u8], o: usize, v: i32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_payload(out: &mut [u8], cursor: &mut usize, offset_at: usize, len_at: usize, data: &[u8]) {
    put_u32(out, offset_at, *cursor as u32);
    put_u32(out, len_at, data.len() as u32);
    out[*cursor..*cursor + data.len()].copy_from_slice(data);
    *cursor += data.len();
}
fn put_string16(out: &mut [u8], cursor: &mut usize, offset_at: usize, len_at: usize, data: &str) {
    put_u32(out, offset_at, *cursor as u32);
    put_u16(out, len_at, data.len() as u16);
    out[*cursor..*cursor + data.len()].copy_from_slice(data.as_bytes());
    *cursor += data.len();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir() -> RenderIr {
        RenderIr {
            vertex_data: alloc::vec![1; 864],
            index_data: alloc::vec![2; 72],
            wgsl: "@location(2) color fn vs_main() {} fn fs_main() {}".into(),
            vertex_entry: "vs_main".into(),
            fragment_entry: "fs_main".into(),
            camera_dynamic_slot: "camera.view_proj".into(),
            output_dynamic_slot: "output.surface".into(),
            pass_label: "SimpleCube".into(),
        }
    }

    #[test]
    fn binary_v1_round_trip() {
        let original = ir();
        let bytes = original.to_bytes();
        let decoded = RenderIrRef::parse(&bytes).unwrap();
        assert_eq!(decoded.vertex_buffer_id(), 1);
        assert_eq!(decoded.index_buffer_id(), 2);
        assert_eq!(decoded.camera_buffer_id(), 3);
        assert_eq!(decoded.vertex_data(), original.vertex_data);
        assert_eq!(decoded.index_data(), original.index_data);
        assert_eq!(decoded.wgsl().unwrap(), original.wgsl);
        assert_eq!(decoded.vertex_entry().unwrap(), "vs_main");
        assert_eq!(decoded.output_dynamic_slot().unwrap(), "output.surface");
        assert_eq!(read_u32(&bytes, 172).unwrap(), 2); // color location
        assert_eq!(read_u32(&bytes, 180).unwrap(), 24); // color byte offset
    }

    #[test]
    fn bad_payload_range_is_rejected() {
        let mut bytes = ir().to_bytes();
        put_u32(&mut bytes, 24, u32::MAX);
        assert_eq!(
            RenderIrRef::parse(&bytes).unwrap_err(),
            RenderIrError::Malformed
        );
    }
}
