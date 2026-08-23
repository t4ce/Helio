#![cfg_attr(not(feature = "std"), no_std)]

//! Stable container for a Helio program after its render graph has been
//! recorded by wgpu. The container deliberately stores relocatable inputs,
//! not target GPU virtual addresses or a pre-patched command buffer.

extern crate alloc;

use alloc::{string::String, vec, vec::Vec};
use core::{fmt, str};

use serde::{Deserialize, Serialize};

mod render_ir;

pub use render_ir::{
    lower_simple_cube_wgpu_trace, RenderIr, RenderIrError, RenderIrRef, TraceFile,
    RENDER_IR_SECTION_NAME,
};

pub const MAGIC: [u8; 8] = *b"HELIOA\0\0";
pub const FORMAT_VERSION: u16 = 1;
const HEADER_LEN: usize = 32;
const ENTRY_FIXED_LEN: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum SectionKind {
    Manifest = 1,
    WgpuTrace = 2,
    ShaderSource = 3,
    IntelXeLpIsa = 4,
    CompilerMetadata = 5,
    RenderIr = 6,
    Other = u16::MAX,
}

impl SectionKind {
    fn from_raw(raw: u16) -> Self {
        match raw {
            1 => Self::Manifest,
            2 => Self::WgpuTrace,
            3 => Self::ShaderSource,
            4 => Self::IntelXeLpIsa,
            5 => Self::CompilerMetadata,
            6 => Self::RenderIr,
            _ => Self::Other,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub engine: String,
    pub program: String,
    pub graph: String,
    pub capture: String,
    pub target_api: String,
    pub target_architecture: String,
    pub surface_format: String,
    pub width: u32,
    pub height: u32,
    pub dynamic_slots: Vec<DynamicSlot>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DynamicSlot {
    pub name: String,
    pub kind: String,
}

impl Manifest {
    pub fn to_json(&self) -> Result<Vec<u8>, Error> {
        serde_json::to_vec_pretty(self).map_err(|_| Error::InvalidManifest)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(bytes).map_err(|_| Error::InvalidManifest)
    }
}

#[derive(Clone, Debug)]
pub struct Section<'a> {
    pub kind: SectionKind,
    pub name: &'a str,
    pub data: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct Artifact<'a> {
    bytes: &'a [u8],
    entries: Vec<Entry<'a>>,
}

#[derive(Clone, Debug)]
struct Entry<'a> {
    kind: SectionKind,
    name: &'a str,
    offset: usize,
    len: usize,
    crc32: u32,
}

impl<'a> Artifact<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_LEN || bytes[..8] != MAGIC {
            return Err(Error::BadMagic);
        }
        let version = read_u16(bytes, 8)?;
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        if read_u16(bytes, 10)? as usize != HEADER_LEN {
            return Err(Error::MalformedHeader);
        }

        let count = read_u32(bytes, 12)? as usize;
        let toc_len = to_usize(read_u64(bytes, 16)?)?;
        let payload_offset = to_usize(read_u64(bytes, 24)?)?;
        let toc_end = HEADER_LEN.checked_add(toc_len).ok_or(Error::OutOfBounds)?;
        if toc_end != payload_offset || payload_offset > bytes.len() {
            return Err(Error::MalformedHeader);
        }
        if count > toc_len / ENTRY_FIXED_LEN {
            return Err(Error::MalformedHeader);
        }

        let mut cursor = HEADER_LEN;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let fixed_end = cursor
                .checked_add(ENTRY_FIXED_LEN)
                .ok_or(Error::OutOfBounds)?;
            if fixed_end > payload_offset {
                return Err(Error::OutOfBounds);
            }
            let name_len = read_u16(bytes, cursor)? as usize;
            let kind = SectionKind::from_raw(read_u16(bytes, cursor + 2)?);
            let offset = to_usize(read_u64(bytes, cursor + 8)?)?;
            let len = to_usize(read_u64(bytes, cursor + 16)?)?;
            let crc32 = read_u32(bytes, cursor + 24)?;
            let name_start = fixed_end;
            let name_end = name_start.checked_add(name_len).ok_or(Error::OutOfBounds)?;
            if name_end > payload_offset {
                return Err(Error::OutOfBounds);
            }
            let name =
                str::from_utf8(&bytes[name_start..name_end]).map_err(|_| Error::InvalidName)?;
            validate_name(name)?;
            let data_end = offset.checked_add(len).ok_or(Error::OutOfBounds)?;
            if offset < payload_offset || data_end > bytes.len() {
                return Err(Error::OutOfBounds);
            }
            if entries.iter().any(|entry: &Entry<'_>| entry.name == name) {
                return Err(Error::DuplicateName);
            }
            entries.push(Entry {
                kind,
                name,
                offset,
                len,
                crc32,
            });
            cursor = align_8(name_end).ok_or(Error::OutOfBounds)?;
        }
        if cursor != payload_offset {
            return Err(Error::MalformedHeader);
        }

        let artifact = Self { bytes, entries };
        for section in artifact.sections() {
            let expected = artifact
                .entries
                .iter()
                .find(|entry| entry.name == section.name)
                .unwrap()
                .crc32;
            if crc32fast::hash(section.data) != expected {
                return Err(Error::ChecksumMismatch);
            }
        }
        if artifact.section("manifest.json").map(|s| s.kind) != Some(SectionKind::Manifest) {
            return Err(Error::MissingManifest);
        }
        Ok(artifact)
    }

    pub fn sections(&self) -> impl Iterator<Item = Section<'a>> + '_ {
        self.entries.iter().map(|entry| Section {
            kind: entry.kind,
            name: entry.name,
            data: &self.bytes[entry.offset..entry.offset + entry.len],
        })
    }

    pub fn section(&self, name: &str) -> Option<Section<'a>> {
        self.sections().find(|section| section.name == name)
    }

    pub fn manifest(&self) -> Result<Manifest, Error> {
        Manifest::from_json(
            self.section("manifest.json")
                .ok_or(Error::MissingManifest)?
                .data,
        )
    }
}

#[derive(Default)]
pub struct Builder {
    sections: Vec<OwnedSection>,
}

struct OwnedSection {
    kind: SectionKind,
    name: String,
    data: Vec<u8>,
}

impl Builder {
    pub fn new(manifest: &Manifest) -> Result<Self, Error> {
        let mut builder = Self::default();
        builder.add(SectionKind::Manifest, "manifest.json", manifest.to_json()?)?;
        Ok(builder)
    }

    pub fn add(
        &mut self,
        kind: SectionKind,
        name: impl Into<String>,
        data: Vec<u8>,
    ) -> Result<(), Error> {
        let name = name.into();
        validate_name(&name)?;
        if self.sections.iter().any(|section| section.name == name) {
            return Err(Error::DuplicateName);
        }
        self.sections.push(OwnedSection { kind, name, data });
        Ok(())
    }

    /// Adds the normalized, pointer-free render program consumed by TRUEOS.
    pub fn add_render_ir(&mut self, ir: &RenderIr) -> Result<(), Error> {
        self.add(SectionKind::RenderIr, RENDER_IR_SECTION_NAME, ir.to_bytes())
    }

    pub fn finish(mut self) -> Result<Vec<u8>, Error> {
        self.sections.sort_by(|a, b| a.name.cmp(&b.name));
        let toc_len = self.sections.iter().try_fold(0usize, |total, section| {
            let entry_len = ENTRY_FIXED_LEN
                .checked_add(section.name.len())
                .ok_or(Error::OutOfBounds)?;
            total
                .checked_add(align_8(entry_len).ok_or(Error::OutOfBounds)?)
                .ok_or(Error::OutOfBounds)
        })?;
        let payload_offset = HEADER_LEN.checked_add(toc_len).ok_or(Error::OutOfBounds)?;
        let payload_len = self.sections.iter().try_fold(0usize, |total, section| {
            total
                .checked_add(section.data.len())
                .ok_or(Error::OutOfBounds)
        })?;
        let total_len = payload_offset
            .checked_add(payload_len)
            .ok_or(Error::OutOfBounds)?;
        let mut out = vec![0u8; total_len];
        out[..8].copy_from_slice(&MAGIC);
        put_u16(&mut out, 8, FORMAT_VERSION);
        put_u16(&mut out, 10, HEADER_LEN as u16);
        put_u32(&mut out, 12, self.sections.len() as u32);
        put_u64(&mut out, 16, toc_len as u64);
        put_u64(&mut out, 24, payload_offset as u64);

        let mut toc_cursor = HEADER_LEN;
        let mut data_cursor = payload_offset;
        for section in self.sections {
            put_u16(&mut out, toc_cursor, section.name.len() as u16);
            put_u16(&mut out, toc_cursor + 2, section.kind as u16);
            put_u64(&mut out, toc_cursor + 8, data_cursor as u64);
            put_u64(&mut out, toc_cursor + 16, section.data.len() as u64);
            put_u32(&mut out, toc_cursor + 24, crc32fast::hash(&section.data));
            let name_start = toc_cursor + ENTRY_FIXED_LEN;
            out[name_start..name_start + section.name.len()]
                .copy_from_slice(section.name.as_bytes());
            toc_cursor = align_8(name_start + section.name.len()).ok_or(Error::OutOfBounds)?;
            let end = data_cursor + section.data.len();
            out[data_cursor..end].copy_from_slice(&section.data);
            data_cursor = end;
        }
        debug_assert_eq!(toc_cursor, payload_offset);
        debug_assert_eq!(data_cursor, total_len);
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    BadMagic,
    UnsupportedVersion(u16),
    MalformedHeader,
    OutOfBounds,
    InvalidName,
    DuplicateName,
    ChecksumMismatch,
    MissingManifest,
    InvalidManifest,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

fn validate_name(name: &str) -> Result<(), Error> {
    if name.is_empty()
        || name.len() > u16::MAX as usize
        || name.starts_with('/')
        || name.contains("..")
        || name.contains('\\')
    {
        return Err(Error::InvalidName);
    }
    Ok(())
}

fn align_8(value: usize) -> Option<usize> {
    value.checked_add(7).map(|v| v & !7)
}

fn to_usize(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::OutOfBounds)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, Error> {
    let raw = bytes.get(offset..offset + 2).ok_or(Error::OutOfBounds)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Error> {
    let raw = bytes.get(offset..offset + 4).ok_or(Error::OutOfBounds)?;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, Error> {
    let raw = bytes.get(offset..offset + 8).ok_or(Error::OutOfBounds)?;
    Ok(u64::from_le_bytes(raw.try_into().unwrap()))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            schema: 1,
            engine: "Helio".into(),
            program: "simple-cube".into(),
            graph: "build_simple_graph".into(),
            capture: "wgpu-trace-v30".into(),
            target_api: "trueos-render".into(),
            target_architecture: "intel-xe-lp".into(),
            surface_format: "Bgra8UnormSrgb".into(),
            width: 1280,
            height: 720,
            dynamic_slots: vec![DynamicSlot {
                name: "camera.view_proj".into(),
                kind: "mat4x4-f32".into(),
            }],
        }
    }

    #[test]
    fn deterministic_round_trip() {
        let mut builder = Builder::new(&manifest()).unwrap();
        builder
            .add(
                SectionKind::WgpuTrace,
                "wgpu/trace.ron",
                b"draw_indexed".to_vec(),
            )
            .unwrap();
        let first = builder.finish().unwrap();

        let mut builder = Builder::new(&manifest()).unwrap();
        builder
            .add(
                SectionKind::WgpuTrace,
                "wgpu/trace.ron",
                b"draw_indexed".to_vec(),
            )
            .unwrap();
        let second = builder.finish().unwrap();
        assert_eq!(first, second);

        let artifact = Artifact::parse(&first).unwrap();
        assert_eq!(artifact.manifest().unwrap(), manifest());
        assert_eq!(
            artifact.section("wgpu/trace.ron").unwrap().data,
            b"draw_indexed"
        );
    }

    #[test]
    fn corrupt_payload_is_rejected() {
        let mut bytes = Builder::new(&manifest()).unwrap().finish().unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        assert_eq!(
            Artifact::parse(&bytes).unwrap_err(),
            Error::ChecksumMismatch
        );
    }

    #[test]
    fn impossible_entry_count_is_rejected_before_allocation() {
        let mut bytes = Builder::new(&manifest()).unwrap().finish().unwrap();
        put_u32(&mut bytes, 12, u32::MAX);
        assert_eq!(Artifact::parse(&bytes).unwrap_err(), Error::MalformedHeader);
    }
}
