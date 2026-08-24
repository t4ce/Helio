use crate::{chunk::*, Compression};
use nebula_core::NebulaError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BinarySerError {
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    #[error("bincode: {0}")]
    Bincode(String),
}

impl From<BinarySerError> for NebulaError {
    fn from(e: BinarySerError) -> Self { NebulaError::Serialize(e.to_string()) }
}

/// Configuration for the compact binary `.nebula` format.
#[derive(Clone, Debug)]
pub struct NebulaBinarySerializer {
    pub compression: Compression,
}

impl Default for NebulaBinarySerializer {
    fn default() -> Self { Self { compression: Compression::Balanced } }
}
