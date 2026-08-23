//! Error types for scene operations.

use thiserror::Error;

/// Error type for scene operations.
///
/// Returned by scene resource management methods when invalid handles are used,
/// resources are still in use, or capacity limits are exceeded.
#[derive(Debug, Error)]
pub enum SceneError {
    /// An invalid handle was used (the resource no longer exists or never existed).
    #[error("invalid {resource} handle")]
    InvalidHandle {
        /// The type of resource that was invalid (e.g., "object", "material", "light").
        resource: &'static str,
    },

    /// A resource cannot be removed because it is still referenced by other resources.
    #[error("{resource} is still in use")]
    ResourceInUse {
        /// The type of resource that is still in use.
        resource: &'static str,
    },

    /// The scene's texture capacity has been exceeded.
    ///
    /// The capacity is selected from the device's complete material binding tier.
    #[error("scene texture capacity exceeded for the active material binding tier")]
    TextureCapacityExceeded,

    /// An authored texture identity is already attached to another live
    /// SceneDB texture entity.
    #[error("texture asset key {asset_key} is already resident")]
    DuplicateTextureAssetKey {
        /// Stable authored identity rejected by SceneDB.
        asset_key: u128,
    },

    /// The monotonic SceneDB texture identity domain reached `u128::MAX`.
    #[error("scene texture asset key space exhausted")]
    TextureAssetKeyExhausted,

    /// Authored Corona definitions exceeded the shader ABI's emitter table.
    #[error("corona emitter capacity exceeded ({requested} requested, maximum {capacity})")]
    CoronaEmitterCapacityExceeded {
        /// Number of definitions supplied by the caller.
        requested: usize,
        /// Fixed emitter-table capacity of the current shader ABI.
        capacity: usize,
    },

    /// An operation was rejected because of an incompatible resource state.
    #[error("invalid operation: {reason}")]
    InvalidOperation {
        /// Human-readable description of why the operation was rejected.
        reason: &'static str,
    },

    /// All GPU coordinate-space slots are in use.
    ///
    /// Sublevels and portals share one small fixed-size GPU buffer
    /// (`libhelio::MAX_COORDINATE_SPACES` slots, slot 0 reserved for world
    /// space). Remove an existing sublevel/portal before adding another once
    /// this is hit — this is not expected to occur in normal use.
    #[error("coordinate space capacity exceeded (all sublevel/portal slots in use)")]
    CoordinateSpaceCapacityExceeded,

    /// Auto-voxel mesh extraction has no free stable output slots.
    #[error("voxel mesh output capacity exceeded (1024 brick slots)")]
    VoxelMeshCapacityExceeded,

    /// A portal mutation would generate more recursive chains than the
    /// renderer's explicitly budgeted chain table can represent.
    #[error("portal chain capacity exceeded for the configured recursion depth")]
    PortalChainCapacityExceeded,
}

/// Result type for scene operations.
///
/// Alias for `std::result::Result<T, SceneError>`.
pub type Result<T> = std::result::Result<T, SceneError>;

/// Helper to construct an [`SceneError::InvalidHandle`] error.
///
/// # Example
/// ```ignore
/// return Err(invalid("object"));
/// ```
pub(super) fn invalid(resource: &'static str) -> SceneError {
    SceneError::InvalidHandle { resource }
}

/// Preserve Helio's public error surface while SceneDB owns the detailed
/// material/texture lifecycle invariant.
pub(super) fn scene_asset(error: helio_scenedb::SceneAssetError) -> SceneError {
    use helio_scenedb::SceneAssetError;

    match error {
        SceneAssetError::MaterialMissingComponent(_)
        | SceneAssetError::MaterialTextureRefsMissing(_) => invalid("material"),
        SceneAssetError::TextureNotResident(_)
        | SceneAssetError::TextureMissingComponent(_)
        | SceneAssetError::TextureAlreadyResident(_) => invalid("texture"),
        SceneAssetError::DuplicateTextureAsset { asset_key, .. } => {
            SceneError::DuplicateTextureAssetKey {
                asset_key: asset_key.0,
            }
        }
        SceneAssetError::InvalidTextureAssetKey => SceneError::InvalidOperation {
            reason: "texture asset key zero is reserved",
        },
        SceneAssetError::TextureInUse { .. } => SceneError::ResourceInUse {
            resource: "texture",
        },
        SceneAssetError::MaterialInUse { .. } => SceneError::ResourceInUse {
            resource: "material",
        },
        SceneAssetError::TextureStore(_) => SceneError::TextureCapacityExceeded,
        SceneAssetError::TextureAssetKeyExhausted => SceneError::TextureAssetKeyExhausted,
        SceneAssetError::TextureRefCountMustStartAtZero(_)
        | SceneAssetError::TextureRefCountOverflow(_)
        | SceneAssetError::TextureRefCountUnderflow(_)
        | SceneAssetError::MaterialRefCountOverflow(_)
        | SceneAssetError::MaterialRefCountUnderflow(_)
        | SceneAssetError::TextureRefCountDiverged { .. }
        | SceneAssetError::UnsupportedTexture(_)
        | SceneAssetError::TextureDataLength { .. } => SceneError::InvalidOperation {
            reason: "SceneDB rejected material/texture state",
        },
    }
}
