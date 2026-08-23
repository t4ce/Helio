macro_rules! define_handle {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name {
            slot: u32,
            generation: u32,
        }

        impl $name {
            /// Sentinel used by descriptors whose `Default` cannot name a
            /// live SceneDB entity. Mutation APIs validate it like any other
            /// stale generation-bearing handle.
            pub const INVALID: Self = Self::from_raw(u32::MAX, u32::MAX);

            pub const fn from_raw(slot: u32, generation: u32) -> Self {
                Self { slot, generation }
            }

            pub const fn slot(self) -> u32 {
                self.slot
            }

            pub const fn generation(self) -> u32 {
                self.generation
            }
        }

        impl super::handles::Handle for $name {
            fn from_parts(slot: u32, generation: u32) -> Self {
                Self::from_raw(slot, generation)
            }

            fn slot(self) -> u32 {
                self.slot
            }

            fn generation(self) -> u32 {
                self.generation
            }
        }
    };
}

pub trait Handle: Copy {
    fn from_parts(slot: u32, generation: u32) -> Self;
    fn slot(self) -> u32;
    fn generation(self) -> u32;
}

/// Encode a Helio generational handle for storage in a SceneDB component.
/// This is the same stable representation used by SceneDB's `Entity`, but is
/// intentionally kept as raw bits for references to non-entity asset pools.
#[inline]
pub(crate) fn bits_from_handle(handle: impl Handle) -> u64 {
    ((handle.generation() as u64) << 32) | handle.slot() as u64
}

/// Decode a handle previously produced by [`bits_from_handle`].
#[inline]
pub(crate) fn handle_from_bits<H: Handle>(bits: u64) -> H {
    H::from_parts(bits as u32, (bits >> 32) as u32)
}

/// Convert a public Helio handle into SceneDB's canonical entity identity.
/// Both contracts use the same `(generation << 32) | slot` representation;
/// keeping conversion here avoids leaking SceneDB types through the public
/// facade while preserving stale-handle checks exactly.
#[inline]
pub(crate) fn entity_from_handle(handle: impl Handle) -> helio_scenedb::Entity {
    helio_scenedb::Entity::from_bits(bits_from_handle(handle))
}

#[inline]
pub(crate) fn handle_from_entity<H: Handle>(entity: helio_scenedb::Entity) -> H {
    H::from_parts(entity.index(), entity.generation())
}

define_handle!(MeshId);
define_handle!(MultiMeshId);
define_handle!(SectionedInstanceId);
define_handle!(MaterialId);
define_handle!(TextureId);
define_handle!(LightId);
define_handle!(ObjectId);
define_handle!(VirtualObjectId);
define_handle!(WaterVolumeId);
define_handle!(WaterHitboxId);
define_handle!(PostProcessVolumeId);
define_handle!(ReflectionCaptureId);
define_handle!(PlanarReflectorId);
define_handle!(VoxelVolumeId);
define_handle!(DecalId);
define_handle!(FoliageTypeId);
define_handle!(FoliageLayerId);
define_handle!(FoliageInteractorId);
define_handle!(SublevelId);
define_handle!(PortalId);

#[cfg(test)]
mod tests {
    use super::{bits_from_handle, entity_from_handle, handle_from_bits, handle_from_entity, ObjectId};

    #[test]
    fn object_id_and_scenedb_entity_share_exact_bits() {
        let id = ObjectId::from_raw(0x89ab_cdef, 0x0123_4567);
        let expected = 0x0123_4567_89ab_cdef;
        assert_eq!(bits_from_handle(id), expected);
        assert_eq!(entity_from_handle(id).bits(), expected);
        assert_eq!(handle_from_entity::<ObjectId>(entity_from_handle(id)), id);
        assert_eq!(handle_from_bits::<ObjectId>(expected), id);
    }
}
