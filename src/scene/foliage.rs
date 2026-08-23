//! Foliage authoring API for [`Scene`](super::Scene).
//!
//! This is the CPU-facing half of the foliage stack: registering foliage *types* (what a
//! grass blade or a tree is), grouping them into *layers* (where they grow), driving the
//! global wind clock, and publishing *interactors* — the moving bodies that push grass
//! aside.
//!
//! # Two things here are load-bearing and easy to break
//!
//! **The `generation` counter must not advance when the wind changes.** `generation` gates
//! re-upload of the foliage type table. Wind changes every single frame; if it bumped
//! `generation`, the type table would re-upload every frame and the residency cache's
//! whole reason for existing — that steady-state foliage costs the CPU nothing — would be
//! gone. Wind travels in the same [`libhelio::FoliageFrameData`] struct but is deliberately outside
//! the versioned part. See `libhelio::FoliageFrameData::generation`.
//!
//! **The `foliage` frame slot is left unwritten when no foliage types are registered.** The
//! foliage passes early-out on the unwritten slot, and that is the entire mechanism behind
//! the zero-overhead-when-absent guarantee. Publishing an empty-but-present struct would
//! silently turn "no foliage in this scene" into "foliage machinery runs and finds nothing".

use glam::Vec3;
use helio_foliage_core::{
    pack_kind_and_flags, FoliageKind, GpuFoliageLayer, GpuFoliageType, FOLIAGE_FLAG_CASTS_SHADOW,
    FOLIAGE_FLAG_RECEIVES_INTERACTION, FOLIAGE_FLAG_TWO_SIDED,
};
pub use helio_foliage_core::GpuFoliageInteractor;
use crate::handles::{FoliageTypeId, MaterialId};

// ─── Public descriptors ─────────────────────────────────────────────────────

/// What a single kind of foliage is.
///
/// All fields are public because the documented call style uses functional-update syntax
/// (`FoliageTypeDescriptor { density: 40.0, ..Default::default() }`), which requires every
/// field to be visible at the call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoliageTypeDescriptor {
    /// Blade, card, or a full mesh driven through virtual geometry.
    pub kind: FoliageKind,
    /// Instances per square metre at full density weight.
    pub density: f32,
    /// Minimum and maximum blade/plant height in metres.
    pub height_range: [f32; 2],
    /// Minimum and maximum width in metres.
    pub width_range: [f32; 2],
    /// Acceptance band on terrain slope, as `[min_angle, max_angle]` in **radians from
    /// horizontal** — `[0.0, 35f32.to_radians()]` means "flat ground up to a 35° incline".
    ///
    /// The GPU field this becomes is a *cosine* band, and `to_gpu` does the conversion.
    /// Do not pass cosines here. Getting this backwards does not warn or render oddly: a
    /// flat plane has `cos(slope) == 1.0`, so a band that looks like `[0.0, 0.61]` in
    /// radians rejects every candidate on level ground and places exactly zero blades,
    /// which is indistinguishable from foliage being disabled.
    pub slope_range: [f32; 2],
    /// Acceptance band on world altitude in metres.
    pub altitude_range: [f32; 2],
    /// Distances at which the four LOD transitions happen, in metres.
    pub lod_distances: [f32; 4],
    /// Per-band wind gain: `[trunk sway, branch flutter, leaf jitter]`.
    pub wind_response: [f32; 3],
    /// How fast a bent plant recovers. Larger is stiffer.
    pub interaction_stiffness: f32,
    /// Generation-bearing canonical material identity. Scene insertion
    /// resolves it to a component-local GPU row and retains the material for
    /// the foliage type's lifetime.
    pub material_id: MaterialId,
    /// Slice in the density texture array driving where this type grows.
    pub density_layer: u32,
    /// Virtual mesh or impostor page, for [`FoliageKind::Mesh`].
    pub mesh_or_impostor_id: u32,
    /// Render both faces. Almost always wanted for vegetation.
    pub two_sided: bool,
    /// Contribute to the shadow atlas.
    pub casts_shadow: bool,
    /// Respond to [`FoliageInteractor`]s.
    pub receives_interaction: bool,
}

impl Default for FoliageTypeDescriptor {
    fn default() -> Self {
        Self {
            kind: FoliageKind::Blade,
            density: 40.0,
            height_range: [0.15, 0.45],
            width_range: [0.01, 0.03],
            slope_range: [0.0, std::f32::consts::FRAC_PI_4],
            altitude_range: [f32::MIN, f32::MAX],
            lod_distances: helio_foliage_core::DEFAULT_LOD_DISTANCES,
            wind_response: [0.0, 0.3, 1.0],
            interaction_stiffness: 6.0,
            material_id: MaterialId::INVALID,
            density_layer: 0,
            mesh_or_impostor_id: u32::MAX,
            two_sided: true,
            casts_shadow: false,
            receives_interaction: true,
        }
    }
}

impl FoliageTypeDescriptor {
    /// Convert to the GPU record.
    ///
    /// Non-finite inputs are clamped to inert values rather than passed through. A NaN
    /// reaching a foliage shader does not produce an error anywhere — it makes every
    /// affected primitive fail its comparisons and vanish, which presents to the user as
    /// "all the grass disappeared" with a clean log. Degrading to zero density is at least
    /// diagnosable.
    pub(crate) fn to_gpu(self, material_row: u32) -> GpuFoliageType {
        fn finite(value: f32, fallback: f32) -> f32 {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        }
        fn finite_pair(pair: [f32; 2], fallback: [f32; 2]) -> [f32; 2] {
            let low = finite(pair[0], fallback[0]);
            let high = finite(pair[1], fallback[1]);
            if low <= high {
                [low, high]
            } else {
                [high, low]
            }
        }

        let flags = u32::from(self.two_sided) * FOLIAGE_FLAG_TWO_SIDED
            | u32::from(self.casts_shadow) * FOLIAGE_FLAG_CASTS_SHADOW
            | u32::from(self.receives_interaction) * FOLIAGE_FLAG_RECEIVES_INTERACTION;

        let mut gpu = GpuFoliageType {
            density: finite(self.density, 0.0).max(0.0),
            height_range: finite_pair(self.height_range, [0.15, 0.45]),
            width_range: finite_pair(self.width_range, [0.01, 0.03]),
            // Angles in, cosines out. `cos` is decreasing over [0, π], so the band
            // inverts: the *max* angle produces the *min* accepted cosine.
            slope_range: {
                let angles =
                    finite_pair(self.slope_range, [0.0, std::f32::consts::FRAC_PI_4]);
                let low = angles[0].clamp(0.0, std::f32::consts::PI);
                let high = angles[1].clamp(0.0, std::f32::consts::PI);
                [high.cos(), low.cos()]
            },
            altitude_range: finite_pair(self.altitude_range, [f32::MIN, f32::MAX]),
            lod_distances: helio_foliage_core::DEFAULT_LOD_DISTANCES,
            wind_response: [
                finite(self.wind_response[0], 0.0),
                finite(self.wind_response[1], 0.0),
                finite(self.wind_response[2], 0.0),
            ],
            interaction_stiffness: finite(self.interaction_stiffness, 6.0).max(0.0),
            material_id: material_row,
            density_layer: self.density_layer,
            kind_and_flags: pack_kind_and_flags(self.kind, flags),
            mesh_or_impostor_id: self.mesh_or_impostor_id,
            _pad: [0; 3],
        };

        // LOD distances must be non-decreasing or `select_blade_lod` picks a level whose
        // band it is not actually inside, and the cross-fade blends two representations
        // that are never both valid.
        let mut previous = 0.0_f32;
        for (slot, raw) in gpu.lod_distances.iter_mut().zip(self.lod_distances) {
            let value = finite(raw, previous).max(previous);
            *slot = value;
            previous = value;
        }
        gpu
    }
}

/// Where a set of foliage types grows.
///
/// Placement accepts a candidate only when its world position lies inside at least one
/// layer's AABB — set `has_infinite_extent` to exempt a layer from that test and carpet
/// the whole ring. An empty layer table keeps the legacy "carpet everything" behaviour,
/// so publishers that predate the layer table are not silently culled.
///
/// Terrain density and exclusion maps are not part of this descriptor yet. The
/// per-type `density_layer` is retained, but a future `FoliageTerrainPass` will
/// need an explicit generation-bearing texture relationship before authored maps
/// can be represented safely.
#[derive(Debug, Clone)]
pub struct FoliageLayer {
    /// Types this layer places.
    pub types: Vec<FoliageTypeId>,
    /// World-space AABB the layer covers, as `[min_xyz, max_xyz]`. Ignored when
    /// `has_infinite_extent` is set.
    pub bounds: [Vec3; 2],
    /// Placement seed. Two layers with the same seed and bounds place identically.
    pub seed: u32,
    /// When set, the layer's bounds are not enforced: every candidate is inside it,
    /// which is the behaviour a scene had before layer bounds were consulted at all.
    pub has_infinite_extent: bool,
}

impl FoliageLayer {
    /// Convert to the GPU record.
    pub(crate) fn to_gpu(&self) -> GpuFoliageLayer {
        let [min, max] = self.bounds;
        GpuFoliageLayer {
            bounds_min: [min.x, min.y, min.z, 0.0],
            bounds_max: [
                max.x,
                max.y,
                max.z,
                if self.has_infinite_extent { 1.0 } else { 0.0 },
            ],
        }
    }
}

/// A moving body that pushes foliage aside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoliageInteractor {
    /// World-space position.
    pub position: Vec3,
    /// Influence radius in metres.
    pub radius: f32,
    /// World-space velocity, used to smear the track across a frame.
    pub velocity: Vec3,
}

impl FoliageInteractor {
    pub(crate) fn to_gpu(self) -> GpuFoliageInteractor {
        // Same reasoning as `FoliageTypeDescriptor::to_gpu`: an inert interactor is far
        // easier to diagnose than a NaN that silently deletes vegetation.
        let position = if self.position.is_finite() {
            self.position
        } else {
            Vec3::ZERO
        };
        let velocity = if self.velocity.is_finite() {
            self.velocity
        } else {
            Vec3::ZERO
        };
        let radius = if self.radius.is_finite() {
            self.radius.max(0.0)
        } else {
            0.0
        };
        GpuFoliageInteractor {
            position_radius: [position.x, position.y, position.z, radius],
            velocity: [velocity.x, velocity.y, velocity.z, 0.0],
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_packs_kind_and_flags() {
        let gpu = FoliageTypeDescriptor {
            kind: FoliageKind::Blade,
            two_sided: true,
            casts_shadow: false,
            receives_interaction: true,
            ..Default::default()
        }
        .to_gpu(0);

        assert_eq!(gpu.kind(), Some(FoliageKind::Blade));
        assert!(gpu.has_flag(FOLIAGE_FLAG_TWO_SIDED));
        assert!(!gpu.has_flag(FOLIAGE_FLAG_CASTS_SHADOW));
        assert!(gpu.has_flag(FOLIAGE_FLAG_RECEIVES_INTERACTION));
    }

    #[test]
    fn non_finite_descriptor_inputs_degrade_to_inert_values() {
        let gpu = FoliageTypeDescriptor {
            density: f32::NAN,
            height_range: [f32::NAN, f32::INFINITY],
            interaction_stiffness: f32::NEG_INFINITY,
            wind_response: [f32::NAN; 3],
            ..Default::default()
        }
        .to_gpu(0);

        assert_eq!(gpu.density, 0.0);
        assert!(gpu.height_range.iter().all(|v| v.is_finite()));
        assert!(gpu.interaction_stiffness.is_finite());
        assert!(gpu.wind_response.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn flat_ground_is_inside_the_default_slope_band() {
        // The regression this pins: the GPU field is a cosine band, the descriptor takes
        // angles. Passing angles straight through rejects every candidate on level ground
        // and places zero blades — which looks exactly like foliage being switched off,
        // with nothing in the log.
        let gpu = FoliageTypeDescriptor {
            slope_range: [0.0, 35f32.to_radians()],
            ..Default::default()
        }
        .to_gpu(0);

        let flat_ground_cos = 1.0_f32;
        assert!(
            flat_ground_cos >= gpu.slope_range[0] && flat_ground_cos <= gpu.slope_range[1],
            "flat ground (cos = 1.0) must be inside {:?}",
            gpu.slope_range
        );

        // And a 60° face must be outside a 35° band.
        let steep_cos = 60f32.to_radians().cos();
        assert!(steep_cos < gpu.slope_range[0]);
    }

    #[test]
    fn steeper_max_angle_widens_the_accepted_band() {
        let gentle = FoliageTypeDescriptor {
            slope_range: [0.0, 10f32.to_radians()],
            ..Default::default()
        }
        .to_gpu(0);
        let steep = FoliageTypeDescriptor {
            slope_range: [0.0, 70f32.to_radians()],
            ..Default::default()
        }
        .to_gpu(0);
        assert!(steep.slope_range[0] < gentle.slope_range[0]);
    }

    #[test]
    fn inverted_ranges_are_ordered_not_rejected() {
        let gpu = FoliageTypeDescriptor {
            height_range: [0.9, 0.1],
            ..Default::default()
        }
        .to_gpu(0);
        assert!(gpu.height_range[0] <= gpu.height_range[1]);
    }

    #[test]
    fn lod_distances_are_forced_non_decreasing() {
        let gpu = FoliageTypeDescriptor {
            lod_distances: [40.0, 10.0, 80.0, 5.0],
            ..Default::default()
        }
        .to_gpu(0);
        let d = gpu.lod_distances;
        assert!(d[0] <= d[1] && d[1] <= d[2] && d[2] <= d[3], "{d:?}");
    }

    #[test]
    fn interactor_rejects_non_finite_transform() {
        let gpu = FoliageInteractor {
            position: Vec3::new(f32::NAN, 0.0, 0.0),
            radius: -3.0,
            velocity: Vec3::splat(f32::INFINITY),
        }
        .to_gpu();

        assert!(gpu.position_radius.iter().all(|v| v.is_finite()));
        assert!(gpu.velocity.iter().all(|v| v.is_finite()));
        assert!(gpu.position_radius[3] >= 0.0);
    }

    #[test]
    fn interactor_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuFoliageInteractor>(), 32);
    }
}
