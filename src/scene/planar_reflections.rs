//! SceneDB-backed planar-reflector CRUD and compact active-row projection.

use glam::Vec3;
use helio_scenedb::{
    SceneIndices, ScenePlanarReflector, ScenePlanarReflectorRow,
};

use crate::handles::{entity_from_handle, handle_from_entity, PlanarReflectorId};
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::Scene;

/// Authored finite rectangle whose receiving surface uses screen-space planar
/// reflection tracing.
///
/// `normal` and `tangent` need not arrive normalized, but they must be finite
/// and linearly independent. The SceneDB row stores the Gram-Schmidt
/// orthonormalized basis so every shader invocation observes identical math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarReflectorDescriptor {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub half_extents: [f32; 2],
    /// Maximum absolute distance between a reconstructed surface point and
    /// the plane. Must be finite and greater than zero.
    pub surface_tolerance: f32,
    /// Maximum angle between the G-buffer normal and reflector normal.
    /// Must be finite and in `(0, pi/2]`.
    pub normal_tolerance_radians: f32,
    /// Higher priority wins when finite reflectors overlap. Equal priorities
    /// resolve to the lower stable SceneDB component row.
    pub priority: f32,
}

impl Default for PlanarReflectorDescriptor {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            half_extents: [50.0, 50.0],
            surface_tolerance: 0.05,
            normal_tolerance_radians: 15.0_f32.to_radians(),
            priority: 0.0,
        }
    }
}

impl PlanarReflectorDescriptor {
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        tangent: [f32; 3],
        half_extents: [f32; 2],
    ) -> Self {
        Self {
            position,
            normal,
            tangent,
            half_extents,
            ..Self::default()
        }
    }

    fn validated_row(self) -> Result<ScenePlanarReflectorRow> {
        let finite3 = |values: [f32; 3]| values.into_iter().all(f32::is_finite);
        if !finite3(self.position) {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector position must be finite",
            });
        }
        if !finite3(self.normal) || !finite3(self.tangent) {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector basis must be finite",
            });
        }
        if !self.half_extents.into_iter().all(|v| v.is_finite() && v > 0.0) {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector half-extents must be finite and positive",
            });
        }
        if !self.surface_tolerance.is_finite() || self.surface_tolerance <= 0.0 {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector surface tolerance must be finite and positive",
            });
        }
        if !self.normal_tolerance_radians.is_finite()
            || self.normal_tolerance_radians <= 0.0
            || self.normal_tolerance_radians > std::f32::consts::FRAC_PI_2
        {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector normal tolerance must be in (0, pi/2]",
            });
        }
        if !self.priority.is_finite() {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector priority must be finite",
            });
        }

        // Scale before taking a length so any finite authored magnitude is
        // accepted without squaring f32::MAX to infinity or a subnormal to
        // zero. Magnitude is not semantic for either basis direction.
        let normalize_scaled = |value: Vec3| {
            let scale = value.abs().max_element();
            if scale == 0.0 {
                return None;
            }
            let scaled = value / scale;
            Some(scaled / scaled.length())
        };
        let Some(normal) = normalize_scaled(Vec3::from_array(self.normal)) else {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector normal must be nonzero",
            });
        };
        let Some(tangent) = normalize_scaled(Vec3::from_array(self.tangent)) else {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector tangent must be nonzero",
            });
        };
        let tangent = tangent - normal * tangent.dot(normal);
        const BASIS_EPSILON_SQ: f32 = 1.0e-12;
        if tangent.length_squared() <= BASIS_EPSILON_SQ {
            return Err(SceneError::InvalidOperation {
                reason: "planar reflector tangent must be independent of its normal",
            });
        }
        let tangent = normalize_scaled(tangent)
            .expect("nondegenerate projected tangent must normalize");

        Ok(ScenePlanarReflectorRow {
            position_tolerance: [
                self.position[0],
                self.position[1],
                self.position[2],
                self.surface_tolerance,
            ],
            normal_cos_threshold: [
                normal.x,
                normal.y,
                normal.z,
                self.normal_tolerance_radians.cos().max(0.0),
            ],
            tangent_priority: [tangent.x, tangent.y, tangent.z, self.priority],
            half_extents_reserved: [self.half_extents[0], self.half_extents[1], 0.0, 0.0],
        })
    }

    fn from_row(row: ScenePlanarReflectorRow) -> Self {
        Self {
            position: row.position_tolerance[..3].try_into().unwrap(),
            normal: row.normal_cos_threshold[..3].try_into().unwrap(),
            tangent: row.tangent_priority[..3].try_into().unwrap(),
            half_extents: row.half_extents_reserved[..2].try_into().unwrap(),
            surface_tolerance: row.position_tolerance[3],
            normal_tolerance_radians: row.normal_cos_threshold[3].clamp(-1.0, 1.0).acos(),
            priority: row.tangent_priority[3],
        }
    }
}

impl Scene {
    pub fn insert_planar_reflector(
        &mut self,
        descriptor: PlanarReflectorDescriptor,
    ) -> Result<PlanarReflectorId> {
        self.insert_planar_reflector_with_tag(descriptor, 0)
    }

    pub fn insert_planar_reflector_with_tag(
        &mut self,
        descriptor: PlanarReflectorDescriptor,
        user_tag: u64,
    ) -> Result<PlanarReflectorId> {
        // Validate and normalize before allocating an Entity or GPU row so a
        // rejected descriptor is fully transactional.
        let reflector = descriptor.validated_row()?;
        let entity = self.authority.insert(ScenePlanarReflector {
            user_tag,
            reflector,
            _reserved: 0,
        });
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .insert_planar_reflector(user_tag, entity);
        let row = self
            .authority
            .gpu_row::<ScenePlanarReflector>(entity)
            .expect("inserted planar reflector must own a mirror row");
        let id = handle_from_entity(entity);
        let compact = self.planar_reflector_projection.insert(id, row);
        let gpu_slot = self.gpu_scene.planar_reflector_indices.push(row);
        debug_assert_eq!(compact, gpu_slot);
        Ok(id)
    }

    pub fn update_planar_reflector(
        &mut self,
        id: PlanarReflectorId,
        descriptor: PlanarReflectorDescriptor,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        if self.authority.get::<ScenePlanarReflector>(entity).is_none() {
            return Err(invalid("planar reflector"));
        }
        // Complete validation precedes mutation, including stale-handle
        // validation above, so neither failure mode can publish partial data.
        let reflector = descriptor.validated_row()?;
        self.authority
            .edit_gpu::<ScenePlanarReflector, _>(entity, |record| {
                record.reflector = reflector;
            })
            .ok_or_else(|| invalid("planar reflector"))?;
        Ok(())
    }

    pub fn remove_planar_reflector(&mut self, id: PlanarReflectorId) -> Result<()> {
        let entity = entity_from_handle(id);
        let record = self
            .authority
            .get::<ScenePlanarReflector>(entity)
            .copied()
            .ok_or_else(|| invalid("planar reflector"))?;
        if !self.authority.despawn(entity) {
            return Err(invalid("planar reflector"));
        }
        self.authority
            .subsystem_mut::<SceneIndices>()
            .expect("SceneIndices is registered during Scene construction")
            .remove_planar_reflector(record.user_tag, entity);
        let compact = self
            .planar_reflector_projection
            .remove(id)
            .expect("live planar reflector must have an active projection");
        debug_assert!(
            self.gpu_scene
                .planar_reflector_indices
                .swap_remove(compact)
                .is_some()
        );
        Ok(())
    }

    pub fn get_planar_reflector(
        &self,
        id: PlanarReflectorId,
    ) -> Option<PlanarReflectorDescriptor> {
        self.authority
            .get::<ScenePlanarReflector>(entity_from_handle(id))
            .map(|record| PlanarReflectorDescriptor::from_row(record.reflector))
    }

    pub fn iter_planar_reflectors(
        &self,
    ) -> impl Iterator<Item = (PlanarReflectorId, PlanarReflectorDescriptor, u64)> + '_ {
        self.authority
            .query::<ScenePlanarReflector>()
            .map(|(entity, record)| {
                (
                    handle_from_entity(entity),
                    PlanarReflectorDescriptor::from_row(record.reflector),
                    record.user_tag,
                )
            })
    }

    pub fn planar_reflector_count(&self) -> usize {
        self.planar_reflector_projection.len()
    }

    pub fn planar_reflector_by_tag(&self, user_tag: u64) -> Option<PlanarReflectorId> {
        self.authority
            .subsystem::<SceneIndices>()
            .and_then(|indices| indices.planar_reflector_by_tag(user_tag))
            .filter(|entity| self.authority.get::<ScenePlanarReflector>(*entity).is_some())
            .map(handle_from_entity)
    }
}

#[cfg(test)]
mod tests {
    use helio_scenedb::ScenePlanarReflector;

    use super::{PlanarReflectorDescriptor, Scene};

    fn create_test_scene() -> Option<Scene> {
        let (device, queue) = crate::test_support::test_gpu()?;
        Some(Scene::new(device, queue))
    }

    #[test]
    fn validation_orthonormalizes_transactionally() {
        let Some(mut scene) = create_test_scene() else {
            eprintln!("skipping planar reflector test: no GPU adapter");
            return;
        };
        let id = scene
            .insert_planar_reflector(PlanarReflectorDescriptor::new(
                [1.0, 2.0, 3.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [4.0, 5.0],
            ))
            .unwrap();
        let normalized = scene.get_planar_reflector(id).unwrap();
        assert_eq!(normalized.normal, [0.0, 1.0, 0.0]);
        assert_eq!(normalized.tangent, [1.0, 0.0, 0.0]);

        scene
            .update_planar_reflector(
                id,
                PlanarReflectorDescriptor {
                    normal: [0.0, f32::MAX, 0.0],
                    tangent: [f32::MAX, f32::MAX, 0.0],
                    ..normalized
                },
            )
            .unwrap();
        let normalized = scene.get_planar_reflector(id).unwrap();
        assert_eq!(normalized.normal, [0.0, 1.0, 0.0]);
        assert_eq!(normalized.tangent, [1.0, 0.0, 0.0]);

        let entity = crate::handles::entity_from_handle(id);
        let before = scene
            .authority
            .get::<ScenePlanarReflector>(entity)
            .unwrap()
            .reflector;
        let invalid = PlanarReflectorDescriptor {
            tangent: [0.0, 9.0, 0.0],
            ..normalized
        };
        assert!(scene.update_planar_reflector(id, invalid).is_err());
        assert_eq!(
            scene
                .authority
                .get::<ScenePlanarReflector>(entity)
                .unwrap()
                .reflector,
            before
        );
    }

    #[test]
    fn sparse_rows_stale_handles_and_both_allocation_epochs_stay_coherent() {
        let Some(mut scene) = create_test_scene() else {
            eprintln!("skipping planar reflector test: no GPU adapter");
            return;
        };
        let first = scene
            .insert_planar_reflector_with_tag(PlanarReflectorDescriptor::default(), 11)
            .unwrap();
        let first_entity = crate::handles::entity_from_handle(first);
        let first_row = scene
            .authority
            .gpu_row::<ScenePlanarReflector>(first_entity)
            .unwrap();
        scene.flush();
        let first_canonical_epoch = scene
            .gpu_scene
            .canonical
            .planar_reflectors
            .as_ref()
            .unwrap()
            .epoch();
        let first_projection_epoch = scene.gpu_scene.planar_reflector_indices.buffer_version();

        let second = scene
            .insert_planar_reflector_with_tag(
                PlanarReflectorDescriptor {
                    position: [0.0, 3.0, 0.0],
                    ..PlanarReflectorDescriptor::default()
                },
                12,
            )
            .unwrap();
        let second_entity = crate::handles::entity_from_handle(second);
        let second_row = scene
            .authority
            .gpu_row::<ScenePlanarReflector>(second_entity)
            .unwrap();
        assert_ne!(first_row, second_row);
        scene.flush();
        let resources = scene.gpu_scene.resources();
        assert!(resources.planar_reflector_buffer_epoch.unwrap() > first_canonical_epoch);
        assert!(resources.planar_reflector_projection_epoch > first_projection_epoch);
        assert_eq!(resources.planar_reflector_count, 2);

        scene.remove_planar_reflector(first).unwrap();
        assert_eq!(scene.planar_reflector_by_tag(11), None);
        assert_eq!(scene.planar_reflector_by_tag(12), Some(second));
        assert_eq!(scene.gpu_scene.planar_reflector_indices.as_slice(), &[second_row]);
        assert_eq!(
            scene
                .authority
                .gpu_row::<ScenePlanarReflector>(second_entity),
            Some(second_row)
        );

        let replacement = scene
            .insert_planar_reflector(PlanarReflectorDescriptor {
                position: [0.0, -4.0, 0.0],
                ..PlanarReflectorDescriptor::default()
            })
            .unwrap();
        let replacement_row = scene
            .authority
            .gpu_row::<ScenePlanarReflector>(crate::handles::entity_from_handle(replacement))
            .unwrap();
        assert_eq!(replacement_row, first_row);
        assert!(scene.get_planar_reflector(first).is_none());
        assert!(scene
            .update_planar_reflector(first, PlanarReflectorDescriptor::default())
            .is_err());
        assert!(scene.remove_planar_reflector(first).is_err());

        scene.clear();
        assert_eq!(scene.planar_reflector_count(), 0);
        assert_eq!(scene.authority.gpu_live_count::<ScenePlanarReflector>(), 0);
        assert_eq!(scene.gpu_scene.resources().planar_reflector_count, 0);
    }
}
