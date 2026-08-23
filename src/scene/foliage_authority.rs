//! SceneDB-backed foliage CRUD and compact shader projections.

use glam::Vec3;
use helio_foliage_core::{
    GpuFoliageLayerProjection, GpuFoliageLayerTypeRelation,
};
use helio_scenedb::{
    Entity, SceneFoliageInteractor, SceneFoliageInteractorRow, SceneFoliageLayer,
    SceneFoliageLayerRow, SceneFoliageLayerTypes, SceneFoliageType, SceneFoliageTypeRow,
    SceneMaterial, SceneWind, SceneWindRow,
};
use libhelio::{FoliageFrameData, Wind};

use super::core::Scene;
use super::errors::{invalid, Result};
use super::foliage::{FoliageInteractor, FoliageLayer, FoliageTypeDescriptor};
use crate::handles::{
    entity_from_handle, handle_from_entity, FoliageInteractorId, FoliageLayerId,
    FoliageTypeId, MaterialId,
};

const MAX_PACKED_FOLIAGE_TYPES: usize = 256;

impl Scene {
    fn bump_foliage_topology(&mut self) {
        self.foliage_generation = self.foliage_generation.wrapping_add(1);
    }

    fn recompute_foliage_type_stats(&mut self) {
        let mut max_height = 0.0f32;
        let mut max_density = 0.0f32;
        for (_, record) in self.authority.query::<SceneFoliageType>() {
            let row = &record.foliage.0;
            if row.height_range[1].is_finite() {
                max_height = max_height.max(row.height_range[1]);
            }
            if row.density.is_finite() && row.density > 0.0 {
                max_density = max_density.max(row.density);
            }
        }
        self.foliage_max_height = max_height;
        self.foliage_max_density = max_density;
    }

    fn validate_foliage_layer_types(&self, types: &[FoliageTypeId]) -> Result<Vec<Entity>> {
        types
            .iter()
            .map(|&id| {
                let entity = entity_from_handle(id);
                self.authority
                    .get::<SceneFoliageType>(entity)
                    .and_then(|_| self.foliage_type_projection.slot(id))
                    .map(|_| entity)
                    .ok_or_else(|| invalid("foliage type"))
            })
            .collect()
    }

    /// Rebuild only compact relationship/index data. The canonical 96-byte and 32-byte
    /// rows remain in SceneDB and are never copied here.
    fn rebuild_foliage_layer_projections(&mut self) {
        let mut layers = Vec::with_capacity(self.foliage_layer_projection.len());
        let mut relations = Vec::new();

        for (compact_layer, &id) in self
            .foliage_layer_projection
            .ids()
            .iter()
            .enumerate()
        {
            let entity = entity_from_handle(id);
            let Some(layer) = self.authority.get::<SceneFoliageLayer>(entity) else {
                debug_assert!(false, "projected foliage layer must remain canonical");
                continue;
            };
            let Some(authored_types) = self.authority.get::<SceneFoliageLayerTypes>(entity) else {
                debug_assert!(false, "foliage layer must retain its type relationship");
                continue;
            };

            let relation_offset = u32::try_from(relations.len())
                .expect("foliage relation projection exceeds u32 rows");
            for &type_entity in &authored_types.types {
                let type_id = handle_from_entity::<FoliageTypeId>(type_entity);
                let compact_type_id = self
                    .foliage_type_projection
                    .slot(type_id)
                    .expect("validated foliage relation became stale");
                let canonical_type_row = self
                    .authority
                    .gpu_row::<SceneFoliageType>(type_entity)
                    .expect("related foliage type must own a canonical GPU row");
                relations.push(GpuFoliageLayerTypeRelation {
                    compact_type_id: u32::try_from(compact_type_id)
                        .expect("foliage compact type id exceeds u32"),
                    canonical_type_row,
                });
            }
            let relation_count = u32::try_from(relations.len())
                .expect("foliage relation projection exceeds u32 rows")
                - relation_offset;
            layers.push(GpuFoliageLayerProjection {
                canonical_layer_row: self
                    .foliage_layer_projection
                    .row(compact_layer)
                    .expect("compact foliage layer must own a canonical row"),
                relation_offset,
                relation_count,
                seed: layer.seed,
            });
        }

        self.gpu_scene.foliage_layer_projections.set_data(layers);
        self.gpu_scene
            .foliage_layer_type_relations
            .set_data(relations);
    }

    pub fn add_foliage_type(
        &mut self,
        descriptor: FoliageTypeDescriptor,
    ) -> Result<FoliageTypeId> {
        if self.foliage_type_projection.len() >= MAX_PACKED_FOLIAGE_TYPES {
            return Err(invalid("foliage type capacity (8-bit blade identity)"));
        }
        let material_entity = entity_from_handle(descriptor.material_id);
        let material_row = self
            .authority
            .gpu_row::<SceneMaterial>(material_entity)
            .ok_or_else(|| invalid("material"))?;
        self.authority
            .retain_material(material_entity)
            .map_err(super::errors::scene_asset)?;

        let entity = self.authority.insert(SceneFoliageType {
            material_entity_bits: material_entity.bits(),
            foliage: SceneFoliageTypeRow::from(descriptor.to_gpu(material_row)),
        });
        let row = self
            .authority
            .gpu_row::<SceneFoliageType>(entity)
            .expect("inserted foliage type must own a canonical GPU row");
        let id = handle_from_entity(entity);
        let compact = self.foliage_type_projection.insert(id, row);
        let gpu_compact = self.gpu_scene.foliage_type_indices.push(row);
        debug_assert_eq!(compact, gpu_compact);
        self.recompute_foliage_type_stats();
        self.bump_foliage_topology();
        Ok(id)
    }

    pub fn update_foliage_type(
        &mut self,
        id: FoliageTypeId,
        descriptor: FoliageTypeDescriptor,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let old = self
            .authority
            .get::<SceneFoliageType>(entity)
            .copied()
            .ok_or_else(|| invalid("foliage type"))?;
        let old_material_entity = Entity::from_bits(old.material_entity_bits);
        let new_material_entity = entity_from_handle(descriptor.material_id);
        let material_row = self
            .authority
            .gpu_row::<SceneMaterial>(new_material_entity)
            .ok_or_else(|| invalid("material"))?;
        let new_row = SceneFoliageTypeRow::from(descriptor.to_gpu(material_row));
        if old.material_entity_bits == new_material_entity.bits()
            && bytemuck::bytes_of(&old.foliage) == bytemuck::bytes_of(&new_row)
        {
            return Ok(());
        }

        if old_material_entity != new_material_entity {
            self.authority
                .retain_material(new_material_entity)
                .map_err(super::errors::scene_asset)?;
            self.authority
                .release_material(old_material_entity)
                .map_err(super::errors::scene_asset)?;
        }
        self.authority
            .edit_gpu::<SceneFoliageType, _>(entity, |record| {
                record.material_entity_bits = new_material_entity.bits();
                record.foliage = new_row;
            })
            .ok_or_else(|| invalid("foliage type"))?;

        if old_material_entity != new_material_entity
            && self
                .authority
                .get::<SceneMaterial>(old_material_entity)
                .is_some_and(|material| material.ref_count == 0)
        {
            let _ = self.remove_material(handle_from_entity::<MaterialId>(old_material_entity));
        }
        self.recompute_foliage_type_stats();
        self.bump_foliage_topology();
        Ok(())
    }

    pub fn remove_foliage_type(&mut self, id: FoliageTypeId) -> Result<()> {
        let entity = entity_from_handle(id);
        let record = self
            .authority
            .get::<SceneFoliageType>(entity)
            .copied()
            .ok_or_else(|| invalid("foliage type"))?;
        let material_entity = Entity::from_bits(record.material_entity_bits);
        self.authority
            .release_material(material_entity)
            .map_err(super::errors::scene_asset)?;
        if !self.authority.despawn(entity) {
            return Err(invalid("foliage type"));
        }

        let compact = self
            .foliage_type_projection
            .remove(id)
            .expect("live foliage type must have a compact projection");
        debug_assert!(self.gpu_scene.foliage_type_indices.swap_remove(compact).is_some());

        let changed_layers: Vec<_> = self
            .authority
            .query::<SceneFoliageLayerTypes>()
            .filter_map(|(layer_entity, relation)| {
                if !relation.types.contains(&entity) {
                    return None;
                }
                let mut types = relation.types.clone();
                types.retain(|&candidate| candidate != entity);
                Some((layer_entity, types))
            })
            .collect();
        for (layer_entity, types) in changed_layers {
            self.authority
                .edit_cpu::<SceneFoliageLayerTypes, _>(layer_entity, |relation| {
                    relation.types = types
                })
                .expect("queried foliage relation must remain live");
        }
        self.rebuild_foliage_layer_projections();

        if self
            .authority
            .get::<SceneMaterial>(material_entity)
            .is_some_and(|material| material.ref_count == 0)
        {
            let _ = self.remove_material(handle_from_entity::<MaterialId>(material_entity));
        }
        self.recompute_foliage_type_stats();
        self.bump_foliage_topology();
        Ok(())
    }

    pub fn foliage_type_count(&self) -> u32 {
        self.foliage_type_projection.len() as u32
    }

    pub fn add_foliage_layer(&mut self, layer: FoliageLayer) -> Result<FoliageLayerId> {
        let type_entities = self.validate_foliage_layer_types(&layer.types)?;
        let entity = self.authority.insert(SceneFoliageLayer {
            foliage: SceneFoliageLayerRow::from(layer.to_gpu()),
            seed: layer.seed,
            _pad: 0,
        });
        assert!(self.authority.insert_cpu(
            entity,
            SceneFoliageLayerTypes {
                types: type_entities,
            },
        ));
        let row = self
            .authority
            .gpu_row::<SceneFoliageLayer>(entity)
            .expect("inserted foliage layer must own a canonical GPU row");
        let id = handle_from_entity(entity);
        self.foliage_layer_projection.insert(id, row);
        self.rebuild_foliage_layer_projections();
        self.bump_foliage_topology();
        Ok(id)
    }

    pub fn update_foliage_layer(
        &mut self,
        id: FoliageLayerId,
        layer: FoliageLayer,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let type_entities = self.validate_foliage_layer_types(&layer.types)?;
        let old_layer = self
            .authority
            .get::<SceneFoliageLayer>(entity)
            .copied()
            .ok_or_else(|| invalid("foliage layer"))?;
        let old_types = self
            .authority
            .get::<SceneFoliageLayerTypes>(entity)
            .ok_or_else(|| invalid("foliage layer"))?;
        let new_row = SceneFoliageLayerRow::from(layer.to_gpu());
        if old_layer.seed == layer.seed
            && bytemuck::bytes_of(&old_layer.foliage) == bytemuck::bytes_of(&new_row)
            && old_types.types == type_entities
        {
            return Ok(());
        }
        self.authority
            .edit_gpu::<SceneFoliageLayer, _>(entity, |record| {
                record.foliage = new_row;
                record.seed = layer.seed;
            })
            .ok_or_else(|| invalid("foliage layer"))?;
        self.authority
            .edit_cpu::<SceneFoliageLayerTypes, _>(entity, |relation| {
                relation.types = type_entities
            })
            .ok_or_else(|| invalid("foliage layer"))?;
        self.rebuild_foliage_layer_projections();
        self.bump_foliage_topology();
        Ok(())
    }

    pub fn remove_foliage_layer(&mut self, id: FoliageLayerId) -> Result<()> {
        let entity = entity_from_handle(id);
        if self.authority.get::<SceneFoliageLayer>(entity).is_none()
            || !self.authority.despawn(entity)
        {
            return Err(invalid("foliage layer"));
        }
        self.foliage_layer_projection
            .remove(id)
            .expect("live foliage layer must have a compact projection");
        self.rebuild_foliage_layer_projections();
        self.bump_foliage_topology();
        Ok(())
    }

    pub fn foliage_layer_count(&self) -> u32 {
        self.foliage_layer_projection.len() as u32
    }

    /// Replace authored wind parameters without re-rolling foliage topology.
    pub fn set_wind(&mut self, wind: Wind) {
        let current = self.wind();
        let mut wind = wind;
        wind.time = current.time;
        wind.prev_time = current.prev_time;
        let edited = self.authority.edit_cpu::<SceneWind, _>(self.wind_entity, |record| {
            record.wind = SceneWindRow::from(wind)
        });
        debug_assert!(edited.is_some(), "global SceneWind entity must remain live");
    }

    pub fn wind(&self) -> Wind {
        self.authority
            .get::<SceneWind>(self.wind_entity)
            .map(|record| Wind::from(record.wind))
            .expect("global SceneWind entity must remain live")
    }

    pub fn advance_wind(&mut self, dt: f32) {
        let edited = self.authority.edit_cpu::<SceneWind, _>(self.wind_entity, |record| {
            let mut wind = Wind::from(record.wind);
            wind.advance(dt);
            record.wind = SceneWindRow::from(wind);
        });
        debug_assert!(edited.is_some(), "global SceneWind entity must remain live");
    }

    pub(in crate::scene) fn foliage_layers_for_debug(
        &self,
    ) -> impl Iterator<Item = FoliageLayer> + '_ {
        self.foliage_layer_projection.ids().iter().filter_map(|&id| {
            let entity = entity_from_handle(id);
            let layer = self.authority.get::<SceneFoliageLayer>(entity)?;
            let types = self.authority.get::<SceneFoliageLayerTypes>(entity)?;
            Some(FoliageLayer {
                types: types
                    .types
                    .iter()
                    .copied()
                    .map(handle_from_entity)
                    .collect(),
                bounds: [
                    Vec3::from_array([
                        layer.foliage.0.bounds_min[0],
                        layer.foliage.0.bounds_min[1],
                        layer.foliage.0.bounds_min[2],
                    ]),
                    Vec3::from_array([
                        layer.foliage.0.bounds_max[0],
                        layer.foliage.0.bounds_max[1],
                        layer.foliage.0.bounds_max[2],
                    ]),
                ],
                seed: layer.seed,
                has_infinite_extent: layer.foliage.0.bounds_max[3] > 0.5,
            })
        })
    }

    pub fn add_foliage_interactor(&mut self, interactor: FoliageInteractor) -> FoliageInteractorId {
        let entity = self.authority.insert(SceneFoliageInteractor {
            interactor: SceneFoliageInteractorRow::from(interactor.to_gpu()),
        });
        let row = self
            .authority
            .gpu_row::<SceneFoliageInteractor>(entity)
            .expect("inserted foliage interactor must own a canonical GPU row");
        let id = handle_from_entity(entity);
        let compact = self.foliage_interactor_projection.insert(id, row);
        let gpu_compact = self.gpu_scene.foliage_interactor_indices.push(row);
        debug_assert_eq!(compact, gpu_compact);
        id
    }

    pub fn update_foliage_interactor(
        &mut self,
        id: FoliageInteractorId,
        position: Vec3,
        velocity: Vec3,
    ) -> Result<()> {
        let entity = entity_from_handle(id);
        let old = self
            .authority
            .get::<SceneFoliageInteractor>(entity)
            .copied()
            .ok_or_else(|| invalid("foliage interactor"))?;
        let new_row = SceneFoliageInteractorRow::from(
            FoliageInteractor {
                position,
                radius: old.interactor.0.position_radius[3],
                velocity,
            }
            .to_gpu(),
        );
        if bytemuck::bytes_of(&old.interactor) == bytemuck::bytes_of(&new_row) {
            return Ok(());
        }
        self.authority
            .edit_gpu::<SceneFoliageInteractor, _>(entity, |record| {
                record.interactor = new_row
            })
            .ok_or_else(|| invalid("foliage interactor"))?;
        Ok(())
    }

    pub fn remove_foliage_interactor(&mut self, id: FoliageInteractorId) -> Result<()> {
        let entity = entity_from_handle(id);
        if self.authority.get::<SceneFoliageInteractor>(entity).is_none()
            || !self.authority.despawn(entity)
        {
            return Err(invalid("foliage interactor"));
        }
        let compact = self
            .foliage_interactor_projection
            .remove(id)
            .expect("live foliage interactor must have a compact projection");
        debug_assert!(self
            .gpu_scene
            .foliage_interactor_indices
            .swap_remove(compact)
            .is_some());
        Ok(())
    }

    pub fn foliage_interactor_count(&self) -> u32 {
        self.foliage_interactor_projection.len() as u32
    }

    pub(in crate::scene) fn foliage_interactors_for_debug(
        &self,
    ) -> impl Iterator<Item = FoliageInteractor> + '_ {
        self.foliage_interactor_projection.ids().iter().filter_map(|&id| {
            let row = &self
                .authority
                .get::<SceneFoliageInteractor>(entity_from_handle(id))?
                .interactor
                .0;
            Some(FoliageInteractor {
                position: Vec3::from_array([
                    row.position_radius[0],
                    row.position_radius[1],
                    row.position_radius[2],
                ]),
                radius: row.position_radius[3],
                velocity: Vec3::from_array([row.velocity[0], row.velocity[1], row.velocity[2]]),
            })
        })
    }

    /// Direct canonical buffers plus compact row projections, or `None` for the
    /// zero-overhead no-types path.
    pub fn foliage_frame_data(&self) -> Option<FoliageFrameData<'_>> {
        if self.foliage_type_projection.is_empty() {
            return None;
        }
        let types = self.gpu_scene.canonical.foliage_types.as_ref()?;
        let layers = self.gpu_scene.canonical.foliage_layers.as_ref()?;
        let interactors = self.gpu_scene.canonical.foliage_interactors.as_ref()?;
        Some(FoliageFrameData {
            types: types.buffer(),
            type_rows: self.gpu_scene.foliage_type_indices.buffer(),
            layers: layers.buffer(),
            layer_projections: self.gpu_scene.foliage_layer_projections.buffer(),
            layer_type_relations: self.gpu_scene.foliage_layer_type_relations.buffer(),
            interactors: interactors.buffer(),
            interactor_rows: self.gpu_scene.foliage_interactor_indices.buffer(),
            type_count: self.foliage_type_projection.len() as u32,
            layer_count: self.foliage_layer_projection.len() as u32,
            layer_relation_count: self.gpu_scene.foliage_layer_type_relations.len() as u32,
            interactor_count: self.foliage_interactor_projection.len() as u32,
            max_height: self.foliage_max_height,
            max_density: self.foliage_max_density,
            type_epoch: types.epoch(),
            layer_epoch: layers.epoch(),
            interactor_epoch: interactors.epoch(),
            type_rows_epoch: self.gpu_scene.foliage_type_indices.buffer_version(),
            layer_projection_epoch: self.gpu_scene.foliage_layer_projections.buffer_version(),
            layer_relation_epoch: self
                .gpu_scene
                .foliage_layer_type_relations
                .buffer_version(),
            interactor_rows_epoch: self
                .gpu_scene
                .foliage_interactor_indices
                .buffer_version(),
            wind: self.wind().to_gpu(),
            generation: self.foliage_generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use glam::Vec3;
    use helio_scenedb::{
        SceneFoliageInteractor, SceneFoliageLayer, SceneFoliageLayerTypes,
        SceneFoliageType, SceneMaterial,
    };

    use super::*;

    fn create_test_scene() -> Scene {
        let (device, queue) = crate::test_support::test_gpu().expect("No test GPU adapter found");
        Scene::new(device, queue)
    }

    fn layer(types: Vec<FoliageTypeId>, seed: u32) -> FoliageLayer {
        FoliageLayer {
            types,
            bounds: [Vec3::splat(-10.0), Vec3::splat(10.0)],
            seed,
            has_infinite_extent: false,
        }
    }

    #[test]
    fn foliage_crud_keeps_canonical_rows_and_relationships_generation_safe() {
        let mut scene = create_test_scene();
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        let first = scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                density: 3.0,
                ..Default::default()
            })
            .unwrap();
        let second = scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                density: 7.0,
                ..Default::default()
            })
            .unwrap();
        let first_row = scene
            .authority
            .gpu_row::<SceneFoliageType>(entity_from_handle(first))
            .unwrap();
        let second_row = scene
            .authority
            .gpu_row::<SceneFoliageType>(entity_from_handle(second))
            .unwrap();

        let authored = scene.add_foliage_layer(layer(vec![first, second], 91)).unwrap();
        assert_eq!(scene.gpu_scene.foliage_layer_projections.len(), 1);
        assert_eq!(scene.gpu_scene.foliage_layer_type_relations.len(), 2);
        assert_eq!(
            scene.gpu_scene.foliage_layer_type_relations.as_slice(),
            &[
                GpuFoliageLayerTypeRelation {
                    compact_type_id: 0,
                    canonical_type_row: first_row,
                },
                GpuFoliageLayerTypeRelation {
                    compact_type_id: 1,
                    canonical_type_row: second_row,
                },
            ]
        );

        scene.remove_foliage_type(first).unwrap();
        assert_eq!(scene.gpu_scene.foliage_type_indices.as_slice(), &[second_row]);
        assert_eq!(
            scene.gpu_scene.foliage_layer_type_relations.as_slice(),
            &[GpuFoliageLayerTypeRelation {
                compact_type_id: 0,
                canonical_type_row: second_row,
            }]
        );
        let relation = scene
            .authority
            .get::<SceneFoliageLayerTypes>(entity_from_handle(authored))
            .unwrap();
        assert_eq!(relation.types, vec![entity_from_handle(second)]);

        assert!(scene.add_foliage_layer(layer(vec![first], 92)).is_err());
        assert!(scene
            .update_foliage_layer(authored, layer(vec![first], 93))
            .is_err());
        assert_eq!(
            scene
                .authority
                .get::<SceneFoliageLayer>(entity_from_handle(authored))
                .unwrap()
                .seed,
            91
        );

        let replacement = scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                ..Default::default()
            })
            .unwrap();
        assert_ne!(replacement, first, "reused entity slots must advance generation");
        assert_eq!(
            scene
                .authority
                .gpu_row::<SceneFoliageType>(entity_from_handle(replacement)),
            Some(first_row),
            "component-local rows may be safely reused after despawn"
        );
    }

    #[test]
    fn foliage_has_no_layer_cap_and_enforces_the_u8_type_boundary() {
        let mut scene = create_test_scene();
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        let foliage_type = scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                ..Default::default()
            })
            .unwrap();

        for seed in 0..65 {
            scene
                .add_foliage_layer(layer(vec![foliage_type], seed))
                .unwrap();
        }
        assert_eq!(scene.foliage_layer_count(), 65);
        assert_eq!(scene.gpu_scene.foliage_layer_projections.len(), 65);
        assert_eq!(scene.gpu_scene.foliage_layer_type_relations.len(), 65);
        for (seed, projection) in scene
            .gpu_scene
            .foliage_layer_projections
            .as_slice()
            .iter()
            .enumerate()
        {
            assert_eq!(projection.seed, seed as u32);
            assert_eq!(projection.relation_offset, seed as u32);
            assert_eq!(projection.relation_count, 1);
        }

        for _ in 1..MAX_PACKED_FOLIAGE_TYPES {
            scene
                .add_foliage_type(FoliageTypeDescriptor {
                    material_id: material,
                    ..Default::default()
                })
                .unwrap();
        }
        assert_eq!(scene.foliage_type_count(), 256);
        assert!(scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                ..Default::default()
            })
            .is_err());
        assert_eq!(
            scene
                .authority
                .get::<SceneMaterial>(entity_from_handle(material))
                .unwrap()
                .ref_count,
            256,
            "a rejected 257th type must not retain the material"
        );
    }

    #[test]
    fn foliage_clear_removes_all_authority_and_wind_is_not_topology() {
        let mut scene = create_test_scene();
        let material = scene.insert_material(libhelio::GpuMaterial::zeroed());
        let foliage_type = scene
            .add_foliage_type(FoliageTypeDescriptor {
                material_id: material,
                ..Default::default()
            })
            .unwrap();
        scene
            .add_foliage_layer(layer(vec![foliage_type], 17))
            .unwrap();
        let first = scene.add_foliage_interactor(FoliageInteractor {
            position: Vec3::X,
            radius: 2.0,
            velocity: Vec3::Y,
        });
        let second = scene.add_foliage_interactor(FoliageInteractor {
            position: Vec3::Z,
            radius: 3.0,
            velocity: Vec3::NEG_X,
        });
        let first_row = scene
            .authority
            .gpu_row::<SceneFoliageInteractor>(entity_from_handle(first))
            .unwrap();
        let second_row = scene
            .authority
            .gpu_row::<SceneFoliageInteractor>(entity_from_handle(second))
            .unwrap();
        scene.remove_foliage_interactor(first).unwrap();
        assert_eq!(scene.gpu_scene.foliage_interactor_indices.as_slice(), &[second_row]);
        let replacement = scene.add_foliage_interactor(FoliageInteractor {
            position: Vec3::ZERO,
            radius: 1.0,
            velocity: Vec3::ZERO,
        });
        assert_ne!(first, replacement);
        assert_eq!(
            scene
                .authority
                .gpu_row::<SceneFoliageInteractor>(entity_from_handle(replacement)),
            Some(first_row)
        );

        let topology = scene.foliage_generation;
        scene.set_wind(Wind {
            direction: Vec3::new(0.5, 0.0, 0.25),
            speed: 4.0,
            ..Default::default()
        });
        scene.advance_wind(0.25);
        assert_eq!(scene.foliage_generation, topology);

        scene.clear();
        assert_eq!(scene.foliage_type_count(), 0);
        assert_eq!(scene.foliage_layer_count(), 0);
        assert_eq!(scene.foliage_interactor_count(), 0);
        assert_eq!(scene.authority.gpu_live_count::<SceneFoliageType>(), 0);
        assert_eq!(scene.authority.gpu_live_count::<SceneFoliageLayer>(), 0);
        assert_eq!(scene.authority.gpu_live_count::<SceneFoliageInteractor>(), 0);
        assert!(scene.gpu_scene.foliage_type_indices.is_empty());
        assert!(scene.gpu_scene.foliage_layer_projections.is_empty());
        assert!(scene.gpu_scene.foliage_layer_type_relations.is_empty());
        assert!(scene.gpu_scene.foliage_interactor_indices.is_empty());
        assert!(scene
            .authority
            .get::<SceneMaterial>(entity_from_handle(material))
            .is_none());
    }
}
