//! Material resource management backed exclusively by SceneDB components.

use helio_core::GpuMaterial;
use helio_scenedb::{SceneMaterial, SceneMaterialTextureRefs, SceneTexture};

use crate::handles::{entity_from_handle, handle_from_entity, MaterialId};
use crate::material::MaterialAsset;

use super::super::errors::{invalid, scene_asset, Result};
use super::super::helpers::scene_material_texture_refs;

#[inline]
fn material_batch_signature(material: GpuMaterial, graph_hash: u64) -> (u32, u64, u32) {
    const BUCKET_FLAGS: u32 = libhelio::FLAG_TRANSPARENT_ONLY | libhelio::FLAG_FORWARD_SHADING;
    (
        material.material_class,
        graph_hash,
        material.flags & BUCKET_FLAGS,
    )
}

#[inline]
fn material_vg_cull_signature(flags: u32) -> u32 {
    flags & libhelio::FLAG_ALPHA_TEST
}

impl super::super::Scene {
    /// Insert a material without texture relations.
    pub fn insert_material(&mut self, material: GpuMaterial) -> MaterialId {
        self.insert_material_asset(material.into())
            .expect("plain GPU materials have no invalid texture relation")
    }

    /// Insert authored material state into SceneDB. The public handle is the
    /// exact generation-bearing Entity bits; objects and material passes use
    /// the independently resolved, component-local `SceneMaterial` GPU row.
    pub fn insert_material_asset(&mut self, material: MaterialAsset) -> Result<MaterialId> {
        let refs = scene_material_texture_refs(&material.textures);
        let entity = self
            .authority
            .insert_material_asset(material.gpu, refs, 0)
            .map_err(scene_asset)?;
        self.cache_material_projection(entity, material.gpu);
        Ok(handle_from_entity(entity))
    }

    /// Update shader parameters while preserving canonical texture relations
    /// and graph identity.
    pub fn update_material(&mut self, id: MaterialId, material: GpuMaterial) -> Result<()> {
        let entity = entity_from_handle(id);
        let canonical = *self
            .authority
            .get::<SceneMaterial>(entity)
            .ok_or_else(|| invalid("material"))?;
        let refs = *self
            .authority
            .get::<SceneMaterialTextureRefs>(entity)
            .ok_or_else(|| invalid("material"))?;
        self.authority
            .update_material_asset(entity, material, refs, canonical.graph_hash)
            .map_err(scene_asset)?;
        self.cache_material_projection(entity, material);
        self.note_vg_material_cull_change(canonical.material.0.flags, material.flags);
        if material_batch_signature(canonical.material.0, canonical.graph_hash)
            != material_batch_signature(material, canonical.graph_hash)
        {
            self.objects_dirty = true;
        }
        Ok(())
    }

    /// Update shader parameters and generation-bearing texture relations.
    pub fn update_material_asset(&mut self, id: MaterialId, material: MaterialAsset) -> Result<()> {
        let entity = entity_from_handle(id);
        let (old_material, graph_hash) = self
            .authority
            .get::<SceneMaterial>(entity)
            .map(|component| (component.material.0, component.graph_hash))
            .ok_or_else(|| invalid("material"))?;
        let refs = scene_material_texture_refs(&material.textures);
        self.authority
            .update_material_asset(entity, material.gpu, refs, graph_hash)
            .map_err(scene_asset)?;
        self.cache_material_projection(entity, material.gpu);
        self.note_vg_material_cull_change(old_material.flags, material.gpu.flags);
        if material_batch_signature(old_material, graph_hash)
            != material_batch_signature(material.gpu, graph_hash)
        {
            self.objects_dirty = true;
        }
        Ok(())
    }

    /// Remove an unused material, release its texture pins, and preserve the
    /// historical cascade that removes now-unreferenced textures.
    pub fn remove_material(&mut self, id: MaterialId) -> Result<()> {
        let entity = entity_from_handle(id);
        let material_row = self
            .authority
            .gpu_row::<SceneMaterial>(entity)
            .ok_or_else(|| invalid("material"))?;
        let (_, refs) = self
            .authority
            .remove_material_asset(entity)
            .map_err(scene_asset)?;
        self.clear_material_projection(material_row as usize);

        let mut unique_textures = Vec::with_capacity(7);
        for reference in refs.references() {
            if !unique_textures.contains(&reference.texture) {
                unique_textures.push(reference.texture);
            }
        }
        for texture in unique_textures {
            if self
                .authority
                .get::<SceneTexture>(texture)
                .is_some_and(|component| component.ref_count == 0)
            {
                self.authority
                    .remove_texture_asset(texture)
                    .map_err(scene_asset)?;
            }
        }
        Ok(())
    }

    pub(in crate::scene) fn cache_material_projection(
        &mut self,
        entity: helio_scenedb::Entity,
        material: GpuMaterial,
    ) {
        let row = self
            .authority
            .gpu_row::<SceneMaterial>(entity)
            .expect("live SceneMaterial must have a registered GPU row") as usize;
        if self.gpu_scene.material_flags.len() <= row {
            self.gpu_scene.material_flags.resize(row + 1, 0);
        }
        self.gpu_scene.material_flags[row] = material.flags;
    }

    pub(in crate::scene) fn note_vg_material_cull_change(
        &mut self,
        old_flags: u32,
        new_flags: u32,
    ) {
        if material_vg_cull_signature(old_flags) != material_vg_cull_signature(new_flags) {
            self.vg_cull_signature_version = self.vg_cull_signature_version.wrapping_add(1);
        }
    }

    pub(in crate::scene) fn note_material_batch_change(
        &mut self,
        old_material: GpuMaterial,
        old_graph_hash: u64,
        new_material: GpuMaterial,
        new_graph_hash: u64,
    ) {
        if material_batch_signature(old_material, old_graph_hash)
            != material_batch_signature(new_material, new_graph_hash)
        {
            self.objects_dirty = true;
        }
    }

    fn clear_material_projection(&mut self, row: usize) {
        if let Some(flags) = self.gpu_scene.material_flags.get_mut(row) {
            *flags = 0;
        }
    }
}
