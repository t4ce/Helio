use helio_voxel_core::{
    GpuVoxelMaterial, GpuVoxelVolume, VoxelEdit, VoxelOctree, BRICK_SIZE,
};
use helio_scenedb::CpuOnlyComponent;

use super::types::VoxelVolumeDescriptor;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelMode {
    Auto = 0,
    Dynamic = 1,
}

impl Default for VoxelMode {
    fn default() -> Self { VoxelMode::Auto }
}

#[derive(Debug)]
pub struct VoxelVolumeRecord {
    pub octree: VoxelOctree,
    pub meshlet_offset: u32,
    pub meshlet_count: u32,
    pub edit_cooldown: u32,
    pub material_palette: Vec<GpuVoxelMaterial>,
}

impl CpuOnlyComponent for VoxelVolumeRecord {}

impl VoxelVolumeRecord {
    pub fn new(descriptor: &VoxelVolumeDescriptor) -> Self {
        let octree = VoxelOctree::new(descriptor.voxel_size, descriptor.root_extent);
        Self {
            octree,
            meshlet_offset: 0,
            meshlet_count: 0,
            edit_cooldown: 0,
            material_palette: descriptor.material_palette.clone(),
        }
    }

    fn grid_dimension(&self) -> u32 {
        let root_extent = self.octree.root.aabb_max[0] - self.octree.root.aabb_min[0];
        (root_extent / self.octree.voxel_size).round().max(1.0) as u32
    }

    pub(crate) fn brick_count(&self) -> Option<u32> {
        let axis = self.grid_dimension().div_ceil(BRICK_SIZE);
        axis.checked_mul(axis)?.checked_mul(axis)
    }

    /// Build the shader-exact authored row. SceneDB owns the returned row;
    /// Helio only retains a compact component-row projection for dense loops.
    pub(crate) fn authored_gpu_row(
        &self,
        local_to_world: glam::Mat4,
        brick_offset: u32,
        palette_offset: u32,
        palette_count: u32,
    ) -> GpuVoxelVolume {
        let dimension = self.grid_dimension();
        GpuVoxelVolume {
            local_to_world: local_to_world.to_cols_array(),
            world_to_local: local_to_world.inverse().to_cols_array(),
            dimensions: [dimension; 3],
            brick_grid_dim: dimension.div_ceil(BRICK_SIZE),
            voxel_size: self.octree.voxel_size,
            palette_offset,
            brick_offset,
            palette_count,
            _pad: [0; 2],
            _pad_tail: [0; 2],
        }
    }

    pub fn edit(&mut self, edit: &VoxelEdit) {
        let center = [
            edit.center.x,
            edit.center.y,
            edit.center.z,
        ];
        self.octree.root.mark_sphere_dirty(center, edit.radius);
        self.edit_cooldown = 0;
    }
}
