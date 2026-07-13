pub mod rendering;

pub use rendering::VirtualGeometryPass;

use bytemuck::{Pod, Zeroable};
use helio_core::GpuInstanceData;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Bindless texture array size per shader stage.
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(any(target_arch = "wasm32", target_os = "macos", target_os = "ios")))]
pub(crate) const MAX_TEXTURES: usize = 256;
#[cfg(any(target_arch = "wasm32", target_os = "macos", target_os = "ios"))]
pub(crate) const MAX_TEXTURES: usize = 16;

pub const LOD_LEVEL_COUNT: u32 = 8;

pub(crate) const INITIAL_MESHLETS: u64 = 1024;
pub(crate) const INITIAL_INSTANCES: u64 = 256;

/// Per-instance values used by meshlet culling. Kept separate from
/// `GpuInstanceData` so the cull shader does not recompute matrix norms for
/// every meshlet in an instance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct InstanceCullData {
    pub max_scale: f32,
    pub min_scale: f32,
    pub cone_cull_enabled: u32,
    pub valid_transform: u32,
}

impl InstanceCullData {
    pub(crate) fn from_instance(instance: &GpuInstanceData) -> Self {
        let model = &instance.model;
        let scale_x = (model[0] * model[0] + model[1] * model[1] + model[2] * model[2]).sqrt();
        let scale_y = (model[4] * model[4] + model[5] * model[5] + model[6] * model[6]).sqrt();
        let scale_z = (model[8] * model[8] + model[9] * model[9] + model[10] * model[10]).sqrt();
        let max_scale = scale_x.max(scale_y).max(scale_z);
        let min_scale = scale_x.min(scale_y).min(scale_z);
        let dot_xy = model[0] * model[4] + model[1] * model[5] + model[2] * model[6];
        let dot_xz = model[0] * model[8] + model[1] * model[9] + model[2] * model[10];
        let dot_yz = model[4] * model[8] + model[5] * model[9] + model[6] * model[10];
        let determinant = model[0] * (model[5] * model[10] - model[6] * model[9])
            - model[4] * (model[1] * model[10] - model[2] * model[9])
            + model[8] * (model[1] * model[6] - model[2] * model[5]);
        let affine = model[3].abs() <= 1.0e-6
            && model[7].abs() <= 1.0e-6
            && model[11].abs() <= 1.0e-6
            && (model[15] - 1.0).abs() <= 1.0e-4;
        let valid = model.iter().all(|value| value.is_finite())
            && max_scale.is_finite()
            && min_scale > 1.0e-8
            && affine;
        let uniform_tolerance = max_scale * 1.0e-4;
        let orthogonal_tolerance = max_scale * max_scale * 1.0e-4;
        let angle_preserving = max_scale - min_scale <= uniform_tolerance
            && dot_xy.abs() <= orthogonal_tolerance
            && dot_xz.abs() <= orthogonal_tolerance
            && dot_yz.abs() <= orthogonal_tolerance;
        let cone_cull_enabled = valid && angle_preserving && determinant > 0.0;

        Self {
            max_scale: if valid { max_scale } else { 0.0 },
            min_scale: if valid { min_scale } else { 0.0 },
            cone_cull_enabled: u32::from(cone_cull_enabled),
            valid_transform: u32::from(valid),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lod quality preset
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LodQuality {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl LodQuality {
    pub fn thresholds(self) -> [f32; 7] {
        match self {
            LodQuality::Low => [0.180, 0.120, 0.080, 0.050, 0.030, 0.015, 0.006],
            LodQuality::Medium => [0.050, 0.035, 0.022, 0.014, 0.008, 0.004, 0.002],
            LodQuality::High => [0.020, 0.014, 0.009, 0.005, 0.003, 0.0015, 0.0006],
            LodQuality::Ultra => [0.008, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU uniform types
// ═══════════════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VgGlobals {
    pub frame: u32,
    pub delta_time: f32,
    pub light_count: u32,
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 4],
    pub rc_world_min: [f32; 4],
    pub rc_world_max: [f32; 4],
    pub csm_splits: [f32; 4],
    pub debug_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullUniforms {
    pub meshlet_count: u32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub hiz_mip_count: u32,
    pub lod_thresholds: [f32; 7],
    _pad3: f32,
}

#[cfg(test)]
mod tests {
    use super::InstanceCullData;
    use helio_core::GpuInstanceData;

    fn instance_with_model(model: [f32; 16]) -> GpuInstanceData {
        GpuInstanceData {
            model,
            normal_mat: [0.0; 12],
            bounds: [0.0; 4],
            mesh_id: 0,
            material_id: 0,
            flags: 0,
            lightmap_index: u32::MAX,
        }
    }

    #[test]
    fn instance_cull_data_is_gpu_aligned() {
        assert_eq!(std::mem::size_of::<InstanceCullData>(), 16);
    }

    #[test]
    fn uniform_scale_keeps_cone_culling_enabled() {
        let instance = instance_with_model([
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 5.0, 6.0, 1.0,
        ]);
        let cull = InstanceCullData::from_instance(&instance);

        assert_eq!(cull.valid_transform, 1);
        assert_eq!(cull.cone_cull_enabled, 1);
        assert_eq!(cull.max_scale, 2.0);
        assert_eq!(cull.min_scale, 2.0);
    }

    #[test]
    fn non_uniform_scale_disables_cone_culling_but_keeps_conservative_radius() {
        let instance = instance_with_model([
            4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let cull = InstanceCullData::from_instance(&instance);

        assert_eq!(cull.valid_transform, 1);
        assert_eq!(cull.cone_cull_enabled, 0);
        assert_eq!(cull.max_scale, 4.0);
        assert_eq!(cull.min_scale, 1.0);
    }

    #[test]
    fn singular_or_non_finite_transform_is_rejected() {
        let singular = instance_with_model([0.0; 16]);
        assert_eq!(
            InstanceCullData::from_instance(&singular).valid_transform,
            0
        );

        let mut non_finite_model = [0.0; 16];
        non_finite_model[0] = f32::NAN;
        let non_finite = instance_with_model(non_finite_model);
        assert_eq!(
            InstanceCullData::from_instance(&non_finite).valid_transform,
            0
        );
    }

    #[test]
    fn shear_or_reflection_disables_cone_culling() {
        let shear = instance_with_model([
            1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(InstanceCullData::from_instance(&shear).cone_cull_enabled, 0);

        let reflection = instance_with_model([
            -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(
            InstanceCullData::from_instance(&reflection).cone_cull_enabled,
            0
        );
    }
}
