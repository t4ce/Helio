// ---- GPU uniform structs (simulation parameters) ---------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DropUniform {
    pub world_center: [f32; 2],
    pub radius: f32,
    pub strength: f32,
    pub volume_row: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DeltaUniform {
    pub delta: [f32; 2],
    /// Pass-owned elapsed time. The shader applies the selected canonical
    /// volume's authored wave speed without retaining a second CPU copy.
    pub time: f32,
    pub time_step: f32,
    pub cascade_patch_size: f32,
    /// Component-local SceneDB row selected for this stable simulation slot.
    pub volume_row: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NormalUniform {
    pub delta: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HitboxCountUniform {
    pub count: u32,
    pub _pad: [u32; 3],
}
