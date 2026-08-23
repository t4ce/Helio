use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelOp {
    SetBox,
    AddSphere,
    SubtractSphere,
    Paint,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelEdit {
    pub op: VoxelOp,
    pub center: Vec3,
    pub radius: f32,
    pub material: u8,
}
