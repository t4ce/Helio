use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct GpuVoxelVolume {
    pub local_to_world: [f32; 16],
    pub world_to_local: [f32; 16],
    pub dimensions: [u32; 3],
    pub brick_grid_dim: u32,
    pub voxel_size: f32,
    pub palette_offset: u32,
    /// First brick-metadata row in the SceneDB voxel residency subsystem.
    pub brick_offset: u32,
    /// Number of authored entries in this volume's stable palette region.
    pub palette_count: u32,
    pub _pad: [u32; 2],
    /// WGSL rounds storage-array struct stride to the 16-byte struct
    /// alignment. Keep those tail bytes explicit so Rust has the same
    /// 176-byte stride and bytemuck never reads implicit padding.
    pub _pad_tail: [u32; 2],
}

const _: () = {
    assert!(std::mem::offset_of!(GpuVoxelVolume, dimensions) == 128);
    assert!(std::mem::offset_of!(GpuVoxelVolume, voxel_size) == 144);
    assert!(std::mem::offset_of!(GpuVoxelVolume, brick_offset) == 152);
    assert!(std::mem::offset_of!(GpuVoxelVolume, palette_count) == 156);
    assert!(std::mem::offset_of!(GpuVoxelVolume, _pad) == 160);
    assert!(std::mem::size_of::<GpuVoxelVolume>() == 176);
    assert!(std::mem::align_of::<GpuVoxelVolume>() == 16);
};

/// One stable output slot in Helio's Auto-voxel mesh projection.
///
/// Rows are indexed by output slot. `generation` identifies the coalesced
/// upload/removal batch that must be processed; graph-rebuilt passes may
/// bootstrap every allocated row regardless of generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelMeshWork {
    /// Component-local `SceneVoxelVolume` GPU row.
    pub volume_row: u32,
    /// Brick index relative to the volume's canonical residency region.
    pub local_brick: u32,
    /// `VOXEL_MESH_WORK_ALLOCATED` when this output slot has a live owner.
    pub flags: u32,
    /// Low 32 bits of the Helio-derived coalesced work generation.
    pub generation: u32,
}

pub const VOXEL_MESH_WORK_ALLOCATED: u32 = 1;

/// Interleaved mesh-pass output vertex. This replaces two independently bound
/// position/normal buffers so extraction remains within the eight-storage
/// minimum limit after binding canonical volume rows.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelMeshVertex {
    /// xyz = world position, w = material id encoded as an exact small f32.
    pub position_material: [f32; 4],
    /// xyz = world normal, w = canonical volume row bitcast to f32.
    pub normal_volume: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<GpuVoxelMeshWork>() == 16);
    assert!(std::mem::size_of::<GpuVoxelMeshVertex>() == 32);
};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBrickMeta {
    pub data_offset: u32,
    pub occupancy: u32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBrickMeshlet {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub brick_index: u32,
    pub volume_id: u32,
    pub _pad: [u32; 2],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelMaterial {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: f32,
    pub _pad: [u32; 2],
}
