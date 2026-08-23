//! Canonical voxel-volume metadata and its shader partner row.

use std::collections::HashMap;
use std::sync::Arc;

use pulsar_scenedb::gpu::SceneGpuStore;
use pulsar_scenedb::page::Pod as SceneDbPod;
use pulsar_scenedb::{Entity, Subsystem};
use pulsar_scenedb_derive::SceneStore;

pub const VOXEL_VOLUME_BUFFER_KEY: &str = "helio.scene.voxel_volumes";
pub const VOXEL_BRICK_BUFFER_KEY: &str = "helio.scene.voxel_bricks";
pub const VOXEL_DATA_BUFFER_KEY: &str = "helio.scene.voxel_data";
pub const VOXEL_PALETTE_BUFFER_KEY: &str = "helio.scene.voxel_palettes";

#[repr(transparent)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneVoxelVolumeRow(pub helio_voxel_core::GpuVoxelVolume);

// SAFETY: the wrapped shader row is bytemuck::Pod, explicitly padded to its
// 176-byte WGSL storage-array stride, and has no invalid bit patterns.
unsafe impl SceneDbPod for SceneVoxelVolumeRow {}

/// Persistent fixed-size voxel-volume state. The octree and material palette
/// are a second CPU-only component on the same Entity because SceneStore's
/// GPU-field contract intentionally remains POD/Copy.
#[repr(C)]
#[derive(Debug, Clone, Copy, SceneStore)]
pub struct SceneVoxelVolume {
    #[gpu(buffer = "helio.scene.voxel_volumes")]
    pub volume: SceneVoxelVolumeRow,
    /// Validated `libhelio::Movability` storage value.
    pub movability: u32,
    /// 0 = Auto mesh path, 1 = Dynamic ray-march path.
    pub mode: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl crate::storage::MutableGpuComponent for SceneVoxelVolume {}

pub fn register_voxel_volume_buffer(store: &mut SceneGpuStore, device: &Arc<wgpu::Device>) {
    SceneVoxelVolume::register_gpu_columns_growable(store, 1, device);
}

/// SceneDB-registered residency for canonical voxel asset payloads.
///
/// Metadata rows remain SceneDB World components; variable-size brick/data
/// bytes live in this subsystem so `GpuScene` never becomes a second owner.
/// Every Auto and Dynamic volume receives stable brick/data and palette
/// regions. Palette regions are power-of-two sized and relocate only when an
/// authored update outgrows them; the owning canonical volume row is then
/// updated before the old region is recycled. Regions coalesce and recycle
/// only for the exact owning Entity generation.
pub struct VoxelResidency {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    brick_buffer: Box<wgpu::Buffer>,
    data_buffer: Box<wgpu::Buffer>,
    palette_buffer: Box<wgpu::Buffer>,
    capacity_bricks: u32,
    capacity_palette_rows: u32,
    next_brick: u32,
    next_palette_row: u32,
    free_regions: Vec<BrickRegion>,
    free_palette_regions: Vec<BrickRegion>,
    region_by_entity: HashMap<Entity, VoxelRegion>,
    epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BrickRegion {
    base: u32,
    count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VoxelRegion {
    bricks: BrickRegion,
    palette: BrickRegion,
    palette_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelResidencyAllocation {
    pub brick_base: u32,
    pub palette_base: u32,
    pub palette_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelResidencyError {
    AlreadyResident,
    NotResident,
    InvalidRegion,
    InvalidBrick,
    PaletteTooLarge,
    CapacityExceeded,
}

impl std::fmt::Display for VoxelResidencyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::AlreadyResident => "voxel volume already has a residency region",
            Self::NotResident => "voxel volume has no residency region",
            Self::InvalidRegion => "voxel residency region must contain at least one brick",
            Self::InvalidBrick => "voxel brick index or payload is invalid",
            Self::PaletteTooLarge => "voxel palette exceeds the 8-bit material domain",
            Self::CapacityExceeded => "voxel residency exceeds the device storage-buffer limit",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for VoxelResidencyError {}

impl VoxelResidency {
    const BRICK_META_STRIDE: u64 = 8;
    const DATA_WORD_BYTES: u64 = 4;
    const PALETTE_ROW_BYTES: u64 = std::mem::size_of::<helio_voxel_core::GpuVoxelMaterial>() as u64;
    // Brick metadata packs the data-word offset into 24 bits. Every brick has
    // 128 words, so this is the exact addressable brick count independent of
    // how differently sized volume regions divide it.
    pub const MAX_ADDRESSABLE_BRICKS: u32 =
        (1 << 24) / helio_voxel_core::RAYMARCH_WORDS_PER_BRICK;
    pub const MAX_PALETTE_ROWS: u32 =
        helio_voxel_core::MAX_VOLUMES * helio_voxel_core::MAX_PALETTE_SIZE;

    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let capacity_bricks = helio_voxel_core::DEFAULT_VOLUME_BRICK_COUNT;
        let (brick_bytes, data_bytes) = Self::buffer_sizes(capacity_bricks)
            .expect("one default voxel region must fit the device-independent u64 domain");
        let brick_buffer = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(VOXEL_BRICK_BUFFER_KEY),
            size: brick_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let data_buffer = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(VOXEL_DATA_BUFFER_KEY),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let capacity_palette_rows = helio_voxel_core::MAX_PALETTE_SIZE;
        let palette_buffer = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(VOXEL_PALETTE_BUFFER_KEY),
            size: u64::from(capacity_palette_rows) * Self::PALETTE_ROW_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        Self {
            device,
            queue,
            brick_buffer,
            data_buffer,
            palette_buffer,
            capacity_bricks,
            capacity_palette_rows,
            next_brick: 0,
            next_palette_row: 0,
            free_regions: Vec::new(),
            free_palette_regions: Vec::new(),
            region_by_entity: HashMap::new(),
            epoch: 0,
        }
    }

    fn buffer_sizes(capacity_bricks: u32) -> Option<(u64, u64)> {
        let bricks = u64::from(capacity_bricks);
        let brick_bytes = bricks.checked_mul(Self::BRICK_META_STRIDE)?;
        let data_bytes = bricks
            .checked_mul(u64::from(helio_voxel_core::RAYMARCH_WORDS_PER_BRICK))?
            .checked_mul(Self::DATA_WORD_BYTES)?;
        Some((brick_bytes, data_bytes))
    }

    fn grow_to_include(&mut self, required_bricks: u32) -> Result<(), VoxelResidencyError> {
        if required_bricks <= self.capacity_bricks {
            return Ok(());
        }
        if required_bricks > Self::MAX_ADDRESSABLE_BRICKS {
            return Err(VoxelResidencyError::CapacityExceeded);
        }
        let mut next = self.capacity_bricks;
        while next < required_bricks {
            next = next
                .checked_mul(2)
                .ok_or(VoxelResidencyError::CapacityExceeded)?;
        }
        next = next.min(Self::MAX_ADDRESSABLE_BRICKS);
        let (old_brick_bytes, old_data_bytes) = Self::buffer_sizes(self.capacity_bricks)
            .ok_or(VoxelResidencyError::CapacityExceeded)?;
        let (brick_bytes, data_bytes) =
            Self::buffer_sizes(next).ok_or(VoxelResidencyError::CapacityExceeded)?;
        let max_binding = u64::from(self.device.limits().max_storage_buffer_binding_size);
        let max_buffer = self.device.limits().max_buffer_size;
        if brick_bytes > max_binding.min(max_buffer) || data_bytes > max_binding.min(max_buffer) {
            return Err(VoxelResidencyError::CapacityExceeded);
        }

        let brick_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(VOXEL_BRICK_BUFFER_KEY),
            size: brick_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let data_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(VOXEL_DATA_BUFFER_KEY),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-residency-grow"),
            });
        encoder.copy_buffer_to_buffer(
            self.brick_buffer.as_ref(),
            0,
            &brick_buffer,
            0,
            old_brick_bytes,
        );
        encoder.copy_buffer_to_buffer(
            self.data_buffer.as_ref(),
            0,
            &data_buffer,
            0,
            old_data_bytes,
        );
        self.queue.submit([encoder.finish()]);
        self.brick_buffer = Box::new(brick_buffer);
        self.data_buffer = Box::new(data_buffer);
        self.capacity_bricks = next;
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    fn grow_palette_to_include(
        &mut self,
        required_rows: u32,
    ) -> Result<(), VoxelResidencyError> {
        if required_rows <= self.capacity_palette_rows {
            return Ok(());
        }
        if required_rows > Self::MAX_PALETTE_ROWS {
            return Err(VoxelResidencyError::CapacityExceeded);
        }
        let mut next = self.capacity_palette_rows;
        while next < required_rows {
            next = next
                .checked_mul(2)
                .ok_or(VoxelResidencyError::CapacityExceeded)?;
        }
        next = next.min(Self::MAX_PALETTE_ROWS);
        let old_bytes = u64::from(self.capacity_palette_rows) * Self::PALETTE_ROW_BYTES;
        let next_bytes = u64::from(next) * Self::PALETTE_ROW_BYTES;
        let max_binding = u64::from(self.device.limits().max_storage_buffer_binding_size);
        if next_bytes > max_binding.min(self.device.limits().max_buffer_size) {
            return Err(VoxelResidencyError::CapacityExceeded);
        }

        let palette_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(VOXEL_PALETTE_BUFFER_KEY),
            size: next_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-palette-residency-grow"),
            });
        encoder.copy_buffer_to_buffer(
            self.palette_buffer.as_ref(),
            0,
            &palette_buffer,
            0,
            old_bytes,
        );
        self.queue.submit([encoder.finish()]);
        self.palette_buffer = Box::new(palette_buffer);
        self.capacity_palette_rows = next;
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    fn clear_region(&self, region: BrickRegion) {
        let (brick_bytes, data_bytes) =
            Self::buffer_sizes(region.count).expect("validated voxel region size");
        let data_word_offset = u64::from(region.base)
            * u64::from(helio_voxel_core::RAYMARCH_WORDS_PER_BRICK);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-residency-clear-region"),
            });
        encoder.clear_buffer(
            self.brick_buffer.as_ref(),
            u64::from(region.base) * Self::BRICK_META_STRIDE,
            Some(brick_bytes),
        );
        encoder.clear_buffer(
            self.data_buffer.as_ref(),
            data_word_offset * Self::DATA_WORD_BYTES,
            Some(data_bytes),
        );
        self.queue.submit([encoder.finish()]);
    }

    fn clear_palette_region(&self, region: BrickRegion) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-palette-residency-clear-region"),
            });
        encoder.clear_buffer(
            self.palette_buffer.as_ref(),
            u64::from(region.base) * Self::PALETTE_ROW_BYTES,
            Some(u64::from(region.count) * Self::PALETTE_ROW_BYTES),
        );
        self.queue.submit([encoder.finish()]);
    }

    fn take_free_region(&mut self, count: u32) -> Option<BrickRegion> {
        let index = self
            .free_regions
            .iter()
            .position(|region| region.count >= count)?;
        let free = self.free_regions[index];
        let allocation = BrickRegion {
            base: free.base,
            count,
        };
        if free.count == count {
            self.free_regions.remove(index);
        } else {
            self.free_regions[index].base += count;
            self.free_regions[index].count -= count;
        }
        Some(allocation)
    }

    fn recycle_region(&mut self, region: BrickRegion) {
        self.free_regions.push(region);
        self.free_regions.sort_unstable_by_key(|entry| entry.base);
        let mut merged: Vec<BrickRegion> = Vec::with_capacity(self.free_regions.len());
        for entry in self.free_regions.drain(..) {
            if let Some(previous) = merged.last_mut() {
                if previous.base + previous.count == entry.base {
                    previous.count += entry.count;
                    continue;
                }
            }
            merged.push(entry);
        }
        self.free_regions = merged;

        while self
            .free_regions
            .last()
            .is_some_and(|tail| tail.base + tail.count == self.next_brick)
        {
            let tail = self.free_regions.pop().expect("tail was just observed");
            self.next_brick = tail.base;
        }
    }

    fn take_free_palette_region(&mut self, count: u32) -> Option<BrickRegion> {
        let index = self
            .free_palette_regions
            .iter()
            .position(|region| region.count >= count)?;
        let free = self.free_palette_regions[index];
        let allocation = BrickRegion {
            base: free.base,
            count,
        };
        if free.count == count {
            self.free_palette_regions.remove(index);
        } else {
            self.free_palette_regions[index].base += count;
            self.free_palette_regions[index].count -= count;
        }
        Some(allocation)
    }

    fn recycle_palette_region(&mut self, region: BrickRegion) {
        self.free_palette_regions.push(region);
        self.free_palette_regions
            .sort_unstable_by_key(|entry| entry.base);
        let mut merged: Vec<BrickRegion> =
            Vec::with_capacity(self.free_palette_regions.len());
        for entry in self.free_palette_regions.drain(..) {
            if let Some(previous) = merged.last_mut() {
                if previous.base + previous.count == entry.base {
                    previous.count += entry.count;
                    continue;
                }
            }
            merged.push(entry);
        }
        self.free_palette_regions = merged;

        while self
            .free_palette_regions
            .last()
            .is_some_and(|tail| tail.base + tail.count == self.next_palette_row)
        {
            let tail = self
                .free_palette_regions
                .pop()
                .expect("palette tail was just observed");
            self.next_palette_row = tail.base;
        }
    }

    /// Allocate one stable contiguous region and return its absolute first
    /// brick row. Surviving regions never move when another volume is removed.
    pub fn allocate(
        &mut self,
        entity: Entity,
        brick_count: u32,
    ) -> Result<u32, VoxelResidencyError> {
        self.allocate_with_palette(entity, brick_count, &[])
            .map(|allocation| allocation.brick_base)
    }

    pub fn allocate_with_palette(
        &mut self,
        entity: Entity,
        brick_count: u32,
        palette: &[helio_voxel_core::GpuVoxelMaterial],
    ) -> Result<VoxelResidencyAllocation, VoxelResidencyError> {
        if brick_count == 0 {
            return Err(VoxelResidencyError::InvalidRegion);
        }
        if palette.len() > helio_voxel_core::MAX_PALETTE_SIZE as usize {
            return Err(VoxelResidencyError::PaletteTooLarge);
        }
        if self.region_by_entity.contains_key(&entity) {
            return Err(VoxelResidencyError::AlreadyResident);
        }
        let region = match self.take_free_region(brick_count) {
            Some(region) => region,
            None => {
                let end = self
                    .next_brick
                    .checked_add(brick_count)
                    .ok_or(VoxelResidencyError::CapacityExceeded)?;
                self.grow_to_include(end)?;
                let region = BrickRegion {
                    base: self.next_brick,
                    count: brick_count,
                };
                self.next_brick = end;
                region
            }
        };
        let palette_capacity = (palette.len().max(1) as u32).next_power_of_two();
        let palette_region = match self.take_free_palette_region(palette_capacity) {
            Some(region) => region,
            None => {
                let Some(end) = self
                    .next_palette_row
                    .checked_add(palette_capacity)
                else {
                    self.recycle_region(region);
                    return Err(VoxelResidencyError::CapacityExceeded);
                };
                if let Err(error) = self.grow_palette_to_include(end) {
                    self.recycle_region(region);
                    return Err(error);
                }
                let region = BrickRegion {
                    base: self.next_palette_row,
                    count: palette_capacity,
                };
                self.next_palette_row = end;
                region
            }
        };
        self.clear_region(region);
        self.clear_palette_region(palette_region);
        if !palette.is_empty() {
            self.queue.write_buffer(
                self.palette_buffer.as_ref(),
                u64::from(palette_region.base) * Self::PALETTE_ROW_BYTES,
                bytemuck::cast_slice(palette),
            );
        }
        let palette_count = palette.len() as u32;
        self.region_by_entity.insert(
            entity,
            VoxelRegion {
                bricks: region,
                palette: palette_region,
                palette_count,
            },
        );
        Ok(VoxelResidencyAllocation {
            brick_base: region.base,
            palette_base: palette_region.base,
            palette_count,
        })
    }

    pub fn release(&mut self, entity: Entity) -> Result<u32, VoxelResidencyError> {
        let region = self
            .region_by_entity
            .remove(&entity)
            .ok_or(VoxelResidencyError::NotResident)?;
        self.recycle_region(region.bricks);
        self.recycle_palette_region(region.palette);
        Ok(region.bricks.base)
    }

    pub fn brick_base(&self, entity: Entity) -> Option<u32> {
        self.region_by_entity
            .get(&entity)
            .map(|region| region.bricks.base)
    }

    pub fn brick_count(&self, entity: Entity) -> Option<u32> {
        self.region_by_entity
            .get(&entity)
            .map(|region| region.bricks.count)
    }

    pub fn palette_base(&self, entity: Entity) -> Option<u32> {
        self.region_by_entity
            .get(&entity)
            .map(|region| region.palette.base)
    }

    pub fn palette_count(&self, entity: Entity) -> Option<u32> {
        self.region_by_entity
            .get(&entity)
            .map(|region| region.palette_count)
    }

    pub fn write_palette(
        &mut self,
        entity: Entity,
        palette: &[helio_voxel_core::GpuVoxelMaterial],
    ) -> Result<(u32, u32), VoxelResidencyError> {
        if palette.len() > helio_voxel_core::MAX_PALETTE_SIZE as usize {
            return Err(VoxelResidencyError::PaletteTooLarge);
        }
        let old_region = self
            .region_by_entity
            .get(&entity)
            .map(|region| region.palette)
            .ok_or(VoxelResidencyError::NotResident)?;
        let required_capacity = (palette.len().max(1) as u32).next_power_of_two();
        let palette_region = if required_capacity <= old_region.count {
            old_region
        } else {
            match self.take_free_palette_region(required_capacity) {
                Some(region) => region,
                None => {
                    let end = self
                        .next_palette_row
                        .checked_add(required_capacity)
                        .ok_or(VoxelResidencyError::CapacityExceeded)?;
                    self.grow_palette_to_include(end)?;
                    let region = BrickRegion {
                        base: self.next_palette_row,
                        count: required_capacity,
                    };
                    self.next_palette_row = end;
                    region
                }
            }
        };
        self.clear_palette_region(palette_region);
        if !palette.is_empty() {
            self.queue.write_buffer(
                self.palette_buffer.as_ref(),
                u64::from(palette_region.base) * Self::PALETTE_ROW_BYTES,
                bytemuck::cast_slice(palette),
            );
        }
        let region = self.region_by_entity
            .get_mut(&entity)
            .expect("validated voxel residency must remain live");
        region.palette = palette_region;
        region.palette_count = palette.len() as u32;
        if palette_region != old_region {
            self.recycle_palette_region(old_region);
        }
        Ok((palette_region.base, palette.len() as u32))
    }

    pub fn write_brick(
        &self,
        entity: Entity,
        local_brick: u32,
        occupied: bool,
        words: &[u32],
    ) -> Result<(), VoxelResidencyError> {
        if words.len() != helio_voxel_core::RAYMARCH_WORDS_PER_BRICK as usize {
            return Err(VoxelResidencyError::InvalidBrick);
        }
        let region = self
            .region_by_entity
            .get(&entity)
            .copied()
            .ok_or(VoxelResidencyError::NotResident)?;
        if local_brick >= region.bricks.count {
            return Err(VoxelResidencyError::InvalidBrick);
        }
        let absolute_brick = u64::from(region.bricks.base) + u64::from(local_brick);
        let data_offset = absolute_brick
            * u64::from(helio_voxel_core::RAYMARCH_WORDS_PER_BRICK);
        let data_offset_u32 =
            u32::try_from(data_offset).map_err(|_| VoxelResidencyError::CapacityExceeded)?;
        if data_offset_u32 > 0x00ff_ffff {
            return Err(VoxelResidencyError::CapacityExceeded);
        }
        let meta = [
            if occupied {
                (1u32 << 24) | data_offset_u32
            } else {
                0
            },
            0,
        ];
        self.queue.write_buffer(
            self.brick_buffer.as_ref(),
            absolute_brick * Self::BRICK_META_STRIDE,
            bytemuck::cast_slice(&meta),
        );
        self.queue.write_buffer(
            self.data_buffer.as_ref(),
            data_offset * Self::DATA_WORD_BYTES,
            bytemuck::cast_slice(words),
        );
        Ok(())
    }

    pub fn brick_buffer(&self) -> &wgpu::Buffer {
        self.brick_buffer.as_ref()
    }

    pub fn data_buffer(&self) -> &wgpu::Buffer {
        self.data_buffer.as_ref()
    }

    pub fn palette_buffer(&self) -> &wgpu::Buffer {
        self.palette_buffer.as_ref()
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn live_count(&self) -> u32 {
        self.region_by_entity.len() as u32
    }

    pub fn capacity_bricks(&self) -> u32 {
        self.capacity_bricks
    }

    /// Clone only the reference-counted wgpu handles needed at the renderer
    /// frame boundary. SceneDB remains the allocation/mutation owner.
    pub fn publication(&self) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, u64, u32, u32) {
        (
            self.brick_buffer.as_ref().clone(),
            self.data_buffer.as_ref().clone(),
            self.palette_buffer.as_ref().clone(),
            self.epoch,
            self.capacity_bricks,
            self.capacity_palette_rows,
        )
    }
}

impl Subsystem for VoxelResidency {
    fn name(&self) -> &'static str {
        "helio.scene.voxel_residency"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

const _: () = {
    assert!(std::mem::size_of::<SceneVoxelVolumeRow>() == 176);
    assert!(std::mem::align_of::<SceneVoxelVolumeRow>() == 16);
    assert!(std::mem::offset_of!(SceneVoxelVolume, volume) == 0);
    assert!(std::mem::offset_of!(SceneVoxelVolume, movability) == 176);
    assert!(std::mem::size_of::<SceneVoxelVolume>() == 192);
};
