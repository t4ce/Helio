//! Ordered authored SDF state owned by SceneDB.
//!
//! SDF boolean operations are not commutative, so storage row order cannot be
//! used as authored evaluation order. This registered subsystem keeps an
//! explicit ordered stream with stable generational edit identities. Its GPU
//! edit buffer is the direct partner of that stream; Helio owns only derived
//! BVH, clipmap, atlas, and dispatch state.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use pulsar_scenedb::Subsystem;

pub const SDF_EDIT_BUFFER_KEY: &str = "helio.scene.sdf.edits";
pub const SDF_TERRAIN_BUFFER_KEY: &str = "helio.scene.sdf.terrain";
pub const MAX_TERRAIN_OCTAVES: u32 = 32;

const INITIAL_EDIT_CAPACITY: u32 = 16;

/// Boolean operation applied between an edit and the accumulated SDF.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union = 0,
    Subtraction = 1,
    Intersection = 2,
}

/// SDF shape discriminant matching the shader ABI.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SdfShapeType {
    Sphere = 0,
    Cube = 1,
    Capsule = 2,
    Torus = 3,
    Cylinder = 4,
}

/// Shape-specific parameters packed into four floats for the shader ABI.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfShapeParams {
    pub param0: f32,
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
}

impl Default for SdfShapeParams {
    fn default() -> Self {
        Self {
            param0: 0.0,
            param1: 0.0,
            param2: 0.0,
            param3: 0.0,
        }
    }
}

impl SdfShapeParams {
    pub fn sphere(radius: f32) -> Self {
        Self {
            param0: radius,
            ..Self::default()
        }
    }

    pub fn cube(half_x: f32, half_y: f32, half_z: f32) -> Self {
        Self {
            param0: half_x,
            param1: half_y,
            param2: half_z,
            param3: 0.0,
        }
    }

    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self {
            param0: radius,
            param1: half_height,
            ..Self::default()
        }
    }

    pub fn torus(major_radius: f32, minor_radius: f32) -> Self {
        Self {
            param0: major_radius,
            param1: minor_radius,
            ..Self::default()
        }
    }

    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        Self {
            param0: radius,
            param1: half_height,
            ..Self::default()
        }
    }
}

/// One authored SDF primitive in explicit evaluation order.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SdfEdit {
    pub shape: SdfShapeType,
    pub op: BooleanOp,
    pub transform: Mat4,
    pub params: SdfShapeParams,
    pub blend_radius: f32,
}

impl SdfEdit {
    pub fn to_gpu(self) -> GpuSdfEdit {
        GpuSdfEdit {
            transform: self.transform.inverse().to_cols_array(),
            shape_type: self.shape as u32,
            boolean_op: self.op as u32,
            blend_radius: self.blend_radius,
            distance_scale: self.similarity_scale().unwrap_or(0.0),
            params: self.params,
        }
    }

    /// Exact primitive distance under an affine transform is available with
    /// no extra shader data only for similarity transforms (translation,
    /// rotation/reflection, and uniform scale). General non-uniform scale or
    /// shear would require a shape-specific distance solver, so authority
    /// validation rejects it instead of silently publishing approximate SDFs.
    fn similarity_scale(self) -> Option<f32> {
        let columns = [
            self.transform.col(0),
            self.transform.col(1),
            self.transform.col(2),
        ];
        if columns.iter().any(|column| column.w.abs() > 1.0e-5)
            || (self.transform.col(3).w - 1.0).abs() > 1.0e-5
        {
            return None;
        }
        let axes = columns.map(|column| column.truncate());
        let scales = axes.map(glam::Vec3::length);
        let maximum = scales.into_iter().fold(0.0f32, f32::max);
        let minimum = scales.into_iter().fold(f32::INFINITY, f32::min);
        if minimum <= 1.0e-6 || maximum - minimum > maximum * 1.0e-4 {
            return None;
        }
        let orthogonality_tolerance = maximum * maximum * 1.0e-4;
        if axes[0].dot(axes[1]).abs() > orthogonality_tolerance
            || axes[0].dot(axes[2]).abs() > orthogonality_tolerance
            || axes[1].dot(axes[2]).abs() > orthogonality_tolerance
        {
            return None;
        }
        Some((scales[0] + scales[1] + scales[2]) / 3.0)
    }

    pub fn bounds(self) -> SdfEditBounds {
        let local_radius = match self.shape {
            SdfShapeType::Sphere => self.params.param0,
            SdfShapeType::Cube => {
                Vec3::new(self.params.param0, self.params.param1, self.params.param2).length()
            }
            SdfShapeType::Capsule | SdfShapeType::Torus => {
                self.params.param0 + self.params.param1
            }
            SdfShapeType::Cylinder => (self.params.param0 * self.params.param0
                + self.params.param1 * self.params.param1)
                .sqrt(),
        };
        let center = self.transform.transform_point3(Vec3::ZERO);
        let max_scale = self
            .transform
            .col(0)
            .truncate()
            .length()
            .max(self.transform.col(1).truncate().length())
            .max(self.transform.col(2).truncate().length());
        SdfEditBounds {
            center_radius: center
                .extend(local_radius * max_scale + self.blend_radius)
                .to_array(),
        }
    }

    fn is_valid(self) -> bool {
        let transform = self.transform.to_cols_array();
        if !transform.into_iter().all(f32::is_finite)
            || !self.transform.determinant().is_finite()
            || self.transform.determinant().abs() <= 1.0e-8
            || self.similarity_scale().is_none()
            || !self.blend_radius.is_finite()
            || self.blend_radius < 0.0
        {
            return false;
        }
        let params = [
            self.params.param0,
            self.params.param1,
            self.params.param2,
            self.params.param3,
        ];
        if !params.into_iter().all(f32::is_finite) {
            return false;
        }
        match self.shape {
            SdfShapeType::Sphere => self.params.param0 > 0.0,
            SdfShapeType::Cube => {
                self.params.param0 > 0.0
                    && self.params.param1 > 0.0
                    && self.params.param2 > 0.0
            }
            SdfShapeType::Capsule
            | SdfShapeType::Torus
            | SdfShapeType::Cylinder => self.params.param0 > 0.0 && self.params.param1 > 0.0,
        }
    }
}

/// Direct shader row for one authored edit (96 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSdfEdit {
    /// World-to-local transform.
    pub transform: [f32; 16],
    pub shape_type: u32,
    pub boolean_op: u32,
    pub blend_radius: f32,
    /// Uniform world-space scale converting local primitive distance to the
    /// exact world-space metric. Reuses the former padding word.
    pub distance_scale: f32,
    pub params: SdfShapeParams,
}

const _: () = assert!(std::mem::size_of::<GpuSdfEdit>() == 96);

/// Conservative world-space sphere used only to build Helio's derived BVH.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfEditBounds {
    pub center_radius: [f32; 4],
}

/// Terrain generation style.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainStyle {
    Rolling = 0,
    Mountains = 1,
    Canyons = 2,
    Dunes = 3,
    Warped = 4,
}

/// Authored procedural terrain configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainConfig {
    pub style: TerrainStyle,
    pub height: f32,
    pub amplitude: f32,
    pub frequency: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    pub warp_amount: f32,
}

impl TerrainConfig {
    pub fn rolling() -> Self {
        Self {
            style: TerrainStyle::Rolling,
            height: -2.0,
            amplitude: 4.0,
            frequency: 0.08,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_amount: 0.0,
        }
    }

    pub fn mountains() -> Self {
        Self {
            style: TerrainStyle::Mountains,
            height: -5.0,
            amplitude: 25.0,
            frequency: 0.03,
            octaves: 7,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_amount: 2.0,
        }
    }

    pub fn canyons() -> Self {
        Self {
            style: TerrainStyle::Canyons,
            height: -2.0,
            amplitude: 15.0,
            frequency: 0.05,
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.55,
            warp_amount: 3.0,
        }
    }

    pub fn dunes() -> Self {
        Self {
            style: TerrainStyle::Dunes,
            height: -1.0,
            amplitude: 6.0,
            frequency: 0.15,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.6,
            warp_amount: 1.0,
        }
    }

    pub fn warped(warp_amount: f32) -> Self {
        Self {
            style: TerrainStyle::Warped,
            height: -2.0,
            amplitude: 4.0,
            frequency: 0.08,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_amount,
        }
    }

    pub fn build_gpu_params(self) -> GpuTerrainParams {
        GpuTerrainParams {
            enabled: 1,
            style: self.style as u32,
            height: self.height,
            amplitude: self.amplitude,
            frequency: self.frequency,
            octaves: self.octaves,
            lacunarity: self.lacunarity,
            persistence: self.persistence,
            warp_amount: self.warp_amount,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        }
    }

    /// Conservative terrain surface interval used by brick classification.
    pub fn y_bounds(self) -> [f32; 2] {
        let relief = match self.style {
            TerrainStyle::Rolling | TerrainStyle::Dunes | TerrainStyle::Warped => {
                self.amplitude
            }
            TerrainStyle::Mountains => self.amplitude * 1.3,
            TerrainStyle::Canyons => self.amplitude + 3.0,
        };
        [self.height - relief, self.height + relief]
    }

    fn is_valid(self) -> bool {
        [
            self.height,
            self.amplitude,
            self.frequency,
            self.lacunarity,
            self.persistence,
            self.warp_amount,
        ]
        .into_iter()
        .all(f32::is_finite)
            && self.amplitude >= 0.0
            && self.frequency > 0.0
            && self.octaves > 0
            && self.octaves <= MAX_TERRAIN_OCTAVES
            && self.lacunarity > 0.0
            && self.persistence > 0.0
            && self.persistence <= 1.0
            && self.warp_amount >= 0.0
    }
}

/// Direct 64-byte terrain uniform owned by SceneDB.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTerrainParams {
    pub enabled: u32,
    pub style: u32,
    pub height: f32,
    pub amplitude: f32,
    pub frequency: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    pub warp_amount: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
    pub _pad6: u32,
}

impl GpuTerrainParams {
    pub fn disabled() -> Self {
        <Self as bytemuck::Zeroable>::zeroed()
    }
}

const _: () = assert!(std::mem::size_of::<GpuTerrainParams>() == 64);

/// Stable identity for one edit independent of its current ordered row.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SdfEditId {
    slot: u32,
    generation: u32,
}

impl SdfEditId {
    pub const fn slot(self) -> u32 {
        self.slot
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SdfAuthorityError {
    InvalidEdit,
    InvalidTerrain,
    InvalidIndex,
    StaleEdit,
    CapacityExceeded,
}

impl std::fmt::Display for SdfAuthorityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidEdit => "SDF edit parameters or transform are invalid",
            Self::InvalidTerrain => "SDF terrain parameters are invalid",
            Self::InvalidIndex => "SDF ordered edit index is out of bounds",
            Self::StaleEdit => "SDF edit identity is stale or has been removed",
            Self::CapacityExceeded => "SDF edit buffer exceeds the device storage limit",
        })
    }
}

impl std::error::Error for SdfAuthorityError {}

/// Result of a read-only CPU ray query against SceneDB's authored SDF stream.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SdfPickResult {
    pub position: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

#[derive(Debug)]
struct EditSlot {
    generation: u32,
    order_index: Option<usize>,
}

/// Allocation-stable read-only publication for renderer integration.
pub struct SdfPublication<'a> {
    pub edit_buffer: wgpu::Buffer,
    pub edit_allocation_epoch: u64,
    pub edit_count: u32,
    pub terrain_buffer: wgpu::Buffer,
    pub terrain_allocation_epoch: u64,
    pub content_generation: u64,
    pub bounds: &'a [SdfEditBounds],
    pub terrain_y_bounds: Option<[f32; 2]>,
    /// `Intersection` cannot be safely reduced to a brick-local BVH result:
    /// a non-overlapping intersection operand can still remove the brick.
    pub requires_canonical_scan: bool,
}

/// SceneDB subsystem owning the ordered SDF edit stream and terrain config.
pub struct SdfAuthority {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    edit_buffer: Box<wgpu::Buffer>,
    terrain_buffer: Box<wgpu::Buffer>,
    edit_capacity: u32,
    edit_allocation_epoch: u64,
    content_generation: u64,
    slots: Vec<EditSlot>,
    free_slots: Vec<u32>,
    order_slots: Vec<u32>,
    edits: Vec<SdfEdit>,
    gpu_edits: Vec<GpuSdfEdit>,
    bounds: Vec<SdfEditBounds>,
    terrain: Option<TerrainConfig>,
    intersection_count: u32,
}

impl SdfAuthority {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let edit_buffer = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(SDF_EDIT_BUFFER_KEY),
            size: u64::from(INITIAL_EDIT_CAPACITY)
                * std::mem::size_of::<GpuSdfEdit>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let terrain_buffer = Box::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(SDF_TERRAIN_BUFFER_KEY),
            size: std::mem::size_of::<GpuTerrainParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(
            terrain_buffer.as_ref(),
            0,
            bytemuck::bytes_of(&GpuTerrainParams::disabled()),
        );
        Self {
            device,
            queue,
            edit_buffer,
            terrain_buffer,
            edit_capacity: INITIAL_EDIT_CAPACITY,
            edit_allocation_epoch: 0,
            content_generation: 1,
            slots: Vec::new(),
            free_slots: Vec::new(),
            order_slots: Vec::new(),
            edits: Vec::new(),
            gpu_edits: Vec::new(),
            bounds: Vec::new(),
            terrain: None,
            intersection_count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn edits(&self) -> &[SdfEdit] {
        &self.edits
    }

    pub fn terrain(&self) -> Option<&TerrainConfig> {
        self.terrain.as_ref()
    }

    /// Evaluate the exact authored boolean stream on CPU.
    pub fn evaluate_sdf(&self, position: Vec3) -> f32 {
        evaluate_sdf_rows(position, &self.edits, &self.gpu_edits, self.terrain.as_ref())
    }

    /// Sphere-trace the canonical authored stream without involving the render
    /// pass. `ray_direction` is normalized so distance is measured in world
    /// units.
    pub fn pick_surface(
        &self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        max_distance: f32,
    ) -> Option<SdfPickResult> {
        if !ray_origin.is_finite()
            || !ray_direction.is_finite()
            || ray_direction.length_squared() <= f32::EPSILON
            || !max_distance.is_finite()
            || max_distance <= 0.0
        {
            return None;
        }
        let ray_direction = ray_direction.normalize();
        let mut distance = 0.0;
        for _ in 0..256 {
            let position = ray_origin + ray_direction * distance;
            let step = self.evaluate_sdf(position);
            if step.abs() < 0.02 {
                return Some(SdfPickResult {
                    position,
                    normal: estimate_normal(self, position),
                    distance,
                });
            }
            distance += step.max(0.01);
            if distance > max_distance {
                break;
            }
        }
        None
    }

    pub fn id_at(&self, order_index: usize) -> Option<SdfEditId> {
        let slot = *self.order_slots.get(order_index)?;
        let generation = self.slots[slot as usize].generation;
        Some(SdfEditId { slot, generation })
    }

    pub fn get(&self, id: SdfEditId) -> Option<&SdfEdit> {
        let index = self.order_index(id).ok()?;
        self.edits.get(index)
    }

    pub fn add(&mut self, edit: SdfEdit) -> Result<SdfEditId, SdfAuthorityError> {
        self.insert(self.edits.len(), edit)
    }

    pub fn insert(
        &mut self,
        order_index: usize,
        edit: SdfEdit,
    ) -> Result<SdfEditId, SdfAuthorityError> {
        if order_index > self.edits.len() {
            return Err(SdfAuthorityError::InvalidIndex);
        }
        if !edit.is_valid() {
            return Err(SdfAuthorityError::InvalidEdit);
        }
        self.ensure_capacity(self.edits.len() + 1)?;
        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let slot = u32::try_from(self.slots.len())
                .map_err(|_| SdfAuthorityError::CapacityExceeded)?;
            self.slots.push(EditSlot {
                generation: 1,
                order_index: None,
            });
            slot
        };
        self.order_slots.insert(order_index, slot);
        if edit.op == BooleanOp::Intersection {
            self.intersection_count += 1;
        }
        self.edits.insert(order_index, edit);
        self.gpu_edits.insert(order_index, edit.to_gpu());
        self.bounds.insert(order_index, edit.bounds());
        self.reindex_from(order_index);
        self.write_from(order_index);
        self.bump_generation();
        Ok(SdfEditId {
            slot,
            generation: self.slots[slot as usize].generation,
        })
    }

    pub fn set(&mut self, id: SdfEditId, edit: SdfEdit) -> Result<(), SdfAuthorityError> {
        if !edit.is_valid() {
            return Err(SdfAuthorityError::InvalidEdit);
        }
        let index = self.order_index(id)?;
        if self.edits[index] == edit {
            return Ok(());
        }
        let old_is_intersection = self.edits[index].op == BooleanOp::Intersection;
        let new_is_intersection = edit.op == BooleanOp::Intersection;
        match (old_is_intersection, new_is_intersection) {
            (false, true) => self.intersection_count += 1,
            (true, false) => self.intersection_count -= 1,
            _ => {}
        }
        self.edits[index] = edit;
        self.gpu_edits[index] = edit.to_gpu();
        self.bounds[index] = edit.bounds();
        self.queue.write_buffer(
            self.edit_buffer.as_ref(),
            index as u64 * std::mem::size_of::<GpuSdfEdit>() as u64,
            bytemuck::bytes_of(&self.gpu_edits[index]),
        );
        self.bump_generation();
        Ok(())
    }

    pub fn remove(&mut self, id: SdfEditId) -> Result<SdfEdit, SdfAuthorityError> {
        let index = self.order_index(id)?;
        let slot = self.order_slots.remove(index);
        let edit = self.edits.remove(index);
        if edit.op == BooleanOp::Intersection {
            self.intersection_count -= 1;
        }
        self.gpu_edits.remove(index);
        self.bounds.remove(index);
        let slot_state = &mut self.slots[slot as usize];
        slot_state.order_index = None;
        slot_state.generation = next_generation(slot_state.generation);
        self.free_slots.push(slot);
        self.reindex_from(index);
        self.write_from(index);
        self.bump_generation();
        Ok(edit)
    }

    pub fn move_edit(
        &mut self,
        id: SdfEditId,
        new_index: usize,
    ) -> Result<(), SdfAuthorityError> {
        if new_index >= self.edits.len() {
            return Err(SdfAuthorityError::InvalidIndex);
        }
        let old_index = self.order_index(id)?;
        if old_index == new_index {
            return Ok(());
        }
        let slot = self.order_slots.remove(old_index);
        let edit = self.edits.remove(old_index);
        let gpu = self.gpu_edits.remove(old_index);
        let bounds = self.bounds.remove(old_index);
        self.order_slots.insert(new_index, slot);
        self.edits.insert(new_index, edit);
        self.gpu_edits.insert(new_index, gpu);
        self.bounds.insert(new_index, bounds);
        let first = old_index.min(new_index);
        self.reindex_from(first);
        self.write_from(first);
        self.bump_generation();
        Ok(())
    }

    pub fn clear(&mut self) {
        if self.edits.is_empty() {
            return;
        }
        for slot in self.order_slots.drain(..) {
            let state = &mut self.slots[slot as usize];
            state.order_index = None;
            state.generation = next_generation(state.generation);
            self.free_slots.push(slot);
        }
        self.edits.clear();
        self.gpu_edits.clear();
        self.bounds.clear();
        self.intersection_count = 0;
        self.bump_generation();
    }

    pub fn set_terrain(
        &mut self,
        terrain: Option<TerrainConfig>,
    ) -> Result<(), SdfAuthorityError> {
        if terrain.is_some_and(|terrain| !terrain.is_valid()) {
            return Err(SdfAuthorityError::InvalidTerrain);
        }
        if self.terrain == terrain {
            return Ok(());
        }
        self.terrain = terrain;
        let gpu = terrain
            .map(TerrainConfig::build_gpu_params)
            .unwrap_or_else(GpuTerrainParams::disabled);
        self.queue.write_buffer(
            self.terrain_buffer.as_ref(),
            0,
            bytemuck::bytes_of(&gpu),
        );
        self.bump_generation();
        Ok(())
    }

    pub fn publication(&self) -> SdfPublication<'_> {
        SdfPublication {
            edit_buffer: self.edit_buffer.as_ref().clone(),
            edit_allocation_epoch: self.edit_allocation_epoch,
            edit_count: self.edits.len() as u32,
            terrain_buffer: self.terrain_buffer.as_ref().clone(),
            terrain_allocation_epoch: 0,
            content_generation: self.content_generation,
            bounds: &self.bounds,
            terrain_y_bounds: self.terrain.map(TerrainConfig::y_bounds),
            requires_canonical_scan: self.intersection_count != 0,
        }
    }

    fn order_index(&self, id: SdfEditId) -> Result<usize, SdfAuthorityError> {
        let slot = self
            .slots
            .get(id.slot as usize)
            .ok_or(SdfAuthorityError::StaleEdit)?;
        if slot.generation != id.generation {
            return Err(SdfAuthorityError::StaleEdit);
        }
        slot.order_index.ok_or(SdfAuthorityError::StaleEdit)
    }

    fn ensure_capacity(&mut self, required: usize) -> Result<(), SdfAuthorityError> {
        let required = u32::try_from(required).map_err(|_| SdfAuthorityError::CapacityExceeded)?;
        if required <= self.edit_capacity {
            return Ok(());
        }
        let max_bytes = self
            .device
            .limits()
            .max_buffer_size
            .min(u64::from(
                self.device.limits().max_storage_buffer_binding_size,
            ));
        let stride = std::mem::size_of::<GpuSdfEdit>() as u64;
        let max_rows = (max_bytes / stride).min(u64::from(u32::MAX)) as u32;
        if required > max_rows {
            return Err(SdfAuthorityError::CapacityExceeded);
        }
        let mut capacity = self.edit_capacity;
        while capacity < required {
            capacity = capacity
                .checked_mul(2)
                .ok_or(SdfAuthorityError::CapacityExceeded)?
                .min(max_rows);
            if capacity < required && capacity == max_rows {
                return Err(SdfAuthorityError::CapacityExceeded);
            }
        }
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(SDF_EDIT_BUFFER_KEY),
            size: u64::from(capacity) * stride,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if !self.gpu_edits.is_empty() {
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("sdf-authority-grow"),
                    });
            encoder.copy_buffer_to_buffer(
                self.edit_buffer.as_ref(),
                0,
                &new_buffer,
                0,
                self.gpu_edits.len() as u64 * stride,
            );
            self.queue.submit([encoder.finish()]);
        }
        self.edit_buffer = Box::new(new_buffer);
        self.edit_capacity = capacity;
        self.edit_allocation_epoch = self.edit_allocation_epoch.wrapping_add(1);
        Ok(())
    }

    fn reindex_from(&mut self, first: usize) {
        for (index, &slot) in self.order_slots.iter().enumerate().skip(first) {
            self.slots[slot as usize].order_index = Some(index);
        }
    }

    fn write_from(&self, first: usize) {
        if first >= self.gpu_edits.len() {
            return;
        }
        self.queue.write_buffer(
            self.edit_buffer.as_ref(),
            first as u64 * std::mem::size_of::<GpuSdfEdit>() as u64,
            bytemuck::cast_slice(&self.gpu_edits[first..]),
        );
    }

    fn bump_generation(&mut self) {
        self.content_generation = self.content_generation.wrapping_add(1).max(1);
    }
}

fn evaluate_sdf_rows(
    position: Vec3,
    edits: &[SdfEdit],
    gpu_edits: &[GpuSdfEdit],
    terrain: Option<&TerrainConfig>,
) -> f32 {
    let mut distance = terrain
        .map(|config| crate::sdf_noise::terrain_sdf(position, config))
        .unwrap_or(1.0e10);
    for (edit, gpu_edit) in edits.iter().zip(gpu_edits) {
        let inverse = Mat4::from_cols_array(&gpu_edit.transform);
        let local_position = inverse.transform_point3(position);
        let shape_distance = evaluate_shape(local_position, edit) * gpu_edit.distance_scale;
        distance = apply_boolean(distance, shape_distance, edit.op, edit.blend_radius);
    }
    distance
}

fn evaluate_shape(position: Vec3, edit: &SdfEdit) -> f32 {
    match edit.shape {
        SdfShapeType::Sphere => position.length() - edit.params.param0,
        SdfShapeType::Cube => {
            let half_extents = Vec3::new(
                edit.params.param0,
                edit.params.param1,
                edit.params.param2,
            );
            let delta = position.abs() - half_extents;
            delta.max(Vec3::ZERO).length() + delta.x.max(delta.y.max(delta.z)).min(0.0)
        }
        SdfShapeType::Capsule => {
            let mut delta = position;
            delta.y -= delta.y.clamp(-edit.params.param1, edit.params.param1);
            delta.length() - edit.params.param0
        }
        SdfShapeType::Torus => {
            let ring = glam::Vec2::new(
                glam::Vec2::new(position.x, position.z).length() - edit.params.param0,
                position.y,
            );
            ring.length() - edit.params.param1
        }
        SdfShapeType::Cylinder => {
            let delta = glam::Vec2::new(
                glam::Vec2::new(position.x, position.z).length(),
                position.y,
            )
            .abs()
                - glam::Vec2::new(edit.params.param0, edit.params.param1);
            delta.x.max(delta.y).min(0.0) + delta.max(glam::Vec2::ZERO).length()
        }
    }
}

fn apply_boolean(left: f32, right: f32, operation: BooleanOp, blend_radius: f32) -> f32 {
    let blended = blend_radius > 0.001;
    match operation {
        BooleanOp::Union if blended => {
            let blend = (0.5 + 0.5 * (right - left) / blend_radius).clamp(0.0, 1.0);
            left * blend + right * (1.0 - blend) - blend_radius * blend * (1.0 - blend)
        }
        BooleanOp::Union => left.min(right),
        BooleanOp::Subtraction if blended => {
            let blend = (0.5 - 0.5 * (right + left) / blend_radius).clamp(0.0, 1.0);
            left * (1.0 - blend) + (-right) * blend + blend_radius * blend * (1.0 - blend)
        }
        BooleanOp::Subtraction => left.max(-right),
        BooleanOp::Intersection if blended => {
            let blend = (0.5 - 0.5 * (right - left) / blend_radius).clamp(0.0, 1.0);
            left * (1.0 - blend) + right * blend + blend_radius * blend * (1.0 - blend)
        }
        BooleanOp::Intersection => left.max(right),
    }
}

fn estimate_normal(authority: &SdfAuthority, position: Vec3) -> Vec3 {
    let epsilon = 0.01;
    let x = Vec3::new(epsilon, 0.0, 0.0);
    let y = Vec3::new(0.0, epsilon, 0.0);
    let z = Vec3::new(0.0, 0.0, epsilon);
    Vec3::new(
        authority.evaluate_sdf(position + x) - authority.evaluate_sdf(position - x),
        authority.evaluate_sdf(position + y) - authority.evaluate_sdf(position - y),
        authority.evaluate_sdf(position + z) - authority.evaluate_sdf(position - z),
    )
    .normalize_or_zero()
}

impl Subsystem for SdfAuthority {
    fn name(&self) -> &'static str {
        "helio.scene.sdf"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn next_generation(generation: u32) -> u32 {
    generation.wrapping_add(1).max(1)
}
