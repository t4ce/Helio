//! SceneDB-owned planetary frame authority.
//!
//! A planet frame is authored/canonical scene state even though its camera-local
//! values normally change every rendered frame. Planetary render passes may keep
//! page residency, lookup tables, clipmaps, extraction work, and indirect buffers,
//! but they must not expose a second mutable frame registry. This subsystem owns
//! stable generational identities, the CPU query projection, and the direct GPU
//! rows consumed by any shader that needs the full frame contract.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::Zeroable;
use helio_planet_voxel_core::{PlanetFrameProjection, PlanetFrameUniform, PlanetId};
use pulsar_scenedb::Subsystem;

pub const PLANET_FRAME_BUFFER_KEY: &str = "helio.scene.planetary_voxel.frames";

const INITIAL_FRAME_CAPACITY: u32 = 16;
static NEXT_AUTHORITY_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Stable SceneDB identity for one authored planet-frame record.
///
/// The slot is also the direct GPU row. It is deliberately independent of a
/// World entity index and can be reused only after its generation advances.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanetFrameId {
    slot: u32,
    generation: u32,
}

impl PlanetFrameId {
    pub const fn slot(self) -> u32 {
        self.slot
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }

    pub const fn to_bits(self) -> u64 {
        self.slot as u64 | ((self.generation as u64) << 32)
    }
}

/// Compact CPU query entry. `gpu_row` remains stable for this identity's
/// lifetime even when removals reorder this compact slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetFrameEntry {
    pub id: PlanetFrameId,
    pub gpu_row: u32,
    pub frame: PlanetFrameUniform,
}

impl PlanetFrameEntry {
    pub const fn projection(self) -> PlanetFrameProjection {
        PlanetFrameProjection {
            identity: self.id.to_bits(),
            gpu_row: self.gpu_row,
            frame: self.frame,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanetFrameUpdateOutcome {
    Applied { previous_frame: Option<u64> },
    Duplicate,
    Stale { newest_frame: u64 },
    FrameConflict,
}

/// Borrowed direct publication from the SceneDB subsystem.
pub struct PlanetFramePublication<'a> {
    pub buffer: wgpu::Buffer,
    /// Process-unique identity of the owning in-memory authority. Moving or
    /// rebuilding the same authority preserves it; constructing a distinct
    /// authority changes it even when local slot generations happen to match.
    pub authority_epoch: u64,
    pub allocation_epoch: u64,
    pub content_generation: u64,
    pub row_span: u32,
    pub entries: &'a [PlanetFrameEntry],
}

impl PlanetFramePublication<'_> {
    /// Refresh an allocation-stable compact renderer projection. Callers can
    /// reuse `output` across publications without retaining authored authority.
    pub fn copy_projections_into(&self, output: &mut Vec<PlanetFrameProjection>) {
        output.clear();
        output.reserve(self.entries.len());
        output.extend(self.entries.iter().copied().map(PlanetFrameEntry::projection));
    }
}

#[derive(Clone, Debug)]
struct FrameSlot {
    generation: u32,
    compact_index: Option<usize>,
}

/// Canonical planetary frame state suitable for both the high-level `Scene`
/// facade and a standalone SceneDB/render-pass integration.
pub struct PlanetFrameAuthority {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    authority_epoch: u64,
    buffer: Box<wgpu::Buffer>,
    buffer_capacity: u32,
    allocation_epoch: u64,
    content_generation: u64,
    row_span: u32,
    slots: Vec<FrameSlot>,
    free_slots: Vec<u32>,
    entries: Vec<PlanetFrameEntry>,
    by_planet: BTreeMap<PlanetId, PlanetFrameId>,
}

impl PlanetFrameAuthority {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let maximum = maximum_frame_rows(&device);
        let capacity = INITIAL_FRAME_CAPACITY.min(maximum).max(1);
        let buffer = create_frame_buffer(&device, capacity);
        Self {
            device,
            queue,
            authority_epoch: next_authority_epoch(),
            buffer: Box::new(buffer),
            buffer_capacity: capacity,
            allocation_epoch: 1,
            content_generation: 1,
            row_span: 0,
            slots: Vec::new(),
            free_slots: Vec::new(),
            entries: Vec::new(),
            by_planet: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[PlanetFrameEntry] {
        &self.entries
    }

    pub const fn content_generation(&self) -> u64 {
        self.content_generation
    }

    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    pub const fn allocation_epoch(&self) -> u64 {
        self.allocation_epoch
    }

    pub const fn row_span(&self) -> u32 {
        self.row_span
    }

    pub fn id_for_planet(&self, planet: PlanetId) -> Option<PlanetFrameId> {
        self.by_planet.get(&planet).copied()
    }

    pub fn get(&self, id: PlanetFrameId) -> Option<&PlanetFrameUniform> {
        let compact_index = self.valid_compact_index(id)?;
        Some(&self.entries[compact_index].frame)
    }

    pub fn frame_for_planet(&self, planet: PlanetId) -> Option<&PlanetFrameUniform> {
        self.id_for_planet(planet).and_then(|id| self.get(id))
    }

    /// Insert one new planet identity. Capacity failure is detected before
    /// slots, maps, generations, or GPU bytes change.
    pub fn insert(
        &mut self,
        frame: PlanetFrameUniform,
    ) -> Result<PlanetFrameId, PlanetFrameAuthorityError> {
        validate_frame(frame)?;
        let planet = frame.planet_id();
        if self.by_planet.contains_key(&planet) {
            return Err(PlanetFrameAuthorityError::DuplicatePlanet(planet));
        }

        let needs_new_slot = self.free_slots.is_empty();
        if needs_new_slot {
            let required = u32::try_from(self.slots.len())
                .ok()
                .and_then(|len| len.checked_add(1))
                .ok_or(PlanetFrameAuthorityError::CapacityExceeded)?;
            self.ensure_capacity(required)?;
        }

        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(FrameSlot {
                generation: 1,
                compact_index: None,
            });
            slot
        };
        let generation = self.slots[slot as usize].generation;
        let id = PlanetFrameId { slot, generation };
        let compact_index = self.entries.len();
        self.entries.push(PlanetFrameEntry {
            id,
            gpu_row: slot,
            frame,
        });
        self.slots[slot as usize].compact_index = Some(compact_index);
        self.by_planet.insert(planet, id);
        self.row_span = self.row_span.max(slot + 1);
        self.write_row(slot, frame);
        self.bump_content_generation();
        Ok(id)
    }

    /// Insert a new planet or update its existing stable identity.
    pub fn upsert(
        &mut self,
        frame: PlanetFrameUniform,
    ) -> Result<(PlanetFrameId, PlanetFrameUpdateOutcome), PlanetFrameAuthorityError> {
        validate_frame(frame)?;
        if let Some(id) = self.id_for_planet(frame.planet_id()) {
            let outcome = self.set(id, frame)?;
            return Ok((id, outcome));
        }
        let id = self.insert(frame)?;
        Ok((
            id,
            PlanetFrameUpdateOutcome::Applied {
                previous_frame: None,
            },
        ))
    }

    /// Replace one frame without changing its stable identity or GPU row.
    pub fn set(
        &mut self,
        id: PlanetFrameId,
        frame: PlanetFrameUniform,
    ) -> Result<PlanetFrameUpdateOutcome, PlanetFrameAuthorityError> {
        validate_frame(frame)?;
        let compact_index = self
            .valid_compact_index(id)
            .ok_or(PlanetFrameAuthorityError::StaleFrame)?;
        let current = self.entries[compact_index].frame;
        if current.planet_id() != frame.planet_id() {
            return Err(PlanetFrameAuthorityError::PlanetIdentityMismatch {
                expected: current.planet_id(),
                actual: frame.planet_id(),
            });
        }
        let current_number = current.frame_number();
        let frame_number = frame.frame_number();
        if frame_number < current_number {
            return Ok(PlanetFrameUpdateOutcome::Stale {
                newest_frame: current_number,
            });
        }
        if frame_number == current_number {
            return Ok(if current == frame {
                PlanetFrameUpdateOutcome::Duplicate
            } else {
                PlanetFrameUpdateOutcome::FrameConflict
            });
        }

        self.entries[compact_index].frame = frame;
        self.write_row(id.slot, frame);
        self.bump_content_generation();
        Ok(PlanetFrameUpdateOutcome::Applied {
            previous_frame: Some(current_number),
        })
    }

    /// Remove one identity and advance its slot generation before reuse.
    pub fn remove(
        &mut self,
        id: PlanetFrameId,
    ) -> Result<PlanetFrameUniform, PlanetFrameAuthorityError> {
        let compact_index = self
            .valid_compact_index(id)
            .ok_or(PlanetFrameAuthorityError::StaleFrame)?;
        let removed = self.entries.swap_remove(compact_index);
        if compact_index < self.entries.len() {
            let moved = self.entries[compact_index];
            self.slots[moved.id.slot as usize].compact_index = Some(compact_index);
        }
        self.by_planet.remove(&removed.frame.planet_id());
        let slot = &mut self.slots[id.slot as usize];
        slot.compact_index = None;
        slot.generation = next_generation(slot.generation);
        self.free_slots.push(id.slot);
        self.write_row(id.slot, PlanetFrameUniform::zeroed());
        if id.slot + 1 == self.row_span {
            self.row_span = self
                .slots
                .iter()
                .rposition(|slot| slot.compact_index.is_some())
                .map_or(0, |index| index as u32 + 1);
        }
        self.bump_content_generation();
        Ok(removed.frame)
    }

    pub fn remove_planet(
        &mut self,
        planet: PlanetId,
    ) -> Result<Option<PlanetFrameUniform>, PlanetFrameAuthorityError> {
        let Some(id) = self.id_for_planet(planet) else {
            return Ok(None);
        };
        self.remove(id).map(Some)
    }

    pub fn clear(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let occupied = self.entries.iter().map(|entry| entry.id.slot).collect::<Vec<_>>();
        for slot_index in occupied {
            let slot = &mut self.slots[slot_index as usize];
            slot.compact_index = None;
            slot.generation = next_generation(slot.generation);
            self.write_row(slot_index, PlanetFrameUniform::zeroed());
        }
        self.entries.clear();
        self.by_planet.clear();
        self.free_slots.clear();
        self.free_slots.extend((0..self.slots.len() as u32).rev());
        self.row_span = 0;
        self.bump_content_generation();
    }

    pub fn publication(&self) -> PlanetFramePublication<'_> {
        PlanetFramePublication {
            buffer: self.buffer.as_ref().clone(),
            authority_epoch: self.authority_epoch,
            allocation_epoch: self.allocation_epoch,
            content_generation: self.content_generation,
            row_span: self.row_span,
            entries: &self.entries,
        }
    }

    /// Recreate only the direct GPU allocation on a replacement device. Stable
    /// ids, CPU state, row assignments, and authored content generation remain
    /// unchanged; allocation consumers rebuild from `allocation_epoch`.
    pub fn recreate_gpu_resources(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<(), PlanetFrameAuthorityError> {
        let maximum = maximum_frame_rows(&device);
        if self.row_span > maximum {
            return Err(PlanetFrameAuthorityError::CapacityExceeded);
        }
        let capacity = self.buffer_capacity.min(maximum).max(self.row_span).max(1);
        let buffer = create_frame_buffer(&device, capacity);
        for entry in &self.entries {
            queue.write_buffer(
                &buffer,
                u64::from(entry.gpu_row) * frame_stride(),
                bytemuck::bytes_of(&entry.frame),
            );
        }
        self.device = device;
        self.queue = queue;
        self.buffer = Box::new(buffer);
        self.buffer_capacity = capacity;
        self.allocation_epoch = self.allocation_epoch.wrapping_add(1).max(1);
        Ok(())
    }

    fn valid_compact_index(&self, id: PlanetFrameId) -> Option<usize> {
        let slot = self.slots.get(id.slot as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        let compact_index = slot.compact_index?;
        (self.entries.get(compact_index)?.id == id).then_some(compact_index)
    }

    fn ensure_capacity(&mut self, required: u32) -> Result<(), PlanetFrameAuthorityError> {
        if required <= self.buffer_capacity {
            return Ok(());
        }
        let maximum = maximum_frame_rows(&self.device);
        if required > maximum {
            return Err(PlanetFrameAuthorityError::CapacityExceeded);
        }
        let mut capacity = self.buffer_capacity;
        while capacity < required {
            capacity = capacity
                .checked_mul(2)
                .unwrap_or(maximum)
                .min(maximum);
            if capacity < required && capacity == maximum {
                return Err(PlanetFrameAuthorityError::CapacityExceeded);
            }
        }
        let buffer = create_frame_buffer(&self.device, capacity);
        if self.row_span != 0 {
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("planet-frame-authority-grow"),
                    });
            encoder.copy_buffer_to_buffer(
                self.buffer.as_ref(),
                0,
                &buffer,
                0,
                u64::from(self.row_span) * frame_stride(),
            );
            self.queue.submit([encoder.finish()]);
        }
        self.buffer = Box::new(buffer);
        self.buffer_capacity = capacity;
        self.allocation_epoch = self.allocation_epoch.wrapping_add(1).max(1);
        Ok(())
    }

    fn write_row(&self, row: u32, frame: PlanetFrameUniform) {
        self.queue.write_buffer(
            self.buffer.as_ref(),
            u64::from(row) * frame_stride(),
            bytemuck::bytes_of(&frame),
        );
    }

    fn bump_content_generation(&mut self) {
        self.content_generation = self.content_generation.wrapping_add(1).max(1);
    }
}

impl Subsystem for PlanetFrameAuthority {
    fn name(&self) -> &'static str {
        "helio.scene.planetary_voxel.frames"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn validate_frame(frame: PlanetFrameUniform) -> Result<(), PlanetFrameAuthorityError> {
    frame
        .validate()
        .map_err(|_| PlanetFrameAuthorityError::InvalidFrame)
}

fn maximum_frame_rows(device: &wgpu::Device) -> u32 {
    let limits = device.limits();
    let maximum_bytes = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size));
    (maximum_bytes / frame_stride()).min(u64::from(u32::MAX)) as u32
}

const fn frame_stride() -> u64 {
    std::mem::size_of::<PlanetFrameUniform>() as u64
}

fn create_frame_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(PLANET_FRAME_BUFFER_KEY),
        size: u64::from(capacity) * frame_stride(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn next_generation(generation: u32) -> u32 {
    generation.wrapping_add(1).max(1)
}

fn next_authority_epoch() -> u64 {
    NEXT_AUTHORITY_EPOCH
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |epoch| epoch.checked_add(1))
        .expect("planet-frame authority epoch space exhausted")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanetFrameAuthorityError {
    #[error("planet {0:?} already has an authored frame")]
    DuplicatePlanet(PlanetId),
    #[error("planet-frame identity is stale")]
    StaleFrame,
    #[error("planet-frame identity belongs to {expected:?}, not {actual:?}")]
    PlanetIdentityMismatch {
        expected: PlanetId,
        actual: PlanetId,
    },
    #[error("planet-frame values do not match the canonical planetary GPU contract")]
    InvalidFrame,
    #[error("planet-frame authority exceeds the device storage-buffer limit")]
    CapacityExceeded,
}
