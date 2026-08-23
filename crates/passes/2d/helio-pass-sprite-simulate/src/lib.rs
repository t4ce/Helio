//! GPU bounce simulation for SceneDB-authored sprites.
//!
//! Authored rows are never mutated by this pass. Position, depth, and current
//! velocity live in a Helio-owned runtime projection keyed by each row's
//! authored epoch. The paired cull and batch passes consume that projection
//! through [`SpriteBufferSource`], while ordinary SceneDB edits remain safe
//! and reset simulation from the newly authored state.

use bytemuck::{Pod, Zeroable};
use helio_core::{Error, PassContext, PrepareContext, RenderPass, Result};
use helio_scenedb::SpriteBufferSource;

const WG_SIZE: u32 = 256;
const RUNTIME_STRIDE: u64 = 24;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniforms {
    bounds_min: [f32; 2],
    bounds_max: [f32; 2],
    dt: f32,
    slot_count: u32,
    _reserved: u32,
    dispatched_threads: u32,
}

/// Pass-derived runtime row. Must match `SpriteRuntime` in every sprite WGSL
/// consumer exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpriteRuntimeRow {
    position: [f32; 2],
    depth: f32,
    authored_epoch: u32,
    velocity: [f32; 2],
}

pub struct SpriteSimulatePass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buf: wgpu::Buffer,
    runtime_buf: wgpu::Buffer,
    runtime_capacity: u32,
    maximum_rows: u32,
    max_workgroups: u32,
    runtime_token: u64,
    source: SpriteBufferSource,
    bound_instances_epoch: u64,
    bound_presence_epoch: u64,
    bound_runtime_epoch: u64,
    bounds_min: [f32; 2],
    bounds_max: [f32; 2],
}

impl SpriteSimulatePass {
    /// Attach simulation to an epoch-aware SceneDB sprite publication.
    /// Sprites opt in by setting a non-zero
    /// `SpriteInstance::with_simulation_velocity` value.
    pub fn try_new(
        device: &wgpu::Device,
        source: SpriteBufferSource,
        bounds_min: [f32; 2],
        bounds_max: [f32; 2],
    ) -> Result<Self> {
        if source.snapshot().runtime.is_some() {
            return Err(Error::Gpu(
                "this sprite source already has a runtime projection owner".into(),
            ));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Simulate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite_simulate.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Simulate BGL"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Simulate PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Simulate Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_simulate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Simulate Uniforms"),
            size: std::mem::size_of::<SimUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let limits = device.limits();
        let maximum_rows = (limits.max_buffer_size / RUNTIME_STRIDE)
            .min(u64::from(limits.max_storage_buffer_binding_size) / RUNTIME_STRIDE)
            .min(u64::from(u32::MAX)) as u32;
        let initial_span = source.snapshot().row_span;
        if initial_span > maximum_rows {
            return Err(Error::Gpu(format!(
                "sprite simulation needs {initial_span} runtime rows, device permits {maximum_rows}"
            )));
        }
        let runtime_capacity = rounded_runtime_capacity(initial_span, maximum_rows)?;
        let runtime_buf = create_runtime_buffer(device, runtime_capacity);
        let runtime_token = source
            .install_runtime(runtime_buf.clone(), runtime_capacity)
            .map_err(|error| Error::Gpu(error.to_string()))?;

        let mut pass = Self {
            pipeline,
            bgl,
            bind_group: None,
            uniform_buf,
            runtime_buf,
            runtime_capacity,
            maximum_rows,
            max_workgroups: limits.max_compute_workgroups_per_dimension.max(1),
            runtime_token,
            source,
            bound_instances_epoch: u64::MAX,
            bound_presence_epoch: u64::MAX,
            bound_runtime_epoch: u64::MAX,
            bounds_min,
            bounds_max,
        };
        pass.refresh_binding(device);
        Ok(pass)
    }

    pub fn new(
        device: &wgpu::Device,
        source: SpriteBufferSource,
        bounds_min: [f32; 2],
        bounds_max: [f32; 2],
    ) -> Self {
        Self::try_new(device, source, bounds_min, bounds_max)
            .expect("SpriteSimulatePass::new exceeded the device runtime-buffer limit")
    }

    fn grow_runtime(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        required: u32,
    ) -> Result<()> {
        if required <= self.runtime_capacity {
            return Ok(());
        }
        if required > self.maximum_rows {
            return Err(Error::Gpu(format!(
                "sprite simulation grew to {required} runtime rows, device permits {}",
                self.maximum_rows
            )));
        }
        let capacity = rounded_runtime_capacity(required, self.maximum_rows)?;
        let new_buffer = create_runtime_buffer(device, capacity);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sprite Simulation Runtime Grow"),
        });
        encoder.copy_buffer_to_buffer(
            &self.runtime_buf,
            0,
            &new_buffer,
            0,
            u64::from(self.runtime_capacity) * RUNTIME_STRIDE,
        );
        encoder.clear_buffer(
            &new_buffer,
            u64::from(self.runtime_capacity) * RUNTIME_STRIDE,
            Some(u64::from(capacity - self.runtime_capacity) * RUNTIME_STRIDE),
        );
        queue.submit([encoder.finish()]);
        self.runtime_buf = new_buffer;
        self.runtime_capacity = capacity;
        if !self.source.replace_runtime(
            self.runtime_token,
            self.runtime_buf.clone(),
            self.runtime_capacity,
        ) {
            return Err(Error::Gpu(
                "sprite simulation runtime publication was replaced by another owner".into(),
            ));
        }
        self.bind_group = None;
        Ok(())
    }

    fn refresh_binding(&mut self, device: &wgpu::Device) {
        let source = self.source.snapshot();
        let runtime_epoch = source.runtime.as_ref().map_or(0, |runtime| runtime.epoch);
        if self.bind_group.is_none()
            || self.bound_instances_epoch != source.instances_epoch
            || self.bound_presence_epoch != source.presence_epoch
            || self.bound_runtime_epoch != runtime_epoch
        {
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Simulate BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: source.presence.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: source.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.runtime_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bound_instances_epoch = source.instances_epoch;
            self.bound_presence_epoch = source.presence_epoch;
            self.bound_runtime_epoch = runtime_epoch;
        }
    }
}

impl Drop for SpriteSimulatePass {
    fn drop(&mut self) {
        self.source.remove_runtime(self.runtime_token);
    }
}

impl RenderPass for SpriteSimulatePass {
    fn name(&self) -> &'static str {
        "SpriteSimulate"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        let row_span = self.source.snapshot().row_span;
        self.grow_runtime(ctx.device, ctx.queue, row_span)?;
        self.refresh_binding(ctx.device);
        let workgroups = row_span.div_ceil(WG_SIZE).min(self.max_workgroups);
        let uniforms = SimUniforms {
            bounds_min: self.bounds_min,
            bounds_max: self.bounds_max,
            dt: ctx.delta_time,
            slot_count: row_span,
            _reserved: 0,
            dispatched_threads: workgroups.saturating_mul(WG_SIZE),
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        self.refresh_binding(ctx.device);
        let slot_count = self.source.snapshot().row_span;
        if slot_count == 0 {
            return Ok(());
        }
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SpriteSimulate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            self.bind_group
                .as_ref()
                .expect("sprite simulation binding is initialized"),
            &[],
        );
        pass.dispatch_workgroups(
            slot_count.div_ceil(WG_SIZE).min(self.max_workgroups),
            1,
            1,
        );
        Ok(())
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_runtime_buffer(device: &wgpu::Device, rows: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sprite Simulation Runtime"),
        size: u64::from(rows.max(1)) * RUNTIME_STRIDE,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn rounded_runtime_capacity(required: u32, maximum_rows: u32) -> Result<u32> {
    if required > maximum_rows {
        return Err(Error::Gpu(format!(
            "sprite simulation needs {required} runtime rows, device permits {maximum_rows}"
        )));
    }
    Ok(required
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(maximum_rows)
        .min(maximum_rows.max(1))
        .max(required))
}

const _: () = {
    assert!(std::mem::size_of::<SpriteRuntimeRow>() == 24);
    assert!(std::mem::size_of::<SimUniforms>() == 32);
};

#[cfg(test)]
mod tests {
    use super::*;
    use helio_scenedb::SceneSpriteRow;
    use wgpu::util::DeviceExt;

    async fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut adapter = None;
        for fallback in [false, true] {
            if let Ok(candidate) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: fallback,
                    apply_limit_buckets: false,
                })
                .await
            {
                adapter = Some(candidate);
                break;
            }
        }
        let adapter = adapter?;
        let mut limits = adapter.limits();
        limits.max_compute_workgroups_per_dimension = 1;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Sprite Simulation Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()
    }

    #[test]
    fn growth_rebind_and_grid_stride_preserve_the_runtime_projection() {
        let Some((device, queue)) = pollster::block_on(gpu()) else {
            eprintln!("skipping sprite simulation GPU test: no adapter");
            return;
        };
        const ROWS: u32 = 300;
        let mut initial = SceneSpriteRow::new([0.0, 0.0], [1.0, 1.0])
            .with_simulation_velocity([1.0, 0.0]);
        initial.set_authored_epoch(1);
        let initial_instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initial Sprite Rows"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let initial_presence = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initial Sprite Presence"),
            contents: bytemuck::bytes_of(&1u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let source = SpriteBufferSource::new(
            initial_instances,
            1,
            initial_presence,
            1,
            1,
        );
        let mut pass = SpriteSimulatePass::try_new(
            &device,
            source.clone(),
            [-10_000.0, -10_000.0],
            [10_000.0, 10_000.0],
        )
        .unwrap();
        assert!(SpriteSimulatePass::try_new(
            &device,
            source.clone(),
            [-1.0; 2],
            [1.0; 2],
        )
        .is_err());

        let rows: Vec<_> = (0..ROWS)
            .map(|index| {
                let mut row = SceneSpriteRow::new([index as f32, 0.0], [1.0, 1.0])
                    .with_simulation_velocity([1.0, 0.0]);
                row.set_authored_epoch(index + 1);
                row
            })
            .collect();
        let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grown Sprite Rows"),
            contents: bytemuck::cast_slice(&rows),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let presence = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grown Sprite Presence"),
            contents: bytemuck::cast_slice(&vec![1u32; ROWS as usize]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        source.publish_authored(instances, 2, presence, 2, ROWS);
        pass.grow_runtime(&device, &queue, ROWS).unwrap();
        pass.refresh_binding(&device);
        let runtime = source.snapshot().runtime.expect("runtime remains published");
        assert!(runtime.row_capacity >= ROWS);
        assert!(runtime.epoch > 1);
        let first_owner_last_epoch = runtime.epoch;

        let uniforms = SimUniforms {
            bounds_min: [-10_000.0; 2],
            bounds_max: [10_000.0; 2],
            dt: 1.0,
            slot_count: ROWS,
            _reserved: 0,
            dispatched_threads: WG_SIZE,
        };
        queue.write_buffer(&pass.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Runtime Tail Readback"),
            size: RUNTIME_STRIDE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sprite Simulation Growth Test"),
        });
        {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sprite Simulation Growth Dispatch"),
                timestamp_writes: None,
            });
            compute.set_pipeline(&pass.pipeline);
            compute.set_bind_group(0, pass.bind_group.as_ref().unwrap(), &[]);
            compute.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &pass.runtime_buf,
            u64::from(ROWS - 1) * RUNTIME_STRIDE,
            &staging,
            0,
            RUNTIME_STRIDE,
        );
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        let tail = *bytemuck::from_bytes::<SpriteRuntimeRow>(&mapped);
        assert_eq!(tail.position, [ROWS as f32, 0.0]);
        assert_eq!(tail.authored_epoch, ROWS);
        drop(mapped);
        staging.unmap();

        drop(pass);
        assert!(source.snapshot().runtime.is_none());
        let replacement = SpriteSimulatePass::try_new(
            &device,
            source.clone(),
            [-10_000.0; 2],
            [10_000.0; 2],
        )
        .unwrap();
        assert!(
            source.snapshot().runtime.unwrap().epoch > first_owner_last_epoch,
            "a replacement owner must not alias a cached runtime epoch"
        );
        drop(replacement);
    }
}
