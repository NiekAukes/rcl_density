// Auto-generated GPU orchestrator — do not edit
#![allow(warnings)]

use std::sync::Arc;

use crate::mathf64::Vec3;
use crate::orchestration::PermutationTables;
use crate::utils::PerlinNoiseSampler;

const GRID_X: u32 = 16;
const GRID_Y: u32 = 384;
const GRID_Z: u32 = 16;
const TOTAL_ELEMENTS: usize = (GRID_X * GRID_Y * GRID_Z) as usize; // 98304
const OUTPUT_BUFFER_SIZE: u64 = (TOTAL_ELEMENTS * size_of::<f32>()) as u64;
const PERM_GENERATOR_SIZE: u64 = (256 * size_of::<i32>() + 3 * size_of::<f32>()) as u64; // 1036
const UNIFORM_VEC3_SIZE: u64 = 16; // vec3 padded to 16 bytes (std140)

pub struct GpuOrchestrator_final_density {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,

    // Compute pipelines
    pipeline_minecraft_noodle_0_98304: wgpu::ComputePipeline,
    pipeline_minecraft_no_interpolate_test_98304: wgpu::ComputePipeline,
    pipeline_final_density_98304: wgpu::ComputePipeline,

    // Output storage buffers (one per shader)
    buf_minecraft_noodle_0_98304_out: wgpu::Buffer,
    buf_minecraft_no_interpolate_test_98304_out: wgpu::Buffer,
    buf_final_density_98304_out: wgpu::Buffer,

    buf_origin: wgpu::Buffer,
    buf_dimensions: wgpu::Buffer,

    // Packed density-input storage buffers
    buf_minecraft_no_interpolate_test_98304_density_inputs: wgpu::Buffer,
    buf_final_density_98304_density_inputs: wgpu::Buffer,

    // Packed permutation-table storage buffers
    buf_minecraft_noodle_0_98304_perm_tables: wgpu::Buffer,

    // Pre-built bind groups
    bind_group_minecraft_noodle_0_98304: wgpu::BindGroup,
    bind_group_minecraft_no_interpolate_test_98304: wgpu::BindGroup,
    bind_group_final_density_98304: wgpu::BindGroup,
}

impl GpuOrchestrator_final_density {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("Failed to request GPU device");

        // --- Load shader modules ---
        let sm_minecraft_noodle_0 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("minecraft_noodle_0"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/minecraft_noodle_0.wgsl").into(),
            ),
        });
        let sm_minecraft_no_interpolate_test =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("minecraft_no_interpolate_test"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/minecraft_no_interpolate_test.wgsl").into(),
                ),
            });
        let sm_final_density = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("final_density"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/final_density.wgsl").into()),
        });

        // --- Create compute pipelines ---
        let pipeline_minecraft_noodle_0_98304 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline_minecraft_noodle_0_98304"),
                layout: None,
                module: &sm_minecraft_noodle_0,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
        let pipeline_minecraft_no_interpolate_test_98304 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline_minecraft_no_interpolate_test_98304"),
                layout: None,
                module: &sm_minecraft_no_interpolate_test,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
        let pipeline_final_density_98304 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline_final_density_98304"),
                layout: None,
                module: &sm_final_density,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Create buffers ---
        let storage_out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let buf_minecraft_noodle_0_98304_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minecraft_noodle_0_98304_out"),
            size: 393216 as u64,
            usage: storage_out_usage,
            mapped_at_creation: false,
        });
        let buf_minecraft_no_interpolate_test_98304_out =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("minecraft_no_interpolate_test_98304_out"),
                size: 393216 as u64,
                usage: storage_out_usage,
                mapped_at_creation: false,
            });
        let buf_final_density_98304_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("final_density_98304_out"),
            size: 393216 as u64,
            usage: storage_out_usage,
            mapped_at_creation: false,
        });

        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let buf_origin = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("origin"),
            size: UNIFORM_VEC3_SIZE,
            usage: uniform_usage,
            mapped_at_creation: false,
        });
        let buf_dimensions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dimensions"),
            size: UNIFORM_VEC3_SIZE,
            usage: uniform_usage,
            mapped_at_creation: false,
        });

        let packed_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buf_minecraft_no_interpolate_test_98304_density_inputs =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("minecraft_no_interpolate_test_98304_density_inputs"),
                size: 393216,
                usage: packed_usage,
                mapped_at_creation: false,
            });
        let buf_final_density_98304_density_inputs =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("final_density_98304_density_inputs"),
                size: 393216,
                usage: packed_usage,
                mapped_at_creation: false,
            });
        let buf_minecraft_noodle_0_98304_perm_tables =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("minecraft_noodle_0_98304_perm_tables"),
                size: 2072,
                usage: packed_usage,
                mapped_at_creation: false,
            });

        // --- Create bind groups ---
        let bind_group_minecraft_noodle_0_98304 =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_minecraft_noodle_0"),
                layout: &pipeline_minecraft_noodle_0_98304.get_bind_group_layout(0),
                entries: &[
                    buf_entry(0, &buf_origin, UNIFORM_VEC3_SIZE),
                    buf_entry_whole(1, &buf_minecraft_noodle_0_98304_out),
                    buf_entry(2, &buf_dimensions, UNIFORM_VEC3_SIZE),
                    buf_entry_whole(3, &buf_minecraft_noodle_0_98304_perm_tables),
                ],
            });
        let bind_group_minecraft_no_interpolate_test_98304 =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_minecraft_no_interpolate_test"),
                layout: &pipeline_minecraft_no_interpolate_test_98304.get_bind_group_layout(0),
                entries: &[
                    buf_entry(0, &buf_origin, UNIFORM_VEC3_SIZE),
                    buf_entry_whole(1, &buf_minecraft_no_interpolate_test_98304_out),
                    buf_entry(2, &buf_dimensions, UNIFORM_VEC3_SIZE),
                    buf_entry_whole(3, &buf_minecraft_no_interpolate_test_98304_density_inputs),
                ],
            });
        let bind_group_final_density_98304 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_final_density"),
            layout: &pipeline_final_density_98304.get_bind_group_layout(0),
            entries: &[
                buf_entry(0, &buf_origin, UNIFORM_VEC3_SIZE),
                buf_entry_whole(1, &buf_final_density_98304_out),
                buf_entry(2, &buf_dimensions, UNIFORM_VEC3_SIZE),
                buf_entry_whole(3, &buf_final_density_98304_density_inputs),
            ],
        });

        let dims: [u32; 4] = [GRID_X, GRID_Y, GRID_Z, 0];
        queue.write_buffer(&buf_dimensions, 0, bytemuck::cast_slice(&dims));

        Self {
            device: Arc::new(device),
            queue,
            pipeline_minecraft_noodle_0_98304,
            pipeline_minecraft_no_interpolate_test_98304,
            pipeline_final_density_98304,
            buf_minecraft_noodle_0_98304_out,
            buf_minecraft_no_interpolate_test_98304_out,
            buf_final_density_98304_out,
            buf_origin,
            buf_dimensions,
            buf_minecraft_no_interpolate_test_98304_density_inputs,
            buf_final_density_98304_density_inputs,
            buf_minecraft_noodle_0_98304_perm_tables,
            bind_group_minecraft_noodle_0_98304,
            bind_group_minecraft_no_interpolate_test_98304,
            bind_group_final_density_98304,
        }
    }

    /// Run the full density pipeline on the GPU and return the target output.
    pub fn orchestrate(&self, origin: Vec3, perm_tables: &PermutationTables) -> DensityHandle {
        let origin_data: [f32; 4] = [origin.x as f32, origin.y as f32, origin.z as f32, 0.0];
        self.queue
            .write_buffer(&self.buf_origin, 0, bytemuck::cast_slice(&origin_data));

        let buf_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: OUTPUT_BUFFER_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut packed = Vec::new();
            packed.extend_from_slice(&perm_generator_bytes(
                &perm_tables.minecraft_noodle_0_octave__8,
            ));
            packed.extend_from_slice(&perm_generator_bytes(
                &perm_tables.minecraft_noodle_1_octave__8,
            ));
            self.queue
                .write_buffer(&self.buf_minecraft_noodle_0_98304_perm_tables, 0, &packed);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("density_encoder"),
            });

        self.queue.submit(std::iter::empty());

        // Wave 0: minecraft_noodle_0
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave_0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_minecraft_noodle_0_98304);
            pass.set_bind_group(0, &self.bind_group_minecraft_noodle_0_98304, &[]);
            pass.dispatch_workgroups(4, GRID_Y / 8, 4);
        }

        // Wave 1: minecraft_no_interpolate_test
        encoder.copy_buffer_to_buffer(
            &self.buf_minecraft_noodle_0_98304_out,
            0,
            &self.buf_minecraft_no_interpolate_test_98304_density_inputs,
            0,
            393216,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave_1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_minecraft_no_interpolate_test_98304);
            pass.set_bind_group(0, &self.bind_group_minecraft_no_interpolate_test_98304, &[]);
            pass.dispatch_workgroups(4, GRID_Y / 8, 4);
        }

        // Wave 2: final_density
        encoder.copy_buffer_to_buffer(
            &self.buf_minecraft_no_interpolate_test_98304_out,
            0,
            &self.buf_final_density_98304_density_inputs,
            0,
            393216,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave_2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_final_density_98304);
            pass.set_bind_group(0, &self.bind_group_final_density_98304, &[]);
            pass.dispatch_workgroups(4, GRID_Y / 8, 4);
        }

        encoder.copy_buffer_to_buffer(
            &self.buf_final_density_98304_out,
            0,
            &buf_staging,
            0,
            OUTPUT_BUFFER_SIZE,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Kick off the async map immediately, but don't wait
        let buf_staging = Arc::new(buf_staging); // requires Arc wrapping
        let buffer_slice = buf_staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        DensityHandle {
            receiver,
            buf_staging,
            device: Arc::clone(&self.device),
        }
    }
}


fn perm_generator_bytes(sampler: &PerlinNoiseSampler) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1036);
    for &b in sampler.permutation.iter() {
        bytes.extend_from_slice(&(b as i32).to_le_bytes());
    }
    bytes.extend_from_slice(&(sampler.origin_x as f32).to_le_bytes());
    bytes.extend_from_slice(&(sampler.origin_y as f32).to_le_bytes());
    bytes.extend_from_slice(&(sampler.origin_z as f32).to_le_bytes());
    bytes
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: Some(std::num::NonZeroU64::new(size).unwrap()),
        }),
    }
}

fn buf_entry_whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
