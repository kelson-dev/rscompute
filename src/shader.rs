use std::ops::Deref;
use std::thread;
use std::time::Duration;
use ash::vk;
use ash::vk::{CommandBufferBeginInfo, ComputePipelineCreateInfo, DescriptorBufferInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, FenceCreateInfo, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, ShaderStageFlags, SubmitInfo, WriteDescriptorSet};
use vk::{DescriptorSet, DescriptorSetLayout, Pipeline, ShaderModule};
use crate::context::VkCtx;
use crate::data::{LinkedMemory, GpuMappedMemory};
use crate::demo::matrix5x5::Matrix5x5;
use crate::demo::multiply5x5_shader::Matrix5x5MultiplicationShader;
use crate::demo::multiply_nx_m_shader::MatrixNxMShader;

#[derive(Clone, Debug)]
pub struct ShaderExecutionContext {
    pub shader_module : ShaderModule,
    pub descriptor_set_layouts: Vec<DescriptorSetLayout>,
    pub descriptor_set: DescriptorSet,
    pub pipeline_layout: PipelineLayout,
    pub pipeline : Pipeline,
    pub write_buffers: Vec<LinkedMemory>,
    pub read_buffers: Vec<LinkedMemory>
}

impl ShaderExecutionContext {
    pub fn destroy(&self, device: &ash::Device, descriptor_set_pool: &vk::DescriptorPool) {
        unsafe {
            println!("Destroying buffers");
            for buffer in self.write_buffers.iter() {
                buffer.destroy(device);
            }
            for buffer in self.read_buffers.iter() {
                buffer.destroy(device);
            }
            println!("Destroying pipeline");
            device.destroy_pipeline(self.pipeline, None);
            println!("Destroying pipeline layout");
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            println!("Freeing descriptor sets");
            device.free_descriptor_sets(*descriptor_set_pool, &[self.descriptor_set]);
            println!("Destroying descriptor set layouts");
            for layout in self.descriptor_set_layouts.iter() {
                device.destroy_descriptor_set_layout(*layout, None);
            }
            println!("Destroying shader module");
            device.destroy_shader_module(self.shader_module, None);
        }
    }
}

pub struct LayoutDescription {
    pub binding: u32,
    pub buffer_size: u64,
    pub index: LayoutDescriptorIndex
}

impl LayoutDescription {

    pub fn create_linked_memory(&self, ctx: &VkCtx) -> Result<LinkedMemory, vk::Result> {
        let buffer = unsafe {
            let buffer_info = vk::BufferCreateInfo {
                size: self.buffer_size,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                ..Default::default()
            };

            ctx.device.create_buffer(&buffer_info, None)
        }?;

        let memory = unsafe {
            let mem_requirements = unsafe {
                ctx.device.get_buffer_memory_requirements(buffer)
            };

            let memory_type_index = unsafe {
                ctx.find_memory_type(
                    mem_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT)
                    .expect("Failed to find suitable memory type")
            };

            let memory_info = vk::MemoryAllocateInfo {
                allocation_size: mem_requirements.size,
                memory_type_index,
                ..Default::default()
            };

            ctx.device.allocate_memory(&memory_info, None)
        }?;

        unsafe {
            ctx.device.bind_buffer_memory(buffer, memory, 0)
        }?;

        Ok(LinkedMemory { binding: self.binding, buffer, memory })
    }
}

pub enum LayoutDescriptorIndex {
    WriteIndex(usize),
    ReadIndex(usize)
}

impl LayoutDescription {
    pub fn get_index(&self) -> usize {
        match self.index {
            LayoutDescriptorIndex::WriteIndex(i) => i,
            LayoutDescriptorIndex::ReadIndex(i) => i
        }
    }
}


pub trait ComputeShader<TPushConstants : Sized> {
    /**
     * Reads or creates the SPIR-V binary for the shader
     */
    fn get_spirv() -> Result<Vec<u32>, vk::Result>;

    fn compile_to_spirv(source: &str, file_name: &str, entry_point: &str) -> Result<Vec<u32>, vk::Result> {
        let mut compiler = shaderc::Compiler::new().expect("Failed to create shader compiler");
        let mut options = shaderc::CompileOptions::new().expect("Failed to create shader compile options");
        let binary_result = compiler.compile_into_spirv(
            source,
            shaderc::ShaderKind::Compute,
            file_name,
            entry_point,
            Some(&options)).expect("Failed to compile shader");
        Ok(binary_result.as_binary().to_vec())
    }

    fn get_write_buffers(&self, ctx: &VkCtx) -> Vec<LinkedMemory> {
        self.get_layout_descriptors().iter()
            .filter(|descriptor| matches!(descriptor.index, LayoutDescriptorIndex::WriteIndex(_)))
            .map(|writable| {
                writable.create_linked_memory(ctx).expect("Failed to create linked memory for writable buffer")
            }).collect()
    }
    fn get_read_buffers(&self, ctx: &VkCtx) -> Vec<LinkedMemory> {
        self.get_layout_descriptors().iter()
            .filter(|descriptor| matches!(descriptor.index, LayoutDescriptorIndex::ReadIndex(_)))
            .map(|readable| {
                readable.create_linked_memory(ctx).expect("Failed to create linked memory for readable buffer")
            }).collect()
    }

    fn get_layout_descriptors(&self) -> Vec<LayoutDescription>;

    fn get_writables(&self) -> Vec<&dyn GpuMappedMemory>;

    fn get_readables(&mut self) -> Vec<&mut dyn GpuMappedMemory>;

    fn get_push_constants(&self) -> Option<TPushConstants> {
        None
    }

    fn get_group_vec(&self) -> (u32, u32, u32) {
        (1, 1, 1)
    }

    /**
     * Writes the contents of the shaders inputs to the GPU
     */
    fn write_inputs(&self, ctx: &VkCtx, write_buffers: &Vec<LinkedMemory>) -> vk::Result {
        let write_descriptors = self.get_layout_descriptors();
        let writables = self.get_writables();

        for descriptor in write_descriptors.iter()
            .filter(|descriptor| matches!(descriptor.index, LayoutDescriptorIndex::WriteIndex(_))) {
            let result = writables[descriptor.get_index()].write(ctx, &write_buffers[descriptor.get_index()]);
            if result != vk::Result::SUCCESS {
                return result;
            }
        }
        vk::Result::SUCCESS
    }

    /**
     * Reads the results of the compute shader from the GPU
     */
    fn read_result(&mut self, ctx: &VkCtx, read_buffers: &Vec<LinkedMemory>) -> vk::Result {
        let read_descriptors = self.get_layout_descriptors();
        let mut readables = self.get_readables();

        for descriptor in read_descriptors.iter()
            .filter(|descriptor| matches!(descriptor.index, LayoutDescriptorIndex::ReadIndex(_))) {
            let result = readables[descriptor.get_index()].read(ctx, &read_buffers[descriptor.get_index()]);
            if result != vk::Result::SUCCESS {
                return result;
            }
        }
        vk::Result::SUCCESS
    }

    /**
     * Dispatch the compute shader to execute the compute operation
     */
    fn run_shader(&self, ctx: &VkCtx, module: &ShaderExecutionContext) -> vk::Result {
        let pipeline = module.pipeline;
        let command_buffer = ctx.command_buffer;
        let descriptor_set = module.descriptor_set;
        let pipeline_layout = module.pipeline_layout;

        let (group_count_x, group_count_y, group_count_z) = self.get_group_vec();

        unsafe {
            let fence =
                ctx.device.create_fence(&FenceCreateInfo {
                    ..Default::default()
                }, None)
                .expect("Failed to create fence");
            ctx.device.begin_command_buffer(command_buffer, &CommandBufferBeginInfo::default());
            ctx.device.cmd_bind_pipeline(command_buffer, PipelineBindPoint::COMPUTE, pipeline);
            ctx.device.cmd_bind_descriptor_sets(command_buffer, PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[descriptor_set], &[]);

            let push_constants = self.get_push_constants();
            if let Some(push_constants) = push_constants {
                let push_constants_as_u8 =
                    std::slice::from_raw_parts(
                        &push_constants as *const TPushConstants as *const u8,
                        std::mem::size_of_val(&push_constants));

                ctx.device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    &push_constants_as_u8);
            }

            ctx.device.cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
            ctx.device.end_command_buffer(command_buffer);
            ctx.device.queue_submit(ctx.queue, &[SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                ..Default::default()
            }], fence);
            ctx.device.wait_for_fences(&[fence], true, std::u64::MAX);
            ctx.device.destroy_fence(fence, None);
        }

        //thread::sleep(Duration::from_secs(1));

        vk::Result::SUCCESS
    }

    fn create_compute_pipeline(layouts: &Vec<DescriptorSetLayout>, pipeline_layout: &PipelineLayout,
                               ctx: &VkCtx, shader_module: &ShaderModule) -> Result<Pipeline, vk::Result> {
        let compute_pipeline_info = ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo {
                module: shader_module.clone(),
                p_name: b"main\0".as_ptr() as *const std::ffi::c_char,
                stage: ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            layout: pipeline_layout.clone(),
            ..Default::default()
        };

        let compute_infos = [compute_pipeline_info];

        let pipelines = unsafe {
            ctx.device.create_compute_pipelines(PipelineCache::null(), &compute_infos, None)
        }.expect("Failed to create compute pipeline");

        Ok(pipelines[0])
    }

    /**
     * Builds the shader context, which can be used multiple times to
     * write to the shader inputs, dispatch the shader, and read the results.
     */
    fn build_shader_context(&self, ctx: &VkCtx) -> Result<ShaderExecutionContext, vk::Result> {
        let shader_module = ctx.create_shader_module(
            Self::get_spirv().expect("Failed to get SPIR-V binary for shader"))
            .expect("Failed to create shader module");

        let layout_descriptors = self.get_layout_descriptors();
        let mut descriptors = layout_descriptors.iter()
            .map(|descriptor| {
                DescriptorSetLayoutBinding {
                    binding: descriptor.binding,
                    descriptor_type: DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: ShaderStageFlags::COMPUTE,
                    ..Default::default()
                }
            })
            .collect::<Vec<DescriptorSetLayoutBinding>>();

        let descriptor_set_layouts = vec![
            unsafe {
                ctx.device.create_descriptor_set_layout(&DescriptorSetLayoutCreateInfo {
                    binding_count: descriptors.len() as u32,
                    p_bindings: descriptors.as_ptr(),
                    ..Default::default()
                }, None)
            }.expect("Failed to create descriptor set layout")
        ];

        let descriptor_set = unsafe {
            ctx.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: ctx.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: descriptor_set_layouts.as_ptr(),
                ..Default::default()
            }).expect("Failed to allocate descriptor sets")[0]
        };

        let push_constant_range = match self.get_push_constants() {
            Some(_) => vk::PushConstantRange {
                stage_flags: ShaderStageFlags::COMPUTE,
                offset: 0,
                size: std::mem::size_of::<TPushConstants>() as u32
            },
            None => vk::PushConstantRange {
                stage_flags: ShaderStageFlags::empty(),
                offset: 0,
                size: 0
            }
        };

        let pipeline_layout_info = PipelineLayoutCreateInfo {
            set_layout_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            ctx.device.create_pipeline_layout(&pipeline_layout_info, None)
        }?;

        let pipeline = Self::create_compute_pipeline(&descriptor_set_layouts, &pipeline_layout, &ctx, &shader_module)
            .expect("Failed to create compute pipeline");

        let module = ShaderExecutionContext {
            shader_module,
            descriptor_set_layouts,
            descriptor_set,
            pipeline_layout,
            pipeline,
            write_buffers: self.get_write_buffers(ctx),
            read_buffers: self.get_read_buffers(ctx)
        };

        let descriptor_set_layout = module.descriptor_set_layouts[0];
        let descriptor_set = module.descriptor_set;

        let buffer_info = layout_descriptors.iter()
            .map(|descriptor| {
                DescriptorBufferInfo {
                    buffer: match descriptor.index {
                        LayoutDescriptorIndex::WriteIndex(i) => module.write_buffers[i].buffer,
                        LayoutDescriptorIndex::ReadIndex(i) => module.read_buffers[i].buffer
                    },
                    offset: 0,
                    range: descriptor.buffer_size
                }
            })
            .collect::<Vec<DescriptorBufferInfo>>();

        let write_descriptor_sets = buffer_info.iter()
            .enumerate()
            .map(|(i, buffer_info)| {
                WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: layout_descriptors[i].binding,
                    descriptor_count: 1,
                    descriptor_type: DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: buffer_info,
                    ..Default::default()
                }
            })
            .collect::<Vec<WriteDescriptorSet>>();

        unsafe {
            ctx.device.update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        Ok(module)
    }
}