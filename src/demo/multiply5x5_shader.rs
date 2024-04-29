use ash::vk;
use ash::vk::{
    DescriptorSet, DescriptorType, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorBufferInfo,
    WriteDescriptorSet,
    Pipeline, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo,
    PipelineShaderStageCreateInfo, ComputePipelineCreateInfo,
    FenceCreateInfo, CommandBufferBeginInfo, PipelineBindPoint, ShaderModule, SubmitInfo,
    ShaderStageFlags };
use crate::context::VkCtx;
use crate::shader::{ComputeShader, LayoutDescription, ShaderExecutionContext};
use crate::data::{GpuMappedMemory, LinkedMemory};
use crate::demo::matrix5x5::Matrix5x5;
use shaderc;
use shaderc::ShaderKind::Compute;
use crate::demo::multiply_nx_m_shader::MatrixPairSizes;
use crate::shader::LayoutDescriptorIndex::{ReadIndex, WriteIndex};

pub struct Matrix5x5MultiplicationShader {
    pub a : Matrix5x5,
    pub b : Matrix5x5,
    pub result : Matrix5x5
}

impl Matrix5x5MultiplicationShader {
    pub fn new(a: Matrix5x5, b: Matrix5x5) -> Matrix5x5MultiplicationShader {
        Matrix5x5MultiplicationShader {
            a,
            b,
            result: Matrix5x5 {
                data: [0.0; 25]
            }
        }
    }

    pub fn source() -> String {
        String::from(
        include_str!("../shaders/matrix_multiplication.comp"))
    }
}


impl ComputeShader<MatrixPairSizes> for Matrix5x5MultiplicationShader {
    fn get_spirv() -> Result<Vec<u32>, vk::Result> {
        Self::compile_to_spirv(
            &Matrix5x5MultiplicationShader::source(),
            "matrix_multiplication.comp",
            "main")
    }

    fn get_layout_descriptors(&self) -> Vec<LayoutDescription> {
        vec![
            LayoutDescription {
                binding: 0,
                buffer_size: self.a.buffer_size(),
                index: WriteIndex(0),
            },
            LayoutDescription {
                binding: 1,
                buffer_size: self.b.buffer_size(),
                index: WriteIndex(1),
            },
            LayoutDescription {
                binding: 2,
                buffer_size: self.result.buffer_size(),
                index: ReadIndex(0),
            },
        ]
    }

    fn get_writables(&self) -> Vec<&dyn GpuMappedMemory> {
        vec![&self.a, &self.b]
    }

    fn get_readables(&mut self) -> Vec<&mut dyn GpuMappedMemory> {
        vec![&mut self.result]
    }

    fn get_group_vec(&self) -> (u32, u32, u32) {
        (5, 5, 1)
    }

    fn get_push_constants(&self) -> Option<MatrixPairSizes> {
        Some(MatrixPairSizes{
            acbr: 5,
            ar: 5,
            bc: 5
        })
    }
}