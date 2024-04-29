use crate::demo::matrix_nx_m::MatrixNxM;
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
use crate::shader::{ComputeShader, LayoutDescription, LayoutDescriptorIndex, ShaderExecutionContext};
use crate::data::{GpuMappedMemory, LinkedMemory};
use shaderc;
use crate::shader::LayoutDescriptorIndex::{ReadIndex, WriteIndex};

pub struct MatrixNxMShader {
    pub a: MatrixNxM,
    pub b: MatrixNxM,
    pub result: MatrixNxM,
}

pub struct MatrixPairSizes {
    pub acbr: u32,
    pub ar: u32,
    pub bc: u32,
}

impl MatrixNxMShader {
    pub fn new(a: MatrixNxM, b: MatrixNxM) -> MatrixNxMShader {
        let a_rows = a.rows;
        let a_columns = a.data.len() / a.rows;
        let b_rows = b.rows;
        let b_columns = b.data.len() / b_rows;
        // assert that a.columns == b.rows
        assert_eq!(a_columns, b_rows);
        MatrixNxMShader {
            a,
            b,
            result: MatrixNxM {
                rows: a_rows,
                data: vec![0.0f32; a_rows * b_columns]
            }
        }
    }

    pub fn source() -> String {
        String::from(
        include_str!("../shaders/matrix_multiplication.comp"))
    }
}

impl ComputeShader<MatrixPairSizes> for MatrixNxMShader {
    fn get_spirv() -> Result<Vec<u32>, vk::Result> {
        Self::compile_to_spirv(
            &MatrixNxMShader::source(),
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
            }
        ]
    }

    fn get_writables(&self) -> Vec<&dyn GpuMappedMemory> {
        vec![&self.a, &self.b]
    }

    fn get_readables(&mut self) -> Vec<&mut dyn GpuMappedMemory> {
        vec![&mut self.result]
    }

    fn get_group_vec(&self) -> (u32, u32, u32) {
        let push_constants = self.get_push_constants().expect("Failed to get push constants when required");
        (push_constants.ar, push_constants.bc, 1)
    }

    fn get_push_constants(&self) -> Option<MatrixPairSizes> {
        let a_columns = self.a.data.len() / self.a.rows;
        let b_columns = self.b.data.len() / self.b.rows;
        Some(MatrixPairSizes {
            acbr: a_columns as u32,
            ar: self.a.rows as u32,
            bc: b_columns as u32,
        })
    }
}