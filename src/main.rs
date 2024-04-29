// Program to compute the multiplication of two matrices using a vulkan compute shader

#![allow(unused)]

pub mod context;
pub mod demo;
pub mod shader;
pub mod data;

use crate::context::VkCtx;
use crate::shader::ComputeShader;

use std::error::Error;
use ash::util::*;
use ash::vk;
use crate::demo::matrix_nx_m::MatrixNxM;
use crate::demo::multiply_nx_m_shader::MatrixNxMShader;

pub fn main() -> Result<(), Box<dyn Error>> {
    let matrix_a = MatrixNxM::new(3, vec![
        1.0, 1.0,
        2.0, 2.0,
        3.0, 3.0,
    ]);

    let matrix_b = MatrixNxM::new(2, vec![
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
    ]);

    let entry = unsafe { ash::Entry::load().expect("Failed to load ash Entry") };

    println!("Creating Vulkan context");
    let ctx = unsafe {
        VkCtx::create_compute_ctx(&entry).expect("Failed to create Vulkan context")
    };

    println!("Creating shader");
    let mut shader = MatrixNxMShader::new(matrix_a, matrix_b);
    {
        println!("Building shader context");
        let shader_ctx = shader.build_shader_context(&ctx).expect("Failed to build shader context");
        println!("Writing inputs to shader");
        shader.write_inputs(&ctx, &shader_ctx.write_buffers);
        println!("Running shader");
        shader.run_shader(&ctx, &shader_ctx);
        println!("Reading results from shader");
        shader.read_result(&ctx, &shader_ctx.read_buffers);
        println!("Destroying shader context");
        shader_ctx.destroy(&ctx.device, &ctx.descriptor_pool);
    }

    println!("Multiplied matrices: ");
    shader.a.print();
    println!("and");
    shader.b.print();
    println!("to get");
    shader.result.print();

    ctx.destroy();

    Ok(())
}

