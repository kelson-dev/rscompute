
use crate::context::VkCtx;
use crate::data::{GpuMappedMemory, LinkedMemory, MappedMemoryPointer};

#[derive(Clone, Copy, Debug)]
pub struct Matrix5x5 {
    pub data: [f32;25],
}

impl Matrix5x5 {
    pub fn print(&self) {
        for i in 0..5 {
            print!("| ");
            for j in 0..5 {
                print!("{:6} ", self.data[i * 5 + j]);
            }
            println!(" |");
        }
    }
}

impl GpuMappedMemory for Matrix5x5 {

    fn write(&self, ctx: &VkCtx, linked_memory: &LinkedMemory) -> ash::vk::Result {
        unsafe {
            let pointer = linked_memory
                .get_mapped_pointer(ctx,25);
            std::ptr::copy_nonoverlapping(
                &self.data as *const f32,
                pointer.ptr,
                std::mem::size_of::<Matrix5x5>());
            pointer.unmap();
        }

        ash::vk::Result::SUCCESS
    }

    fn read(&mut self, ctx: &VkCtx, linked_memory: &LinkedMemory) -> ash::vk::Result {
        unsafe {
            let pointer = linked_memory
                .get_mapped_pointer(ctx, 25);
            std::ptr::copy_nonoverlapping(
                pointer.ptr,
                &mut self.data as *mut [f32; 25] as *mut f32,
                std::mem::size_of::<Matrix5x5>());
            pointer.unmap();
        };
        ash::vk::Result::SUCCESS
    }

    fn buffer_size(&self) -> u64 {
        std::mem::size_of::<[f32;25]>() as u64
    }
}

