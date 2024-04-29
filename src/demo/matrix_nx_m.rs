use crate::context::VkCtx;
use crate::data::{GpuMappedMemory, LinkedMemory, MappedMemoryPointer};

pub struct MatrixNxM {
    pub rows: usize,
    pub data: Vec<f32>
}

impl MatrixNxM {
    pub fn print(&self) {
        let columns = self.data.len() / self.rows;
        for i in 0..self.rows {
            print!("| ");
            for j in 0..columns {
                let index = (i * columns) + j;
                print!("{:6} ", self.data[index]);
            }
            println!(" |");
        }
    }

    pub fn new(rows: usize, data: Vec<f32>) -> MatrixNxM {
        assert_eq!(data.len() % rows, 0);
        MatrixNxM {
            rows,
            data
        }
    }
}

impl GpuMappedMemory for MatrixNxM {
    fn write(&self, ctx: &VkCtx, linked_memory: &LinkedMemory) -> ash::vk::Result {
        unsafe {
            let pointer = linked_memory
                .get_mapped_pointer(ctx, self.data.len() as u64);
            std::ptr::copy_nonoverlapping(
                self.data.as_ptr(),
                pointer.ptr,
                std::mem::size_of::<f32>() * self.data.len());
            pointer.unmap();
        }

        ash::vk::Result::SUCCESS
    }

    fn read(&mut self, ctx: &VkCtx, linked_memory: &LinkedMemory) -> ash::vk::Result {
        unsafe {
            let pointer = linked_memory
                .get_mapped_pointer(ctx, self.data.len() as u64);
            std::ptr::copy_nonoverlapping(
                pointer.ptr,
                self.data.as_mut_ptr(),
                std::mem::size_of::<f32>() * self.data.len());
            pointer.unmap();
        };
        ash::vk::Result::SUCCESS
    }

    fn buffer_size(&self) -> u64 {
        std::mem::size_of::<f32>() as u64
            * self.data.len() as u64
    }
}