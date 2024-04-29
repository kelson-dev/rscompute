use crate::context::VkCtx;
use ash;
use ash::vk::{Buffer, DeviceMemory};

#[derive(Clone, Debug)]
pub struct LinkedMemory {
    pub binding: u32,
    pub buffer: Buffer,
    pub memory: DeviceMemory,
}

impl LinkedMemory {
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }

    pub fn get_mapped_pointer<'a, T>(&self, ctx: &VkCtx, length: u64) -> MappedMemoryPointer<'a, T> {
        let ptr = unsafe {
            ctx.device.map_memory(
                self.memory,
                0,
                std::mem::size_of::<T>() as u64 * length,
                ash::vk::MemoryMapFlags::empty())
        }.expect("Failed to map pointer for host memory") as *mut T;

        MappedMemoryPointer {
            ptr,
            device: ctx.device.clone(),
            memory: self.memory,
            _marker: std::marker::PhantomData,
        }
    }
}

pub struct MappedMemoryPointer<'a, T> {
    pub ptr: *mut T,
    pub device: ash::Device,
    pub memory: DeviceMemory,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> MappedMemoryPointer<'a, T> {
    pub fn unmap(&self) {
        unsafe {
            self.device.unmap_memory(self.memory)
        };
    }
}

pub trait GpuMappedMemory {

    fn write(&self, ctx: &VkCtx, buffer: &LinkedMemory) -> ash::vk::Result;

    fn read(&mut self, ctx: &VkCtx, buffer: &LinkedMemory) -> ash::vk::Result;

    fn buffer_size(&self) -> u64;
}