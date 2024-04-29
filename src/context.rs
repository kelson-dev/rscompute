use std::ffi::{CStr, CString};
use ash::vk;
use ash::vk::{API_VERSION_1_1, DescriptorPoolCreateFlags};
use vk::ApplicationInfo;

pub struct VkCtx {
    pub command_buffer: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
    pub device : ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub instance: ash::Instance,
    pub queue: vk::Queue,
    pub descriptor_pool: vk::DescriptorPool,
}

impl VkCtx {
    pub fn destroy(&self) {
        println!("Destroying VkCtx");
        unsafe {
            println!("Destroying descriptor pool {:?}", self.descriptor_pool);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            println!("Destroying command buffers {:?}", self.command_buffer);
            self.device.free_command_buffers(self.command_pool, &[self.command_buffer]);
            println!("Destroying command pool {:?}", self.command_pool);
            self.device.destroy_command_pool(self.command_pool, None);
            println!("Destroying device {:?}", self.device.handle());
            self.device.destroy_device(None);
            println!("Destroying instance");
            self.instance.destroy_instance(None);
        }
    }
}

impl VkCtx {
    pub unsafe fn create_compute_ctx(entry: &ash::Entry) -> Result<VkCtx, ash::LoadingError> {
        // set up validation layers

        let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").expect("Failed to create CString")];

        // Check if validation layers are available

        let layer_properties = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("Failed to enumerate instance layer properties");
        for layer_name in layer_names.iter() {
            let mut layer_found = false;
            for layer_property in layer_properties.iter() {
                let test_layer_name = unsafe { CStr::from_ptr(layer_property.layer_name.as_ptr()) };
                if test_layer_name == layer_name.as_c_str() {
                    layer_found = true;
                    break;
                }
            }
            if layer_found == false {
                panic!("Validation layer {:?} not available", layer_name);
            }
        }

        let app_name = CString::new("matrix-multiplication").expect("Failed to create CString");
        let engine_name = CString::new("No Engine").expect("Failed to create CString");

        let app_info = ApplicationInfo {
            p_application_name : app_name.as_ptr(),
            application_version: 0,
            engine_version: 0,
            p_engine_name: engine_name.as_ptr(),
            api_version: API_VERSION_1_1,
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layer_names.len() as u32,
            pp_enabled_layer_names: layer_names.as_ptr() as *const *const std::ffi::c_char,
            ..Default::default()
        };

        print!("Creating instance... ");
        let instance = unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");
        println!("handle is {:?}", instance.handle());
        print!("Getting a Physical Device... ");
        let physical_device = unsafe { instance.enumerate_physical_devices() }
            .expect("Failed to enumerate physical devices")
            .into_iter()
            .next()
            .expect("No physical devices found");
        println!("found {:?}", physical_device);


        print!("Creating device... ");
        let device = unsafe {
            let queue_create_info = vk::DeviceQueueCreateInfo {
                queue_family_index: 0,
                queue_count: 1,
                p_queue_priorities: &1.0f32,
                ..Default::default()
            };
            let device_create_info = vk::DeviceCreateInfo {
                queue_create_info_count: 1,
                p_queue_create_infos: &queue_create_info,
                ..Default::default()
            };
            instance.create_device(physical_device, &device_create_info, None)
        }.expect("Failed to create device");
        println!("handle is {:?}", device.handle());

        print!("Getting a queue... ");
        let queue = unsafe { device.get_device_queue(0, 0) };
        println!("got queue {:?}", queue);

        print!("Creating command pool... ");
        let command_pool = unsafe {
            let command_pool_create_info = vk::CommandPoolCreateInfo {
                queue_family_index: 0,
                ..Default::default()
            };
            device.create_command_pool(&command_pool_create_info, None)
        }.expect("Failed to create command pool");
        println!("handle is {:?}", command_pool);

        print!("Creating command buffer... ");
        let command_buffer = unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            device.allocate_command_buffers(&command_buffer_allocate_info)
        }.expect("Failed to allocate command buffer")[0];
        println!("handle is {:?}", command_buffer);
        print!("Creating descriptor pool... ");
        let descriptor_pool = unsafe {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 3,
                },
            ];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
                flags: DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                ..Default::default()
            };
            device.create_descriptor_pool(&descriptor_pool_create_info, None)
        }.expect("Failed to create descriptor pool");
        println!("handle is {:?}", descriptor_pool);

        Ok(VkCtx {
            command_buffer,
            command_pool,
            device,
            physical_device,
            instance,
            queue,
            descriptor_pool,
        })
    }

    pub fn create_shader_module(&self, source: Vec<u32>) -> Result<vk::ShaderModule, vk::Result> {
        println!("Creating shader module");
        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: source.len() * 4,
            p_code: source.as_ptr(),
            ..Default::default()
        };
        unsafe {
            self.device.create_shader_module(&shader_info, None)
        }
    }


    pub unsafe fn find_memory_type(
        &self,
        memory_type_bits: u32,
        flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let memory_properties = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };
        for i in 0..memory_properties.memory_type_count {
            if (memory_type_bits & (1 << i)) != 0
                && (memory_properties.memory_types[i as usize].property_flags & flags) == flags {
                return Some(i);
            }
        }
        None
    }
}