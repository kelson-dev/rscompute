# rscompute

This repo is me learning how to run compute shaders using Vulkan from Rust using the Ash bindings.
Ash is mostly direct bindings to the Vulkan C functions so the code here should be mostly analogous to a similar program in C.

## Heap corruption issue.

I'm currently facing heap corruption issues that I can't track down.

On Ubuntu 22.04 with a Radeon graphics card the program encounters a seemingly random crash at some point after "read_results" is called (main.rs:48).

On Windows 11 with an NVidia graphics card the program succeeds about 50% of the time, and fails the other 50% with "exit code: 0xc0000374, STATUS_HEAP_CORRUPTION"

Sample outputs in the [runs](./runs/) folder.

## Program flow

1. Ash is initialized with Entry::load and then a context.rs/VkCtx struct is built.
2. The VkCtx struct holds the Instance, Device, Command Pool/Buffer, Queue, and Descriptor Pool.
3. A demo::matrix_nx_m::MatrixNxMShader is created with the sample matricies.
4. A ShaderExecutionContext is created, which contains the Shader Module, Descript Set/Layouts, Pipeline/Pipeline Layout, and Buffers and Memory objects for each of the input and output buffers of the shader. The shader code itself is in the [shaders](./shaders/) folder.
5. The input buffers are written to the GPU by mapping memory, copying the matrix data, and then unmapping the memory.
6. The shader is ran, using a fence to wait for shader execution to finish.
7. The output buffer is read from the GPU by mapping memory, copying the matrix data, and then unmaping the memory.
8. The ShaderExecutionContext is cleaned up before it gets dropped, destroying its buffers, freeing its memory, and then destroying the Pipeline, Pipeline Layout, freeing the Descriptor Sets, destroying the Descriptor Set Layout, and destroying the Shader Module
9. The results of the matrix multiplication are printed
10. The VkCtx is cleaned up by destroying the Descriptor Pool, Command Buffer, Command Pool, Device, and then the Instance.

