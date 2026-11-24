// Inkling 2 Mini S8: programmatic dependent launch for the inkling kernels (k1_primitives.cu's pdl_enter /
// launch_pdl convention). A launch_pdl kernel lets the stream's next kernel launch, then waits for the previous
// kernel's grid and its writes (pdl_enter), first in every block, before any global memory access: its results and
// the order of its arithmetic are those of a plain launch (bit-equal). The next kernel's launch and block dispatch
// overlap this grid; in a captured decode graph the stream order becomes programmatic edges.
#pragma once

#include <cuda_runtime.h>

namespace cere::inkling::pdl {

// trigger(): the stream's next kernel may launch. wait(): the previous kernel's grid has completed and its writes are
// visible. Between the two a kernel reads only the model's weights (sealed before any forward call, never written by
// a kernel), so its weight loads overlap the previous kernel's tail; every other access follows wait().
__device__ __forceinline__ void trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Called by every thread of every block, first (a kernel without a weight prologue).
__device__ __forceinline__ void enter() {
    trigger();
    wait();
}

// The largest grid launched as a programmatic dependent launch (k1_primitives.cu's kPdlMaxBlocks: a larger, prefill
// grid's blocks, resident early and waiting, would hold SM slots the predecessor's later waves need). A larger grid
// launches plainly; its enter() wait is then a no-op and the stream orders it as before.
constexpr unsigned long long kMaxBlocks = 1024;

template <typename... Params, typename... Args>
cudaError_t launch(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                   Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1ull * grid.x * grid.y * grid.z <= kMaxBlocks ? 1 : 0;
    const cudaError_t e = cudaLaunchKernelEx(&config, kernel, args...);
    const cudaError_t last = cudaGetLastError();
    return e != cudaSuccess ? e : last;
}

}  // namespace cere::inkling::pdl
