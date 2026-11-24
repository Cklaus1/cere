// P3b (m23 design note M23-D1): the grouped-GEMM runner behind p3b_runner.cu's CutlassMoeFCRunner (NVFP4 x NVFP4
// -> BF16); its SM120 kernels are the 32 instantiations executor/CMakeLists.txt writes from p3b_instance.cu.in.
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h"

namespace tensorrt_llm::kernels::cutlass_kernels {
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
}
