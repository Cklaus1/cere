#!/usr/bin/env python3
"""AmdAmd PAR-6: generate HIP kernel sources from the CUDA ones (executor/cuda/src/...), never editing them.

Each listed file goes through ROCm's hipify-perl, then through PATCHES: (file, old, new, expected count) text edits
for what has no HIP counterpart, each asserted to apply exactly `count` times so an upstream change fails the build
loudly instead of compiling something else. Output: build/hip/gen/<same relative path>, .cu -> .hip.

usage (in the llama-amd image, repo root as cwd): python3 executor/hip/gen_hip.py
"""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path("executor/cuda/src")
OUT = Path("build/hip/gen")
FILES = ["kernels/k1_primitives.h", "kernels/k1_primitives.cu", "kernels/k5_gdn_chunk.h", "kernels/p4_attention.h", "kernels/k3_hc_math.cuh",
         "fault/fault_inject.h",
         "kernels/k2_small_m.h", "kernels/k2_fast.h", "kernels/k2_fp8.h", "kernels/k2_mxfp8.h"]
# The model forward and the device (step 9): plain CUDA runtime / graph API, hipified like the kernels. vision.cpp
# (with kv_vision, plain CUDA too) included.
FILES += [f"model/{f}" for f in ("forward.cpp", "model.cpp", "model.h", "mtp.cu", "mtp.h", "layout.h", "moe_backend.h",
                                  "mxfp8_plan.h", "fp8_weights.cpp", "vision.h", "vision.cpp")]
FILES += [f"device/{f}" for f in ("cuda_device.cpp", "cuda_device.h", "memory_plan.h", "memory_plan.cpp",
                                   "context_generation.h")]  # chunk, p4: header only (kernels: executor/hip/src/*_hip.cpp)
# Plain-CUDA kernels (no tensor-core PTX, cp.async, clusters or PDL): hipify plus the GENERIC transforms only.
# (k0_nvfp4: NVIDIA's FP4 quantizer; its cvt / rcp PTX get software equivalents below, so K7's W4A4 MoE can run.)
PLAIN = ["kv_vision", "k13_moe_bf16", "k0_nvfp4", "k7_moe", "k3_hc", "k9_qsa", "k6_attention", "k4_nvfp4", "k2_gemv", "k4_ple", "k5_gdn", "k5_gdn_layer", "k5_gdn_decode", "k12_sampler", "k13s_spec_accept", "lora"]
# Headers replaced by a HIP stand-in instead of hipified (the CUDA one wraps NVIDIA-only libraries).
OVERRIDES = {"kernels/k2_gemm.h": Path("executor/hip/src/k2_gemm_adapter.h")}
for name in PLAIN:
    for ext in (".h", ".cuh", ".cu"):
        if (SRC / "kernels" / (name + ext)).exists():
            FILES.append(f"kernels/{name}{ext}")
# Mechanical transforms applied to every file, any number of times: HIP's *_sync warp shuffles take a 64-bit lane
# mask (all lanes of a 32-wide wave is all ones either way).
GENERIC = [
    (re.compile(r"(__(?:shfl(?:_xor|_down|_up)?|ballot|any|all|match_any|reduce_(?:max|min|add|and|or|xor))_sync)\(0x[fF]{8}[uU]?"), r"\1(0xffffffffffffffffull"),
    (re.compile(r"constexpr unsigned kFull = 0xffffffffu;"), "constexpr unsigned long long kFull = ~0ull;"),
    (re.compile(r"\b__ldg\("), "__cere_ldg("),
    (re.compile(r"\b__float2bfloat16\("), "__cere_float2bfloat16("),  # CUDA's canonical NaN (cuda_compat.h)
    (re.compile(r"\b__byte_perm\("), "__cere_byte_perm("),  # HIP's indexes a byte array (scratch); cuda_compat.h's is v_perm
    (re.compile(r"#include <hipblaslt\.h>"), "#include <hipblaslt/hipblaslt.h>"),  # hipify's cublasLt.h mapping
    # hipify-perl maps CUDA's __NV_E4M3 / __NV_E5M2 (OCP FP8) to HIP's FNUZ variants (bias 8 / 16: MI300's formats);
    # CUDA's are OCP, which HIP names __HIP_E4M3 / __HIP_E5M2 (probe on gfx1151: 0x38 -> 1.0, 0x7e -> 448).
    (re.compile(r"\b__HIP_E4M3_FNUZ\b"), "__HIP_E4M3"),
    (re.compile(r"\b__HIP_E5M2_FNUZ\b"), "__HIP_E5M2"),
    # ... and the FP8 types the same way (__nv_fp8_e4m3 is OCP; hipify gives the FNUZ type: K0's block scales)
    (re.compile(r"\b__hip_fp8_e4m3_fnuz\b"), "__hip_fp8_e4m3"),
    (re.compile(r"\b__hip_fp8_e5m2_fnuz\b"), "__hip_fp8_e5m2"),
    # never-contracted rounding intrinsics (CUDA's guarantee) -> cuda_compat.h's contract(off) versions
    (re.compile(r"\b__(fadd|fsub|fmul|dadd|dmul)_rn\("), r"__cere_\1_rn("),
    # PTX cache-hinted loads (ld.global.nc, L2 prefetch hint): plain loads of the same width.
    (re.compile(r'asm\("ld\.global\.nc\.L2::256B\.v4\.u32 \{%0,%1,%2,%3\}, \[%4\];" : "=r"\((\w+)\.x\), "=r"\(\1\.y\), "=r"\(\1\.z\), "=r"\(\1\.w\) : "l"\((\w+)\)\);'),
     r"\1 = *static_cast<const uint4 *>(\2);"),
    (re.compile(r'asm\("ld\.global\.nc\.L2::256B\.v2\.u32 \{%0,%1\}, \[%2\];" : "=r"\((\w+)\.x\), "=r"\(\1\.y\) : "l"\((\w+)\)\);'),
     r"\1 = *static_cast<const uint2 *>(\2);"),
    (re.compile(r'asm\("ld\.global\.nc\.L2::256B\.u32 %0, \[%1\];" : "=r"\((\w+)\) : "l"\((\w+)\)\);'),
     r"\1 = *static_cast<const unsigned int *>(\2);"),
    (re.compile(r'asm\("ld\.global\.nc\.L2::256B\.u16 %0, \[%1\];" : "=h"\((\w+)\) : "l"\((\w+)\)\);'),
     r"\1 = *static_cast<const unsigned short *>(\2);"),
    # PTX approximate math -> the gfx11 hardware approximations (v_exp_f32, v_log_f32, v_rsq_f32, v_sqrt_f32);
    # bits differ from NVIDIA's either way, so kernels using them are held to their FP64 tolerance.
    (re.compile(r'asm\("ex2\.approx(?:\.ftz)?\.f32 %0, %1;" : "=f"\((\w+)\) : "f"\((\w+)\)\);'), r"\1 = __builtin_amdgcn_exp2f(\2);"),
    (re.compile(r'asm\("lg2\.approx(?:\.ftz)?\.f32 %0, %1;" : "=f"\((\w+)\) : "f"\((\w+)\)\);'), r"\1 = __builtin_amdgcn_logf(\2);"),
    (re.compile(r'asm\("rsqrt\.approx(?:\.ftz)?\.f32 %0, %1;" : "=f"\((\w+)\) : "f"\((\w+)\)\);'), r"\1 = __builtin_amdgcn_rsqf(\2);"),
    (re.compile(r'asm\("sqrt\.approx(?:\.ftz)?\.f32 %0, %1;" : "=f"\((\w+)\) : "f"\((\w+)\)\);'), r"\1 = __builtin_amdgcn_sqrtf(\2);"),
    # div.full (2-ulp full-range division) -> IEEE division (correctly rounded: never less accurate)
    (re.compile(r'asm\("div\.full\.f32 %0, %1, %2;" : "=f"\((\w+)\) : "f"\((\w+)\), "f"\((\w+)\)\);'), r"\1 = \2 / \3;"),
    # fma.rn.f32.bf16 d, a, a, c with a a bf16 bit pattern: fmaf on the exact widening
    (re.compile(r'asm\("fma\.rn\.f32\.bf16 %0, %1, %1, %2;" : "=f"\((\w+)\) : "h"\(([^)]*)\), "f"\((.*)\)\);'),
     r"\1 = fmaf(__cere_bf16f(\2), __cere_bf16f(\2), \3);"),
]

PDL_PLAIN = '''template <typename... Params, typename... Args>
hipError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, hipStream_t stream,
                       Args... args) {
    // AmdAmd: HIP has no programmatic dependent launch; a plain launch computes the same (see pdl_enter).
    hipLaunchKernelGGL(kernel, grid, block, smem, stream, args...);
    return hipGetLastError();
}'''
PATCHES = [
    # K6 (dense attention): sm_89+ cvt.rn.f16x2.e4m3x2 -> an exact software OCP E4M3 decode (every E4M3 value is exact
    # in FP16 and FP32, so the floats are those of the hardware convert); cp.async (with src-size 0 zero fill) ->
    # plain 16-byte copies into shared memory, commit / wait dropped: each copy completes in program order and every
    # wait is already followed by the __syncthreads that publishes the tile.
    ("kernels/k6_attention.cu", re.compile(r"__device__ __forceinline__ float2 fp8x2_to_float2\(uint32_t two\) \{\n.*?\n\}\n", re.S),
     """__device__ __forceinline__ float cere_e4m3(uint32_t c) {  // AmdAmd: OCP E4M3 -> float, exact
    // OCP E4M3 -> float, exact and branch-free: the 7 magnitude bits at FP32 bit 20 are 2^(e-127)(1 + m/8)
    // (or the subnormal m 2^-129), times 2^120 is 2^(e-7)(1 + m/8) (m 2^-9); 0x7f is NaN; the sign is OR-ed in.
    const uint32_t a = c & 0x7fu;
    const float v = a == 0x7fu ? __uint_as_float(0x7fc00000u) : __uint_as_float(a << 20) * 0x1p120f;
    return __uint_as_float(__float_as_uint(v) | ((c & 0x80u) << 24));
}
__device__ __forceinline__ float2 fp8x2_to_float2(uint32_t two) {
    return make_float2(cere_e4m3(two & 0xffu), cere_e4m3((two >> 8) & 0xffu));
}
""", 1),
    ("kernels/k6_attention.cu", re.compile(r"    const uint32_t ks_addr = static_cast<uint32_t>\(__cvta_generic_to_shared\(ks\)\);\n    const uint32_t vs_addr = static_cast<uint32_t>\(__cvta_generic_to_shared\(vs\)\);\n"), "", 1),
    ("kernels/k6_attention.cu", re.compile(r"const uint32_t (k|v)dst = \1s_addr \+ static_cast<uint32_t>\("), r"uint8_t* const \1dst = \1s + (", 2),
    ("kernels/k6_attention.cu", re.compile(r'asm volatile\("cp\.async\.cg\.shared\.global \[%0\], \[%1\], 16, %2;\\n" ::"r"\((\w+) \+ 16 \* i\),\s*"l"\((\w) \+ off \+ 16 \* i\), "r"\(vsz\)\);'),
     r"*reinterpret_cast<uint4*>(\1 + 16 * i) = vsz ? *reinterpret_cast<const uint4*>(\2 + off + 16 * i) : make_uint4(0, 0, 0, 0);", 2),
    ("kernels/k6_attention.cu", re.compile(r'\s*asm volatile\("cp\.async\.commit_group;\\n" ::\);'), "", 2),
    ("kernels/k6_attention.cu", re.compile(r'        if constexpr \(kSmemK\) asm volatile\("cp\.async\.wait_group 1;\\n" ::\);\n'), "", 1),
    ("kernels/k6_attention.cu", re.compile(r'        asm volatile\("cp\.async\.wait_all;\\n" ::\);\n'), "", 1),
    # K6 on gfx1151's 64 KB of LDS per block (sm_120 opts in to 99 KB): the FP8 step blocks keep K in registers (the
    # kSmemK false variant the chains already run, same arithmetic); split calls of 4-16 rows run the 1-token blocks
    # (Smem<4> is 84 KB); and every prefill-class call (> 16 rows, or its cls) runs P4 whatever use_p4 says, since the
    # CUDA-core prefill kernel needs 83 KB: on HIP P4 is the only prefill path (executor/hip/src/p4_attention_hip.cpp).
    # K9 (QSA): CUDA's nvcuda::wmma -> rocWMMA (the same fragment API on gfx11; F32 accumulation, its own k order);
    # cp.async -> plain copies (as K6); the thread-block-cluster top-k is dropped and its calls run topk_kernel (the
    # header: every top-k kernel writes the same bytes).
    ("kernels/k9_qsa.cu", re.compile(r"#include <mma\.h>\n"), """#include <rocwmma/rocwmma.hpp>
namespace nvcuda::wmma {  // AmdAmd: CUDA's wmma names on rocWMMA, BF16 pointers as rocWMMA's bfloat16_t
using namespace rocwmma;
template <class F>
__device__ inline void load_matrix_sync(F& f, const __hip_bfloat16* p, uint32_t ld) {
    rocwmma::load_matrix_sync(f, reinterpret_cast<const rocwmma::bfloat16_t*>(p), ld);
}
}  // namespace nvcuda::wmma
""", 1),
    ("kernels/k9_qsa.cu", re.compile(r"16, 16, 16, __hip_bfloat16, wmma::"), "16, 16, 16, rocwmma::bfloat16_t, wmma::", 6),
    ("kernels/k9_qsa.cu", re.compile(r"__device__ __forceinline__ void cp16\(void\* dst, const void\* src, bool valid\) \{\n.*?\n\}\n__device__ __forceinline__ void cp_commit\(\) \{.*?\}\n(// .*?\n)?__device__ __forceinline__ void cp_wait0\(\) \{.*?\}\n", re.S),
     """__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {  // AmdAmd: a plain copy
    *static_cast<uint4*>(dst) = valid ? *static_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
}
__device__ __forceinline__ void cp_commit() {}
__device__ __forceinline__ void cp_wait0() {}
""", 1),
    ("kernels/k9_qsa.cu", re.compile(r"__global__ void __cluster_dims__\(kTkCluster, 1, 1\) __launch_bounds__\(kTopkThreads\)\n    topk_cluster_kernel\(.*?\n\}\n", re.S), "", 1),
    ("kernels/k9_qsa.cu", re.compile(r"return ld <= kTkMaxBlocks \? TopkKernel::cluster : TopkKernel::block;"), "return TopkKernel::block;", 1),
    ("kernels/k9_qsa.cu", re.compile(r"    if \(kernel == TopkKernel::cluster\) \{\n        topk_cluster_kernel<<<.*?\n    \} else if", re.S), "    if (kernel == TopkKernel::cluster) {\n        return hipErrorNotSupported;\n    } else if", 1),
    ("kernels/k9_qsa.cu", re.compile(r"    const uint32_t (k|v)_addr = static_cast<uint32_t>\(__cvta_generic_to_shared\(&sm\.(kp\.k|v)\[0\]\[0\]\)\);\n"), r"    uint8_t* const \1_addr = &sm.\2[0][0];\n", 2),
    ("kernels/k9_qsa.cu", re.compile(r'asm volatile\("cp\.async\.cg\.shared\.global \[%0\], \[%1\], 16, %2;\\n" ::"r"\(((?:k|v)_addr \+ j \* \w+ \+ c)\), "l"\(src\),\s*"r"\(row >= 0 \? 16 : 0\)\);'),
     r"*reinterpret_cast<uint4*>(\1) = row >= 0 ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);", 2),
    ("kernels/k9_qsa.cu", re.compile(r'    asm volatile\("cp\.async\.(commit_group|wait_group [01]);\\n" ::\);\n'), "", 4),
    # K9 split attention on 64 KB of LDS: 32 splits of at most 3 key tiles (96 keys; ceil(2,051 / 32) = 65) instead of
    # 16 of 5 (SplitSmem 96 KB -> 63.9 KB); the kernel's static_asserts check the coupled constants.
    ("kernels/k9_qsa.cu", re.compile(r"constexpr int kSplits = 16;"), "constexpr int kSplits = 32;  // AmdAmd: 16 on sm_120", 1),
    ("kernels/k9_qsa.cu", re.compile(r"constexpr int kSaTiles = 5;"), "constexpr int kSaTiles = 3;  // AmdAmd: 5 on sm_120", 1),
    ("kernels/k9_qsa.cu", re.compile(r'static_assert\(sizeof\(SplitSmem\) <= 99 \* 1024, "split_attention_kernel: sm_120'), 'static_assert(sizeof(SplitSmem) <= 64 * 1024, "split_attention_kernel: gfx1151', 1),
    # K9 prefill-class calls run the HIP P4 (WMMA sparse GQA) whatever use_p4 says, as K6's: its CUDA-core rows kernel
    # runs at 0.5 TFLOP/s here (4,096 rows x 2,051 keys: 422 ms vs 67 ms on P4, k9 timing in amd-amd design note).
    # K6 FP8 decode: on gfx1151 the split rows kernels without shared-memory K spill 1.5-5.7 KB per lane (the K row in
    # registers); a one-token FP8 call that is not a verify chain runs the per-KV-head tile kernel instead (Smem<1>
    # fits), whose arithmetic the rows kernels reproduce bit for bit (k6_attention.cu), so every output is unchanged
    # (k6_test checks dense_attention_rows' rows against these one-token calls). Chains keep the rows kernel.
    ("kernels/k6_attention.cu", re.compile(r"    if constexpr \(kTokens == 1 && std::is_same_v<KvT, uint8_t>\) \{\n        // FP8 cache rows"),
     "    if constexpr (kTokens == 1 && std::is_same_v<KvT, uint8_t> && kChain) {  // AmdAmd: non-chain -> tile kernel\n        // FP8 cache rows", 1),
    ("kernels/k6_attention.cu", re.compile(r"tokens >= 4 \? launch<4, uint8_t, true>"), "false ? launch<4, uint8_t, true>", 1),  # AmdAmd: Smem<4> is 84 KB
    # A verify chain on HIP is its rows' one-token calls, launch by launch: row t == dense_attention at prefix pos0 + t
    # (host positions) or the step call at positions[t] (device positions), as the contract states, so every output
    # bit is the one-token call's by construction; the spilling rows kernel is not used (8 rows: 3.7 -> see k6_test).
    ("kernels/k6_attention.cu", re.compile(r"(hipError_t attention_chain_impl\(const bf16\* q, int64_t tokens, int64_t pos0, const int64_t\* pos,\n.*?\) \{\n).*?\n\}\n", re.S),
     r"""\1    for (int64_t t = 0; t < tokens; ++t) {  // AmdAmd: the rows' one-token calls
        const hipError_t e =
            pos == nullptr
                ? attention_impl(q + t * kQWidth, 1, pos0 + t, k_cache, int64_t{kKvWidth}, v_cache, int64_t{kKvWidth},
                                 slots, scale, workspace, workspace_bytes, out + t * kQWidth, stream, 1)
                : attention_step_impl(q + t * kQWidth, 1, k_cache, v_cache, slots, pos + t, scale, workspace,
                                      workspace_bytes, out + t * kQWidth, stream);
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}
""", 1),
    ("kernels/k9_qsa.cu", re.compile(r"    if \(use_p4 && sel >= p4::kMinRows && q_stride % 8 == 0"), "    if (sel >= p4::kMinRows && q_stride % 8 == 0  /* AmdAmd: P4 is HIP's prefill path */", 1),
    # K3 (hyper-connections): programmatic dependent launch -> none (each PDL wait / trigger a no-op; stream order
    # stands), and hc_combine_norm's one-launch cluster kernel dropped: the call takes its documented two-launch path
    # (hc_combine, then the K1 per-stream norm), bit-identical by its contract.
    ("kernels/k3_hc.cu", re.compile(r'__device__ __forceinline__ void (launch_dependents|dependency_wait)\(\) \{ asm volatile\("griddepcontrol\.\w+;" ::: "memory"\); \}'),
     r"__device__ __forceinline__ void \1() {}", 2),
    ("kernels/k3_hc.cu", re.compile(r"__global__ void __cluster_dims__\(1, kHc, 1\) __launch_bounds__\(kThreads\).*?\n\}\n", re.S), "", 1),
    ("kernels/k3_hc.cu", re.compile(r"    const bool fused = m <= kCombineSplitMaxM &&"), "    const bool fused = false && m <= kCombineSplitMaxM &&  // AmdAmd: no clusters\n       ", 1),
    ("kernels/k3_hc.cu", re.compile(r"    hipLaunchAttribute attr\{\};.*?return e == hipSuccess && hipGetLastError\(\) == hipSuccess \? CERE_OK : CERE_ERR_INTERNAL;", re.S),
     "    (void)stage;\n    return CERE_ERR_INTERNAL;  // AmdAmd: unreachable (fused is false)", 1),
    # K0 (NVFP4 helpers): rcp.approx.ftz.f32 -> gfx11's v_rcp_f32 (also an approximate reciprocal: it may differ from
    # NVIDIA's in the last place, so a block scale can round to a neighbouring FP8 code; HIP is deterministic run to
    # run); cvt.rn.satfinite.e2m1x2.f32 -> an exact software round-to-nearest-even onto E2M1 {0, .5, 1, 1.5, 2, 3, 4,
    # 6}, saturating at 6, the sign kept (so -0 for small negatives), NaN -> +6.
    ("kernels/k0_nvfp4.cuh", re.compile(r'    float b;\n    asm volatile\("rcp\.approx\.ftz\.f32 %0, %1;\\n" : "=f"\(b\) : "f"\(a\)\);\n    return b;'),
     "    return __builtin_amdgcn_rcpf(a);  // AmdAmd: v_rcp_f32", 1),
    ("kernels/k0_nvfp4.cuh", re.compile(r"__device__ __forceinline__ uint32_t e2m1x8\(const float \(&a\)\[8\]\) \{\n.*?\n\}\n", re.S),
     """__device__ __forceinline__ uint32_t e2m1_rn_satfinite(float x) {  // AmdAmd: cvt.rn.satfinite.e2m1 in software
    const uint32_t sign = (__float_as_uint(x) >> 28) & 8u;
    const float a = fabsf(x);
    uint32_t c;
    if (!(a == a)) c = 7u;  // NaN
    else if (a <= 0.25f) c = 0u;  // ties to even (code 0)
    else if (a < 0.75f) c = 1u;
    else if (a <= 1.25f) c = 2u;
    else if (a < 1.75f) c = 3u;
    else if (a <= 2.5f) c = 4u;
    else if (a < 3.5f) c = 5u;
    else if (a <= 5.0f) c = 6u;
    else c = 7u;  // satfinite
    return sign | c;
}
__device__ __forceinline__ uint32_t e2m1x8(const float (&a)[8]) {  // element 2j in the low nibble of byte j
    uint32_t val = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) val |= e2m1_rn_satfinite(a[i]) << (4 * i);
    return val;
}
""", 1),
    # K0's e4m3_to_float through HIP's FP8 type is a long software conversion (VGPR pressure, scratch in K7's GEMM);
    # the exact branch-free decode K6 / P4 use instead (every E4M3 value exact in FP32; 0x7f NaN).
    ("kernels/k0_nvfp4.cuh", re.compile(r"__device__ __forceinline__ float e4m3_to_float\(uint8_t bits\) \{\n.*?\n\}\n", re.S),
     """__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {  // AmdAmd: exact, branch-free
    const uint32_t c = bits, a = c & 0x7fu;
    const float v = a == 0x7fu ? __uint_as_float(0x7fc00000u) : __uint_as_float(a << 20) * 0x1p120f;
    return __uint_as_float(__float_as_uint(v) | ((c & 0x80u) << 24));
}
""", 1),
    # K7 (NVFP4 MoE): PDL -> plain launches (k1's PDL_PLAIN arity: no smem argument here), fma.rn.ftz -> fmaf (equal
    # for every non-denormal operand and result).
    ("kernels/k7_moe.cu", re.compile(r'    asm volatile\("griddepcontrol\.launch_dependents;" ::: "memory"\);\n    asm volatile\("griddepcontrol\.wait;" ::: "memory"\);\n'), "", 1),
    ("kernels/k7_moe.cu", re.compile(r"template <class\.\.\. Params, class\.\.\. Args>\nhipError_t launch_pdl\(.*?\n\}", re.S),
     """template <class... Params, class... Args>
hipError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, hipStream_t stream, Args... args) {
    hipLaunchKernelGGL(kernel, grid, block, 0, stream, args...);  // AmdAmd: no PDL on HIP; a plain launch
    return hipGetLastError();
}""", 1),
    ("kernels/k7_moe.cu", re.compile(r'    float d;\n    asm\("fma\.rn\.ftz\.f32 %0, %1, %2, %3;" : "=f"\(d\) : "f"\(a\), "f"\(b\), "f"\(c\)\);\n    return d;'),
     "    return fmaf(a, b, c);  // AmdAmd", 1),
    # K13 (BF16 MoE of the MTP draft; NVFP4 small tail): PDL waits / triggers dropped and launch_dependent a plain
    # launch; cvt.rn.f16x2.e2m1x2 -> the exact FP16 encodings of the 8 E2M1 magnitudes (low nibble -> low half).
    ("kernels/k13_moe_bf16.cu", re.compile(r'\s*asm volatile\("griddepcontrol\.(?:wait|launch_dependents);" ::: "memory"\);'), "", 4),
    ("kernels/k13_moe_bf16.cu", re.compile(r'    asm\("\{\\n"\n        "\.reg \.b8 b0, b1, b2, b3;\\n"\n.*?: "r"\(codes\)\);\n', re.S),
     """    {  // AmdAmd: cvt.rn.f16x2.e2m1x2 in software (every E2M1 value is exact in FP16)
        constexpr uint32_t kHalf[8] = {0x0000u, 0x3800u, 0x3C00u, 0x3E00u, 0x4000u, 0x4200u, 0x4400u, 0x4600u};
        auto h16 = [&](uint32_t c) { return kHalf[c & 7u] | ((c & 8u) << 12); };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = (codes >> (8 * i)) & 0xFFu;
            h[i] = h16(b & 0xFu) | (h16(b >> 4) << 16);
        }
    }
""", 1),
    ("kernels/k13_moe_bf16.cu", re.compile(r"    hipLaunchAttribute attr\{\};\n    attr\.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n.*?return cudaLaunchKernelEx\(&config, kernel, g, f\);", re.S),
     "    hipLaunchKernelGGL(kernel, base.gridDim, base.blockDim, base.dynamicSmemBytes, base.stream, g, f);  // AmdAmd: no PDL\n    return hipGetLastError();", 1),
    # The device: CUDA_MODULE_LOADING=EAGER is required (no first-launch module loads inside graph captures or timed
    # steps); HIP's counterpart is HIP_ENABLE_DEFERRED_LOADING (default 1: deferred), so the check requires it set to 0.
    ("device/cuda_device.cpp", re.compile(r"#include <cudaTypedefs\.h>\n"), "", 1),
    ("device/cuda_device.cpp", re.compile(r"cere_status require_eager_loading\(\) \{\n.*?\n\}\n", re.S),
     """cere_status require_eager_loading() {  // AmdAmd: HIP_ENABLE_DEFERRED_LOADING=0 is HIP's eager code-object loading
    const char *v = std::getenv("HIP_ENABLE_DEFERRED_LOADING");
    return v != nullptr && std::strcmp(v, "0") == 0 ? CERE_OK : CERE_ERR_UNSUPPORTED;
}
""", 1),
    # The forward's graphs: HIP's hipGraphExecUpdate (error node + result enum) and hipGraphInstantiateWithFlags for
    # CUDA 12's result-info update and flags instantiate.
    ("model/forward.cpp", re.compile(r"            cudaGraphExecUpdateResultInfo info\{\};\n            updated = hipGraphExecUpdate\(g\.exec\[i\], gs\.closed\[i\], &info\) == hipSuccess;"),
     "            hipGraphNode_t error_node = nullptr;  // AmdAmd: HIP's signature\n            hipGraphExecUpdateResult info{};\n            updated = hipGraphExecUpdate(g.exec[i], gs.closed[i], &error_node, &info) == hipSuccess;", 1),
    ("model/forward.cpp", re.compile(r"made = hipGraphInstantiate\(&g\.exec\[i\], gs\.closed\[i\], 0\);"), "made = hipGraphInstantiateWithFlags(&g.exec[i], gs.closed[i], 0);", 1),
    # The device: CUDA's sticky device-fault codes HIP does not have (its faults arrive as the codes kept in the list);
    # the read-only host-register support from hipDeviceProp_t (no hipDeviceAttribute for it).
    ("device/cuda_device.cpp", re.compile(r"    case cudaError(?:HardwareStackError|IllegalInstruction|MisalignedAddress|InvalidAddressSpace|InvalidPc):\n"), "", 5),
    ("device/cuda_device.cpp", re.compile(r"hipDeviceGetAttribute\(&read_only, cudaDevAttrHostRegisterReadOnlySupported, ordinal_\)"),
     "[&] { hipDeviceProp_t p{}; const hipError_t r = hipGetDeviceProperties(&p, ordinal_); read_only = p.hostRegisterReadOnlySupported; return r; }()", 1),
    # Open: gfx1151 instead of sm_120; unifiedAddressing is reported 0 by HIP on this APU although ROCm's address space
    # is one flat space (pinned host memory is device-accessible at its host pointer: canUseHostPointerForRegisteredMem
    # 1, which allocate_shared checks where it matters), so the open check asks for that instead.
    ("device/cuda_device.cpp", re.compile(r"if \(prop\.major != 12 \|\| prop\.minor != 0 \|\| !prop\.canMapHostMemory \|\| !prop\.unifiedAddressing\)"),
     'int host_ptr = 0;  // AmdAmd\n    (void)hipDeviceGetAttribute(&host_ptr, hipDeviceAttributeCanUseHostPointerForRegisteredMem, ordinal);\n'
     '    if (std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0 || !prop.canMapHostMemory || host_ptr == 0)', 1),
    # K7: tiles of <= 8 rows (m <= 8) run k7_small_gemv.inc's GEMV form, larger calls k7_seg_gemm.inc's per-expert
    # int8 WMMA form; both are grouped_gemm_kernel's arithmetic operation for operation (the same bits; k7_test
    # checks across kernels). grouped_gemm_kernel stays compiled as the reference.
    ("kernels/k7_moe.cu", re.compile(r"\n// -- shared expert and combine -"), "\n#include \"k7_small_gemv.inc\"  // AmdAmd\n#include \"k7_seg_gemm.inc\"  // AmdAmd\n\n// -- shared expert and combine -", 1),
    ("kernels/k7_moe.cu", re.compile(r"( +)grouped_gemm_kernel<(true|false)><<<dim3\(\((d\.\w+) \+ kTileCols - 1\) / kTileCols, (static_cast<unsigned>\(tiles\))\), kGemmThreads, 0, stream>>>\((g\d)\);"),
     r"""\1if (m <= static_cast<uint32_t>(kGemvMaxRows) && \5.k <= kGemvMaxK)  // AmdAmd
\1    launch_grouped_gemv<\2>(dim3((\3 + kGemvThreads / 8 - 1) / (kGemvThreads / 8), \4), \5, m, stream);
\1else  // per expert segment on int8 WMMA (k7_seg_gemm.inc), grouped_gemm_kernel's bits
\1    segment_gemm_kernel<\2><<<dim3((\3 + kTileCols - 1) / kTileCols, d.num_experts), kSegThreads, 0, stream>>>(\5);""", 2),
    # K7 grouping of small calls in one block (k7_route_small.inc: the four launches' integers, ~17 us -> ~3 at m 1).
    ("kernels/k7_moe.cu", re.compile(r"\n// -- W4A4 grouped GEMM -"), "\n#include \"k7_route_small.inc\"  // AmdAmd\n\n// -- W4A4 grouped GEMM -", 1),
    ("kernels/k7_moe.cu", re.compile(r"( +)(if \(hipMemsetAsync\(w\.counts, 0, .*?place_kernel<<<d\.num_experts, kSortThreads, 0, stream>>>\(w\.topk_ids, w\.expert_offset, w\.perm, w\.pos, pairs\);)", re.S),
     r"""\1if (pairs <= kRouteSmallPairs && d.num_experts <= kMaxExperts) {  // AmdAmd: one block (k7_route_small.inc)
\1    route_small_kernel<<<1, kSortThreads, 0, stream>>>(w.topk_ids, w.counts, w.expert_offset, w.tile_offset, w.perm,
\1                                                        w.pos, static_cast<int>(pairs), static_cast<int>(d.num_experts));
\1} else {
\1\2
\1}""", 1),
    # P4 sparse attention on the FP8 cache decodes the call's K / V once into qsa_attention's dead scratch
    # (p4_attention_hip.cpp step 16): the declaration gains defaulted scratch arguments, K9 passes its own.
    ("kernels/p4_attention.h", re.compile(r"(const int64_t\* slots, float scale, bf16\* out, int64_t out_stride,\s*hipStream_t stream)\);"),
     r"\1, void* scratch = nullptr, size_t scratch_bytes = 0);  // AmdAmd", 1),
    ("kernels/k9_qsa.cu", re.compile(r"(p4::sparse_attention\(q, q_stride, indices, kRowWidth, rows, k, k_stride, v, v_stride, k_cache,\s*v_cache, slots, scale, out, out_stride, stream)\)"),
     r"\1, scratch, scratch_bytes)", 1),
    ("kernels/k7_moe.cu", re.compile(r"\nbool launched\(\) \{"), "\n#include \"k7_combine_split.inc\"  // AmdAmd\n\nbool launched() {", 1),
    ("kernels/k7_moe.cu", re.compile(r"combine_kernel<Routed::kSum><<<m, kCombineThreads, 0, stream>>>\("),
     "combine_gate_kernel<<<m, kCombineThreads, 0, stream>>>(x, L.shared_expert_gate, reinterpret_cast<float *>(w.x_codes), hidden);  // AmdAmd: x_codes is dead after GEMM1\n"
     "        combine_split_kernel<<<dim3(m, (d.hidden + kSplitElems - 1) / kSplitElems), kCombineThreads, 0, stream>>>(reinterpret_cast<const float *>(w.x_codes), ", 1),
    ("kernels/k7_moe.cu", re.compile(r"(combine_split_kernel<<<.*?\(reinterpret_cast<const float \*>\(w\.x_codes\), )\n\s*x, L\.shared_expert_gate, ", re.S), r"\1\n            ", 1),
    ("kernels/k6_attention.cu", re.compile(r"launch_rows<kStepRows, true, kStep, kChain>"), "launch_rows<kStepRows, false, kStep, kChain>", 1),
    ("kernels/k6_attention.cu", re.compile(r"cudaError_t err = sel >= 4 \? launch<4>|hipError_t err = sel >= 4 \? launch<4>"),
     "hipError_t err = false ? launch<4>", 1),
    ("kernels/k6_attention.cu", re.compile(r"    if \(use_p4 && sel >= p4::kMinRows && aligned16\(out\)\)\n"), "    if (sel >= p4::kMinRows)  // AmdAmd: P4 is HIP's prefill path\n", 1),
    ("kernels/k1_primitives.cu", re.compile(r'    asm volatile\("griddepcontrol\.(launch_dependents|wait);" ::: "memory"\);\n'),
     "", 2),
    ("kernels/k1_primitives.cu", re.compile(r"template <typename\.\.\. Params, typename\.\.\. Args>\nhipError_t launch_pdl\(.*?\n}", re.S),
     PDL_PLAIN, 1),
    # Thread-block clusters (Hopper/Blackwell) have no RDNA counterpart. top2's result is a function of the
    # candidate set, not of the split or merge order (see the kernel's comment), so one block per row computes the
    # same answer: no cluster attribute, the block's own sb[0]/ss[0] is the row's top-2, grid x = 1.
    ("kernels/k1_primitives.cu", re.compile(r"__global__ void __cluster_dims__\(kTop2Cluster, 1, 1\) __launch_bounds__"),
     "__global__ void __launch_bounds__", 1),
    ("kernels/k1_primitives.cu", re.compile(
        r"    cg::cluster_group cl = cg::this_cluster\(\);\n.*?\n    cl\.sync\(\);  // no rank exits while rank 0 reads its shared memory\n", re.S),
     """    __syncthreads();  // AmdAmd: one block per row, no cluster merge
    if (warp == 0 && lane < 2) {
        const uint64_t p = lane == 0 ? sb[0] : ss[0];
        const uint32_t i = p == 0 ? kNone : ~static_cast<uint32_t>(p);
        const int64_t o = r * out_stride + lane;
        ids[o] = i;
        scores[o] = p == 0 ? -INFINITY : __bfloat162float(row[i]);
    }
""", 1),
    ("kernels/k1_primitives.cu", re.compile(r"dim3 top2_grid\(int64_t rows\) \{ return dim3\(kTop2Cluster, static_cast<unsigned>\(rows\)\); \}"),
     "dim3 top2_grid(int64_t rows) { return dim3(1, static_cast<unsigned>(rows)); }", 1),
]


def main() -> int:
    for rel, adapter in OVERRIDES.items():
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(adapter.read_text())
        print("installed", dst, "from", adapter)
    for rel in FILES:
        src = SRC / rel
        text = subprocess.run(["hipify-perl", str(src)], capture_output=True, text=True, check=True).stdout
        for pat, new in GENERIC:
            text = pat.sub(new, text)
        for f, pat, new, count in PATCHES:
            if f != rel:
                continue
            text, n = pat.subn((lambda _m: new) if '\\' not in new else new, text)
            if n != count:
                raise SystemExit(f"{rel}: patch {pat.pattern[:50]!r} applied {n} times, expected {count}")
        dst = OUT / (rel[:-3] + ".hip" if rel.endswith(".cu") else rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print("generated", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
