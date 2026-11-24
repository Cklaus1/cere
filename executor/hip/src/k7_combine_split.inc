// AmdAmd PAR-6 step 11: combine_kernel<Routed::kSum> split over the hidden row (included by gen_hip.py into the
// hipified k7_moe.hip after combine_kernel). combine_kernel runs one block per token, so a decode token's combine
// is one 128-thread block walking hidden x top_k loads (44 us at m 1 on gfx1151). Here block (t, c) forms elements
// [c * kSplitElems, (c + 1) * kSplitElems) of token t, 8 consecutive elements per thread with 16-byte expert-row
// loads: the token's gate from combine_gate_kernel (combine_gate, computed once per token) and, per
// element, combine_kernel's expressions (acc += w_k * y_k in slot order, the BF16 rounding, + gate * shared), so
// every output bit equals combine_kernel's.
constexpr int kSplitElems = kCombineThreads * 8;

// The token's gate sigmoid(x . g) once per token (combine_gate, its fixed reduction order) into gates[t], so the
// split blocks of a token read it instead of each recomputing the 2,560-element dot (a third of a prefill combine).
__global__ void __launch_bounds__(kCombineThreads)
    combine_gate_kernel(const __hip_bfloat16 *__restrict__ x, const __hip_bfloat16 *__restrict__ gate_weight,
                        float *__restrict__ gates, int hidden) {
    __shared__ float scratch[kCombineThreads / 32];
    const float gate = combine_gate(x, gate_weight, blockIdx.x, hidden, scratch);
    if (threadIdx.x == 0) gates[blockIdx.x] = gate;
}

__global__ void __launch_bounds__(kCombineThreads)
    combine_split_kernel(const float *__restrict__ gates,
                         const __hip_bfloat16 *__restrict__ y, const int32_t *__restrict__ pos,
                         const float *__restrict__ topk_weights, const __hip_bfloat16 *__restrict__ shared,
                         __hip_bfloat16 *__restrict__ out, __hip_bfloat16 *__restrict__ routed_out, int hidden, int top_k) {
    const int64_t t = blockIdx.x;
    const float gate = gates[t];  // combine_gate_kernel's value: the same bits as combine_kernel's gate
    const int j0 = static_cast<int>(blockIdx.y) * kSplitElems + static_cast<int>(threadIdx.x) * 8;  // 8 elements
    if (j0 >= hidden) return;
    const bool vec = (hidden % 8) == 0 && j0 + 8 <= hidden && (reinterpret_cast<uintptr_t>(y) & 15) == 0 &&
                     (reinterpret_cast<uintptr_t>(shared) & 15) == 0 && (reinterpret_cast<uintptr_t>(out) & 15) == 0;
    float acc[8] = {};
    for (int k = 0; k < top_k; ++k) {  // slot order, as combine_kernel
        const int64_t r = pos[t * top_k + k];
        const float wk = topk_weights[t * top_k + k];
        __hip_bfloat16 v[8];
        if (vec) {
            *reinterpret_cast<uint4 *>(v) = *reinterpret_cast<const uint4 *>(y + r * hidden + j0);
        } else {
            for (int q = 0; q < 8; ++q) v[q] = j0 + q < hidden ? y[r * hidden + j0 + q] : __cere_float2bfloat16(0.0f);
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] += wk * __bfloat162float(v[q]);
    }
    for (int q = 0; q < 8 && j0 + q < hidden; ++q) {
        const int j = j0 + q;
        const __hip_bfloat16 routed_bf16 = __cere_float2bfloat16(acc[q]);
        if (routed_out != nullptr) routed_out[t * hidden + j] = routed_bf16;
        const float routed = __bfloat162float(routed_bf16);
        out[t * hidden + j] = __cere_float2bfloat16(routed + gate * __bfloat162float(shared[t * hidden + j]));
    }
}
