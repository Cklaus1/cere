// AmdAmd PAR-6 step 11: a small-m form of K7's W4A4 grouped GEMM (included by gen_hip.py into the hipified k7_moe.hip
// after grouped_gemm_kernel; it uses that file's GemmArgs, tile map, kTileRows and the K0 helpers). moe_block runs it
// for m <= kGemvMaxRows (every expert tile then has at most that many rows): decode and short verify steps.
// grouped_gemm_kernel's 128 threads are 64 columns x 2 groups of 16 rows, so a 1-row tile leaves half the threads
// loading weights for nothing and the other half looping over 16 row slots. Here the tile's <= 8 A rows are staged
// once in LDS for the whole k (int8 2 * e2m1 codes, float block scales: grouped_gemm_kernel's staged values) and
// 8 lanes serve one column (weight row): per step of 8 NVFP4 blocks lane j forms block j's exact integer dot (its 8
// code bytes contiguous, so a column's 8 lanes read 64 B together) and the scale product, and every lane of the
// group then runs the same FP32 chain over the 8 blocks in k order with the values shuffled in. Each block's isum is
// an integer (any summation order gives it exactly); the chain is grouped_gemm_kernel's per output element,
// acc += float(isum) * (a_scale * w_scale) block after block, then the 0.25 * alpha BF16 epilogue (SwiGLU for GEMM1),
// so the outputs are bit-identical to that kernel's (k7_test checks).
constexpr int kGemvThreads = 256;
constexpr int kGemvMaxRows = 8;
constexpr int kGemvMaxK = 2560;  // hidden (GEMM1's k); GEMM2's k is the expert intermediate (640)

// 8 E2M1 codes (code i in bits 4 i .. 4 i + 3 of w) -> 8 int8 2 * e2m1 (code i in byte i of {lo, hi}), the values
// of nvfp4::e2m1x4_to_s8x4 (code 8 -> 0) with byte selectors v_perm takes directly: even / odd codes split into bytes,
// two table lookups (positive, negative) per half, the sign picks, two perms interleave. ~20 ops for 8 codes.
__device__ __forceinline__ void k7_e2m1x8_to_s8(uint32_t w, uint32_t &lo, uint32_t &hi) {
    const uint32_t ev = w & 0x0F0F0F0Fu, od = (w >> 4) & 0x0F0F0F0Fu;  // codes 0, 2, 4, 6 / 1, 3, 5, 7 in bytes 0..3
    auto lut = [](uint32_t c) {
        const uint32_t idx = c & 0x07070707u;
        const uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, idx);  // {0, 1, 2, 3, 4, 6, 8, 12}
        const uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, idx);  // negated (code 8 -> 0)
        const uint32_t m = ((c >> 3) & 0x01010101u) * 0xFFu;
        return (pos & ~m) | (neg & m);
    };
    const uint32_t e = lut(ev), o = lut(od);
    lo = __builtin_amdgcn_perm(o, e, 0x05010400u);  // {e0, o0, e1, o1}: codes 0 1 2 3
    hi = __builtin_amdgcn_perm(o, e, 0x07030602u);  // {e2, o2, e3, o3}: codes 4 5 6 7
}

__device__ __forceinline__ int k7_dot16(const uint32_t (&w)[4], const uint32_t *a) {  // exact integer block dot
    int isum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) isum = __dp4a(static_cast<int>(w[j]), static_cast<int>(a[j]), isum);
    return isum;
}

// kMR: the rows a tile can hold (min(m, kGemvMaxRows) rounded up to 1, 4 or 8): the accumulators of rows that
// cannot exist are not allocated (GEMM1 at kMR 8 needed 256 VGPRs and spilled; decode is m 1).
template <bool kGlu, int kMR>
__global__ void __launch_bounds__(kGemvThreads) grouped_gemv_kernel(GemmArgs g) {
    constexpr int kLanes = 8;                          // lanes per column = blocks per step
    constexpr int kCols = kGemvThreads / kLanes;       // columns per block (32)
    __shared__ __align__(16) uint32_t a_s8[kGemvMaxRows][kGemvMaxK / 4];
    __shared__ float a_sf[kGemvMaxRows][kGemvMaxK / 16];
    __shared__ int64_t a_row[kGemvMaxRows];
    __shared__ int32_t t_off[kMaxExperts + 1];
    const int tile = blockIdx.y;
    // the tile's expert: the tile offsets loaded into LDS in one round, then the binary search there (in global
    // memory its ~9 dependent loads were most of a decode GEMM's time on gfx1151's ~1 us load latency)
    for (int i = threadIdx.x; i <= g.num_experts; i += kGemvThreads) t_off[i] = g.tile_offset[i];
    __syncthreads();
    if (tile >= t_off[g.num_experts]) return;  // block-uniform
    int lo = 0, hi = g.num_experts;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (t_off[mid] <= tile) lo = mid; else hi = mid;
    }
    const int e = lo;
    const int32_t row0 = g.expert_offset[e] + (tile - t_off[e]) * kTileRows;
    const int rows = min(kMR, g.expert_offset[e + 1] - row0);  // host: <= kMR rows per tile
    const int j = threadIdx.x % kLanes;                 // this lane's block within a step
    const int gl = (threadIdx.x % 32) / kLanes * kLanes;  // the group's first lane in the wave
    const int col = blockIdx.x * kCols + threadIdx.x / kLanes;
    const bool ok = col < g.n;                          // a group is uniform; groups may differ within a wave
    const int64_t w_row = static_cast<int64_t>(e) * g.n + (ok ? col : 0);
    const uint2 *w0 = reinterpret_cast<const uint2 *>(g.w0 + w_row * (g.k / 2));  // block b's 8 code bytes: w0[b]
    const uint2 *w1 = kGlu ? reinterpret_cast<const uint2 *>(g.w1 + w_row * (g.k / 2)) : nullptr;
    const uint8_t *s0 = g.s0 + w_row * (g.k / 16);
    const uint8_t *s1 = kGlu ? g.s1 + w_row * (g.k / 16) : nullptr;
    // kD steps of weights in flight (a rolling prefetch: slot i reloads right after step i is consumed), the first
    // ones issued before the A staging they do not depend on (GEMM2's k 640 is 5 steps: all of them)
    constexpr int kD = 5;
    const int blocks = g.k / 16;
    uint2 q0[kD], q1[kD];
    uint8_t sc0[kD], sc1[kD];
#pragma unroll
    for (int i = 0; i < kD; ++i) {
        const int blk = i * kLanes + j;
        if (i * kLanes < blocks) {
            q0[i] = w0[blk]; sc0[i] = s0[blk];
            if constexpr (kGlu) { q1[i] = w1[blk]; sc1[i] = s1[blk]; }
        }
    }
    if (threadIdx.x < kGemvMaxRows) {
        const int r = threadIdx.x;
        a_row[r] = r < rows ? (g.a_perm != nullptr ? g.a_perm[row0 + r] / g.a_div : row0 + r) : -1;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < rows * (g.k / 8); idx += kGemvThreads) {
        const int r = idx / (g.k / 8), wd = idx % (g.k / 8);
        const uint32_t v = *reinterpret_cast<const uint32_t *>(g.a_codes + a_row[r] * (g.k / 2) + wd * 4);
        a_s8[r][2 * wd] = nvfp4::e2m1x4_to_s8x4(v & 0xFFFFu);
        a_s8[r][2 * wd + 1] = nvfp4::e2m1x4_to_s8x4(v >> 16);
    }
    for (int idx = threadIdx.x; idx < rows * (g.k / 16); idx += kGemvThreads) {
        const int r = idx / (g.k / 16), b = idx % (g.k / 16);
        a_sf[r][b] = nvfp4::e4m3_to_float(g.a_scales[a_row[r] * (g.k / 16) + b]);
    }
    __syncthreads();
    float acc0[kMR], acc1[kMR];
#pragma unroll
    for (int r = 0; r < kMR; ++r) acc0[r] = acc1[r] = 0.0f;
    for (int b00 = 0; b00 < blocks; b00 += kD * kLanes) {
#pragma unroll
        for (int i = 0; i < kD; ++i) {
            const int b0 = b00 + i * kLanes;
            if (b0 >= blocks) continue;  // uniform (continue keeps the slot loop unrollable)
            const int blk = b0 + j;
            uint32_t c0[4], c1[4] = {0, 0, 0, 0};
            k7_e2m1x8_to_s8(q0[i].x, c0[0], c0[1]);
            k7_e2m1x8_to_s8(q0[i].y, c0[2], c0[3]);
            const float ws0 = nvfp4::e4m3_to_float(sc0[i]);
            float ws1 = 0.0f;
            if constexpr (kGlu) {
                k7_e2m1x8_to_s8(q1[i].x, c1[0], c1[1]);
                k7_e2m1x8_to_s8(q1[i].y, c1[2], c1[3]);
                ws1 = nvfp4::e4m3_to_float(sc1[i]);
            }
            if (b0 + kD * kLanes < blocks) {  // refill slot i with the step kD ahead
                q0[i] = w0[blk + kD * kLanes]; sc0[i] = s0[blk + kD * kLanes];
                if constexpr (kGlu) { q1[i] = w1[blk + kD * kLanes]; sc1[i] = s1[blk + kD * kLanes]; }
            }
#pragma unroll
            for (int r = 0; r < kMR; ++r) {
                if (r >= rows) break;  // block-uniform
                const uint32_t *a = &a_s8[r][4 * blk];
                const float as = a_sf[r][blk];
                const int is0 = k7_dot16(c0, a);
                const float p0 = as * ws0;
                int is1 = 0;
                float p1 = 0.0f;
                if constexpr (kGlu) { is1 = k7_dot16(c1, a); p1 = as * ws1; }
#pragma unroll
                for (int t = 0; t < kLanes; ++t) {  // blocks b0 .. b0 + 7 in k order, every lane of the group alike
                    acc0[r] += static_cast<float>(__shfl(is0, gl + t, 32)) * __shfl(p0, gl + t, 32);
                    if constexpr (kGlu) acc1[r] += static_cast<float>(__shfl(is1, gl + t, 32)) * __shfl(p1, gl + t, 32);
                }
            }
        }
    }
    if (!ok || j != 0) return;
    const float alpha = g.alphas[e];
#pragma unroll
    for (int r = 0; r < kMR; ++r) {
        if (r >= rows) continue;
        const float g0 = __bfloat162float(__cere_float2bfloat16((acc0[r] * 0.25f) * alpha));
        float result;
        if constexpr (kGlu) {
            const float up = __bfloat162float(__cere_float2bfloat16((acc1[r] * 0.25f) * alpha));
            result = (g0 / (1.0f + expf(-g0))) * up;  // SiLU(gate) * linear in FP32
        } else {
            result = g0;
        }
        g.out[static_cast<int64_t>(row0 + r) * g.n + col] = __cere_float2bfloat16(result);
    }
}

// The launch (m <= kGemvMaxRows: an expert's tile holds at most m rows, one per token routed to it).
template <bool kGlu>
void launch_grouped_gemv(dim3 grid, const GemmArgs &g, uint32_t m, hipStream_t stream) {
    if (m <= 1) grouped_gemv_kernel<kGlu, 1><<<grid, kGemvThreads, 0, stream>>>(g);
    else if (m <= 4) grouped_gemv_kernel<kGlu, 4><<<grid, kGemvThreads, 0, stream>>>(g);
    else grouped_gemv_kernel<kGlu, kGemvMaxRows><<<grid, kGemvThreads, 0, stream>>>(g);
}
