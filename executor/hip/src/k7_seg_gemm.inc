// AmdAmd PAR-6 step 12: K7's W4A4 grouped GEMM for larger calls (m > kGemvMaxRows: prefill) on gfx11's int8 WMMA,
// per expert segment instead of per 32-row tile (included by gen_hip.py into the hipified k7_moe.hip after
// grouped_gemm_kernel). Block (64-column tile, expert e), 4 waves of 16 columns; the expert's rows run in passes of
// 64 (four 16-row WMMA tiles), so one weight fragment, unpacked once per 16-element NVFP4 block, feeds four WMMAs;
// the fragment's two halves are unpacked by the two half-waves and exchanged (gfx11's wave32 WMMA wants the B
// operand in both). One v_wmma_i32_16x16x16_iu8 k step is one block's exact integer dot; then per output element
// acc += float(isum) * (a_scale * w_scale) block after block in k order and the 0.25 * alpha BF16 epilogue (SwiGLU for
// GEMM1): grouped_gemm_kernel's arithmetic operation for operation, so every output bit is that kernel's (k7_test
// checks). The A chunk (grouped_gemm_kernel's staged values) is double-buffered in LDS with the next chunk's A and
// weights loaded into registers while this chunk computes (one barrier per chunk).
constexpr int kSegThreads = 128;
constexpr int kSegRows = 64;  // rows per pass: 4 WMMA row tiles
typedef int k7sv4i __attribute__((ext_vector_type(4)));
typedef int k7sv8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint32_t k7s_word(const uint4 (&v)[4], int i) {  // i compile-time after unrolling
    const uint4 q = v[i / 4];
    return (i % 4) == 0 ? q.x : (i % 4) == 1 ? q.y : (i % 4) == 2 ? q.z : q.w;
}

// The B operand of one block: 16 codes (words w_lo, w_hi: codes 0..7, 8..15) as 16 int8, built by the two half-waves
// (lane < 16 unpacks codes 0..7, lane >= 16 codes 8..15) and exchanged, so every lane holds all 16.
__device__ __forceinline__ k7sv4i k7s_bfrag(uint32_t w_lo, uint32_t w_hi, int hl) {
    uint32_t a, b;
    k7_e2m1x8_to_s8(hl == 0 ? w_lo : w_hi, a, b);
    const uint32_t oa = __shfl_xor(a, 16, 32), ob = __shfl_xor(b, 16, 32);
    return hl == 0 ? k7sv4i{int(a), int(b), int(oa), int(ob)} : k7sv4i{int(oa), int(ob), int(a), int(b)};
}

template <bool kGlu>
__global__ void __launch_bounds__(kSegThreads) segment_gemm_kernel(GemmArgs g) {
    static_assert(kChunk == 128 && kSegThreads == 128, "the staging map");
    __shared__ __align__(16) uint32_t a_s8[2][kSegRows][kChunk / 4 + 4];
    // activation scales by (block, 16-row tile, half-wave, row pair): a lane's 8 rows (2 i + hl) are contiguous
    __shared__ __align__(16) float a_sf[2][kChunk / 16][kSegRows / 16][2][8];
    const int e = blockIdx.y;
    const int32_t seg0 = g.expert_offset[e], seg_rows = g.expert_offset[e + 1] - seg0;
    if (seg_rows <= 0) return;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, rl = lane % 16, hl = lane / 16;
    const int col = blockIdx.x * kTileCols + wave * 16 + rl;
    const bool col_ok = col < g.n;
    const int64_t w_row = static_cast<int64_t>(e) * g.n + (col_ok ? col : 0);
    const k7sv8i zero = {};
    // staging map: thread t stages code words (t % 16) of rows t / 16 + 8 q (q < 8) and scales (t % 8) of rows
    // t / 8 + 16 q (q < 4) of the pass
    const int ra = threadIdx.x / 16, wd = threadIdx.x % 16, rs = threadIdx.x / 8, sb = threadIdx.x % 8;
    for (int p0 = 0; p0 < seg_rows; p0 += kSegRows) {
        const int rows = min(kSegRows, seg_rows - p0);
        int64_t src_a[8], src_s[4];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int r = ra + 8 * q;
            src_a[q] = r < rows ? (g.a_perm != nullptr ? g.a_perm[seg0 + p0 + r] / g.a_div : seg0 + p0 + r) : -1;
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = rs + 16 * q;
            src_s[q] = r < rows ? (g.a_perm != nullptr ? g.a_perm[seg0 + p0 + r] / g.a_div : seg0 + p0 + r) : -1;
        }
        uint32_t na[8];
        uint8_t ns[4];
        uint4 nq0[4] = {}, nq1[4] = {};
        uint2 nw0 = {0, 0}, nw1 = {0, 0};
        auto fetch_a = [&](int kk) {
            if (kk >= g.k) return;
#pragma unroll
            for (int q = 0; q < 8; ++q)
                na[q] = src_a[q] >= 0 ? *reinterpret_cast<const uint32_t *>(g.a_codes + src_a[q] * (g.k / 2) + kk / 2 + wd * 4) : 0u;
#pragma unroll
            for (int q = 0; q < 4; ++q) ns[q] = src_s[q] >= 0 ? g.a_scales[src_s[q] * (g.k / 16) + kk / 16 + sb] : uint8_t(0);
        };
        auto fetch_w = [&](int kk) {
            if (kk >= g.k) return;
            const uint4 *p = reinterpret_cast<const uint4 *>(g.w0 + w_row * (g.k / 2) + kk / 2);
#pragma unroll
            for (int q = 0; q < 4; ++q) nq0[q] = p[q];
            nw0 = *reinterpret_cast<const uint2 *>(g.s0 + w_row * (g.k / 16) + kk / 16);
            if constexpr (kGlu) {
                const uint4 *p1 = reinterpret_cast<const uint4 *>(g.w1 + w_row * (g.k / 2) + kk / 2);
#pragma unroll
                for (int q = 0; q < 4; ++q) nq1[q] = p1[q];
                nw1 = *reinterpret_cast<const uint2 *>(g.s1 + w_row * (g.k / 16) + kk / 16);
            }
        };
        float acc0[4][8], acc1[4][8];
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc0[t][i] = acc1[t][i] = 0.0f;
        const int tiles = (rows + 15) / 16;  // block-uniform
        __syncthreads();  // the previous pass's LDS reads are done
        fetch_a(0);
        fetch_w(0);
        for (int k0 = 0; k0 < g.k; k0 += kChunk) {
            const int buf = (k0 / kChunk) & 1;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                a_s8[buf][ra + 8 * q][2 * wd] = nvfp4::e2m1x4_to_s8x4(na[q] & 0xFFFFu);
                a_s8[buf][ra + 8 * q][2 * wd + 1] = nvfp4::e2m1x4_to_s8x4(na[q] >> 16);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = rs + 16 * q;
                a_sf[buf][sb][r / 16][r % 2][(r % 16) / 2] = src_s[q] >= 0 ? nvfp4::e4m3_to_float(ns[q]) : 0.0f;
            }
            uint4 wq0[4], wq1[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) { wq0[q] = nq0[q]; wq1[q] = nq1[q]; }
            const uint2 ws0 = nw0, ws1 = nw1;
            fetch_a(k0 + kChunk);
            // GEMM1 (two weight matrices) loads the next weights after this chunk's math: holding both chunks' gate
            // and up fragments with the 64 accumulators exceeded the VGPRs (scratch spills); GEMM2 overlaps them.
            if constexpr (!kGlu) fetch_w(k0 + kChunk);
            __syncthreads();
#pragma unroll
            for (int b = 0; b < kChunk / 16; ++b) {
                const k7sv4i bw0 = k7s_bfrag(k7s_word(wq0, 2 * b), k7s_word(wq0, 2 * b + 1), hl);
                const float sw0 = nvfp4::e4m3_to_float(static_cast<uint8_t>((b < 4 ? ws0.x : ws0.y) >> (8 * (b % 4))));
                k7sv4i bw1 = {};
                float sw1 = 0.0f;
                if constexpr (kGlu) {
                    bw1 = k7s_bfrag(k7s_word(wq1, 2 * b), k7s_word(wq1, 2 * b + 1), hl);
                    sw1 = nvfp4::e4m3_to_float(static_cast<uint8_t>((b < 4 ? ws1.x : ws1.y) >> (8 * (b % 4))));
                }
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t >= tiles) break;
                    const uint4 av = *reinterpret_cast<const uint4 *>(&a_s8[buf][t * 16 + rl][4 * b]);
                    const k7sv4i a = {int(av.x), int(av.y), int(av.z), int(av.w)};
                    const float4 s_lo = *reinterpret_cast<const float4 *>(&a_sf[buf][b][t][hl][0]);
                    const float4 s_hi = *reinterpret_cast<const float4 *>(&a_sf[buf][b][t][hl][4]);
                    const float as[8] = {s_lo.x, s_lo.y, s_lo.z, s_lo.w, s_hi.x, s_hi.y, s_hi.z, s_hi.w};
                    const k7sv8i d0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bw0, zero, false);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc0[t][i] += static_cast<float>(d0[i]) * (as[i] * sw0);
                    if constexpr (kGlu) {
                        const k7sv8i d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bw1, zero, false);
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc1[t][i] += static_cast<float>(d1[i]) * (as[i] * sw1);
                    }
                }
            }
            if constexpr (kGlu) fetch_w(k0 + kChunk);
        }
        if (col_ok) {
            const float alpha = g.alphas[e];
#pragma unroll
            for (int t = 0; t < 4; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = t * 16 + 2 * i + hl;
                    if (r >= rows) continue;
                    const float g0 = __bfloat162float(__cere_float2bfloat16((acc0[t][i] * 0.25f) * alpha));
                    float result;
                    if constexpr (kGlu) {
                        const float up = __bfloat162float(__cere_float2bfloat16((acc1[t][i] * 0.25f) * alpha));
                        result = (g0 / (1.0f + expf(-g0))) * up;  // SiLU(gate) * linear in FP32
                    } else {
                        result = g0;
                    }
                    g.out[static_cast<int64_t>(seg0 + p0 + r) * g.n + col] = __cere_float2bfloat16(result);
                }
        }
    }
}
