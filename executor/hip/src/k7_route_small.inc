// AmdAmd PAR-6 step 15: K7's expert grouping for small calls in one block (included by gen_hip.py into the hipified
// k7_moe.hip after place_kernel). moe_block groups the (token, slot) pairs by expert with a memset, count_kernel,
// offsets_kernel (one thread's serial pass over the 512 experts) and place_kernel: four launches, ~17 us of a decode
// token's 0.25 ms MoE layer on gfx1151, almost all fixed latency. For pairs <= kRouteSmallPairs one block does the
// same: counts (written to `counts` as count_kernel leaves them), the exclusive prefix sums of the counts and of their
// 32-row tiles (offsets_kernel's values: integer sums, any order), and each pair's row = its expert's first row + the
// number of earlier pairs of that expert (place_kernel's stable ascending order). An id outside [0, num_experts) is
// not counted or placed and gets pos 0, as count_kernel. Every output integer is the four kernels'.
constexpr int kRouteSmallPairs = kSortThreads;  // m * top_k: decode and verify steps

__global__ void __launch_bounds__(kSortThreads)
    route_small_kernel(const int32_t *__restrict__ ids, int32_t *__restrict__ counts,
                       int32_t *__restrict__ expert_offset, int32_t *__restrict__ tile_offset,
                       int32_t *__restrict__ perm, int32_t *__restrict__ pos, int pairs, int num_experts) {
    __shared__ int32_t c[kMaxExperts];
    __shared__ int32_t off[kMaxExperts];
    __shared__ int32_t sid[kRouteSmallPairs];
    __shared__ int32_t scan_r[kSortThreads], scan_t[kSortThreads];
    const int tid = threadIdx.x;
    for (int e = tid; e < num_experts; e += kSortThreads) c[e] = 0;
    if (tid < pairs) sid[tid] = ids[tid];
    __syncthreads();
    if (tid < pairs) {
        const int32_t id = sid[tid];
        if (static_cast<uint32_t>(id) < static_cast<uint32_t>(num_experts)) atomicAdd(&c[id], 1);
        else pos[tid] = 0;
    }
    __syncthreads();
    // thread tid owns experts [tid * per, (tid + 1) * per)
    const int per = (num_experts + kSortThreads - 1) / kSortThreads;
    const int e0 = min(tid * per, num_experts), e1 = min(e0 + per, num_experts);
    int32_t rows = 0, tiles = 0;
    for (int e = e0; e < e1; ++e) { rows += c[e]; tiles += (c[e] + kTileRows - 1) / kTileRows; }
    scan_r[tid] = rows;
    scan_t[tid] = tiles;
    __syncthreads();
    for (int d = 1; d < kSortThreads; d *= 2) {  // inclusive Hillis-Steele scan
        const int32_t vr = tid >= d ? scan_r[tid - d] : 0, vt = tid >= d ? scan_t[tid - d] : 0;
        __syncthreads();
        scan_r[tid] += vr;
        scan_t[tid] += vt;
        __syncthreads();
    }
    int32_t r = scan_r[tid] - rows, t = scan_t[tid] - tiles;  // exclusive
    for (int e = e0; e < e1; ++e) {
        counts[e] = c[e];
        expert_offset[e] = r;
        tile_offset[e] = t;
        off[e] = r;
        r += c[e];
        t += (c[e] + kTileRows - 1) / kTileRows;
    }
    if (tid == kSortThreads - 1) {
        expert_offset[num_experts] = scan_r[tid];
        tile_offset[num_experts] = scan_t[tid];
    }
    __syncthreads();
    if (tid < pairs) {
        const int32_t id = sid[tid];
        if (static_cast<uint32_t>(id) < static_cast<uint32_t>(num_experts)) {
            int32_t rank = 0;
            for (int q = 0; q < tid; ++q) rank += sid[q] == id;
            const int32_t row = off[id] + rank;
            perm[row] = tid;
            pos[tid] = row;
        }
    }
}
