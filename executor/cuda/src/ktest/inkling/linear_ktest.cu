// Inkling 2 Mini S8: kernel entry point of libcere_ktest.so over the BF16 projections (kernels/inkling/linear.h), for
// executor/tests/gpu/test_inkling_kernels_linear.py. segs: host int64 [count, 6], per segment (x, ldx, w, y, ldy, n)
// with device pointers as integers; path: LinearPath (0 auto, 1 narrow, 2 wide). Enqueues on `stream` and returns the
// launch status (0 = cudaSuccess).
#include <climits>
#include <cstdint>

#include "kernels/inkling/linear.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace ink = cere::inkling;

CERE_KTEST int cere_ktest_inkling_linear(const int64_t *segs, int64_t count, int64_t m, int64_t k, int64_t path,
                                         cudaStream_t stream) {
    if (segs == nullptr || count < 0 || count > INT_MAX || path < 0 || path > 2) return int(cudaErrorInvalidValue);
    ink::LinearArgs a;
    a.m = m;
    a.k = k;
    a.count = int(count);
    for (int64_t s = 0; s < count && s < ink::kLinearMaxSegs; ++s) {
        const int64_t *d = segs + 6 * s;
        a.seg[s].x = reinterpret_cast<const __nv_bfloat16 *>(d[0]);
        a.seg[s].ldx = d[1];
        a.seg[s].w = reinterpret_cast<const __nv_bfloat16 *>(d[2]);
        a.seg[s].y = reinterpret_cast<__nv_bfloat16 *>(d[3]);
        a.seg[s].ldy = d[4];
        a.seg[s].n = d[5];
    }
    return int(ink::linear(a, stream, static_cast<ink::LinearPath>(path)));
}
