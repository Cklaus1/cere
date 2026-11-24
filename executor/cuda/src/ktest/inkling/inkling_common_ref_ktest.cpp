// Inkling 2 Mini (inkling design note G1): CPU entry points of libcere_ktest.so over the shared numeric contract
// (executor/ref/inkling/common_ref.h), so Python GPU tests compare a kernel's output with the reference bit for bit.
// Host memory only; each returns 0, or -1 for an argument it refuses (null pointer, unknown op, width or format).
#include <cstdint>

#include "inkling/common_ref.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

// y[i] = op(x[i]): op 0 exp, 1 log, 2 tanh, 3 sigmoid, 4 rsqrt.
CERE_KTEST int cere_ktest_inkling_ref_unary(uint32_t op, const float *x, float *y, uint64_t n) {
    if ((x == nullptr || y == nullptr) && n != 0) return -1;
    if (op > 4) return -1;
    for (uint64_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = op == 0 ? ref::exp(v) : op == 1 ? ref::log(v) : op == 2 ? ref::tanh(v) : op == 3 ? ref::sigmoid(v)
                                                                                          : ref::rsqrt(v);
    }
    return 0;
}

namespace {
template <int L> float reduce(uint32_t op, const float *a, const float *b, int n) {
    return op == 0 ? ref::reduce_sum<L>(a, n) : op == 1 ? ref::reduce_dot<L>(a, b, n) : ref::reduce_max<L>(a, n);
}
}  // namespace

// *out = reduce_<op><width>(a[, b], n): op 0 sum, 1 dot (b required), 2 max; width 16, 32, 64 or 256.
CERE_KTEST int cere_ktest_inkling_ref_reduce(uint32_t op, uint32_t width, const float *a, const float *b, int32_t n,
                                             float *out) {
    if (out == nullptr || n < 0 || (a == nullptr && n != 0) || op > 2 || (op == 1 && b == nullptr && n != 0)) {
        return -1;
    }
    switch (width) {
    case 16: *out = reduce<16>(op, a, b, n); return 0;
    case 32: *out = reduce<32>(op, a, b, n); return 0;
    case 64: *out = reduce<64>(op, a, b, n); return 0;
    case 256: *out = reduce<256>(op, a, b, n); return 0;
    default: return -1;
    }
}

// out[i] = decode of element i: format 0 bf16 (in holds n little-endian uint16), 1 FP4 E2M1 (the low nibble of byte
// i), 2 FP8 E4M3 fn, 3 UE8M0.
CERE_KTEST int cere_ktest_inkling_ref_decode(uint32_t format, const uint8_t *in, float *out, uint64_t n) {
    if ((in == nullptr || out == nullptr) && n != 0) return -1;
    if (format > 3) return -1;
    for (uint64_t i = 0; i < n; ++i) {
        if (format == 0) out[i] = ref::decode_bf16(uint16_t(in[2 * i] | (in[2 * i + 1] << 8)));
        else if (format == 1) out[i] = ref::decode_e2m1(in[i] & 0x0f);
        else if (format == 2) out[i] = ref::decode_e4m3(in[i]);
        else out[i] = ref::decode_ue8m0(in[i]);
    }
    return 0;
}
