// Probe: gfx11 WMMA bf16 16x16x16 (wave32) operand/accumulator layout on gfx1151.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#ifndef MAPROW
#define MAPROW (2 * i + l / 16)
#endif
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

__global__ void probe(const __bf16 *A, const __bf16 *B, float *C) {  // A 16x16 row-major, B 16x16 (k x n) row-major
    int l = threadIdx.x, r = l % 16;
    bf16x16 a, b;
    for (int k = 0; k < 16; ++k) { a[k] = A[r * 16 + k]; b[k] = B[k * 16 + r]; }  // lane: A row r, B column r
    f32x8 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    for (int i = 0; i < 8; ++i) C[MAPROW * 16 + r] = c[i];
}
int main() {
    __bf16 hA[256], hB[256]; float ref[256], out[256];
    for (int i = 0; i < 256; ++i) { hA[i] = (__bf16)(float)((i * 7) % 13 - 6); hB[i] = (__bf16)(float)((i * 5) % 11 - 5); }
    for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) { float s = 0; for (int k = 0; k < 16; ++k) s += (float)hA[m*16+k] * (float)hB[k*16+n]; ref[m*16+n] = s; }
    __bf16 *dA, *dB; float *dC;
    hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dC, 1024);
    hipMemcpy(dA, hA, 512, hipMemcpyHostToDevice); hipMemcpy(dB, hB, 512, hipMemcpyHostToDevice);
    probe<<<1, 32>>>(dA, dB, dC); hipMemcpy(out, dC, 1024, hipMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < 256; ++i) if (fabsf(out[i] - ref[i]) > 1e-4f * (1.f + fabsf(ref[i]))) { if (bad < 12) printf("m%d n%d got %.9g want %.9g\n", i / 16, i % 16, out[i], ref[i]); ++bad; }
    if (getenv("DUMP")) for (int m = 0; m < 16; ++m) { for (int n = 0; n < 16; ++n) printf("%5.0f/%-5.0f", out[m*16+n], ref[m*16+n]); printf("\n"); }
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("%s wmma layout %s (%d elements outside 1e-4 rel; row = 2i + lane/16, col = lane%%16)\n", p.gcnArchName, bad ? "WRONG" : "OK", bad);
    return bad != 0;
}
