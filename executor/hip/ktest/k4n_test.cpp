// AmdAmd PAR-6 step 14c: K4's NVFP4 PLE gather (the S3 / FP4-Local table format; executor/cuda/src/kernels/k4_ple.cu,
// hipified) on gfx1151: out[i] = row ids[i] decoded as k4_nvfp4.h (E2M1 x E4M3 exact, x F32 global, BF16 RNE), the
// block scales and global in mapped pinned host memory as the binder places them. Random bytes everywhere (incl.
// E4M3 NaN scales, -0 codes); ids include out-of-range ones (zero bytes: +0). Compared bit for bit with the header's
// host decode_row.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k4_nvfp4.h"
#include "kernels/k4_ple.h"

namespace k4 = cere::k4;
namespace n4 = cere::k4::nvfp4;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

int main() {
    std::mt19937 g(11);
    const int64_t rows = 5000, n = 4096;
    std::vector<uint8_t> table(size_t(rows) * n4::kRowBytes), scales(size_t(rows) * n4::kRowScales);
    for (auto &b : table) b = uint8_t(g());
    for (auto &b : scales) b = uint8_t(g());
    std::vector<int64_t> ids(n);
    for (auto &i : ids) i = int64_t(g() % (rows + 40)) - 20;  // some < 0, some >= rows
    int fails = 0;
    for (float global : {0.0123f, 1.0f, 3.5e-3f}) {
        uint8_t *dt, *hs; float *hg; int64_t *di; uint16_t *dout;
        HCK(hipMalloc(&dt, table.size())); HCK(hipMemcpy(dt, table.data(), table.size(), hipMemcpyHostToDevice));
        HCK(hipHostMalloc(&hs, scales.size(), hipHostMallocMapped)); std::memcpy(hs, scales.data(), scales.size());
        HCK(hipHostMalloc(&hg, 4, hipHostMallocMapped)); *hg = global;
        uint8_t *ds; float *dg; HCK(hipHostGetDevicePointer((void **)&ds, hs, 0)); HCK(hipHostGetDevicePointer((void **)&dg, hg, 0));
        HCK(hipMalloc(&di, n * 8)); HCK(hipMemcpy(di, ids.data(), n * 8, hipMemcpyHostToDevice));
        HCK(hipMalloc(&dout, size_t(n) * 160 * 2));
        if (k4::gather_nvfp4(dt, ds, dg, rows, di, n, reinterpret_cast<__hip_bfloat16 *>(dout), nullptr) != CERE_OK) return 2;
        HCK(hipDeviceSynchronize());
        std::vector<uint16_t> out(size_t(n) * 160), ref(160);
        HCK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
        const uint8_t zr[n4::kRowBytes] = {}, zs[n4::kRowScales] = {};
        long bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            const bool in = ids[i] >= 0 && ids[i] < rows;
            n4::decode_row(in ? &table[ids[i] * n4::kRowBytes] : zr, in ? &scales[ids[i] * n4::kRowScales] : zs, global, ref.data());
            if (std::memcmp(ref.data(), &out[i * 160], 320) != 0) {
                if (++bad <= 3) for (int e = 0; e < 160; ++e) if (ref[e] != out[i * 160 + e]) { std::printf("  id %lld e %d ref %04x got %04x\n", (long long)ids[i], e, ref[e], out[i * 160 + e]); break; }
            }
        }
        std::printf("%-22s %s  4096 ids (incl. out of range), global %g: %ld rows differ\n", "gather_nvfp4", bad ? "FAIL" : "PASS", global, bad);
        fails += bad != 0;
        HCK(hipFree(dt)); HCK(hipHostFree(hs)); HCK(hipHostFree(hg)); HCK(hipFree(di)); HCK(hipFree(dout));
    }
    std::printf("%s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
