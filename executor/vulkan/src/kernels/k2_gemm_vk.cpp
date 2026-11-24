// VulkanAmd P2: see k2_gemm_vk.h.
#include "kernels/k2_gemm_vk.h"

#include <cstring>
#if defined(CERE_VK_K2_CAPTURE)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#endif

#include "kernels/k2_mgemv_vk.h"

namespace cere::vkk2 {
namespace {

struct GemmPush {
    uint64_t x, w, s, y, xinfo, winfo;
    uint32_t m, n, k, out_f32, group, force;
};
static_assert(sizeof(GemmPush) == 72, "push layout matches k2_gemm.comp / k2_gemm_f32.comp");

struct ScanPush {
    uint64_t src, scales, info;
    uint32_t rows, k, split, pad1;
};
static_assert(sizeof(ScanPush) == 40, "push layout matches k2_gemm_scan.comp");

constexpr uint32_t kTile = 128;

bool a16(uint64_t a) { return a % 16 == 0; }
uint64_t round16(uint64_t b) { return (b + 15) / 16 * 16; }
constexpr uint64_t k4G = uint64_t(1) << 32;

// The SPIR-V names of the GEMM's pipelines, one selection for the dispatches and for pipeline_specs.
const char *scan_spv(bool mx) { return mx ? "k2_gemm_scan_mx.spv" : "k2_gemm_scan_bf16.spv"; }
const char *gemm_spv(bool mx) { return mx ? "k2_gemm_mx.spv" : "k2_gemm_bf16.spv"; }
const char *gemm_f32_spv(bool mx) { return mx ? "k2_gemm_f32_mx.spv" : "k2_gemm_f32_bf16.spv"; }

// split: x rows may be SPLIT (Context::k2_tail_split); weights never split (a w row SPLIT was FB on the FP32 route
// anyway), so a prepared weight's info is the same whatever the lever.
void scan(vk::Context &ctx, bool mx, uint64_t src, uint64_t scales, uint64_t info, uint32_t rows, uint32_t k,
          bool split = false) {
    const ScanPush pc{src, scales, info, rows, k, split ? 1u : 0u, 0};
    ctx.dispatch(ctx.pipeline(scan_spv(mx), sizeof(pc), 32), &pc, (rows + 7) / 8);
}

#if defined(CERE_VK_K2_CAPTURE)
// TEST BUILD ONLY (never in the default library: AGENTS.md 14). P4-B activation capture: a gemm() call with
// m >= CERE_VK_K2_CAPTURE_MINM (default 1024) whose (n, k) is in CERE_VK_K2_CAPTURE_SHAPES (default
// "10240:320,2560:6144": K3's hc up projection and K5's out_proj) copies its x operand into a host-visible buffer; a
// later gemm() or run() call on the same context, after that submit completed (Context::last_completed_nonce), writes it to
// CERE_VK_K2_CAPTURE_DIR/<seq>_n<n>_k<k>_m<m>.bf16 (raw BF16 [m, k]) and frees it. Total bytes are capped by
// CERE_VK_K2_CAPTURE_MAX_MB (default 1800). Nothing is captured without CERE_VK_K2_CAPTURE_DIR.
struct CapturePending { vk::Context *ctx; vk::Buffer buf; uint64_t nonce; uint32_t m, n, k; uint64_t seq; };
struct CaptureState {
    bool init = false;
    std::string dir;
    uint32_t min_m = 1024;
    uint64_t max_bytes = 1800ull << 20, bytes = 0, seq = 0;
    std::vector<std::pair<uint32_t, uint32_t>> shapes;
    std::vector<CapturePending> pending;
};
CaptureState &capture_state() {
    static CaptureState s;
    if (!s.init) {
        s.init = true;
        if (const char *d = std::getenv("CERE_VK_K2_CAPTURE_DIR")) s.dir = d;
        if (const char *v = std::getenv("CERE_VK_K2_CAPTURE_MINM")) s.min_m = uint32_t(std::strtoul(v, nullptr, 10));
        if (const char *v = std::getenv("CERE_VK_K2_CAPTURE_MAX_MB")) s.max_bytes = uint64_t(std::strtoull(v, nullptr, 10)) << 20;
        const char *sh = std::getenv("CERE_VK_K2_CAPTURE_SHAPES");
        std::string list = sh ? sh : "10240:320,2560:6144";
        size_t p = 0;
        while (p < list.size()) {
            size_t e = list.find(',', p);
            if (e == std::string::npos) e = list.size();
            const std::string item = list.substr(p, e - p);
            const size_t c = item.find(':');
            if (c != std::string::npos)
                s.shapes.emplace_back(uint32_t(std::strtoul(item.c_str(), nullptr, 10)), uint32_t(std::strtoul(item.c_str() + c + 1, nullptr, 10)));
            p = e + 1;
        }
    }
    return s;
}
void capture_flush(vk::Context &ctx) {
    CaptureState &s = capture_state();
    for (size_t i = 0; i < s.pending.size();) {
        CapturePending &p = s.pending[i];
        if (p.ctx != &ctx || ctx.last_completed_nonce() <= p.nonce) { ++i; continue; }
        char name[512];
        std::snprintf(name, sizeof name, "%s/%06llu_n%u_k%u_m%u.bf16", s.dir.c_str(), static_cast<unsigned long long>(p.seq), p.n, p.k, p.m);
        if (FILE *f = std::fopen(name, "wb")) {
            std::fwrite(p.buf.map, 1, size_t(p.m) * p.k * 2, f);
            std::fclose(f);
        }
        ctx.free(p.buf);
        s.pending.erase(s.pending.begin() + long(i));
    }
}
void capture(vk::Context &ctx, uint64_t x, uint32_t m, uint32_t n, uint32_t k) {
    CaptureState &s = capture_state();
    if (s.dir.empty()) return;
    capture_flush(ctx);
    if (m < s.min_m) return;
    if (std::find(s.shapes.begin(), s.shapes.end(), std::make_pair(n, k)) == s.shapes.end()) return;
    const uint64_t bytes = uint64_t(m) * k * 2;
    if (s.bytes + bytes > s.max_bytes) return;
    CapturePending p{&ctx, {}, ctx.last_completed_nonce(), m, n, k, s.seq++};
    if (ctx.allocate(bytes, p.buf) != CERE_OK || p.buf.map == nullptr) return;
    s.bytes += bytes;
    struct { uint64_t src, dst; uint32_t vecs, pad0; } pc{x, p.buf.addr, uint32_t(bytes / 16), 0};
    ctx.dispatch(ctx.pipeline("k2_gemm_capture.spv", sizeof pc), &pc, uint32_t((bytes / 16 + 255) / 256));
    ctx.barrier();
    s.pending.push_back(p);
}
#endif

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(bool bf16, bool mx, const std::vector<uint32_t> &gemv_bf16_ks, const std::vector<uint32_t> &gemv_mx_ks) {
    std::vector<vk::PipelineSpec> out;
    out.push_back({"k2_gemm_probe.spv", 16, 32, {}});  // gemm_probe's push block: { idx, out } (two addresses)
    if (bf16 || mx) out.push_back({scan_spv(false), sizeof(ScanPush), 32, {}});  // gemm() always scans the BF16 activations x
    for (int f = 0; f < 2; ++f) {
        if (!(f ? mx : bf16)) continue;
        if (f) out.push_back({scan_spv(true), sizeof(ScanPush), 32, {}});
        out.push_back({gemm_spv(f != 0), sizeof(GemmPush), 32, {}});
        out.push_back({gemm_f32_spv(f != 0), sizeof(GemmPush), 0, {}});
    }
    // run()'s class <= 16 side is vkk2::gemv at the caller's k values.
    append_unique(out, pipeline_specs(gemv_bf16_ks, gemv_mx_ks, false));
    return out;
}

cere_status gemm_probe(vk::Context &ctx) {
    if (!ctx.info().coopmat) {
        ctx.set_probe(vk::Probe::kK2GemmLayout, vk::ProbeState::kFail);
        return CERE_ERR_UNSUPPORTED;
    }
    vk::Buffer idx, out;
    cere_status st = ctx.allocate(256 * 4, idx);
    if (st == CERE_OK) st = ctx.allocate((32 * 16 + 32) * 4, out);
    if (st != CERE_OK) {
        ctx.free(idx);
        return st;
    }
    for (int i = 0; i < 256; ++i) static_cast<float *>(idx.map)[i] = float(i);
    std::memset(out.map, 0xFF, (32 * 16 + 32) * 4);
    struct { uint64_t idx, out; } pc{idx.addr, out.addr};
    const vk::Pipeline *p = nullptr;
    st = ctx.pipeline(vk::PipelineSpec{"k2_gemm_probe.spv", sizeof(pc), 32, {}}, &p);
    if (st == CERE_OK) st = ctx.begin();
    if (st == CERE_OK) {
        ctx.dispatch(*p, &pc, 1);
        ctx.barrier();
        st = ctx.submit_wait();  // a latched recording error comes back here
    }
    bool ok = st == CERE_OK;
    const float *o = static_cast<const float *>(out.map);
    for (int l = 0; l < 32 && ok; ++l) {
        ok = o[32 * 16 + l] == 8.0f;
        for (int i = 0; i < 8 && ok; ++i) ok = o[l * 16 + i] == float((2 * i + l / 16) * 16 + l % 16);
    }
    ctx.free(idx);
    ctx.free(out);
    if (st != CERE_OK) return st;  // not probed (launch failure): the state stays as it was
    ctx.set_probe(vk::Probe::kK2GemmLayout, ok ? vk::ProbeState::kPass : vk::ProbeState::kFail);
    return ok ? CERE_OK : CERE_ERR_UNSUPPORTED;
}

bool gemm_ready(const vk::Context &ctx) {
    return ctx.info().coopmat && ctx.probe(vk::Probe::kK2GemmLayout) == vk::ProbeState::kPass;
}

size_t gemm_workspace_bytes(uint32_t m, uint32_t n) { return round16(uint64_t(m) * 4) + round16(uint64_t(n) * 4); }

cere_status gemm_prepare_weight(vk::Context &ctx, uint64_t w, uint64_t scales, uint32_t n, uint32_t k, uint64_t winfo) {
    const bool mx = scales != 0;
    if (!w || !winfo || n == 0 || k == 0 || k % 64 != 0 || !a16(w) || !a16(winfo) ||
        uint64_t(n) * k * (mx ? 1 : 2) >= k4G)
        return CERE_ERR_ARGUMENT;
    scan(ctx, mx, w, scales, winfo, n, k);
    return CERE_OK;
}

cere_status gemm_scan_rows(vk::Context &ctx, uint64_t x, uint32_t m, uint32_t k, uint64_t info) {
    if (!x || !info || m == 0 || k == 0 || k % 64 != 0 || !a16(x) || !a16(info) || uint64_t(m) * k * 2 >= k4G)
        return CERE_ERR_ARGUMENT;
    scan(ctx, false, x, 0, info, m, k, ctx.k2_tail_split());
    return CERE_OK;
}

cere_status gemm(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t scales, uint64_t y, uint32_t m, uint32_t n, uint32_t k,
                 bool out_f32, uint64_t ws, uint64_t winfo, const GemmOptions *opts) {
    const GemmOptions o = opts ? *opts : GemmOptions{};
    const bool mx = scales != 0;
    if (!x || !w || !y || !ws || m == 0 || n == 0 || k == 0 || k % 64 != 0 || !a16(x) || !a16(w) || !a16(y) || !a16(ws) ||
        !a16(winfo) || uint64_t(m) * k * 2 >= k4G || uint64_t(n) * k * (mx ? 1 : 2) >= k4G ||
        uint64_t(m) * n * (out_f32 ? 4 : 2) >= k4G)
        return CERE_ERR_ARGUMENT;
    if (o.route != GemmRoute::kAuto && o.route != GemmRoute::kFp32) return CERE_ERR_ARGUMENT;
    if (o.profile_stamp0 >= 0 && uint32_t(o.profile_stamp0) + 4 > vk::Context::kStamps) return CERE_ERR_ARGUMENT;
    if (!gemm_ready(ctx)) return CERE_ERR_UNSUPPORTED;

    uint32_t ps = o.profile_stamp0 >= 0 ? uint32_t(o.profile_stamp0) : 0;
    auto stamp = [&] { if (o.profile_stamp0 >= 0) ctx.stamp(ps++); };
#if defined(CERE_VK_K2_CAPTURE)
    capture(ctx, x, m, n, k);
#endif
    // An armed K2 x-info tap (vk_launch.h) takes the x-row info words: the same scan, written where the caller counts.
    const uint64_t tap = ctx.take_k2_info(x, m, k);
    const uint64_t xinfo = tap != 0 ? tap : ws, wi = winfo ? winfo : ws + round16(uint64_t(m) * 4);
    const uint32_t group = n > 8192 ? 32 : 1;  // block order only (k2_wmma's): no effect on any output bit
    const uint32_t tiles = ((m + kTile - 1) / kTile) * ((n + kTile - 1) / kTile);
    stamp();
    scan(ctx, false, x, 0, xinfo, m, k, ctx.k2_tail_split());
    if (!winfo) scan(ctx, mx, w, scales, wi, n, k);
    ctx.barrier();
    stamp();
    const bool force = o.route == GemmRoute::kFp32;
    if (!force) {
        const GemmPush pc{x, w, scales, y, xinfo, wi, m, n, k, out_f32 ? 1u : 0u, group, 0};
        ctx.dispatch(ctx.pipeline(gemm_spv(mx), sizeof(pc), 32), &pc, tiles);
        ctx.barrier();
    }
    stamp();
    const GemmPush pf{x, w, scales, y, xinfo, wi, m, n, k, out_f32 ? 1u : 0u, group, force ? 1u : 0u};
    ctx.dispatch(ctx.pipeline(gemm_f32_spv(mx), sizeof(pf)), &pf, tiles);
    stamp();
    return CERE_OK;
}

cere_status run(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t scales, uint64_t y, uint32_t m, uint32_t n, uint32_t k,
                bool out_f32, uint32_t cls, uint64_t ws, uint64_t winfo) {
    if (m == 0 || (cls != 0 && cls < m)) return CERE_ERR_ARGUMENT;
    const uint32_t choice = cls != 0 ? cls : m;
#if defined(CERE_VK_K2_CAPTURE)
    if (!capture_state().dir.empty()) capture_flush(ctx);  // decode steps after a prefill write its captures
#endif
    if (choice < kGemmMinM) return gemv(ctx, x, w, scales, y, m, n, k, out_f32);
    if (k % 64 != 0) return CERE_ERR_UNSUPPORTED;
    return gemm(ctx, x, w, scales, y, m, n, k, out_f32, ws, winfo);
}

}  // namespace cere::vkk2
