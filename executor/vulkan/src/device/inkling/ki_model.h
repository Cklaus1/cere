// Inkling S11 (inkling design note 3, 4.2): the Inkling 2 Mini text forward on Vulkan from the ki_* kernels, shared
// by the library's device (vk_inkling_device.cpp) and the standalone G3 / G5 harness (ktest/ki_forward.cpp).
//
// Graph (REF modeling_inkling2_loop.py @046bdb8c :856-1149 layers, :1359-1447 model; S5 CONTRACT visit plan): 48 visits,
// v < 6 the layers 0-5, 6 <= v < 42 the loop layers 6-17 in three passes, v >= 42 the layers 18-23. A session owns per
// visit its KV cache (sliding: a ring of 1,024 slots; full, layers 5/11/17/23: one row per position) and its four FP32
// conv states; the forward owns per visit its output rows, so a block state is just an earlier visit's output buffer
// (no copies). Per layer:
//   a = l == 0 ? x : depth_mix(B ++ [x]; attention queries)  (full layers: 3 queries, one per q / k / v input)
//   passes 2-3: a_g = msg_gate(a_g, message_g) with messages = depth_mix([prev pass x_in(12), prev pass out]; 4 queries)
//   h_g = rms_w(a_g) BF16; q / rel / sink / gate from h_q, k from h_k, v from h_v; attention; o_proj; post-attn conv
//   r = boundary ? float(conv) : x + float(conv); B' = boundary ? B ++ [x] : B
//   mm = depth_mix(B' ++ [r]; mlp query) (msg_gate with the mlp message in passes 2-3); experts of rms_w(mm) BF16;
//   post-MLP conv; out = bf16(r + float(conv)).
// final: depth_mix(B ++ [x]; output query) -> rms_w(norm) BF16; logits(): unembed BF16 -> 30 tanh(x / 30) FP32.
// Dense projections run on vki::dense (one fma chain per output, bit-equal to the CPU references) or, with
// Options::k2, on K2 (vkk2::run; not bit-equal); the experts on ki_moe (arm Q, the checkpoint's own NVFP4 W4A4 / MXFP8
// activation quantization). One submit per visit (the amdgpu job timeout, vk_launch.h kMaxSubmitGpuMs).
//
// Memory: every buffer is one vk::Buffer (<= 4 GiB each) and must be host-visible (UMA: AMD's RADV, lavapipe); init
// refuses other devices (CERE_ERR_UNSUPPORTED). Errors are returned, never fatal: a failed call leaves no command
// buffer open; a session whose step failed must be discarded by the caller (its state is undefined).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cere_executor.h"
#include "kernels/ki_ops_vk.h"
#include "launch/vk_launch.h"

namespace cere::vkink {

constexpr uint32_t kHidden = 2304, kLayers = 24, kVisits = 48, kLoop0 = 6, kLoop1 = 18, kPasses = 3;
constexpr uint32_t kVocab = 200058;        // unpadded logits (the checkpoint's embed / unembed have 201,024 rows)
constexpr uint32_t kVocabRows = 201024;
constexpr uint32_t kQRow = 3072, kKvRow = 1024, kRelRow = 384, kHeads = 24, kKvHeads = 8, kInter = 1152, kExperts = 128;
constexpr uint32_t kMaxRows = 4096;        // rows of one step call
constexpr uint32_t kGroupedMoeTokens = 16; // step rows from which the MoE uses the expert-grouped tiles (same bits)
// (query row, key) pairs of one attention submit: a 512-row full-attention chunk at ~5K context outran AMD's ~2 s
// amdgpu ring timeout in one command buffer (p07); rows are independent, so blocks change nothing but the submits.
constexpr uint64_t kAttnKeyRows = uint64_t(1) << 19;
// Short steps (<= kOneSubmitRows rows ending at <= kOneSubmitCtx positions) may group visits per command buffer
// (Options::visits_per_submit); long steps submit every visit under AMD's ~2 s ring timeout.
constexpr uint32_t kOneSubmitRows = 16;
constexpr uint64_t kOneSubmitCtx = 32768;
constexpr uint64_t kMaxContext = uint64_t(1) << 20;  // config max_position_embeddings
// Command buffers in flight (the device's vk::CreateOptions::max_in_flight; D-I-EPB (a2)): a step's intermediate
// submits do not wait, so the host records the next command buffer while the GPU runs the previous one; a step and a
// logits() call retire everything before they return (the host then reads or rewrites mapped buffers).
constexpr uint32_t kSubmitsInFlight = 2;

// The visit -> physical layer map (S5 CONTRACT section 2).
constexpr uint32_t phys(uint32_t v) { return v < kLoop0 ? v : v < 42 ? kLoop0 + (v - kLoop0) % 12 : v - 24; }

// A checkpoint tensor already on the device: its buffer address, persistent host mapping and size in bytes.
struct TensorView {
    uint64_t addr = 0;
    const void *map = nullptr;
    uint64_t bytes = 0;
};
// Looks a checkpoint tensor up by its full name ("model.llm.layers.0.attn.wq_du.weight"); false when absent.
using Find = std::function<bool(const std::string &name, TensorView *out)>;

struct Layer {
    bool full = false;
    uint32_t fmt = 0;  // 0 NVFP4, 1 MXFP8
    uint64_t attn_q = 0, mlp_q = 0, norm_q = 0, norm_k = 0, norm_v = 0, mlp_norm = 0;
    uint64_t wq = 0, wk = 0, wv = 0, wr = 0, wo = 0, wg = 0, ws = 0, q_scale = 0, rel_scale = 0, rel_proj = 0;
    uint64_t k_conv = 0, v_conv = 0, k_dynp = 0, v_dynp = 0, attn_conv = 0, mlp_conv = 0;
    uint64_t router_w = 0, router_b = 0;
    vki::ExpertW w13, w2;
    uint64_t ctrl[4][3] = {};  // q, k, v, mlp x down, forget, gain (loop layers)
};

// The tensor names the text forward binds (checkpoint names, never repacked), and whether each must hold exactly the
// bytes of its shape: a binder refuses a pack that lacks one. Vision (`model.visual.*`) and the drafter are not here.
std::vector<std::string> text_tensor_names();

// One session's state: per visit the KV cache and the conv states (ping-pong), all on the forward's context.
class Session {
public:
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    uint32_t max_ctx() const { return max_ctx_; }
    uint64_t bytes() const { return bytes_; }

private:
    friend class Forward;
    Session() = default;
    vk::Context *ctx_ = nullptr;
    uint32_t max_ctx_ = 0;
    uint64_t bytes_ = 0;
    struct Visit {
        uint64_t kc = 0, vc = 0;  // KV cache (BF16 rows of 1,024)
        uint64_t st[4][2] = {};   // conv states FP32 [3][C], ping-pong: k, v, post-attn, post-mlp
        int cur[4] = {};
    } v_[kVisits];
    std::vector<vk::Buffer> owned_;
};

class Forward {
public:
    struct Options {
        uint32_t max_rows = 256;    // rows of one step call (1 .. kMaxRows): the prefill chunk
        uint32_t logit_rows = 1;    // rows one logits() call may score (1 .. max_rows)
        bool k2 = false;            // dense projections on K2 (not bit-equal; G5 compares logits)
        bool trace = false;         // every barrier becomes submit_wait + begin; every op printed to stderr (debug)
        bool prof = false;          // GPU ms summed per submit site (perf); with prof_split every barrier is a submit too
        bool prof_split = true;     // (per-op attribution, but the extra submits change the timing)
        uint32_t visits_per_submit = 1;  // short steps: visits per command buffer (1-48; AMD measured 1 fastest)
        // Fail-first G3 mutants (ki_forward only; the library device runs none): the MoE mutant of every layer, and
        // passes 2-3 without messages (no message mix, no message gates: every loop visit runs as in pass 1).
        vki::Mutant moe_mutant = vki::Mutant::kNone;
        bool no_messages = false;
        // D-I-EPB (a2) per-item pairs (ki_forward only; the library device keeps kAuto): the attention and message
        // gate routes. Every route gives the same bits.
        vki::AttnRoute attn_route = vki::AttnRoute::kAuto;
        vki::MsgRoute msg_route = vki::MsgRoute::kAuto;
    };

    explicit Forward(vk::Context &ctx) : ctx_(ctx) {}
    ~Forward();
    Forward(const Forward &) = delete;
    Forward &operator=(const Forward &) = delete;

    // Binds the text weights through `find` (and builds the merged message queries), allocates the scratch and warms
    // every pipeline. ARGUMENT (why: the tensor) for a missing or wrongly sized tensor, UNSUPPORTED on a device whose
    // memory is not host-visible.
    cere_status init(const Find &find, const Options &opt, std::string *why);
    // Bytes init allocates besides the weights, and one session's state at max_ctx positions.
    static uint64_t scratch_bytes(const Options &opt);
    static uint64_t session_bytes(uint64_t max_ctx);

    // A fresh session (zero conv states) for positions [0, max_ctx).
    cere_status create_session(uint32_t max_ctx, std::unique_ptr<Session> *out);

    // M (1 .. max_rows) tokens at positions pos0 .. pos0 + M - 1 through the model (ids < kVocab; pos0 + M <= the
    // session's max_ctx: the caller checks both). The final normed rows stay in the forward for logits().
    cere_status step(Session &s, const uint32_t *ids, uint32_t M, uint32_t pos0);
    // Softcapped FP32 logits of rows [first, first + rows) of the last step (rows <= logit_rows) into logits_host()
    // row-major [rows][kVocab].
    cere_status logits(uint32_t first, uint32_t rows);
    const float *logits_host() const { return static_cast<const float *>(host(s_.logits)); }
    // Visit v's output rows of the last step, BF16 [M][kHidden] (debug: G2 against S1's taps).
    const uint16_t *visit_out_host(uint32_t v) const { return static_cast<const uint16_t *>(host(out_[v])); }
    uint64_t weight_bytes() const { return weight_bytes_; }
    // opt.prof: GPU milliseconds and submit counts per ki_model.cpp source line of the submit / barrier that closed
    // each command buffer (the op group recorded since the previous one), since the last prof_reset().
    const std::map<int, std::pair<double, uint64_t>> &prof() const { return prof_; }
    void prof_reset() { prof_.clear(); }

private:
    struct Scratch {
        uint64_t ids, emb, tau, a, msgs, t3, md, hq, hk, hv, q, r, sink, gate, kraw, vraw, kd, vd, kcv, vcv, k, v, ao, o, co,
            res, mm, hn, rids, rwts, rlg, rtab, apart, xq, xs, xa, gu, vq, vs, va, y, dm, dc, fin, fn, logits_bf, logits;
    };

    cere_status bind(const Find &find, std::string *why);
    cere_status alloc(uint64_t bytes, uint64_t *addr, std::vector<vk::Buffer> *owner);
    void *host(uint64_t addr) const;
    cere_status begin();
    cere_status submit(int line = __builtin_LINE());
    cere_status drain();  // retires every submit in flight
    // The scratch buffers init allocates for max_rows M and logit_rows LR (one source for init and scratch_bytes).
    static std::vector<std::pair<uint64_t Scratch::*, uint64_t>> scratch_plan(uint64_t M, uint64_t LR);
    cere_status bar(int line = __builtin_LINE());
    cere_status visit_cut(uint32_t M, uint32_t pos0, uint32_t v);
    cere_status fail(cere_status st);
    cere_status dense(uint64_t x, uint64_t w, uint64_t y, uint32_t M, uint32_t n, uint32_t k);
    cere_status conv(Session &s, uint32_t v, int c, uint64_t x, uint64_t w, uint64_t dyn, uint64_t out, uint32_t M,
                     uint32_t C, bool dynamic);
    cere_status rms(vki::Tensor x, uint64_t w, uint64_t out, uint32_t M);
    cere_status layer(Session &s, uint32_t v, const std::vector<uint64_t> &B, uint64_t x, uint32_t M, uint32_t pos0,
                      const uint64_t *prev);

    vk::Context &ctx_;
    Options opt_;
    bool ready_ = false, open_ = false;
    Layer L_[kLayers];
    uint64_t embed_ = 0, embed_norm_ = 0, norm_ = 0, unembed_ = 0, out_q_ = 0, cross_q_ = 0;  // cross_q FP32 [12][4][H]
    uint64_t weight_bytes_ = 0;
    uint64_t out_[kVisits] = {};
    Scratch s_{};
    uint64_t k2_ws_ = 0;
    std::vector<std::pair<uint64_t, uint64_t>> winfo_;  // K2 row info per weight address
    std::vector<vk::Buffer> owned_;
    std::map<int, std::pair<double, uint64_t>> prof_;
    std::vector<int> stamp_lines_;  // prof without split: the bar() line of each stamp of the open command buffer
};

}  // namespace cere::vkink
