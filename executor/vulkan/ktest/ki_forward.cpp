// Inkling S11: the Inkling 2 Mini text forward on Vulkan (src/device/inkling/ki_model.h, inkling design note 3, 4.2,
// G3 / G5), standalone: it loads the S0 artifact's safetensors (exact checkpoint bytes), runs one prompt's prefill in
// chunks and then a teacher-forced decode, and writes the logits for executor/vulkan/tools/ki_g3_compare.py.
//
//   ki_forward SPV_DIR PACK_DIR PROMPT_IDS OUT [--forced IDS] [--chunk N] [--ctx N] [--k2] [--mutant NAME] [--keep N]
//              [--last-row] [--attn auto|split|tiled] [--msg auto|row|tiled] [--in-flight N]
//   PROMPT_IDS / IDS: raw little-endian int32 token ids. OUT: float32 rows of V = 200,058 softcapped logits, first one
//   row per prompt position, then one row per forced id (the logits after feeding it: row i predicts forced[i + 1]).
//   --k2: dense projections on K2 (AMD's fast path; G5 compares logits only).
//   --mutant softmax | globalscale | nomsg: G3's fail-first mutants (ki_model.h Options; each must FAIL G3);
//   --keep N: OUT holds only the last N prompt rows (then the forced rows), and OUT.amax every prompt row's maxima:
//   per row int32 count, then that many int32 ids (ascending) whose logit equals the row max (long prompts: p07).
//   --last-row: cered's prefill path (the AMD prefill bar, inkling design note D-I-EPB): each multi-row step scores
//   only its last row; OUT holds the last prompt row, then the forced rows.
//   --attn / --msg / --in-flight (D-I-EPB (a2) per-item pairs; the library uses auto, auto, vkink::kSubmitsInFlight):
//   the attention route, the message gate route, and the command buffers in flight (1: every submit waits).
// Env: KI_TRACE=1 every barrier becomes submit_wait + begin and each recorded op is printed (a device fault, on
// lavapipe an out-of-bounds shader access is a host SIGSEGV, is pinned to the ops since the last barrier);
// KI_DUMP=path every visit's output rows, BF16 [48][M][H] per step (G2 against S1's taps: ki_g3_compare.py g2).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "device/inkling/ki_model.h"
#include "ki_pack.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"

namespace vk = cere::vk;
namespace vkink = cere::vkink;
using kipack::Pack;
using kipack::Weights;

namespace {
using kipack::die;

std::vector<int32_t> read_ids(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open " + path);
    std::vector<int32_t> v;
    int32_t x;
    while (std::fread(&x, 4, 1, f) == 1) v.push_back(x);
    std::fclose(f);
    return v;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::printf("usage: ki_forward SPV_DIR PACK_DIR PROMPT_IDS OUT [--forced IDS] [--chunk N] [--ctx N] [--k2] "
                    "[--mutant softmax|globalscale|nomsg] [--keep N] [--last-row] [--attn auto|split|tiled] "
                    "[--msg auto|row|tiled] [--in-flight N]\n");
        return 2;
    }
    std::string forced_path;
    uint32_t chunk = 512, max_ctx = 4096, keep = 0, in_flight = vkink::kSubmitsInFlight;
    bool last_row = false;
    vkink::Forward::Options opt;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--forced" && i + 1 < argc) forced_path = argv[++i];
        else if (a == "--chunk" && i + 1 < argc) chunk = uint32_t(std::atoi(argv[++i]));
        else if (a == "--ctx" && i + 1 < argc) max_ctx = uint32_t(std::atoi(argv[++i]));
        else if (a == "--k2") opt.k2 = true;
        else if (a == "--keep" && i + 1 < argc) keep = uint32_t(std::atoi(argv[++i]));
        else if (a == "--last-row") last_row = true;
        else if (a == "--in-flight" && i + 1 < argc) in_flight = uint32_t(std::atoi(argv[++i]));
        else if (a == "--attn" && i + 1 < argc) {
            const std::string r = argv[++i];
            if (r == "auto") opt.attn_route = cere::vki::AttnRoute::kAuto;
            else if (r == "split") opt.attn_route = cere::vki::AttnRoute::kSplit;
            else if (r == "tiled") opt.attn_route = cere::vki::AttnRoute::kTiled;
            else die("unknown --attn " + r);
        } else if (a == "--msg" && i + 1 < argc) {
            const std::string r = argv[++i];
            if (r == "auto") opt.msg_route = cere::vki::MsgRoute::kAuto;
            else if (r == "row") opt.msg_route = cere::vki::MsgRoute::kRow;
            else if (r == "tiled") opt.msg_route = cere::vki::MsgRoute::kTiled;
            else die("unknown --msg " + r);
        } else if (a == "--mutant" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "softmax") opt.moe_mutant = cere::vki::Mutant::kSoftmaxRouting;
            else if (m == "globalscale") opt.moe_mutant = cere::vki::Mutant::kGlobalActScale;
            else if (m == "nomsg") opt.no_messages = true;
            else die("unknown mutant " + m);
        } else {
            die("unknown argument " + a);
        }
    }
    const std::vector<int32_t> prompt = read_ids(argv[3]);
    const std::vector<int32_t> forced = forced_path.empty() ? std::vector<int32_t>{} : read_ids(forced_path);
    if (prompt.empty() || prompt.size() + forced.size() > max_ctx) die("prompt empty or longer than --ctx");
    for (const std::vector<int32_t> *ids : {&prompt, &forced})
        for (int32_t id : *ids)
            if (id < 0 || id >= int32_t(vkink::kVocab)) die("token id out of range");
    if (chunk == 0 || chunk > vkink::kMaxRows) die("--chunk must be 1 .. 4096");
    opt.max_rows = chunk;
    opt.logit_rows = std::min<uint32_t>(chunk, 512);  // scored in pieces (a 4,096-row block is 3.3 GB of logits)
    opt.trace = std::getenv("KI_TRACE") != nullptr;
    opt.prof = std::getenv("KI_PROF") != nullptr;  // KI_PROF=2: per submit only (barriers stay barriers)
    opt.prof_split = !(opt.prof && std::strcmp(std::getenv("KI_PROF"), "2") == 0);
    if (const char *o = std::getenv("KI_VPS")) opt.visits_per_submit = uint32_t(std::max(1, std::atoi(o)));
    FILE *dump = nullptr;
    if (const char *d = std::getenv("KI_DUMP"))
        if (!(dump = std::fopen(d, "wb"))) die(std::string("cannot write ") + d);

    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    auto ctx = vk::testing::create_or_die(argv[1], 0, in_flight);
    std::printf("device: %s\n", ctx->info().name.c_str());
    Pack pk;
    pk.open(std::string(argv[2]) + "/model-00001-of-00001.safetensors");
    Weights w{*ctx, pk};
    double t0 = now_s();
    for (const std::string &n : vkink::text_tensor_names()) w.load(n);
    std::printf("weights: %.2f GB in %.1f s\n", double(w.total) / 1e9, now_s() - t0);

    auto fwd = std::make_unique<vkink::Forward>(*ctx);
    std::string why;
    const vkink::Find find = w.find();
    if (const cere_status st = fwd->init(find, opt, &why); st != CERE_OK)
        die("init failed (" + std::to_string(int(st)) + "): " + why + " " + ctx->last_error());
    std::unique_ptr<vkink::Session> s;
    if (fwd->create_session(max_ctx, &s) != CERE_OK) die("create_session: " + ctx->last_error());

    FILE *out = std::fopen(argv[4], "wb");
    if (!out) die("cannot write output");
    FILE *amax = nullptr;
    if (keep && !(amax = std::fopen((std::string(argv[4]) + ".amax").c_str(), "wb"))) die("cannot write OUT.amax");
    const size_t T = prompt.size();
    auto run = [&](const int32_t *ids, uint32_t M, uint32_t pos0) {
        if (fwd->step(*s, reinterpret_cast<const uint32_t *>(ids), M, pos0) != CERE_OK) die("step: " + ctx->last_error());
        const bool lastrow = last_row && M > 1;
        for (uint32_t f = lastrow ? M - 1 : 0; f < M; f += opt.logit_rows) {
            const uint32_t nr = std::min(opt.logit_rows, M - f);
            if (fwd->logits(f, nr) != CERE_OK) die("logits: " + ctx->last_error());
            const float *lg = fwd->logits_host();
            for (uint32_t r = f; r < f + nr; ++r) {
                const size_t row = size_t(pos0) + r;
                const float *x = lg + size_t(r - f) * vkink::kVocab;
                if (amax && row < T) {
                    const float mx = *std::max_element(x, x + vkink::kVocab);
                    std::vector<int32_t> ties;
                    for (uint32_t c = 0; c < vkink::kVocab; ++c)
                        if (x[c] == mx) ties.push_back(int32_t(c));
                    const int32_t n = int32_t(ties.size());
                    std::fwrite(&n, 4, 1, amax);
                    std::fwrite(ties.data(), 4, ties.size(), amax);
                }
                if (lastrow ? row + 1 == T : (!keep || row >= T || row + keep >= T))
                    std::fwrite(x, 4, vkink::kVocab, out);
            }
        }
        if (dump)
            for (uint32_t v = 0; v < vkink::kVisits; ++v)
                std::fwrite(fwd->visit_out_host(v), 2, size_t(M) * vkink::kHidden, dump);
    };
    // KI_PROF: GPU ms per ki_model.cpp barrier / submit line (the op group it closes), prefill then decode.
    auto prof = [&](const char *phase, size_t steps) {
        if (!opt.prof || steps == 0) return;
        double tot = 0;
        for (const auto &e : fwd->prof()) tot += e.second.first;
        std::printf("prof %s: GPU %.2f ms per step over %zu steps\n", phase, tot / double(steps), steps);
        for (const auto &e : fwd->prof())
            std::printf("prof %s line %4d: %8.3f ms/step %5.1f %% (%llu submits)\n", phase, e.first,
                        e.second.first / double(steps), 100.0 * e.second.first / tot,
                        static_cast<unsigned long long>(e.second.second));
        fwd->prof_reset();
    };
    t0 = now_s();
    for (uint32_t p0 = 0; p0 < prompt.size(); p0 += chunk)
        run(&prompt[p0], std::min<uint32_t>(chunk, uint32_t(prompt.size()) - p0), p0);
    const double tp = now_s() - t0;
    prof("prefill", (prompt.size() + chunk - 1) / chunk);
    t0 = now_s();
    for (size_t i = 0; i < forced.size(); ++i) run(&forced[i], 1, uint32_t(prompt.size() + i));
    const double td = now_s() - t0;
    prof("decode", forced.size());
    std::fclose(out);
    if (amax) std::fclose(amax);
    if (dump) std::fclose(dump);
    std::printf("prefill %zu tokens %.2f s, decode %zu steps %.2f s (%.2f tok/s)\n", prompt.size(), tp, forced.size(), td,
                forced.empty() ? 0.0 : double(forced.size()) / td);
    s.reset();
    fwd.reset();
    return 0;
}
