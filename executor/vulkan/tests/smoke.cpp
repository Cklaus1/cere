// VulkanAmd P3 T2: smoke test of libcere_executor_vulkan.so, loaded as cered loads it (dlopen, ABI version first),
// as executor/hip/tests/smoke.cpp does for the HIP library. Rows print PASS / FAIL; exit 1 on any FAIL.
//
// Usage: smoke LIBRARY HEADER EXPORTS_MAP
//   LIBRARY      build/vulkan/libcere_executor_vulkan.so
//   HEADER       executor/include/cere_executor.h (the ABI's function declarations: the expected export set)
//   EXPORTS_MAP  executor/cuda/src/host/exports.map (its global patterns filter the header's functions)
// Rows: exported symbol set == the header's ABI functions matching the version script; ABI version; device id grammar
// (open "0", refuse others without killing the process); model-create refusals through the real ABI (incomplete model,
// out-of-scope plans, levers, budget); handle misuse. The GPU is opened (Vulkan device 0); no model memory is kept.
#include <dlfcn.h>
#include <elf.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "cere_executor.h"
#include "host/support.h"

namespace {

int g_fail = 0, g_pass = 0;
void row(const std::string &name, bool ok, const std::string &detail = "") {
    std::printf("%-64s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    (ok ? g_pass : g_fail)++;
}

std::string slurp(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// Defined GLOBAL symbols of the ELF's .dynsym (type FUNC and the rest), by parsing the file.
bool exported(const std::string &path, std::set<std::string> *funcs, std::set<std::string> *others) {
    const std::string b = slurp(path);
    if (b.size() < sizeof(Elf64_Ehdr) || std::memcmp(b.data(), ELFMAG, SELFMAG) != 0) return false;
    const auto *eh = reinterpret_cast<const Elf64_Ehdr *>(b.data());
    const auto *sh = reinterpret_cast<const Elf64_Shdr *>(b.data() + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; ++i) {
        if (sh[i].sh_type != SHT_DYNSYM) continue;
        const Elf64_Shdr &str = sh[sh[i].sh_link];
        const auto *sym = reinterpret_cast<const Elf64_Sym *>(b.data() + sh[i].sh_offset);
        for (size_t k = 0; k < sh[i].sh_size / sizeof(Elf64_Sym); ++k) {
            const unsigned bind = ELF64_ST_BIND(sym[k].st_info), type = ELF64_ST_TYPE(sym[k].st_info);
            if (sym[k].st_shndx == SHN_UNDEF || (bind != STB_GLOBAL && bind != STB_WEAK)) continue;
            const std::string name = b.data() + str.sh_offset + sym[k].st_name;
            (type == STT_FUNC ? funcs : others)->insert(name);
        }
        return true;
    }
    return false;
}

// The ABI functions: every `cere_status|uint32_t cere_x(` declaration at the start of a header line whose name matches
// a `cere_*_*;` global pattern of the version script.
std::set<std::string> abi_functions(const std::string &header, const std::string &map) {
    std::vector<std::string> prefixes;
    {
        std::istringstream in(slurp(map));
        std::string line;
        bool global = false;
        while (std::getline(in, line)) {
            if (line.find("global:") != std::string::npos) global = true;
            if (line.find("local:") != std::string::npos) global = false;
            const size_t star = line.find('*');
            const size_t c = line.find("cere_");
            if (global && c != std::string::npos && star != std::string::npos && star > c) {
                prefixes.push_back(line.substr(c, star - c));
            }
        }
    }
    std::set<std::string> out;
    std::istringstream in(slurp(header));
    std::string line;
    while (std::getline(in, line)) {
        for (const char *ret : {"cere_status ", "uint32_t "}) {
            if (line.rfind(ret, 0) != 0) continue;
            const size_t a = std::strlen(ret), p = line.find('(', a);
            if (p == std::string::npos) continue;
            const std::string name = line.substr(a, p - a);
            if (name.find(' ') != std::string::npos || name.rfind("cere_", 0) != 0) continue;
            for (const std::string &pre : prefixes) {
                if (name.rfind(pre, 0) == 0) out.insert(name);
            }
        }
    }
    return out;
}

template <typename T>
T sym(void *lib, const char *name) {
    return reinterpret_cast<T>(dlsym(lib, name));
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: smoke LIBRARY HEADER EXPORTS_MAP\n");
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    row("dlopen", lib != nullptr, lib ? "" : dlerror());
    if (lib == nullptr) return 1;

    // ---- symbols ----
    std::set<std::string> funcs, others;
    const bool parsed = exported(argv[1], &funcs, &others);
    const std::set<std::string> want = abi_functions(argv[2], argv[3]);
    std::string extra, missing;
    for (const std::string &f : funcs) {
        if (!want.count(f)) extra += " " + f;
    }
    for (const std::string &f : want) {
        if (!funcs.count(f)) missing += " " + f;
    }
    row("exported functions == ABI functions of cere_executor.h", parsed && extra.empty() && missing.empty() && !want.empty(),
        std::to_string(funcs.size()) + " exported, " + std::to_string(want.size()) + " in the ABI" +
            (extra.empty() ? "" : "; extra:" + extra) + (missing.empty() ? "" : "; missing:" + missing));
    std::string other_names;
    for (const std::string &o : others) other_names += " " + o;
    row("no exported non-function symbols", parsed && others.empty(), other_names);

    auto abi = sym<uint32_t (*)()>(lib, "cere_executor_abi_version");
    auto create = sym<cere_status (*)(cere_executor **, const cere_executor_options *)>(lib, "cere_executor_create");
    auto destroy = sym<cere_status (*)(cere_executor *)>(lib, "cere_executor_destroy");
    auto model_create = sym<cere_status (*)(cere_executor *, const cere_model_desc *, cere_model **)>(lib, "cere_model_create");
    auto model_release = sym<cere_status (*)(cere_executor *, cere_model *)>(lib, "cere_model_release");
    auto caps = sym<cere_status (*)(cere_executor *, const cere_model *, cere_execution_capabilities *)>(
        lib, "cere_executor_capabilities");
    auto session_release = sym<cere_status (*)(cere_executor *, cere_session *)>(lib, "cere_session_release");
    if (!abi || !create || !destroy || !model_create || !model_release || !caps || !session_release) {
        row("dlsym of the entry points", false);
        return 1;
    }
    row("cere_executor_abi_version == CERE_EXECUTOR_ABI_VERSION", abi() == CERE_EXECUTOR_ABI_VERSION,
        std::to_string(abi()));

    const auto make = [&](const char *id, uint64_t budget, cere_executor **out) {
        cere_executor_options o{};
        o.struct_size = sizeof o;
        o.abi_version = CERE_EXECUTOR_ABI_VERSION;
        o.device_id = id;
        o.worker_epoch = 1;
        o.memory_budget_bytes = budget;
        return create(out, &o);
    };

    // ---- device id grammar ----
    {
        cere_executor *e = nullptr;
        const cere_status st = make("0", 1ull << 30, &e);
        row("create(\"0\") OK", st == CERE_OK && e != nullptr, "status " + std::to_string(st));
        if (e) row("destroy(\"0\") OK", destroy(e) == CERE_OK);
    }
    const std::string share = "0,host_share=/dev/shm/cere,artifact=" + std::string(64, 'a');
    for (const std::string &id : std::vector<std::string>{"99", "7", "x", "vk:0", "hip:0", "0,release_device=1", share}) {
        cere_executor *e = nullptr;
        const cere_status st = make(id.c_str(), 1ull << 30, &e);
        row("create(\"" + (id.size() > 40 ? id.substr(0, 40) + "..." : id) + "\") UNSUPPORTED, process alive",
            st == CERE_ERR_UNSUPPORTED && !e,
            "status " + std::to_string(st));
        if (e) destroy(e);
    }
    for (const char *id : {"", "-1", "+0", "00", "0,bogus=1", "0,release_device=2", "0,flag", "1234567890"}) {
        cere_executor *e = nullptr;
        const cere_status st = make(id, 1ull << 30, &e);
        row(std::string("create(\"") + id + "\") ARGUMENT", st == CERE_ERR_ARGUMENT && !e, "status " + std::to_string(st));
        if (e) destroy(e);
    }

    // ---- model creation refusals (through the real ABI, on one executor) ----
    cere_executor *e = nullptr;
    if (make("0", uint64_t(100) << 30, &e) != CERE_OK || e == nullptr) {
        row("create(\"0\") for the model rows", false);
        return 1;
    }
    const cere_qwen4_exp_config config = cere::host::a4_support().config;
    // An incomplete A4 text model: its config and a few tensors with their real shapes (the layout accepts it; the
    // name binding refuses it: the forward could never run it).
    struct T {
        const char *name;
        uint32_t dtype, placement;
        std::vector<uint64_t> shape;
        uint64_t esize;
    };
    const std::vector<T> ts = {
        {"model.language_model.embed_tokens.weight", CERE_DTYPE_BF16, CERE_PLACEMENT_DEVICE, {248320, 2560}, 2},
        {"model.language_model.layers.0.attn_hyper_connection.hc_norm.weight", CERE_DTYPE_BF16, CERE_PLACEMENT_DEVICE,
         {10240}, 2},
        {"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", CERE_DTYPE_U8,
         CERE_PLACEMENT_HOST_PINNED, {1000, 80}, 1},
        {"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight_scale", CERE_DTYPE_F8_E4M3,
         CERE_PLACEMENT_HOST_PINNED, {1000, 10}, 1},
    };
    std::vector<cere_tensor_desc> tensors;
    for (const T &t : ts) {
        cere_tensor_desc d{};
        d.struct_size = sizeof d;
        d.abi_version = CERE_EXECUTOR_ABI_VERSION;
        d.name = t.name;
        d.dtype = t.dtype;
        d.placement = t.placement;
        d.ndim = uint32_t(t.shape.size());
        uint64_t n = 1;
        for (size_t i = 0; i < t.shape.size(); ++i) {
            d.shape[i] = t.shape[i];
            n *= t.shape[i];
        }
        d.bytes = n * t.esize;
        tensors.push_back(d);
    }
    const auto desc = [&]() {
        cere_model_desc d{};
        d.struct_size = sizeof d;
        d.abi_version = CERE_EXECUTOR_ABI_VERSION;
        d.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
        d.family = CERE_FAMILY_QWEN4_EXP;
        d.config = &config;
        d.tensors = tensors.data();
        d.tensor_count = tensors.size();
        d.max_sessions = 1;
        d.max_context_tokens = 32768;
        return d;
    };
    const auto expect = [&](const std::string &name, const cere_model_desc &d, cere_status want_st) {
        cere_model *m = nullptr;
        const cere_status st = model_create(e, &d, &m);
        row(name, st == want_st && m == nullptr, "status " + std::to_string(st));
        if (m) model_release(e, m);
    };
    expect("model: incomplete A4 text model UNSUPPORTED (binding)", desc(), CERE_ERR_UNSUPPORTED);
    {
        cere_model_desc d = desc();
        d.max_context_tokens = 2048;
        expect("model: max_context_tokens 2,048 UNSUPPORTED (OD-5)", d, CERE_ERR_UNSUPPORTED);
        d.max_context_tokens = 32769;
        expect("model: max_context_tokens 32,769 UNSUPPORTED", d, CERE_ERR_UNSUPPORTED);
    }
    {
        cere_model_desc d = desc();
        d.max_sessions = 2;
        expect("model: max_sessions 2 UNSUPPORTED", d, CERE_ERR_UNSUPPORTED);
    }
    {
        cere_model_desc d = desc();
        d.prefix_state_slots = 1;
        expect("model: prefix_state_slots 1 UNSUPPORTED", d, CERE_ERR_UNSUPPORTED);
    }
    {
        cere_model_desc d = desc();
        d.snapshot_host_bytes = 1 << 20;
        expect("model: snapshot arena UNSUPPORTED", d, CERE_ERR_UNSUPPORTED);
    }
    {
        static const uint32_t map[4] = {0, 1, 2, 3};
        cere_model_desc d = desc();
        d.speculative_draft_tokens = 4;
        d.draft_token_map = map;
        d.draft_token_map_count = 4;
        // T6: drafts are built; this desc has no mtp.* tensors, so the binding refuses it (as every incomplete model).
        expect("model: speculative_draft_tokens 4 without the MTP tensors UNSUPPORTED (binding)", d,
               CERE_ERR_UNSUPPORTED);
        d.speculative_draft_tokens = 3;
        expect("model: speculative_draft_tokens 3 refused (the ABI's values are 0 and 4)", d, CERE_ERR_UNSUPPORTED);
    }
    for (const auto &[slot, impl, label] : std::vector<std::tuple<uint32_t, uint32_t, const char *>>{
             {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_GRAPH, "decode_step graph"},
             {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M, "k2_gemm small_m"},
             {CERE_SLOT_VERIFY_ATTENTION, CERE_LEVER_VERIFY_BATCHED, "verify_attention batched"},
             {CERE_SLOT_K2_WEIGHTS, CERE_LEVER_K2_WEIGHTS_FP8, "k2_weights fp8"},
             {CERE_SLOT_MOE_GEMM, CERE_LEVER_MOE_P3B, "moe_gemm p3b"}}) {
        cere_slot_selection sel{};
        sel.struct_size = sizeof sel;
        sel.abi_version = CERE_EXECUTOR_ABI_VERSION;
        sel.block_version = CERE_SLOT_BLOCK_VERSION;
        sel.count = 1;
        sel.choices[0].slot = slot;
        sel.choices[0].impl = impl;
        cere_model_desc d = desc();
        d.selection = &sel;
        expect(std::string("model: lever ") + label + " UNSUPPORTED", d, CERE_ERR_UNSUPPORTED);
    }
    {
        cere_executor *small = nullptr;
        if (make("0", uint64_t(1) << 30, &small) == CERE_OK) {
            cere_model *m = nullptr;
            const cere_model_desc d = desc();
            const cere_status st = model_create(small, &d, &m);
            row("model: device budget 1 GiB < the plan's blocks CAPACITY", st == CERE_ERR_CAPACITY && !m,
                "status " + std::to_string(st));
            destroy(small);
        } else {
            row("model: small-budget executor", false);
        }
    }
    {
        cere_model *m = nullptr;
        const cere_model_desc d = desc();
        tensors[0].bytes += 2;  // bytes != shape x dtype
        const cere_status st = model_create(e, &d, &m);
        tensors[0].bytes -= 2;
        row("model: tensor bytes != shape ARGUMENT", st == CERE_ERR_ARGUMENT && !m, "status " + std::to_string(st));
    }
    expect("model: the refusals left the executor usable (incomplete model again)", desc(), CERE_ERR_UNSUPPORTED);

    // ---- handle misuse (the host layer's rules) ----
    {
        cere_execution_capabilities c{};
        c.struct_size = sizeof c;
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        row("capabilities of an unknown model ARGUMENT", caps(e, reinterpret_cast<const cere_model *>(&c), &c) ==
                                                              CERE_ERR_ARGUMENT);
        row("model_release of an unknown model ARGUMENT",
            model_release(e, reinterpret_cast<cere_model *>(&c)) == CERE_ERR_ARGUMENT);
        row("session_release of an unknown session ARGUMENT",
            session_release(e, reinterpret_cast<cere_session *>(&c)) == CERE_ERR_ARGUMENT);
        row("destroy(null) ARGUMENT", destroy(nullptr) == CERE_ERR_ARGUMENT);
    }
    row("destroy(executor) OK", destroy(e) == CERE_OK);
    std::printf("smoke: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
