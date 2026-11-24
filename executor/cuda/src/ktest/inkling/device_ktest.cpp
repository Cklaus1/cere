// Inkling 2 Mini S5: the CUDA device (device/inkling/inkling_device.cpp) end to end through the ABI v5 runtime of
// libcere_executor (cere_executor_create, cere_model_create / put / seal, cere_session_create, one PREFILL work per
// prompt submitted and polled to its terminal completion), for executor/tests/gpu/inkling_forward.py rows `device`
// (its greedy ids must equal cere_ktest_inkling_generate's: the forward without the device and host scheduler),
// `sampling` (S6g: the sampler and logprobs) and `perf` (per-prompt wall times on one sealed model: decode tok/s as
// the difference of two token budgets); and the device's KV-byte admission (S10, row `admission`: pool fill,
// CAPACITY, release).
// Tensors, names and meta as forward_ktest.cpp: the caller's `model.llm.*` checkpoint tensors on GPU0, copied to the
// host one at a time and put into the model (a second copy of the weights on the device).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cere_executor.h"
#include "cere_model_constants.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

// Bit i set when layer i of the compiled descriptor's per-layer list equals `value` (host/api.cpp's layer_mask).
template <size_t N> uint32_t layer_mask(const char *const (&layers)[N], const char *value) {
    uint32_t mask = 0;
    for (size_t i = 0; i < N; ++i) {
        if (std::strcmp(layers[i], value) == 0) mask |= uint32_t(1) << i;
    }
    return mask;
}

// The config the library was compiled for: what the runtime fills from config.json (host/api.cpp kInkling2Config).
cere_inkling2_config compiled_config() {
    namespace g = cere::gen;
    return cere_inkling2_config{
        sizeof(cere_inkling2_config), CERE_EXECUTOR_ABI_VERSION, g::kVocabSize, g::kUnpaddedVocabSize, g::kHiddenSize,
        g::kNumHiddenLayers, g::kNumAttentionHeads, g::kNumKeyValueHeads, g::kHeadDim, g::kMoeIntermediateSize,
        g::kNumExperts, g::kNumExpertsPerTok, g::kSlidingWindowSize, g::kRelativePositionRelExtent,
        g::kRelativePositionDRel, g::kLoopCoreStart, g::kLoopCoreEnd, g::kLoopNumPasses, g::kLoopGateRank,
        g::kNLayersPerAttnResBlock, g::kConvKernelSize, g::kEosTokenId,
        layer_mask(g::kLayerMixer, "sliding_attention"), layer_mask(g::kLayerExperts, "mxfp8"), g::kRmsNormEps,
        g::kFinalLogitSoftcapping, g::kRouteScale, g::kMaxPositionEmbeddings};
}

uint64_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case CERE_DTYPE_F32: return 4;
    case CERE_DTYPE_BF16: return 2;
    case CERE_DTYPE_F8_E4M3:
    case CERE_DTYPE_U8: return 1;
    default: return 0;
    }
}

// The executor, its model and the sessions of one call; released in reverse order.
struct Abi {
    cere_executor *executor = nullptr;
    cere_model *model = nullptr;
    ~Abi() {
        if (model != nullptr) cere_model_release(executor, model);
        if (executor != nullptr) cere_executor_destroy(executor);
    }
};

// One request on a fresh session: PREFILL of the prompt with max_new_tokens, polled to terminal. `sampling` null:
// greedy; `logprobs` null: none (else [gen] entries with `top_logprobs` top entries each).
cere_status generate(const Abi &abi, const cere_execution_capabilities &caps, uint64_t context, const uint32_t *prompt,
                     uint64_t len, uint64_t gen, const cere_sampling_params *sampling, uint32_t top_logprobs,
                     uint32_t *out, cere_token_score *scores, cere_token_logprobs *logprobs, int64_t *result) {
    cere_session *session = nullptr;
    if (const cere_status st = cere_session_create(abi.executor, abi.model, context, &session); st != CERE_OK) {
        return st;
    }
    cere_work work{};
    work.struct_size = sizeof(work);
    work.abi_version = CERE_EXECUTOR_ABI_VERSION;
    work.operation = CERE_OP_PREFILL;
    work.expected_worker_epoch = caps.worker_epoch;
    work.session = session;
    work.tokens = prompt;
    work.token_count = len;
    work.max_new_tokens = gen;
    work.out_tokens = out;
    work.out_capacity = gen;
    work.out_scores = scores;
    work.sampling = sampling;
    work.out_logprobs = logprobs;
    work.top_logprobs = logprobs != nullptr ? top_logprobs : 0;
    cere_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
    batch.expected_worker_epoch = caps.worker_epoch;
    batch.capability_profile_generation = caps.profile_generation;
    batch.items = &work;
    batch.item_count = 1;
    cere_completion_ticket ticket{};
    cere_status st = cere_executor_submit(abi.executor, &batch, &ticket);
    cere_completion done{};
    while (st == CERE_OK) {
        done = cere_completion{};
        done.struct_size = sizeof(done);
        done.abi_version = CERE_EXECUTOR_ABI_VERSION;
        const cere_status p = cere_executor_poll(abi.executor, ticket, &done);
        if (p == CERE_OK) break;
        if (p != CERE_PENDING) st = p;
    }
    const cere_status released = cere_session_release(abi.executor, session);
    if (st != CERE_OK) return st;
    result[0] = int64_t(done.committed_tokens);
    result[1] = int64_t(done.finish);
    if (done.finish != CERE_FINISH_LENGTH && done.finish != CERE_FINISH_STOP_TOKEN) {
        return done.status != CERE_OK ? done.status : CERE_ERR_INTERNAL;
    }
    return released;
}

// The tensors' descriptors (CERE_PLACEMENT_DEVICE; names and meta as forward_ktest.cpp) and the largest one's bytes.
cere_status describe(const void *names, const void *meta, int64_t count, std::vector<cere_tensor_desc> *descs,
                     uint64_t *largest) {
    const auto *dims = static_cast<const int64_t *>(meta);
    const char *name = static_cast<const char *>(names);
    descs->assign(static_cast<size_t>(count), cere_tensor_desc{});
    *largest = 0;
    for (int64_t i = 0; i < count; ++i) {
        cere_tensor_desc &t = (*descs)[size_t(i)];
        const int64_t *d = dims + 6 * i;
        if (d[1] < 0 || d[1] > 4 || dtype_size(uint32_t(d[0])) == 0) return CERE_ERR_ARGUMENT;
        t.struct_size = sizeof(t);
        t.abi_version = CERE_EXECUTOR_ABI_VERSION;
        t.name = name;
        name += std::strlen(name) + 1;
        t.dtype = uint32_t(d[0]);
        t.placement = CERE_PLACEMENT_DEVICE;
        t.ndim = uint32_t(d[1]);
        t.bytes = dtype_size(t.dtype);
        for (uint32_t j = 0; j < t.ndim; ++j) {
            t.shape[j] = uint64_t(d[2 + j]);
            t.bytes *= t.shape[j];
        }
        *largest = std::max(*largest, t.bytes);
    }
    return CERE_OK;
}

// An executor on GPU0 with `budget` bytes and its Inkling model (max_sessions `sessions`, max_context_tokens
// `context`, kv_pool_tokens `pool`, 0 = the derived pool, speculative_draft_tokens `window`: 0 = no drafter, else
// the MTP drafter's tensors are among the descriptors) created from the descriptors: CERE_ERR_CAPACITY when the
// device's estimate exceeds the budget. With `ptrs`, the tensors are put (copied to the host one at a time) and the
// model sealed; `used` = device bytes in use after the seal, `caps` = the model's capabilities.
cere_status open_model(const std::vector<cere_tensor_desc> &descs, uint64_t largest, const uint64_t *ptrs,
                       uint64_t sessions, uint64_t context, uint64_t pool, uint64_t budget, Abi *abi,
                       cere_execution_capabilities *caps, int64_t *used, uint32_t window = 0) {
    const cere_inkling2_config config = compiled_config();
    const uint32_t stop = cere::gen::kEosTokenId;
    cere_model_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
    desc.model_kind = CERE_MODEL_INKLING2_NVFP4;
    desc.config = reinterpret_cast<const cere_qwen4_exp_config *>(&config);
    desc.tensors = descs.data();
    desc.tensor_count = uint64_t(descs.size());
    desc.max_sessions = sessions;
    desc.max_context_tokens = context;
    desc.kv_pool_tokens = pool;
    desc.stop_token_ids = &stop;
    desc.stop_token_count = 1;
    desc.family = CERE_FAMILY_INKLING2;
    desc.speculative_draft_tokens = window;

    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = "0";
    options.worker_epoch = 1;
    options.memory_budget_bytes = budget;
    if (const cere_status st = cere_executor_create(&abi->executor, &options); st != CERE_OK) return st;
    if (const cere_status st = cere_model_create(abi->executor, &desc, &abi->model); st != CERE_OK) return st;
    if (ptrs == nullptr) return CERE_OK;
    std::vector<uint8_t> host(largest);
    for (size_t i = 0; i < descs.size(); ++i) {
        const cere_tensor_desc &t = descs[i];
        if (cudaMemcpy(host.data(), reinterpret_cast<const void *>(ptrs[i]), t.bytes, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        if (const cere_status st = cere_model_put(abi->executor, abi->model, uint64_t(i), 0, host.data(), t.bytes);
            st != CERE_OK) {
            return st;
        }
    }
    host = std::vector<uint8_t>();
    if (const cere_status st = cere_model_seal(abi->executor, abi->model); st != CERE_OK) return st;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return CERE_ERR_INTERNAL;
    *used = int64_t(total_b - free_b);
    caps->struct_size = sizeof(*caps);
    caps->abi_version = CERE_EXECUTOR_ABI_VERSION;
    return cere_executor_capabilities(abi->executor, abi->model, caps);
}

}  // namespace

// The device's generation of `n` prompts (uint32 host, back to back; lens int64 host [n]), one fresh session each,
// gen tokens per prompt or gens[p] (int64 host [n], nullable, each in [1, gen]) (stop token eos 200006 as the runtime
// passes it): out_ids uint32 host [n, gen] (0 past a stop or the prompt's budget), out_result int64 host [n, 2] =
// (committed tokens, CERE_FINISH_*), out_bytes int64 host [1] = device bytes in use after the model's seal,
// out_seconds f64 host [n] (nullable) = each prompt's wall time from session create to release (prefill and decode;
// load and seal excluded). Executor budget `budget` bytes, model max_context_tokens `context`, max_sessions 1.
// params: null = greedy, else double host [8] = (temperature, top_p, min_p, top_k, frequency_penalty,
// presence_penalty, repetition_penalty, seed) for every prompt (prompt p's seed is seed + p). out_logprobs: null, or
// cere_token_logprobs host [n, gen] with `top_logprobs` top entries each (needs params).
CERE_KTEST int cere_ktest_inkling_device_generate(const void *tensors, const void *names, const void *meta,
                                                  int64_t count, const void *prompts, const void *lens, int64_t n,
                                                  int64_t gen, const void *gens, int64_t context, int64_t budget,
                                                  const void *params, int64_t top_logprobs, void *out_ids,
                                                  void *out_result, void *out_bytes, void *out_logprobs,
                                                  void *out_seconds, cudaStream_t stream) {
    if (tensors == nullptr || names == nullptr || meta == nullptr || count <= 0 || prompts == nullptr ||
        lens == nullptr || n < 1 || gen < 1 || context < 1 || budget < 1 || out_ids == nullptr ||
        out_result == nullptr || out_bytes == nullptr || top_logprobs < 0 || top_logprobs > CERE_MAX_TOP_LOGPROBS ||
        (out_logprobs != nullptr && params == nullptr) || (out_logprobs == nullptr && top_logprobs != 0)) {
        return CERE_ERR_ARGUMENT;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CERE_ERR_INTERNAL;
    std::vector<cere_tensor_desc> descs;
    uint64_t largest = 0;
    if (const cere_status st = describe(names, meta, count, &descs, &largest); st != CERE_OK) return st;
    Abi abi;
    cere_execution_capabilities caps{};
    if (const cere_status st = open_model(descs, largest, static_cast<const uint64_t *>(tensors), 1, uint64_t(context),
                                          0, uint64_t(budget), &abi, &caps, static_cast<int64_t *>(out_bytes));
        st != CERE_OK) {
        return st;
    }
    const auto *tok = static_cast<const uint32_t *>(prompts);
    const auto *len = static_cast<const int64_t *>(lens);
    auto *ids = static_cast<uint32_t *>(out_ids);
    auto *res = static_cast<int64_t *>(out_result);
    const auto *budgets = static_cast<const int64_t *>(gens);
    auto *seconds = static_cast<double *>(out_seconds);
    auto *lp = static_cast<cere_token_logprobs *>(out_logprobs);
    std::fill(ids, ids + n * gen, 0u);
    if (lp != nullptr) std::fill(lp, lp + n * gen, cere_token_logprobs{});
    std::vector<cere_token_score> scores(static_cast<size_t>(gen));
    cere_sampling_params sp{};
    const auto *pv = static_cast<const double *>(params);
    if (pv != nullptr) {
        sp.struct_size = sizeof(sp);
        sp.abi_version = CERE_EXECUTOR_ABI_VERSION;
        sp.temperature = float(pv[0]);
        sp.top_p = float(pv[1]);
        sp.min_p = float(pv[2]);
        sp.top_k = uint32_t(pv[3]);
        sp.frequency_penalty = float(pv[4]);
        sp.presence_penalty = float(pv[5]);
        sp.repetition_penalty = float(pv[6]);
    }
    for (int64_t p = 0; p < n; ++p) {
        const int64_t g = budgets != nullptr ? budgets[p] : gen;
        if (len[p] < 1 || g < 1 || g > gen || len[p] + g > context) return CERE_ERR_ARGUMENT;
        if (pv != nullptr) sp.seed = uint64_t(pv[7]) + uint64_t(p);
        const auto t0 = std::chrono::steady_clock::now();
        if (const cere_status st = generate(abi, caps, uint64_t(context), tok, uint64_t(len[p]), uint64_t(g),
                                            pv != nullptr ? &sp : nullptr, uint32_t(top_logprobs), ids + p * gen,
                                            scores.data(), lp != nullptr ? lp + p * gen : nullptr, res + 2 * p);
            st != CERE_OK) {
            return st;
        }
        if (seconds != nullptr) {
            seconds[p] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        tok += len[p];
    }
    return CERE_OK;
}

// S9 phase B (inkling design note D-I-S9B): greedy generation as cere_ktest_inkling_device_generate (stop token
// eos, out_ids, out_result and out_bytes the same), on a model with speculative_draft_tokens `window` (0 = none, the
// plain path; else 2..8, the drafter's mtp.safetensors tensors among the caller's) and max_sessions `concurrent`:
// the prompts run in groups of `concurrent` in order, every prompt of a group on its own session submitted together
// (one ticket each) and polled round robin to terminal, so the host batches their rounds. out_seconds f64 host [n]:
// each prompt's wall from its group's first session create to its terminal poll. out_stats int64 host [n, 2]
// (nullable): its completion's (verify_steps, accepted_drafts).
CERE_KTEST int cere_ktest_inkling_device_generate_draft(const void *tensors, const void *names, const void *meta,
                                                        int64_t count, const void *prompts, const void *lens,
                                                        int64_t n, int64_t gen, const void *gens, int64_t context,
                                                        int64_t budget, int64_t window, int64_t concurrent,
                                                        void *out_ids, void *out_result, void *out_bytes,
                                                        void *out_seconds, void *out_stats, cudaStream_t stream) {
    if (tensors == nullptr || names == nullptr || meta == nullptr || count <= 0 || prompts == nullptr ||
        lens == nullptr || n < 1 || gen < 1 || context < 1 || budget < 1 || window < 0 || window > 8 ||
        concurrent < 1 || concurrent > 64 || out_ids == nullptr || out_result == nullptr || out_bytes == nullptr ||
        out_seconds == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CERE_ERR_INTERNAL;
    std::vector<cere_tensor_desc> descs;
    uint64_t largest = 0;
    if (const cere_status st = describe(names, meta, count, &descs, &largest); st != CERE_OK) return st;
    Abi abi;
    cere_execution_capabilities caps{};
    if (const cere_status st = open_model(descs, largest, static_cast<const uint64_t *>(tensors),
                                          uint64_t(concurrent), uint64_t(context), 0, uint64_t(budget), &abi, &caps,
                                          static_cast<int64_t *>(out_bytes), uint32_t(window));
        st != CERE_OK) {
        return st;
    }
    const auto *len = static_cast<const int64_t *>(lens);
    const auto *budgets = static_cast<const int64_t *>(gens);
    auto *ids = static_cast<uint32_t *>(out_ids);
    auto *res = static_cast<int64_t *>(out_result);
    auto *seconds = static_cast<double *>(out_seconds);
    auto *stats = static_cast<int64_t *>(out_stats);
    std::fill(ids, ids + n * gen, 0u);
    std::vector<const uint32_t *> start(static_cast<size_t>(n));
    const auto *tok = static_cast<const uint32_t *>(prompts);
    for (int64_t p = 0; p < n; ++p) {
        const int64_t g = budgets != nullptr ? budgets[p] : gen;
        if (len[p] < 1 || g < 1 || g > gen || len[p] + g > context) return CERE_ERR_ARGUMENT;
        start[size_t(p)] = tok;
        tok += len[p];
    }
    struct Item {
        cere_session *session = nullptr;
        cere_work work{};
        cere_completion_ticket ticket{};
        bool done = false;
    };
    for (int64_t p0 = 0; p0 < n; p0 += concurrent) {
        const int64_t group = std::min(concurrent, n - p0);
        std::vector<Item> items(static_cast<size_t>(group));
        cere_status st = CERE_OK;
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < group && st == CERE_OK; ++i) {
            const int64_t p = p0 + i;
            Item &it = items[size_t(i)];
            st = cere_session_create(abi.executor, abi.model, uint64_t(context), &it.session);
            if (st != CERE_OK) break;
            cere_work &w = it.work;
            w.struct_size = sizeof(w);
            w.abi_version = CERE_EXECUTOR_ABI_VERSION;
            w.operation = CERE_OP_PREFILL;
            w.expected_worker_epoch = caps.worker_epoch;
            w.session = it.session;
            w.tokens = start[size_t(p)];
            w.token_count = uint64_t(len[p]);
            w.max_new_tokens = uint64_t(budgets != nullptr ? budgets[p] : gen);
            w.out_tokens = ids + p * gen;
            w.out_capacity = uint64_t(gen);
            cere_batch batch{};
            batch.struct_size = sizeof(batch);
            batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
            batch.expected_worker_epoch = caps.worker_epoch;
            batch.capability_profile_generation = caps.profile_generation;
            batch.items = &w;
            batch.item_count = 1;
            st = cere_executor_submit(abi.executor, &batch, &it.ticket);
        }
        for (int64_t left = group; st == CERE_OK && left > 0;) {
            for (int64_t i = 0; i < group && st == CERE_OK; ++i) {
                Item &it = items[size_t(i)];
                if (it.done) continue;
                cere_completion done{};
                done.struct_size = sizeof(done);
                done.abi_version = CERE_EXECUTOR_ABI_VERSION;
                const cere_status poll = cere_executor_poll(abi.executor, it.ticket, &done);
                if (poll == CERE_PENDING) continue;
                if (poll != CERE_OK) {
                    st = poll;
                    break;
                }
                const int64_t p = p0 + i;
                it.done = true;
                --left;
                seconds[p] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                res[2 * p] = int64_t(done.committed_tokens);
                res[2 * p + 1] = int64_t(done.finish);
                if (stats != nullptr) {
                    stats[2 * p] = int64_t(done.verify_steps);
                    stats[2 * p + 1] = int64_t(done.accepted_drafts);
                }
                if (done.finish != CERE_FINISH_LENGTH && done.finish != CERE_FINISH_STOP_TOKEN) {
                    st = done.status != CERE_OK ? done.status : CERE_ERR_INTERNAL;
                }
            }
        }
        for (Item &it : items) {
            if (it.session == nullptr) continue;
            const cere_status released = cere_session_release(abi.executor, it.session);
            if (st == CERE_OK) st = released;
        }
        if (st != CERE_OK) return st;
    }
    return CERE_OK;
}

// KV-byte admission of the device (inkling design note section 9.2, S10): an executor of `budget` bytes first
// creates the model with kv_pool_tokens `refused_pool` (out_refused int64 host [1] = its status: CERE_ERR_CAPACITY
// when the estimate counts the pool), then loads and seals it with max_sessions `sessions`, max_context_tokens
// `context` and kv_pool_tokens `pool`, and runs `ops` (int64 host [n_ops]) in order: op > 0 creates a session of op
// tokens, op < 0 releases the session op index -op - 1 created. out_status int64 host [n_ops] = each op's status;
// out_bytes int64 host [n_ops + 1] = device bytes in use after the seal and after each op. The sessions still open
// are released at the end.
CERE_KTEST int cere_ktest_inkling_device_admission(const void *tensors, const void *names, const void *meta,
                                                   int64_t count, int64_t sessions, int64_t context, int64_t pool,
                                                   int64_t refused_pool, int64_t budget, const void *ops,
                                                   int64_t n_ops, void *out_status, void *out_bytes,
                                                   void *out_refused, cudaStream_t stream) {
    if (tensors == nullptr || names == nullptr || meta == nullptr || count <= 0 || sessions < 1 || context < 1 ||
        pool < 0 || refused_pool < 0 || budget < 1 || ops == nullptr || n_ops < 1 || out_status == nullptr ||
        out_bytes == nullptr || out_refused == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CERE_ERR_INTERNAL;
    std::vector<cere_tensor_desc> descs;
    uint64_t largest = 0;
    if (const cere_status st = describe(names, meta, count, &descs, &largest); st != CERE_OK) return st;
    {
        Abi refused;
        static_cast<int64_t *>(out_refused)[0] =
            open_model(descs, largest, nullptr, uint64_t(sessions), uint64_t(context), uint64_t(refused_pool),
                       uint64_t(budget), &refused, nullptr, nullptr);
    }
    Abi abi;
    cere_execution_capabilities caps{};
    auto *bytes = static_cast<int64_t *>(out_bytes);
    if (const cere_status st = open_model(descs, largest, static_cast<const uint64_t *>(tensors), uint64_t(sessions),
                                          uint64_t(context), uint64_t(pool), uint64_t(budget), &abi, &caps, bytes);
        st != CERE_OK) {
        return st;
    }
    const auto *op = static_cast<const int64_t *>(ops);
    auto *status = static_cast<int64_t *>(out_status);
    std::vector<cere_session *> open(static_cast<size_t>(n_ops), nullptr);
    cere_status result = CERE_OK;
    for (int64_t i = 0; i < n_ops && result == CERE_OK; ++i) {
        if (op[i] > 0) {
            status[i] = cere_session_create(abi.executor, abi.model, uint64_t(op[i]), &open[size_t(i)]);
        } else if (op[i] < 0 && -op[i] - 1 < i && open[size_t(-op[i] - 1)] != nullptr) {
            status[i] = cere_session_release(abi.executor, open[size_t(-op[i] - 1)]);
            if (status[i] == CERE_OK) open[size_t(-op[i] - 1)] = nullptr;
        } else {
            result = CERE_ERR_ARGUMENT;
        }
        size_t free_b = 0, total_b = 0;
        if (result == CERE_OK && cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) result = CERE_ERR_INTERNAL;
        bytes[i + 1] = int64_t(total_b - free_b);
    }
    for (cere_session *s : open) {
        if (s != nullptr && cere_session_release(abi.executor, s) != CERE_OK && result == CERE_OK) {
            result = CERE_ERR_INTERNAL;
        }
    }
    return result;
}
