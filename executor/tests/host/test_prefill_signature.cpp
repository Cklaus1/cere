// BC1 leftover (batch-concurrency design note "Prefill selection signature"): cere_model_prefill_signature on the
// toy device (the SHA-256 of the format v1 text, cere_executor.h), its refusals, and the CUDA-free selection_text of
// model/prefill_signature.h (one changed pin range changes the text and the digest).
#include "check.h"

#include "cere_executor.h"
#include "host/device.h"
#include "host/sha256.h"
#include "host/state.h"
#include "model/prefill_signature.h"
#include "toy/toy_device.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using Digest = std::array<uint8_t, 32>;

std::string hex(const Digest &d) {
    static const char kDigits[] = "0123456789abcdef";
    std::string s;
    for (const uint8_t b : d) {
        s += kDigits[b >> 4];
        s += kDigits[b & 15];
    }
    return s;
}

Digest digest_of(const std::string &text) { return cere::host::sha256(text.data(), text.size()); }

cere_executor *make_executor(const char *device_id) {
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = device_id;
    options.worker_epoch = 1;
    options.memory_budget_bytes = uint64_t(1) << 30;
    cere_executor *executor = nullptr;
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_OK);
    CHECK(executor != nullptr);
    return executor;
}

// A one-tensor toy model, sealed when `seal`.
struct Model {
    cere_qwen4_exp_config config = cere::toy::fixture_config();
    cere_tensor_desc tensor{};
    cere_model_desc desc{};
    cere_model *model = nullptr;

    Model(cere_executor *executor, bool seal) {
        tensor.struct_size = sizeof(tensor);
        tensor.abi_version = CERE_EXECUTOR_ABI_VERSION;
        tensor.name = "t0";
        tensor.dtype = CERE_DTYPE_U8;
        tensor.placement = CERE_PLACEMENT_DEVICE;
        tensor.ndim = 1;
        tensor.shape[0] = 16;
        tensor.bytes = 16;
        desc.struct_size = sizeof(desc);
        desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
        desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
        desc.config = &config;
        desc.tensors = &tensor;
        desc.tensor_count = 1;
        desc.max_sessions = 1;
        desc.max_context_tokens = 64;
        CHECK_EQ(cere_model_create(executor, &desc, &model), CERE_OK);
        const std::vector<uint8_t> bytes(16, 0x5a);
        CHECK_EQ(cere_model_put(executor, model, 0, 0, bytes.data(), bytes.size()), CERE_OK);
        if (seal) CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    }
};

// A device with no prefill selection (host::Device's default): every call the test makes goes to the toy device it
// wraps, prefill_selection is not overridden.
class PlainDevice final : public cere::host::Device {
public:
    explicit PlainDevice(std::unique_ptr<cere::host::Device> inner) : inner_(std::move(inner)) {}
    std::vector<const cere::host::Support *> supported() const override { return inner_->supported(); }
    uint64_t host_budget_bytes() const override { return inner_->host_budget_bytes(); }
    cere::host::MemoryEstimate estimate(const cere::host::ModelPlan &plan) const override {
        return inner_->estimate(plan);
    }
    cere_status allocate(const cere::host::ModelPlan &plan, std::unique_ptr<cere::host::DeviceModel> *out) override {
        return inner_->allocate(plan, out);
    }
    cere_status write(cere::host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                      uint64_t len) override {
        return inner_->write(model, tensor_index, offset, bytes, len);
    }
    uint32_t moe_lever() const override { return inner_->moe_lever(); }
    cere_status set_levers(cere::host::DeviceModel &model, const cere::host::Levers &levers) override {
        return inner_->set_levers(model, levers);
    }
    cere_status instantiate(cere::host::DeviceModel &model) override { return inner_->instantiate(model); }
    bool forward_supported() const override { return inner_->forward_supported(); }
    bool sampler_supported() const override { return inner_->sampler_supported(); }
    cere_status create_session(cere::host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<cere::host::DeviceSession> *out) override {
        return inner_->create_session(model, max_context_tokens, lazy, out);
    }
    cere_status step(cere::host::DeviceModel &model, cere::host::DeviceSession &session, const uint32_t *tokens,
                     uint64_t count, uint64_t limit, const cere::host::StepImages *images, uint64_t *consumed,
                     const cere::host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                     const cere::host::EndStep *end, cere::host::StepSlice *slice) override {
        return inner_->step(model, session, tokens, count, limit, images, consumed, sampling, out, logprobs, end,
                            slice);
    }
    cere_status step_rows(cere::host::DeviceModel &model, const cere::host::RowArgs *rows, uint32_t row_count,
                          cere_token_score *out, cere_status *row_status) override {
        return inner_->step_rows(model, rows, row_count, out, row_status);
    }
    cere_status set_rope_delta(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                               int64_t rope_delta) override {
        return inner_->set_rope_delta(model, session, rope_delta);
    }

private:
    std::unique_ptr<cere::host::Device> inner_;
};

constexpr const char *kToyText =
    "cere-prefill-signature-v1\n"
    "host plan_captures_rule 1\n"
    "host prefill_chunk 4096\n"
    "toy prefill-selection v1\n";
// SHA-256 of kToyText: the default toy's pinned signature.
constexpr const char *kToyPinned = "63c07240fed82246042181ae7d6429d75511153a0a8fd50089fa704873ad7a7f";

Digest signature(cere_executor *executor, cere_model *model, cere_status expect = CERE_OK) {
    Digest d{};
    CHECK_EQ(cere_model_prefill_signature(executor, model, d.data()), expect);
    return d;
}

void test_toy_signature() {
    CHECK(hex(digest_of(kToyText)) == kToyPinned);
    cere_executor *executor = make_executor("toy");
    Model m(executor, true);
    CHECK(hex(signature(executor, m.model)) == kToyPinned);
    CHECK_EQ(cere_model_release(executor, m.model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // The prefill chunk is a line of the text.
    executor = make_executor("toy,prefill_chunk=512");
    Model chunk(executor, true);
    const std::string chunk_text =
        "cere-prefill-signature-v1\nhost plan_captures_rule 1\nhost prefill_chunk 512\ntoy prefill-selection v1\n";
    CHECK(signature(executor, chunk.model) == digest_of(chunk_text));
    CHECK_EQ(cere_model_release(executor, chunk.model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Fail first: one changed "pinned bucket" (",prefill_sig=2") gives another digest, so the default's pin no longer
// matches.
void test_changed_bucket_breaks_the_pin() {
    cere_executor *executor = make_executor("toy,prefill_sig=2");
    Model m(executor, true);
    const Digest d = signature(executor, m.model);
    CHECK(d == digest_of(std::string(kToyText) + "toy bucket 2\n"));
    CHECK(hex(d) != kToyPinned);
    CHECK_EQ(cere_model_release(executor, m.model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = "toy,prefill_sig=0";
    cere_executor *bad = nullptr;
    CHECK_EQ(cere_executor_create(&bad, &options), CERE_ERR_ARGUMENT);
}

void test_refusals() {
    cere_executor *executor = make_executor("toy");
    Model sealed(executor, true);
    Digest d{};
    d.fill(0xa5);
    CHECK_EQ(cere_model_prefill_signature(nullptr, sealed.model, d.data()), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefill_signature(executor, sealed.model, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefill_signature(executor, nullptr, d.data()), CERE_ERR_ARGUMENT);
    // Unsealed: as cere_executor_capabilities, CERE_ERR_ARGUMENT, out untouched.
    Model open(executor, false);
    CHECK_EQ(cere_model_prefill_signature(executor, open.model, d.data()), CERE_ERR_ARGUMENT);
    Digest untouched{};
    untouched.fill(0xa5);
    CHECK(d == untouched);
    CHECK_EQ(cere_model_release(executor, open.model), CERE_OK);
    CHECK_EQ(cere_model_release(executor, sealed.model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // A device without a prefill selection (host::Device's default): CERE_ERR_UNSUPPORTED on a sealed model.
    executor = make_executor("toy");
    executor->device = std::make_unique<PlainDevice>(std::move(executor->device));
    Model plain(executor, true);
    signature(executor, plain.model, CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_release(executor, plain.model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// selection_text (the CUDA device's lines): fixed order, one fact per line; a changed pin range changes the text and
// the digest.
void test_selection_text() {
    using namespace cere::model::prefill_sig;
    PrefillSelection s;
    s.cublaslt_version = 130101;
    PinTable pins;
    pins.n = 640;
    pins.k = 2560;
    pins.out = "bf16";
    pins.cublaslt_version = 130101;
    pins.ranges.push_back({17, 28, {21, 18, 20, 18, 2, 14}});
    pins.ranges.push_back({29, 32, {21, 18, 20, 17, 2, 14}});
    s.pins.push_back(pins);
    PlanSelection plan;
    plan.name = "attn_o";
    plan.n = 2560;
    plan.k = 4096;
    plan.out = "bf16";
    plan.max_m = 2;
    BucketAlgo b0;
    b0.m_hi = 1;
    b0.algo_id = 13;
    b0.custom = 75;
    BucketAlgo b1;
    b1.m_hi = 2;
    b1.algo_id = 21;
    b1.tile = 18;
    b1.stages = 20;
    b1.splitk = 1;
    b1.swizzle = 1;
    b1.inner = 3;
    b1.cluster = 2;
    plan.buckets = {b0, b1};
    s.plans.push_back(plan);
    s.k5_chunk_tokens = 64;
    s.k6_max_context = 2048;

    const std::string k7 = selection_text(s);
    CHECK(k7 ==
          "cublaslt 130101\n"
          "pins 640 2560 bf16 cublaslt 130101 ranges 2\n"
          "pin 17 28 algo 21 tile 18 stages 20 splitk 18 reduction 2 partials 14 custom 0\n"
          "pin 29 32 algo 21 tile 18 stages 20 splitk 17 reduction 2 partials 14 custom 0\n"
          "plan attn_o 2560 4096 bf16 max_m 2 buckets 2\n"
          "bucket 1 algo 13 tile 0 stages 0 splitk 0 reduction 0 custom 75 swizzle 0 inner 0 cluster 0\n"
          "bucket 2 algo 21 tile 18 stages 20 splitk 1 reduction 0 custom 0 swizzle 1 inner 3 cluster 2\n"
          "p3b none\n"
          "k5 chunk_tokens 64\n"
          "k6 max_context 2048\n");
    CHECK(selection_text(s) == k7);  // deterministic

    s.p3b = true;
    s.p3b_buckets.assign(cere::p3b::kPinnedBuckets, cere::p3b::kPinnedBuckets + cere::p3b::kPinnedBucketCount);
    const std::string p3b = selection_text(s);
    const std::string p3b_lines = "p3b 4\n"
                                  "p3b_bucket 6 gemm1 128 32 64 1 0 gemm2 128 32 64 1 0\n"
                                  "p3b_bucket 512 gemm1 128 32 128 1 0 gemm2 128 32 128 1 0\n"
                                  "p3b_bucket 2048 gemm1 128 32 128 1 0 gemm2 128 128 64 0 0\n"
                                  "p3b_bucket 4294967295 gemm1 256 128 64 1 0 gemm2 256 128 64 1 0\n";
    CHECK(p3b.find(p3b_lines) != std::string::npos);
    CHECK(p3b.find("p3b none") == std::string::npos);

    // Fail first: one pinned range moved (m_hi 28 -> 27) changes the text and so the digest.
    PrefillSelection moved = s;
    moved.pins[0].ranges[0].m_hi = 27;
    CHECK(selection_text(moved) != p3b);
    CHECK(digest_of(selection_text(moved)) != digest_of(p3b));
    // So does one bucket's algorithm attribute.
    PrefillSelection swizzled = s;
    swizzled.plans[0].buckets[1].swizzle = 0;
    CHECK(digest_of(selection_text(swizzled)) != digest_of(p3b));
    // And one pinned range's custom option (r3-prefill design note: the bucket re-pins pin one).
    PrefillSelection custom = s;
    custom.pins[0].ranges[1].config.custom = 91;
    CHECK(selection_text(custom).find("partials 14 custom 91\n") != std::string::npos);
    CHECK(digest_of(selection_text(custom)) != digest_of(p3b));
}

}  // namespace

int main() {
    test_toy_signature();
    test_changed_bucket_breaks_the_pin();
    test_refusals();
    test_selection_text();
    return cere::test::finish("test_prefill_signature");
}
