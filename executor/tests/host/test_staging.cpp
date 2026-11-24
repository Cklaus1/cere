// Put staging (LoadFast2, m216 design note M216S-11): cere_executor_register_staging / unregister_staging through the
// C ABI against the CPU toy device. A round trip and its argument errors; a DEVICE put whose source lies inside a
// registered range takes the direct path (write_staged) and every other put the plain write, with the tensors
// byte-exact whatever the tiling; a staging range is reusable as soon as its put returns; a put refused before the
// copy releases the range; a range with a put in flight cannot be unregistered (BUSY) and an executor with a
// registered range cannot be destroyed (BUSY).
#include "check.h"

#include "cere_executor.h"
#include "host/state.h"
#include "toy/toy_device.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kMiB = uint64_t(1) << 20;
// DEVICE: one large tensor (several staging buffers, a ragged tail) and one small; HOST_PINNED; DISCARD.
constexpr uint64_t kLarge = 3 * kMiB + 4099;
constexpr uint64_t kSmall = 17;
constexpr uint64_t kHost = kMiB + 3;

cere_status create_executor(const std::string &id, cere_executor **out) {
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = id.c_str();
    options.worker_epoch = 1;
    options.memory_budget_bytes = uint64_t(1) << 30;
    *out = nullptr;
    return cere_executor_create(out, &options);
}

struct Desc {
    cere_qwen4_exp_config config = cere::toy::fixture_config();
    std::vector<std::string> names{"large", "small", "ple", "drop"};
    std::vector<uint64_t> sizes{kLarge, kSmall, kHost, 16};
    std::vector<uint32_t> placements{CERE_PLACEMENT_DEVICE, CERE_PLACEMENT_DEVICE, CERE_PLACEMENT_HOST_PINNED,
                                     CERE_PLACEMENT_DISCARD};
    std::vector<cere_tensor_desc> tensors;
    cere_model_desc desc{};

    Desc() {
        for (size_t i = 0; i < names.size(); ++i) {
            cere_tensor_desc t{};
            t.struct_size = sizeof(t);
            t.abi_version = CERE_EXECUTOR_ABI_VERSION;
            t.name = names[i].c_str();
            t.dtype = CERE_DTYPE_U8;
            t.placement = placements[i];
            t.ndim = 1;
            t.shape[0] = sizes[i];
            t.bytes = sizes[i];
            tensors.push_back(t);
        }
        desc.struct_size = sizeof(desc);
        desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
        desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
        desc.config = &config;
        desc.tensors = tensors.data();
        desc.tensor_count = tensors.size();
        desc.max_sessions = 1;
        desc.max_context_tokens = 64;
    }
};

std::vector<uint8_t> pattern(uint64_t len, uint8_t seed) {
    std::vector<uint8_t> bytes(len);
    for (uint64_t i = 0; i < len; ++i) bytes[i] = uint8_t(seed + 31 * i + (i >> 12));
    return bytes;
}

cere::toy::ToyDevice &toy(cere_executor *executor) { return static_cast<cere::toy::ToyDevice &>(*executor->device); }

void test_round_trip() {
    cere_executor *executor = nullptr;
    CHECK_EQ(create_executor("toy", &executor), CERE_OK);
    std::vector<uint8_t> a(kMiB), b(kMiB);
    CHECK_EQ(cere_executor_register_staging(nullptr, a.data(), a.size()), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_register_staging(executor, nullptr, a.size()), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_register_staging(executor, a.data(), 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_unregister_staging(executor, a.data()), CERE_ERR_ARGUMENT);  // never registered
    CHECK_EQ(cere_executor_register_staging(executor, a.data(), a.size()), CERE_OK);
    CHECK_EQ(toy(executor).staging_ranges(), 1);
    // Overlaps: the same range, one inside it, one straddling its start; the range after it is disjoint.
    CHECK_EQ(cere_executor_register_staging(executor, a.data(), a.size()), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_register_staging(executor, a.data() + 4096, 4096), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_register_staging(executor, b.data(), b.size()), CERE_OK);
    CHECK_EQ(toy(executor).staging_ranges(), 2);
    CHECK_EQ(cere_executor_unregister_staging(executor, a.data() + 1), CERE_ERR_ARGUMENT);  // not a base
    CHECK_EQ(cere_executor_unregister_staging(nullptr, a.data()), CERE_ERR_ARGUMENT);
    // Unregistering before release is enforced: the executor outlives no registration.
    CHECK_EQ(cere_executor_destroy(executor), CERE_ERR_BUSY);
    CHECK_EQ(cere_executor_unregister_staging(executor, a.data()), CERE_OK);
    CHECK_EQ(cere_executor_unregister_staging(executor, a.data()), CERE_ERR_ARGUMENT);  // already ended
    CHECK_EQ(cere_executor_unregister_staging(executor, b.data()), CERE_OK);
    CHECK_EQ(toy(executor).staging_ranges(), 0);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_direct_path_tiling() {
    Desc d;
    cere_executor *executor = nullptr;
    CHECK_EQ(create_executor("toy", &executor), CERE_OK);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    // A 2 MiB allocation whose first MiB is registered: a piece starting in it and running past its end is not
    // inside one range, so it takes the plain write.
    std::vector<uint8_t> arena(2 * kMiB);
    uint8_t *staging = arena.data();
    CHECK_EQ(cere_executor_register_staging(executor, staging, kMiB), CERE_OK);
    const auto large = pattern(kLarge, 1);
    const auto small = pattern(kSmall, 2);
    const auto host = pattern(kHost, 3);
    uint64_t staged = 0, staged_bytes = 0;
    // The large tensor in ragged pieces, each copied to a varying offset inside the registered MiB; the staging bytes
    // are overwritten right after each put returns (the put has finished reading them).
    const uint64_t pieces[] = {kMiB, 1, 4096, kMiB - 4097, 12345, kMiB - 12345, 7, 4092};
    uint64_t at = 0;
    size_t k = 0;
    while (at < kLarge - 64) {
        const uint64_t want = pieces[k % (sizeof(pieces) / sizeof(pieces[0]))];
        const uint64_t n = std::min<uint64_t>(want, kLarge - 64 - at);
        const uint64_t shift = (k * 977) % (kMiB - n + 1);
        std::memcpy(staging + shift, large.data() + at, n);
        CHECK_EQ(cere_model_put(executor, model, 0, at, staging + shift, n), CERE_OK);
        std::memset(staging + shift, 0xA5, n);
        ++staged;
        staged_bytes += n;
        at += n;
        ++k;
    }
    // The last 64 bytes straddle the registered range's end: plain write.
    std::memcpy(staging + kMiB - 32, large.data() + at, 64);
    CHECK_EQ(cere_model_put(executor, model, 0, at, staging + kMiB - 32, 64), CERE_OK);
    CHECK_EQ(toy(executor).staged_puts(), staged);
    CHECK_EQ(toy(executor).staged_bytes(), staged_bytes);
    // The small tensor from unregistered memory: plain write.
    CHECK_EQ(cere_model_put(executor, model, 1, 0, small.data(), small.size()), CERE_OK);
    CHECK_EQ(toy(executor).staged_puts(), staged);
    // HOST_PINNED from the registered range: a host copy, never the direct path.
    for (uint64_t off = 0; off < kHost; off += kMiB / 2) {
        const uint64_t n = std::min<uint64_t>(kMiB / 2, kHost - off);
        std::memcpy(staging, host.data() + off, n);
        CHECK_EQ(cere_model_put(executor, model, 2, off, staging, n), CERE_OK);
    }
    CHECK_EQ(toy(executor).staged_puts(), staged);
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    if (model != nullptr) {
        CHECK(toy(executor).tensor(*model->device, 0) == large);
        CHECK(toy(executor).tensor(*model->device, 1) == small);
        CHECK(toy(executor).tensor(*model->device, 2) == host);
    }
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_unregister_staging(executor, staging), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// A put refused before its copy (range, sealed model) releases the staging range like a finished one.
void test_refused_put_releases_range() {
    Desc d;
    cere_executor *executor = nullptr;
    CHECK_EQ(create_executor("toy", &executor), CERE_OK);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    std::vector<uint8_t> staging(kMiB, 1);
    CHECK_EQ(cere_executor_register_staging(executor, staging.data(), staging.size()), CERE_OK);
    CHECK_EQ(cere_model_put(executor, model, 1, 1, staging.data(), kSmall), CERE_ERR_ARGUMENT);  // past the end
    CHECK_EQ(cere_model_put(executor, model, 9, 0, staging.data(), 1), CERE_ERR_ARGUMENT);       // no such tensor
    CHECK_EQ(toy(executor).staged_puts(), 0);
    CHECK_EQ(cere_executor_unregister_staging(executor, staging.data()), CERE_OK);
    CHECK_EQ(cere_executor_register_staging(executor, staging.data(), staging.size()), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_unregister_staging(executor, staging.data()), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// A put held inside the toy's write_staged (staging_gate=<file>) keeps its range busy until it returns.
void test_unregister_in_use() {
    char tmpl[] = "/tmp/cere-staging-gate-XXXXXX";
    const int fd = mkstemp(tmpl);
    CHECK(fd >= 0);
    if (fd < 0) return;
    close(fd);
    const std::string gate = tmpl;
    Desc d;
    cere_executor *executor = nullptr;
    CHECK_EQ(create_executor("toy,staging_gate=" + gate, &executor), CERE_OK);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    const auto small = pattern(kSmall, 4);
    std::vector<uint8_t> staging(kMiB);
    std::memcpy(staging.data(), small.data(), small.size());
    CHECK_EQ(cere_executor_register_staging(executor, staging.data(), staging.size()), CERE_OK);
    cere_status put = CERE_ERR_INTERNAL;
    std::thread putter([&] { put = cere_model_put(executor, model, 1, 0, staging.data(), small.size()); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (toy(executor).staged_holding() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK_EQ(toy(executor).staged_holding(), 1);
    CHECK_EQ(cere_executor_unregister_staging(executor, staging.data()), CERE_ERR_BUSY);
    CHECK_EQ(toy(executor).staging_ranges(), 1);  // still registered
    std::remove(gate.c_str());
    putter.join();
    CHECK_EQ(put, CERE_OK);
    CHECK_EQ(toy(executor).staged_puts(), 1);
    CHECK_EQ(cere_executor_unregister_staging(executor, staging.data()), CERE_OK);
    CHECK_EQ(toy(executor).staging_ranges(), 0);
    if (model != nullptr) CHECK(toy(executor).tensor(*model->device, 1) == small);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

}  // namespace

int main() {
    test_round_trip();
    test_direct_path_tiling();
    test_refused_put_releases_range();
    test_unregister_in_use();
    return cere::test::finish("test_staging");
}
