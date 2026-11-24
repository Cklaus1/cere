// SHA-NI SHA-256 (executor/cuda/src/host/sha256_ni.{h,cpp}; the ktest state digests, the O47 chunk digests): the SHA-NI
// path and the scalar path
// (cere::host::Sha256) give the FIPS 180-4 digests and equal digests on random buffers of every length 0..4097
// (whole and in uneven pieces) and on one buffer of 64 MiB + 13 bytes; CERE_SHA256_FORCE_SCALAR=1 selects the scalar
// path. The Makefile also builds this test with -DCERE_SHA256_NI_MUTANT (a broken SHA-NI round) and requires that
// build to FAIL. `--cpu-has-sha-ni` exits 0 when the CPU has SHA-NI, else 1 (without it only the scalar path runs).
#include "check.h"

#include "host/sha256.h"
#include "host/sha256_ni.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Digest = std::array<uint8_t, 32>;

std::string hex(const Digest &d) {
    static const char *digits = "0123456789abcdef";
    std::string out;
    for (uint8_t b : d) {
        out += digits[b >> 4];
        out += digits[b & 15];
    }
    return out;
}

Digest ktest_digest(bool ni, const uint8_t *data, size_t len) {
    cere::host::Sha256Ni s(ni);
    s.update(data, len);
    return s.finish();
}

// The digest from updates of 1, 2, 3, ... bytes (wrapping at 97), so every buffered/block boundary is crossed.
Digest pieces_digest(bool ni, const uint8_t *data, size_t len) {
    cere::host::Sha256Ni s(ni);
    size_t at = 0;
    for (size_t step = 1; at < len; step = step % 97 + 1) {
        const size_t take = std::min(step, len - at);
        s.update(data + at, take);
        at += take;
    }
    return s.finish();
}

std::vector<uint8_t> random_bytes(size_t len, uint64_t seed) {
    std::vector<uint8_t> out(len);
    uint64_t x = seed * 0x9E3779B97F4A7C15ULL + 1;
    for (size_t i = 0; i < len; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out[i] = uint8_t(x >> 32);
    }
    return out;
}

void check_vector(bool cpu, const std::string &message, const char *want) {
    const auto *p = reinterpret_cast<const uint8_t *>(message.data());
    CHECK(hex(cere::host::sha256(p, message.size())) == want);
    CHECK(hex(ktest_digest(false, p, message.size())) == want);
    CHECK(hex(pieces_digest(false, p, message.size())) == want);
    if (cpu) {
        CHECK(hex(ktest_digest(true, p, message.size())) == want);
        CHECK(hex(pieces_digest(true, p, message.size())) == want);
    }
}

}  // namespace

int main(int argc, char **argv) {
    const bool cpu = cere::host::cpu_has_sha_ni();
    if (argc == 2 && std::strcmp(argv[1], "--cpu-has-sha-ni") == 0) return cpu ? 0 : 1;
    if (!cpu) std::printf("test_sha256_ni: the CPU has no SHA-NI; the scalar path only\n");

    // CERE_SHA256_FORCE_SCALAR=1, read once by the first selection: the scalar path.
    CHECK(::setenv("CERE_SHA256_FORCE_SCALAR", "1", 1) == 0);
    CHECK(!cere::host::sha_ni_selected());
    CHECK(!cere::host::Sha256Ni().ni());
    CHECK(::setenv("CERE_SHA256_FORCE_SCALAR", "0", 1) == 0);
    CHECK(!cere::host::sha_ni_selected());  // read once
    CHECK(cere::host::sha_ni_wanted(true, nullptr));
    CHECK(cere::host::sha_ni_wanted(true, "0"));
    CHECK(!cere::host::sha_ni_wanted(true, "1"));
    CHECK(!cere::host::sha_ni_wanted(false, nullptr));

    // FIPS 180-4 / NIST CAVS vectors.
    check_vector(cpu, "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check_vector(cpu, "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    check_vector(cpu, "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                 "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    check_vector(cpu, std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    if (cpu) {
        const std::vector<uint8_t> random = random_bytes(4097, 1);
        for (size_t len = 0; len <= random.size(); ++len) {
            const Digest want = cere::host::sha256(random.data(), len);
            CHECK(ktest_digest(true, random.data(), len) == want);
            CHECK(pieces_digest(true, random.data(), len) == want);
        }
        const std::vector<uint8_t> big = random_bytes((size_t(64) << 20) + 13, 2);
        CHECK(ktest_digest(true, big.data(), big.size()) == cere::host::sha256(big.data(), big.size()));
    }
    return cere::test::finish("test_sha256_ni");
}
