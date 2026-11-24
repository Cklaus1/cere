#include "host/sha256_ni.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace cere::host {
namespace {

constexpr uint32_t kInit[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

#if defined(__x86_64__)
alignas(16) constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

// FIPS 180-4 compression of `n` 64-byte blocks into h[8] (a..h) with the SHA extensions. The instructions keep the
// state as ABEF / CDGH halves and run two rounds per sha256rnds2; message words 16..63 come from sha256msg1/msg2.
__attribute__((target("sha,ssse3,sse4.1"))) void blocks_ni(uint32_t h[8], const uint8_t *p, size_t n) {
    const __m128i bswap = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i t = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)), 0xB1);  // CDAB
    __m128i s1 = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h + 4)), 0x1B);  // EFGH
    __m128i s0 = _mm_alignr_epi8(t, s1, 8);  // ABEF
    s1 = _mm_blend_epi16(s1, t, 0xF0);       // CDGH
    for (; n > 0; --n, p += 64) {
        const __m128i abef = s0;
        const __m128i cdgh = s1;
        __m128i m[4];
        for (int i = 0; i < 4; ++i) {
            m[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16 * i)), bswap);
        }
        for (int r = 0; r < 16; ++r) {
            const __m128i msg = _mm_add_epi32(m[r & 3], _mm_load_si128(reinterpret_cast<const __m128i *>(kK + 4 * r)));
            s1 = _mm_sha256rnds2_epu32(s1, s0, msg);
#if defined(CERE_SHA256_NI_MUTANT)
            // The mutant check's broken round (executor/tests/host/test_sha256_ni.cpp): the last group's second
            // round pair is skipped, so the digest must differ from the scalar path's.
            if (r == 15) continue;
#endif
            s0 = _mm_sha256rnds2_epu32(s0, s1, _mm_shuffle_epi32(msg, 0x0E));
            if (r < 12) {
                // words 4(r+4)..4(r+4)+3 = msg2(msg1(w[r], w[r+1]) + alignr(w[r+3], w[r+2], 4), w[r+3])
                __m128i next = _mm_sha256msg1_epu32(m[r & 3], m[(r + 1) & 3]);
                next = _mm_add_epi32(next, _mm_alignr_epi8(m[(r + 3) & 3], m[(r + 2) & 3], 4));
                m[r & 3] = _mm_sha256msg2_epu32(next, m[(r + 3) & 3]);
            }
        }
        s0 = _mm_add_epi32(s0, abef);
        s1 = _mm_add_epi32(s1, cdgh);
    }
    t = _mm_shuffle_epi32(s0, 0x1B);    // FEBA
    s1 = _mm_shuffle_epi32(s1, 0xB1);   // DCHG
    s0 = _mm_blend_epi16(t, s1, 0xF0);  // DCBA
    s1 = _mm_alignr_epi8(s1, t, 8);     // HGFE
    _mm_storeu_si128(reinterpret_cast<__m128i *>(h), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(h + 4), s1);
}
#else
void blocks_ni(uint32_t *, const uint8_t *, size_t) { std::abort(); }
#endif

}  // namespace

bool cpu_has_sha_ni() {
#if defined(__x86_64__)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d) == 0 || ((b >> 29) & 1) == 0) return false;  // SHA
    if (__get_cpuid(1, &a, &b, &c, &d) == 0) return false;
    return ((c >> 9) & 1) != 0 && ((c >> 19) & 1) != 0;  // SSSE3, SSE4.1
#else
    return false;
#endif
}

bool sha_ni_wanted(bool cpu, const char *force_scalar) {
    return cpu && !(force_scalar != nullptr && std::strcmp(force_scalar, "1") == 0);
}

bool sha_ni_selected() {
    static const bool selected = sha_ni_wanted(cpu_has_sha_ni(), std::getenv("CERE_SHA256_FORCE_SCALAR"));
    return selected;
}

Sha256Ni::Sha256Ni(bool ni) : ni_(ni) {
    if (ni_ && !cpu_has_sha_ni()) std::abort();
    std::memcpy(h_, kInit, sizeof(h_));
}

void Sha256Ni::update(const void *data, size_t len) {
    if (!ni_) {
        scalar_.update(data, len);
        return;
    }
    const uint8_t *p = static_cast<const uint8_t *>(data);
    total_ += len;
    if (buffered_ != 0) {
        const size_t take = std::min(len, sizeof(buffer_) - buffered_);
        std::memcpy(buffer_ + buffered_, p, take);
        buffered_ += take;
        p += take;
        len -= take;
        if (buffered_ < sizeof(buffer_)) return;
        blocks_ni(h_, buffer_, 1);
        buffered_ = 0;
    }
    if (len >= 64) {
        blocks_ni(h_, p, len / 64);
        p += len / 64 * 64;
        len %= 64;
    }
    std::memcpy(buffer_, p, len);
    buffered_ = len;
}

std::array<uint8_t, 32> Sha256Ni::finish() {
    if (!ni_) return scalar_.finish();
    const uint64_t bits = total_ * 8;
    uint8_t pad[72] = {0x80};
    const size_t pad_len = (buffered_ < 56 ? 56 : 120) - buffered_;
    for (int i = 0; i < 8; ++i) pad[pad_len + i] = uint8_t(bits >> (56 - 8 * i));
    update(pad, pad_len + 8);
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = uint8_t(h_[i] >> 24);
        out[4 * i + 1] = uint8_t(h_[i] >> 16);
        out[4 * i + 2] = uint8_t(h_[i] >> 8);
        out[4 * i + 3] = uint8_t(h_[i]);
    }
    return out;
}

}  // namespace cere::host
