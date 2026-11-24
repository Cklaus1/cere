// FL-13 (fp4local design note D8, D14): the K4 NVFP4 PLE row decode (kernels/k4_nvfp4.h, the functions the K4
// gather kernel runs) against an independent CPU reference built from the format definitions (E2M1 and E4M3 by
// ldexp in double, the global product rounded to F32 by the compiler's IEEE conversion, BF16 RNE by neighbour
// comparison), on synthetic NVFP4 rows: every E2M1 code at both nibble positions against every E4M3 scale byte
// (0, subnormals 0x01-0x07, the largest 0x7E, NaN 0x7F, negatives), under F32 globals that are 1, powers of two,
// not powers of two, subnormal, overflowing, negative and -0; plus hand-computed BF16 bits and the RNE tie rule.
#include "check.h"

#include "kernels/k4_nvfp4.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace n = cere::k4::nvfp4;

float bits_f32(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

uint32_t f32_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

// ---- independent reference ----

double ref_e2m1(uint32_t code) {
    static const double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double m = kMag[code & 7u];
    return (code & 8u) ? -m : m;
}

double ref_e4m3(uint32_t byte) {
    const uint32_t e = (byte >> 3) & 15u;
    const uint32_t m = byte & 7u;
    if (e == 15u && m == 7u) return std::nan("");
    const double mag = e == 0 ? std::ldexp(double(m) / 8.0, -6) : std::ldexp(1.0 + double(m) / 8.0, int(e) - 7);
    return (byte & 0x80u) ? -mag : mag;
}

// BF16 RNE by choosing between the two truncation neighbours; NaN -> 0x7FFF.
uint16_t ref_bf16(float v) {
    if (std::isnan(v)) return 0x7FFFu;
    const uint32_t b = f32_bits(v);
    const uint32_t lo = b >> 16;
    const uint32_t rem = b & 0xFFFFu;
    if (rem < 0x8000u) return uint16_t(lo);
    if (rem > 0x8000u) return uint16_t(lo + 1);
    return uint16_t((lo & 1u) ? lo + 1 : lo);
}

uint16_t ref_element(uint32_t code, uint32_t scale, float global) {
    const double block = ref_e2m1(code) * ref_e4m3(scale);  // exact: <= 5 significant bits
    const float block32 = float(block);
    CHECK(std::isnan(block) || double(block32) == block);
    const float v = float(double(block32) * double(global));  // exact double product, one F32 rounding
    return ref_bf16(v);
}

// ---- cases ----

// Every code x every scale byte, through decode_row: row r has scale bytes (10 r + b) & 0xFF for block b and element
// i holds code (i + r) % 16, so each block holds all 16 codes and each code sits at both nibble positions.
void exhaustive(float global, int *compared) {
    constexpr int kRows = 26;  // 260 blocks >= 256 scale bytes
    for (int r = 0; r < kRows; ++r) {
        uint8_t packed[n::kRowBytes];
        uint8_t scales[n::kRowScales];
        uint32_t codes[n::kRowElements];
        for (int i = 0; i < n::kRowElements; ++i) codes[i] = uint32_t((i + r) % 16);
        for (int j = 0; j < n::kRowBytes; ++j) packed[j] = uint8_t(codes[2 * j] | (codes[2 * j + 1] << 4));
        for (int b = 0; b < n::kRowScales; ++b) scales[b] = uint8_t((10 * r + b) & 0xFF);
        uint16_t out[n::kRowElements];
        n::decode_row(packed, scales, global, out);
        for (int i = 0; i < n::kRowElements; ++i) {
            CHECK_EQ(n::code_of(packed, i), codes[i]);
            const uint16_t want = ref_element(codes[i], scales[i / n::kBlock], global);
            if (out[i] != want) {
                std::fprintf(stderr, "global %a row %d element %d code %u scale 0x%02x: 0x%04x, expected 0x%04x\n",
                             double(global), r, i, codes[i], scales[i / n::kBlock], out[i], want);
                ++cere::test::failures();
            }
            ++*compared;
        }
    }
}

uint16_t one(uint32_t code, uint32_t scale, float global) {
    uint8_t packed[n::kRowBytes] = {};
    uint8_t scales[n::kRowScales] = {};
    packed[0] = uint8_t(code & 0xFu);
    scales[0] = uint8_t(scale);
    uint16_t out[n::kRowElements];
    n::decode_row(packed, scales, global, out);
    return out[0];
}

}  // namespace

int main() {
    // Format tables.
    for (uint32_t c = 0; c < 16; ++c) CHECK(n::e2m1(c) == float(ref_e2m1(c)) && std::signbit(n::e2m1(c)) == bool(c & 8));
    for (uint32_t s = 0; s < 256; ++s) {
        const double want = ref_e4m3(s);
        if (std::isnan(want)) {
            CHECK(std::isnan(n::e4m3(s)));
        } else {
            CHECK(double(n::e4m3(s)) == want && std::signbit(n::e4m3(s)) == bool(s & 0x80u));
        }
    }
    CHECK(n::e4m3(0x7E) == 448.0f);
    CHECK(n::e4m3(0x01) == std::ldexp(1.0f, -9));
    CHECK(n::e4m3(0x07) == 7.0f * std::ldexp(1.0f, -9));
    CHECK(n::e4m3(0x08) == std::ldexp(1.0f, -6));
    CHECK(n::e4m3(0x00) == 0.0f && !std::signbit(n::e4m3(0x00)) && std::signbit(n::e4m3(0x80)));

    // Nibble order: element 2j is the low nibble of byte j.
    {
        uint8_t packed[n::kRowBytes] = {};
        packed[0] = 0x21;
        packed[79] = 0xF7;
        CHECK_EQ(n::code_of(packed, 0), 1);
        CHECK_EQ(n::code_of(packed, 1), 2);
        CHECK_EQ(n::code_of(packed, 158), 7);
        CHECK_EQ(n::code_of(packed, 159), 15);
    }

    // Hand-computed BF16 bits.
    CHECK_EQ(one(0x7, 0x7E, 1.0f), 0x4528);              // 6 x 448 = 2688 (largest magnitude)
    CHECK_EQ(one(0xF, 0x7E, 1.0f), 0xC528);              // -2688
    CHECK_EQ(one(0x1, 0x01, 1.0f), 0x3A80);              // 0.5 x 2^-9 = 2^-10 (smallest nonzero magnitude)
    CHECK_EQ(one(0x7, 0x07, 1.0f), 0x3DA8);              // 6 x 7 x 2^-9 = 42 x 2^-9 = 1.3125 x 2^-4
    CHECK_EQ(one(0x5, 0x00, 1.0f), 0x0000);              // scale 0 -> +0
    CHECK_EQ(one(0xD, 0x00, 1.0f), 0x8000);              // -3 x +0 -> -0
    CHECK_EQ(one(0x0, 0x7E, 1.0f), 0x0000);              // code 0 -> +0
    CHECK_EQ(one(0x8, 0x7E, 1.0f), 0x8000);              // code 8 = -0
    CHECK_EQ(one(0x3, 0x7F, 1.0f), 0x7FFF);              // E4M3 NaN -> canonical BF16 NaN
    CHECK_EQ(one(0x2, 0x38, 0.1f), 0x3DCD);              // 1 x 1 x 0.1f (0x3DCCCCCD) rounds up
    CHECK_EQ(one(0x2, 0x38, 2.0f), 0x4000);              // the global multiplies: 1 x 1 x 2
    CHECK_EQ(one(0x2, 0x38, bits_f32(0x3F808000)), 0x3F80);  // tie, even neighbour below
    CHECK_EQ(one(0x2, 0x38, bits_f32(0x3F818000)), 0x3F82);  // tie, even neighbour above
    CHECK_EQ(one(0x7, 0x7E, FLT_MAX), 0x7F80);           // overflow in F32 -> +inf
    CHECK_EQ(one(0x1, 0x01, std::ldexp(1.0f, -140)), 0x0000);  // 2^-150 rounds to +0 in F32 (tie to even)
    CHECK_EQ(one(0x2, 0x38, std::ldexp(1.0f, -133)), 0x0001);  // F32 subnormal 2^-133 (bits 0x10000) = BF16 0x0001
    CHECK_EQ(n::bf16_rne_bits(bits_f32(0xFFC00001u)), 0x7FFF);

    // Exhaustive codes x scales under each global.
    const float globals[] = {
        1.0f,
        2.0f,
        std::ldexp(1.0f, -20),
        0.1f,
        bits_f32(0x3A1D4952u),  // a non-power-of-two global of the order of the FP4-Local table's (F8)
        3.0f,
        1.0e38f,                // products overflow to +-inf
        std::ldexp(1.0f, -130), // products land in the F32 subnormal range
        bits_f32(0x00000001u),  // smallest F32 subnormal
        -1.0f,
        -0.0f,
    };
    int compared = 0;
    for (const float g : globals) exhaustive(g, &compared);
    CHECK_EQ(compared, int(sizeof(globals) / sizeof(globals[0])) * 26 * n::kRowElements);
    std::printf("test_k4_nvfp4: %d elements compared\n", compared);
    return cere::test::finish("test_k4_nvfp4");
}
