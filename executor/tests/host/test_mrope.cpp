// M2.13 C8 (m213 design note C8, D11, M213S-04): cere::mrope::positions equals SGLang's image-only mRoPE path
// (`QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets`, checked there against `get_rope_index`)
// on every case of tests/fixtures/native/mrope_cases.json (scripts/native/gen_mrope_cases.py, pinned image, CPU),
// position for position, plus fixture-free properties.
#include "check.h"

#include "host/mrope.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using cere::mrope::ImageSpan;

// ---- minimal JSON (the fixture is machine-written by gen_mrope_cases.py: integers, ASCII strings, no escapes) ----

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    long long number = 0;
    std::string text;
    std::vector<Json> items;
    std::map<std::string, Json> fields;

    const Json &operator[](const std::string &key) const {
        static const Json missing;
        const auto it = fields.find(key);
        return it == fields.end() ? missing : it->second;
    }
    const Json &operator[](size_t i) const { return items[i]; }
};

struct Parser {
    const std::string &s;
    size_t i = 0;
    bool ok = true;

    void ws() {
        while (i < s.size() && std::strchr(" \t\r\n", s[i])) ++i;
    }
    char peek() { return i < s.size() ? s[i] : '\0'; }
    bool take(const char *word) {
        const size_t n = std::strlen(word);
        if (s.compare(i, n, word) != 0) return false;
        i += n;
        return true;
    }
    Json value() {
        ws();
        Json out;
        const char c = peek();
        if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            out.kind = c == '{' ? Json::Object : Json::Array;
            ++i;
            ws();
            if (peek() == close) return ++i, out;
            while (ok) {
                if (out.kind == Json::Object) {
                    ws();
                    const std::string key = string();
                    ws();
                    if (peek() != ':') ok = false;
                    ++i;
                    out.fields[key] = value();
                } else {
                    out.items.push_back(value());
                }
                ws();
                if (peek() == ',') {
                    ++i;
                } else {
                    if (peek() != close) ok = false;
                    ++i;
                    break;
                }
            }
        } else if (c == '"') {
            out.kind = Json::String;
            out.text = string();
        } else if (take("true")) {
            out.kind = Json::Bool;
            out.number = 1;
        } else if (take("false")) {
            out.kind = Json::Bool;
        } else if (take("null")) {
        } else {
            out.kind = Json::Number;
            char *end = nullptr;
            out.number = std::strtoll(s.c_str() + i, &end, 10);
            if (end == s.c_str() + i) ok = false;
            i = size_t(end - s.c_str());
        }
        return out;
    }
    std::string string() {
        std::string out;
        if (peek() != '"') ok = false;
        ++i;
        while (ok && i < s.size() && s[i] != '"') {
            if (s[i] == '\\') ok = false;
            out += s[i++];
        }
        ++i;
        return out;
    }
};

std::string fixtures_dir() {
    const std::string layout = CERE_LAYOUT_FIXTURE;
    return layout.substr(0, layout.rfind('/'));
}

// ---- helpers ----

struct Result {
    bool ok = false;
    std::vector<int64_t> positions;  // [3][L]
    int64_t rope_delta = 0;
};

Result run(uint64_t input_len, const std::vector<ImageSpan> &spans) {
    Result r;
    r.positions.assign(3 * input_len, -7);
    r.rope_delta = 12345;
    r.ok = cere::mrope::positions(input_len, spans, r.positions.data(), &r.rope_delta);
    return r;
}

// ---- fixture ----

void test_fixture() {
    std::ifstream in(fixtures_dir() + "/native/mrope_cases.json", std::ios::binary);
    CHECK(bool(in));
    if (!in) return;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    Parser parser{text};
    const Json doc = parser.value();
    CHECK(parser.ok);
    CHECK(doc["schema"].text == "cere-mrope-cases-v1");
    CHECK(doc["model_type"].text == "qwen4_exp");
    CHECK_EQ(doc["spatial_merge_size"].number, cere::mrope::kSpatialMergeSize);
    const Json &cases = doc["cases"];
    CHECK(cases.kind == Json::Array && cases.items.size() >= 10);

    size_t images_max = 0, checked = 0;
    bool text_only = false, image_first = false, image_last = false, adjacent = false;
    for (const Json &c : cases.items) {
        const uint64_t len = uint64_t(c["input_len"].number);
        std::vector<ImageSpan> spans;
        for (const Json &image : c["images"].items) {
            ImageSpan span{};
            span.token_offset = uint64_t(image["offset"].number);
            span.grid_t = uint32_t(image["grid"][0].number);
            span.grid_h = uint32_t(image["grid"][1].number);
            span.grid_w = uint32_t(image["grid"][2].number);
            CHECK_EQ(cere::mrope::image_tokens(span), image["tokens"].number);
            spans.push_back(span);
        }
        const Json &want = c["positions"];
        CHECK_EQ(want.items.size(), 3);
        for (const Json &row : want.items) CHECK_EQ(row.items.size(), len);
        if (want.items.size() != 3) continue;

        const Result got = run(len, spans);
        CHECK(got.ok);
        size_t mismatches = 0;
        for (size_t axis = 0; axis < 3 && axis < want.items.size(); ++axis) {
            for (size_t j = 0; j < len && j < want[axis].items.size(); ++j) {
                if (got.positions[axis * len + j] != want[axis][j].number) {
                    if (mismatches++ == 0) {
                        std::fprintf(stderr, "%s: positions[%zu][%zu] = %lld, want %lld\n", c["name"].text.c_str(),
                                     axis, j, (long long)got.positions[axis * len + j], want[axis][j].number);
                    }
                }
            }
        }
        CHECK_EQ(mismatches, 0);
        CHECK_EQ(got.rope_delta, c["rope_delta"].number);
        ++checked;

        images_max = std::max(images_max, spans.size());
        text_only |= spans.empty();
        if (!spans.empty()) {
            image_first |= spans.front().token_offset == 1;
            image_last |= spans.back().token_offset + cere::mrope::image_tokens(spans.back()) + 1 == len;
        }
        for (size_t k = 1; k < spans.size(); ++k) {
            // `<|vision_end|><|vision_start|>` is all that separates adjacent images
            adjacent |= spans[k - 1].token_offset + cere::mrope::image_tokens(spans[k - 1]) + 2 == spans[k].token_offset;
        }
    }
    CHECK_EQ(checked, cases.items.size());
    CHECK_EQ(images_max, 4);
    CHECK(text_only && image_first && image_last && adjacent);
    std::printf("test_mrope: %zu fixture cases equal SGLang\n", checked);
}

// ---- fixture-free ----

void test_text_only_is_arange() {
    for (uint64_t len : {1u, 2u, 17u, 4096u}) {
        const Result r = run(len, {});
        CHECK(r.ok);
        CHECK_EQ(r.rope_delta, 0);
        bool arange = true;
        for (uint64_t axis = 0; axis < 3; ++axis) {
            for (uint64_t j = 0; j < len; ++j) arange &= r.positions[axis * len + j] == int64_t(j);
        }
        CHECK(arange);
    }
}

// Hand-computed: 2 text tokens, one image t=2 h=4 w=6 (2 x 2 x 3 = 12 pads) at offset 2, then 3 text tokens.
void test_hand_case() {
    const Result r = run(17, {{2, 2, 4, 6}});
    CHECK(r.ok);
    const int64_t want[3][17] = {
        {0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 5, 6, 7},
        {0, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 5, 6, 7},
        {0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6, 7},
    };
    for (int axis = 0; axis < 3; ++axis) {
        for (int j = 0; j < 17; ++j) CHECK_EQ(r.positions[axis * 17 + j], want[axis][j]);
    }
    // max position 7 + 1 - 17
    CHECK_EQ(r.rope_delta, -9);
}

// Pads ending at the last prompt token are accepted (SGLang refuses only end >= input_len).
void test_image_at_end() {
    const Result tail = run(3, {{1, 1, 2, 4}});  // 1 x 1 x 2 = 2 pads at rows 1..2, the last prompt token
    CHECK(tail.ok);
    CHECK_EQ(tail.positions[0 * 3 + 2], 1);
    CHECK_EQ(tail.positions[2 * 3 + 2], 2);
    CHECK_EQ(tail.rope_delta, 3 - 3);
    CHECK(!run(2, {{1, 1, 2, 4}}).ok);  // pads would run past the prompt
}

// rope_delta = next position - L <= 0 for any valid layout: an image advances by max(t, h/2, w/2) <= its token count.
void test_rope_delta_nonpositive() {
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    auto next = [&seed](uint32_t mod) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return uint32_t(seed % mod);
    };
    for (int trial = 0; trial < 500; ++trial) {
        std::vector<ImageSpan> spans;
        uint64_t len = 1 + next(5);
        const uint32_t images = next(5);
        for (uint32_t k = 0; k < images; ++k) {
            ImageSpan span{len + 1, 1 + next(2), 2 * (1 + next(40)), 2 * (1 + next(40))};
            spans.push_back(span);
            len = span.token_offset + cere::mrope::image_tokens(span) + 1 + next(4);
        }
        const Result r = run(len, spans);
        CHECK(r.ok);
        CHECK(r.rope_delta <= 0);
        int64_t max = -1;
        for (int64_t p : r.positions) max = std::max(max, p);
        CHECK_EQ(r.rope_delta, max + 1 - int64_t(len));
    }
}

// Inputs SGLang's image-only path refuses (returns None) or that no image grid produces; outputs stay untouched.
void test_invalid() {
    const std::vector<std::pair<uint64_t, std::vector<ImageSpan>>> bad = {
        {0, {}},                                    // empty prompt
        {10, {{3, 1, 4, 4}, {5, 1, 2, 2}}},         // overlapping spans
        {20, {{8, 1, 2, 2}, {3, 1, 2, 2}}},         // not in prompt order
        {10, {{1, 1, 3, 4}}},                       // h not a multiple of the merge size
        {10, {{1, 1, 4, 5}}},                       // w not a multiple of the merge size
        {10, {{1, 0, 4, 4}}},                       // t = 0
        {10, {{1, 1, 0, 4}}},                       // no pads
        {10, {{9, 1, 2, 4}}},                       // pads past the prompt
        {10, {{uint64_t(1) << 62, 1, 2, 2}}},       // offset far past the prompt
        {10, {{1, 0xffffffffu, 0xfffffffeu, 0xfffffffeu}}},  // token count overflow
    };
    for (const auto &[len, spans] : bad) {
        std::vector<int64_t> positions(3 * len, -7);
        int64_t delta = 12345;
        CHECK(!cere::mrope::positions(len, spans, positions.data(), &delta));
        CHECK_EQ(delta, 12345);
        CHECK(std::all_of(positions.begin(), positions.end(), [](int64_t p) { return p == -7; }));
    }
}

}  // namespace

int main() {
    test_fixture();
    test_text_only_is_arange();
    test_hand_case();
    test_image_at_end();
    test_rope_delta_nonpositive();
    test_invalid();
    return cere::test::finish("test_mrope");
}
