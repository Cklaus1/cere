// Minimal assertion helpers for the host-layer tests (no external test framework).
#pragma once

#include <cstdio>

namespace cere::test {
inline int &failures() {
    static int count = 0;
    return count;
}
inline int finish(const char *suite) {
    if (failures() != 0) {
        std::fprintf(stderr, "%s: %d failure(s)\n", suite, failures());
        return 1;
    }
    std::printf("%s: PASS\n", suite);
    return 0;
}
}  // namespace cere::test

#define CHECK(cond)                                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                             \
            ++cere::test::failures();                                                                                  \
        }                                                                                                              \
    } while (0)

#define CHECK_EQ(actual, expected)                                                                                    \
    do {                                                                                                               \
        const long long cere_actual_ = (long long)(actual);                                                            \
        const long long cere_expected_ = (long long)(expected);                                                        \
        if (cere_actual_ != cere_expected_) {                                                                          \
            std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s = %lld, expected %s = %lld\n", __FILE__, __LINE__,         \
                         #actual, cere_actual_, #expected, cere_expected_);                                            \
            ++cere::test::failures();                                                                                  \
        }                                                                                                              \
    } while (0)
