// OH-08: the fault-injection sites' trigger (fault/fault_inject.h). Compiled only by a CERE_FAULT_INJECTION build
// (executor/CMakeLists.txt adds this file then, and nowhere else).
#include "fault/fault_inject.h"

#if !defined(CERE_FAULT_INJECTION)
#error "fault/fault_inject.cu belongs to a CERE_FAULT_INJECTION build only"
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cere::fault {
namespace {

constexpr const char *kSites[] = {"forward_launch", "forward_sticky", "seal", "graph_instantiate", "moe_fork",
                                  "bound_fault"};
constexpr unsigned kSiteCount = sizeof(kSites) / sizeof(kSites[0]);

struct Armed {
    unsigned site = kSiteCount;  // kSiteCount: nothing armed
    uint64_t at = 0;
};

[[noreturn]] void malformed(const char *value) {
    std::fprintf(stderr, "cere: fault_inject malformed CERE_FAULT_INJECT=%s (want <site>:<n>, n >= 1, site one of",
                 value);
    for (const char *name : kSites) std::fprintf(stderr, " %s", name);
    std::fprintf(stderr, ")\n");
    std::abort();
}

Armed parse() {
    const char *value = std::getenv("CERE_FAULT_INJECT");
    if (value == nullptr || *value == '\0') return {};
    const char *colon = std::strchr(value, ':');
    if (colon == nullptr) malformed(value);
    Armed armed;
    for (unsigned s = 0; s < kSiteCount; ++s) {
        if (std::strlen(kSites[s]) == size_t(colon - value) && std::strncmp(kSites[s], value, colon - value) == 0) {
            armed.site = s;
        }
    }
    const char *digits = colon + 1;
    if (armed.site == kSiteCount || *digits == '\0' || std::strlen(digits) > 18) malformed(value);
    for (const char *p = digits; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') malformed(value);
        armed.at = armed.at * 10 + uint64_t(*p - '0');
    }
    if (armed.at == 0) malformed(value);
    return armed;
}

std::atomic<uint64_t> g_passes{0};  // passes of the armed site so far

__global__ void trap_kernel() { __trap(); }

}  // namespace

bool fire(Site site) {
    static const Armed armed = parse();
    if (unsigned(site) != armed.site) return false;
    const uint64_t pass = g_passes.fetch_add(1) + 1;
    if (pass != armed.at) return false;
    std::fprintf(stderr, "cere: fault_inject site=%s hit=%llu\n", kSites[armed.site], (unsigned long long)pass);
    return true;
}

void trap(cudaStream_t stream) { trap_kernel<<<1, 1, 0, stream>>>(); }

}  // namespace cere::fault
