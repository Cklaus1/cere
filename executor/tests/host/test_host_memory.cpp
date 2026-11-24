// M2.11 snapshot arena memory (m211 design note D5, R1): the CUDA plan keeps the arena out of the model's mapped
// host block (device/memory_plan.h host_layout) while the host estimate still counts it, and the arena's own
// allocation (host/locked_memory.h) is page-aligned, zero-filled, mlock'd, excluded from core dumps and fork children,
// and fails closed (CERE_ERR_CAPACITY naming RLIMIT_MEMLOCK) when the lock limit is too small.
#include "check.h"

#include "device/memory_plan.h"
#include "host/locked_memory.h"
#include "host/memory_ledger.h"

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using cere::device::host_layout;
using cere::device::HostLayout;
using cere::host::LockedMemory;
using cere::host::Held;
using cere::host::LedgerEntry;
using cere::host::MemoryLedger;

constexpr uint64_t kGiB = uint64_t(1) << 30;

// The mapped block (cudaHostAllocMapped: HOST_PINNED tensors and staging) ends at the aligned end of its regions;
// the snapshot arena is not inside it. The host estimate (pinned_bytes) is the block plus the arena, exactly the value
// of the former layout that placed the arena at the block's 256-aligned end.
void test_layout_keeps_arena_out_of_mapped_block() {
    const HostLayout big = host_layout(1000, 4 * kGiB);
    CHECK(!big.overflow);
    CHECK_EQ(big.mapped_bytes, 1024);
    CHECK_EQ(big.snapshot_bytes, 4 * kGiB);
    CHECK_EQ(big.pinned_bytes, 1024 + 4 * kGiB);

    const HostLayout odd = host_layout(1000, 100);
    CHECK(!odd.overflow);
    CHECK_EQ(odd.mapped_bytes, 1024);
    CHECK_EQ(odd.snapshot_bytes, 100);
    CHECK_EQ(odd.pinned_bytes, 1280);  // align(1024 + 100, 256)

    const HostLayout none = host_layout(1000, 0);
    CHECK(!none.overflow);
    CHECK_EQ(none.mapped_bytes, 1024);
    CHECK_EQ(none.snapshot_bytes, 0);
    CHECK_EQ(none.pinned_bytes, 1024);

    CHECK(host_layout(UINT64_MAX - 10, 0).overflow);
    CHECK(host_layout(1024, UINT64_MAX).overflow);
}

// VmLck of this process in bytes.
uint64_t locked_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmLck:", 0) == 0) return std::strtoull(line.c_str() + 6, nullptr, 10) * 1024;
    }
    return UINT64_MAX;
}

// The VmFlags of the mapping that starts at `at` (/proc/self/smaps), space-separated; empty when not found.
std::string vm_flags(const void *at) {
    std::ifstream smaps("/proc/self/smaps");
    std::string line;
    bool in = false;
    const auto want = reinterpret_cast<uintptr_t>(at);
    while (std::getline(smaps, line)) {
        uintptr_t start = 0;
        uintptr_t end = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx ", &start, &end) == 2 && line.find(':') > line.find(' ')) {
            in = start == want;
        } else if (in && line.rfind("VmFlags:", 0) == 0) {
            return line.substr(8) + " ";
        }
    }
    return {};
}

bool has_flag(const std::string &flags, const char *flag) {
    return flags.find(std::string(" ") + flag + " ") != std::string::npos;
}

void test_locked_memory() {
    const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
    const uint64_t before = locked_bytes();
    {
        LockedMemory m;
        std::string error;
        CHECK_EQ(m.allocate(3 * page + 100, &error), CERE_OK);
        CHECK(error.empty());
        CHECK(m.data() != nullptr);
        CHECK_EQ(m.bytes(), 3 * page + 100);
        CHECK_EQ(reinterpret_cast<uintptr_t>(m.data()) % page, 0);
        bool zero = true;
        for (uint64_t i = 0; i < 4 * page; ++i) zero &= m.data()[i] == 0;
        CHECK(zero);
        CHECK_EQ(m.locked_bytes(), 4 * page);  // whole pages
        CHECK_EQ(locked_bytes(), before + m.locked_bytes());  // VmLck grows by exactly the locked pages
        const std::string flags = vm_flags(m.data());
        CHECK(has_flag(flags, "lo"));  // mlock'd
        CHECK(has_flag(flags, "dd"));  // MADV_DONTDUMP
        CHECK(has_flag(flags, "dc"));  // MADV_DONTFORK
        for (uint64_t i = 0; i < m.bytes(); ++i) m.data()[i] = 0xA5;
        CHECK_EQ(m.allocate(page, &error), CERE_ERR_INTERNAL);  // already holds memory
    }
    CHECK_EQ(locked_bytes(), before);

    LockedMemory empty;
    CHECK_EQ(empty.allocate(0, nullptr), CERE_OK);
    CHECK(empty.data() == nullptr);
    CHECK_EQ(empty.bytes(), 0);
    CHECK_EQ(empty.locked_bytes(), 0);

    LockedMemory huge;
    std::string error;
    CHECK_EQ(huge.allocate(uint64_t(1) << 62, &error), CERE_ERR_CAPACITY);
    CHECK(huge.data() == nullptr);
    CHECK(error.find("mmap") != std::string::npos);
    CHECK_EQ(huge.locked_bytes(), 0);
    CHECK_EQ(locked_bytes(), before);
}

// R1 [REVISED 2026-10-11]: the executor's own memory ledger holds exactly the bytes of the live entries, per kind;
// an entry drops its bytes once (drop, re-hold or destruction) and kinds never mix.
void test_memory_ledger() {
    MemoryLedger ledger;
    for (const Held k : {Held::kDevice, Held::kRing, Held::kMapped, Held::kLocked}) CHECK_EQ(ledger.held(k), 0);
    {
        LedgerEntry device;
        LedgerEntry ring;
        device.hold(ledger, Held::kDevice, 81610599680);
        ring.hold(ledger, Held::kRing, 4 * (uint64_t(64) << 20));
        CHECK_EQ(device.bytes(), 81610599680);
        CHECK_EQ(ledger.held(Held::kDevice), 81610599680);
        CHECK_EQ(ledger.held(Held::kRing), 268435456);
        CHECK_EQ(ledger.held(Held::kMapped), 0);
        CHECK_EQ(ledger.held(Held::kLocked), 0);
        {
            LedgerEntry arena;
            arena.hold(ledger, Held::kLocked, 4 * kGiB);
            CHECK_EQ(ledger.held(Held::kLocked), 4 * kGiB);
            arena.hold(ledger, Held::kLocked, kGiB);  // re-hold replaces, never adds
            CHECK_EQ(ledger.held(Held::kLocked), kGiB);
        }
        CHECK_EQ(ledger.held(Held::kLocked), 0);
        device.drop();
        device.drop();  // once only
        CHECK_EQ(device.bytes(), 0);
        CHECK_EQ(ledger.held(Held::kDevice), 0);
        CHECK_EQ(ledger.held(Held::kRing), 268435456);
    }
    for (const Held k : {Held::kDevice, Held::kRing, Held::kMapped, Held::kLocked}) CHECK_EQ(ledger.held(k), 0);
}

// mlock beyond RLIMIT_MEMLOCK fails closed with a message naming the limit and the bytes needed. Run in a child that
// lowers the limit to 64 KiB and, as root, drops CAP_IPC_LOCK (which ignores the limit) by switching to uid 65534,
// so this process keeps its own limit and privileges.
void test_locked_memory_rlimit() {
    std::fflush(nullptr);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        const int failed_before = cere::test::failures();
        const rlimit limit{64 << 10, 64 << 10};
        CHECK_EQ(setrlimit(RLIMIT_MEMLOCK, &limit), 0);
        if (geteuid() == 0) {
            CHECK_EQ(setgid(65534), 0);
            CHECK_EQ(setuid(65534), 0);
        }
        const uint64_t before = locked_bytes();
        LockedMemory m;
        std::string error;
        CHECK_EQ(m.allocate(uint64_t(1) << 20, &error), CERE_ERR_CAPACITY);
        CHECK(m.data() == nullptr);
        CHECK_EQ(m.bytes(), 0);
        CHECK(error.find("RLIMIT_MEMLOCK") != std::string::npos);
        CHECK(error.find("1048576") != std::string::npos);  // the bytes the limit must allow
        CHECK(error.find("65536") != std::string::npos);    // the current soft limit
        CHECK_EQ(locked_bytes(), before);
        std::fprintf(stderr, "rlimit child: %s\n", error.c_str());
        std::fflush(nullptr);
        _exit(cere::test::failures() == failed_before ? 0 : 1);
    }
    int status = 0;
    CHECK_EQ(waitpid(pid, &status, 0), pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

}  // namespace

int main() {
    test_layout_keeps_arena_out_of_mapped_block();
    test_locked_memory();
    test_memory_ledger();
    test_locked_memory_rlimit();
    return cere::test::finish("test_host_memory");
}
