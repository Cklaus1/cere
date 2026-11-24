// Shared HOST_PINNED segment (build-loop-ple design note, opportunity O47), driven through the C ABI against the CPU toy
// device in forked processes: two processes of one artifact map one segment (one file; each counts the region in Rss
// and about half of it in Pss), a tampered segment or differing put bytes fail closed, another artifact gets another
// segment, a stale segment without mappers is reused only when its digest verifies, the last detach unlinks, and the
// host budget counts the region only for the creator. Without the options the toy keeps private tensors.
#include "host/host_share.h"
#include "check.h"

#include "cere_executor.h"
#include "host/state.h"
#include "toy/toy_device.h"

#include <sys/resource.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kEpoch = 3;
const std::string kArtifactA(64, 'a');
const std::string kArtifactB = std::string(63, 'a') + "b";

// HOST_PINNED region of the test model: two tensors, about 4 MiB.
constexpr uint64_t kHostA = uint64_t(3) << 20;
constexpr uint64_t kHostB = (uint64_t(1) << 20) + 3;

// The largest region a test here maps (test_chunks_hash_during_the_pass: 2 chunks + 1 MiB), plus headroom for the
// smaller segments a test may hold beside it.
constexpr uint64_t kLargestRegion = 2 * cere::host::SharedSegment::kChunkBytes + (uint64_t(1) << 20);
constexpr uint64_t kDirHeadroom = uint64_t(64) << 20;

// Free bytes of the filesystem holding `dir` (0 when statvfs fails).
uint64_t free_bytes(const char *dir) {
    struct statvfs vfs{};
    if (statvfs(dir, &vfs) != 0) return 0;
    return uint64_t(vfs.f_bavail) * vfs.f_frsize;
}

std::string make_dir() {
    // tmpfs when it has room for every region here, so the segment is shmem as on the lanes (RssShmem); docker's
    // default /dev/shm (64 MiB, the CI container) is too small, so then $TMPDIR or /tmp (ALLOCATE refuses a short
    // filesystem, CERE_ERR_CAPACITY, which is the product's behaviour and not what these tests exercise).
    const char *tmp = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    const bool shm = access("/dev/shm", W_OK) == 0 && free_bytes("/dev/shm") >= kLargestRegion + kDirHeadroom;
    const char *base = shm ? "/dev/shm" : tmp;
    std::string tmpl = std::string(base) + "/cere-share-test-XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (mkdtemp(buf.data()) == nullptr) {
        std::perror("mkdtemp");
        std::exit(2);
    }
    return std::string(buf.data());
}

void remove_dir(const std::string &dir) {
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string name = e->d_name;
            if (name == "." || name == "..") continue;
            const std::string path = dir + "/" + name;
            chmod(path.c_str(), 0600);
            unlink(path.c_str());
        }
        closedir(d);
    }
    rmdir(dir.c_str());
}

// CHECK_EQ that ends the test (removing `dir`) on a mismatch, so a refused open reports a failure instead of the test
// dereferencing a null segment.
#define REQUIRE_EQ(actual, expected, dir)                                                                             \
    do {                                                                                                               \
        const long long cere_actual_ = (long long)(actual);                                                            \
        const long long cere_expected_ = (long long)(expected);                                                        \
        if (cere_actual_ != cere_expected_) {                                                                          \
            std::fprintf(stderr, "%s:%d: REQUIRE_EQ failed: %s = %lld, expected %s = %lld\n", __FILE__, __LINE__,       \
                         #actual, cere_actual_, #expected, cere_expected_);                                            \
            ++cere::test::failures();                                                                                  \
            remove_dir(dir);                                                                                           \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

// Published segments in `dir` (names "cere-host-<64 hex>", no suffix).
std::vector<std::string> segments(const std::string &dir) {
    std::vector<std::string> out;
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string name = e->d_name;
            if (name.rfind("cere-host-", 0) == 0 && name.size() == 10 + 64) out.push_back(name);
        }
        closedir(d);
    }
    return out;
}

std::string device_id(const std::string &dir, const std::string &artifact, const std::string &extra = "") {
    return "toy" + extra + ",host_share=" + dir + ",artifact=" + artifact;
}

cere_status create_executor(const std::string &id, cere_executor **out) {
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = id.c_str();
    options.worker_epoch = kEpoch;
    options.memory_budget_bytes = uint64_t(1) << 30;
    *out = nullptr;
    return cere_executor_create(out, &options);
}

// The test model: DEVICE, two HOST_PINNED, DISCARD.
struct Desc {
    cere_qwen4_exp_config config = cere::toy::fixture_config();
    std::vector<std::string> names{"dev", "ple.a", "ple.b", "drop"};
    std::vector<uint64_t> sizes{4096, kHostA, kHostB, 16};
    std::vector<uint32_t> placements{CERE_PLACEMENT_DEVICE, CERE_PLACEMENT_HOST_PINNED, CERE_PLACEMENT_HOST_PINNED,
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

// Puts every tensor (`seed` shifts the HOST_PINNED bytes) in 1 MiB pieces; the first put status other than CERE_OK.
cere_status put_all(cere_executor *executor, cere_model *model, const Desc &d, uint8_t seed) {
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        const auto bytes = pattern(d.sizes[i], uint8_t(seed + i));
        for (uint64_t at = 0; at < bytes.size(); at += uint64_t(1) << 20) {
            const uint64_t n = std::min<uint64_t>(bytes.size() - at, uint64_t(1) << 20);
            if (const cere_status st = cere_model_put(executor, model, i, at, bytes.data() + at, n); st != CERE_OK) {
                return st;
            }
        }
    }
    return CERE_OK;
}

// A child process: runs `body` (its CHECK failures are the exit status) and exits without returning.
pid_t spawn(const std::function<void()> &body) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        body();
        std::fflush(nullptr);
        _exit(cere::test::failures() == 0 ? 0 : 1);
    }
    return pid;
}

int wait_exit(pid_t pid) {
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

struct Pipe {
    int fd[2] = {-1, -1};
    Pipe() {
        if (pipe(fd) != 0) std::exit(2);
    }
    ~Pipe() {
        close(fd[0]);
        close(fd[1]);
    }
    void signal() const { (void)!write(fd[1], "x", 1); }
    bool wait() const {
        char c = 0;
        return read(fd[0], &c, 1) == 1;
    }
};

// A lane: creates the executor and model, puts, seals, checks its HOST_PINNED bytes, signals `ready` and waits for
// `go`; then releases the model and destroys the executor (the detach) unless `crash` (exit without destructors, as a
// killed process).
void lane(const std::string &id, const Pipe &ready, const Pipe &go, bool crash = false) {
    Desc d;
    cere_executor *executor = nullptr;
    CHECK_EQ(create_executor(id, &executor), CERE_OK);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    CHECK_EQ(put_all(executor, model, d, 0), CERE_OK);
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    if (model != nullptr) {
        auto &toy = static_cast<cere::toy::ToyDevice &>(*executor->device);
        for (uint64_t i = 1; i <= 2; ++i) CHECK(toy.tensor(*model->device, i) == pattern(d.sizes[i], uint8_t(i)));
        CHECK(toy.tensor(*model->device, 0) == pattern(d.sizes[0], 0));
    }
    ready.signal();
    go.wait();
    if (crash) {
        std::fflush(nullptr);
        _exit(cere::test::failures() == 0 ? 0 : 1);
    }
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Rss and Pss (kB) of the mapping of `path` in process `pid`; {-1, -1} when it has none. A smaps line whose first
// word ends with ':' is a field of the mapping above it; any other line starts a mapping ("start-end perms ... path").
std::pair<long long, long long> rss_pss(pid_t pid, const std::string &path) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/smaps");
    std::string line;
    bool inside = false;
    long long rss = -1, pss = -1;
    while (std::getline(in, line)) {
        const std::string word = line.substr(0, line.find(' '));
        if (word.empty() || word.back() != ':') {
            inside = line.size() >= path.size() && line.compare(line.size() - path.size(), path.size(), path) == 0;
            continue;
        }
        if (!inside) continue;
        long long v = 0;
        if (std::sscanf(line.c_str(), "Rss: %lld kB", &v) == 1) rss = (rss < 0 ? 0 : rss) + v;
        if (std::sscanf(line.c_str(), "Pss: %lld kB", &v) == 1) pss = (pss < 0 ? 0 : pss) + v;
    }
    return {rss, pss};
}

// Two lanes of one artifact started together: one segment, both lanes map it (Rss = the region in each, Pss about
// half), both read their bytes; the segment survives the first detach and the last one unlinks it.
void test_two_processes_share_one_segment() {
    const std::string dir = make_dir();
    const std::string id = device_id(dir, kArtifactA);
    Pipe ready1, go1, ready2, go2;
    const pid_t a = spawn([&] { lane(id, ready1, go1); });
    const pid_t b = spawn([&] { lane(id, ready2, go2); });
    CHECK(ready1.wait());
    CHECK(ready2.wait());
    const auto segs = segments(dir);
    CHECK_EQ(segs.size(), 1);
    if (segs.size() == 1) {
        const std::string path = dir + "/" + segs[0];
        const long long region_kib = (kHostA + 256 + kHostB) / 1024;  // at least the two tensors
        const auto [rss_a, pss_a] = rss_pss(a, path);
        const auto [rss_b, pss_b] = rss_pss(b, path);
        std::printf("segment rss_a=%lld pss_a=%lld rss_b=%lld pss_b=%lld kB (region >= %lld kB)\n", rss_a, pss_a,
                    rss_b, pss_b, region_kib);
        CHECK(rss_a >= region_kib);
        CHECK(rss_b >= region_kib);
        // Pss splits the shared pages between the two mappers: about half of Rss in each, summing to one copy.
        CHECK(pss_a * 2 <= rss_a + 8 && pss_a * 2 >= rss_a - 8);
        CHECK(pss_b * 2 <= rss_b + 8 && pss_b * 2 >= rss_b - 8);
        CHECK(pss_a + pss_b <= rss_a + 8);
        struct stat st{};
        CHECK_EQ(stat(path.c_str(), &st), 0);
        CHECK_EQ(st.st_mode & 0777, 0400);
    }
    go1.signal();
    CHECK_EQ(wait_exit(a), 0);
    CHECK_EQ(segments(dir).size(), 1);  // b still maps it
    go2.signal();
    CHECK_EQ(wait_exit(b), 0);
    CHECK_EQ(segments(dir).size(), 0);  // the last detach unlinked it
    remove_dir(dir);
}

// m216 design note L4: with host_share_keep=1 the last detach leaves the published segment; a later lane attaches to
// it (digest verified), and the next last detach of a lane started without the option removes it.
void test_keep_leaves_the_published_segment() {
    const std::string dir = make_dir();
    Pipe ready1, go1;
    const pid_t a = spawn([&] { lane(device_id(dir, kArtifactA) + ",host_share_keep=1", ready1, go1); });
    CHECK(ready1.wait());
    go1.signal();
    CHECK_EQ(wait_exit(a), 0);
    CHECK_EQ(segments(dir).size(), 1);  // kept
    Pipe ready2, go2;
    const pid_t b = spawn([&] { lane(device_id(dir, kArtifactA) + ",host_share_keep=1", ready2, go2); });
    CHECK(ready2.wait());
    go2.signal();
    CHECK_EQ(wait_exit(b), 0);
    CHECK_EQ(segments(dir).size(), 1);  // the attacher keeps it too
    Pipe ready3, go3;
    const pid_t c = spawn([&] { lane(device_id(dir, kArtifactA), ready3, go3); });
    CHECK(ready3.wait());
    go3.signal();
    CHECK_EQ(wait_exit(c), 0);
    CHECK_EQ(segments(dir).size(), 0);  // a lane without keep is the last mapper: removed
    remove_dir(dir);
}

// m216 design note L4 (C3): cere_model_host_resident covers a HOST_PINNED tensor of an attached model without a put
// and returns the segment's bytes; UNSUPPORTED for a creator and a private copy; ARGUMENT for a DEVICE or DISCARD
// tensor, a covered tensor, a null `out` and a sealed model.
void test_host_resident() {
    const std::string dir = make_dir();
    const std::string keep = device_id(dir, kArtifactA) + ",host_share_keep=1";
    const auto unsupported_before_put = [](const std::string &id) {
        Desc d;
        cere_executor *executor = nullptr;
        CHECK_EQ(create_executor(id, &executor), CERE_OK);
        cere_model *model = nullptr;
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
        const void *p = &d;
        CHECK_EQ(cere_model_host_resident(executor, model, 1, &p), CERE_ERR_UNSUPPORTED);
        CHECK(p == nullptr);
        CHECK_EQ(put_all(executor, model, d, 0), CERE_OK);  // the failed call covered nothing
        CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    };
    // A private copy, then the creator of the kept segment.
    CHECK_EQ(wait_exit(spawn([&] { unsupported_before_put("toy"); })), 0);
    CHECK_EQ(wait_exit(spawn([&] { unsupported_before_put(keep); })), 0);
    CHECK_EQ(segments(dir).size(), 1);
    // The attacher: no put of either HOST_PINNED tensor.
    CHECK_EQ(wait_exit(spawn([&] {
                 Desc d;
                 cere_executor *executor = nullptr;
                 CHECK_EQ(create_executor(device_id(dir, kArtifactA), &executor), CERE_OK);
                 cere_model *model = nullptr;
                 CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
                 const void *p = nullptr;
                 CHECK_EQ(cere_model_host_resident(executor, model, 0, &p), CERE_ERR_ARGUMENT);  // DEVICE
                 CHECK_EQ(cere_model_host_resident(executor, model, 3, &p), CERE_ERR_ARGUMENT);  // DISCARD
                 CHECK_EQ(cere_model_host_resident(executor, model, 1, nullptr), CERE_ERR_ARGUMENT);
                 for (uint64_t i = 1; i <= 2; ++i) {
                     p = nullptr;
                     CHECK_EQ(cere_model_host_resident(executor, model, i, &p), CERE_OK);
                     const auto want = pattern(d.sizes[i], uint8_t(i));
                     CHECK(p != nullptr && std::memcmp(p, want.data(), want.size()) == 0);
                     CHECK_EQ(cere_model_host_resident(executor, model, i, &p), CERE_ERR_ARGUMENT);  // covered
                     const uint8_t byte = 0;
                     CHECK_EQ(cere_model_put(executor, model, i, 0, &byte, 1), CERE_ERR_ARGUMENT);
                 }
                 for (uint64_t i : {uint64_t(0), uint64_t(3)}) {
                     const auto bytes = pattern(d.sizes[i], uint8_t(i));
                     CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
                 }
                 CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
                 CHECK_EQ(cere_model_host_resident(executor, model, 1, &p), CERE_ERR_ARGUMENT);  // sealed
                 auto &toy = static_cast<cere::toy::ToyDevice &>(*executor->device);
                 for (uint64_t i = 0; i <= 2; ++i) CHECK(toy.tensor(*model->device, i) == pattern(d.sizes[i], uint8_t(i)));
                 CHECK_EQ(cere_model_release(executor, model), CERE_OK);
                 CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
             })),
             0);
    CHECK_EQ(segments(dir).size(), 0);  // the attacher without keep was the last mapper
    remove_dir(dir);
}

// A different artifact gets a different segment (both lanes alive at once).
void test_different_artifact_gets_another_segment() {
    const std::string dir = make_dir();
    Pipe ready1, go1, ready2, go2;
    const pid_t a = spawn([&] { lane(device_id(dir, kArtifactA), ready1, go1); });
    CHECK(ready1.wait());
    const pid_t b = spawn([&] { lane(device_id(dir, kArtifactB), ready2, go2); });
    CHECK(ready2.wait());
    CHECK_EQ(segments(dir).size(), 2);
    go1.signal();
    go2.signal();
    CHECK_EQ(wait_exit(a), 0);
    CHECK_EQ(wait_exit(b), 0);
    CHECK_EQ(segments(dir).size(), 0);
    remove_dir(dir);
}

// Flips one byte of the region of the published segment `path` (the header is the first 4,096 bytes).
void tamper(const std::string &path) {
    CHECK_EQ(chmod(path.c_str(), 0600), 0);
    const int fd = open(path.c_str(), O_RDWR);
    CHECK(fd >= 0);
    uint8_t byte = 0;
    CHECK_EQ(pread(fd, &byte, 1, 4096 + 100), 1);
    byte ^= 0xff;
    CHECK_EQ(pwrite(fd, &byte, 1, 4096 + 100), 1);
    close(fd);
    CHECK_EQ(chmod(path.c_str(), 0400), 0);
}

// m216 design note L4c: an attach checks the header at ALLOCATE and the content digest while UPLOAD runs; a tampered
// segment fails closed at INSTANTIATE: the model is never sealed (release only), nothing private instead, and the
// segment stays for the operator. `id` attaches; the HOST_PINNED tensors are covered without a put, so the digest is
// the only check that sees the tampered byte.
void attach_fails_at_seal(const std::string &id) {
    Desc d;
    cere_executor *executor = nullptr;
    CHECK_EQ(create_executor(id, &executor), CERE_OK);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    for (uint64_t i = 1; i <= 2; ++i) {
        const void *p = nullptr;
        CHECK_EQ(cere_model_host_resident(executor, model, i, &p), CERE_OK);
    }
    for (uint64_t i : {uint64_t(0), uint64_t(3)}) {
        const auto bytes = pattern(d.sizes[i], uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_ERR_INTERNAL);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// A tampered segment fails closed (above) and stays for the operator; the lane that created it keeps running.
void test_tampered_segment_fails_closed() {
    const std::string dir = make_dir();
    const std::string id = device_id(dir, kArtifactA);
    Pipe ready1, go1;
    const pid_t a = spawn([&] { lane(id, ready1, go1); });
    CHECK(ready1.wait());
    const auto segs = segments(dir);
    CHECK_EQ(segs.size(), 1);
    if (segs.size() == 1) tamper(dir + "/" + segs[0]);
    CHECK_EQ(wait_exit(spawn([&] { attach_fails_at_seal(id); })), 0);
    CHECK_EQ(segments(dir).size(), 1);
    go1.signal();
    CHECK_EQ(wait_exit(a), 0);
    remove_dir(dir);
}

// Put bytes that differ from an attached segment (an artifact whose host bytes differ under the same key) fail closed
// at the put.
void test_differing_put_fails_closed() {
    const std::string dir = make_dir();
    const std::string id = device_id(dir, kArtifactA);
    Pipe ready1, go1;
    const pid_t a = spawn([&] { lane(id, ready1, go1); });
    CHECK(ready1.wait());
    const pid_t b = spawn([&] {
        Desc d;
        cere_executor *executor = nullptr;
        CHECK_EQ(create_executor(id, &executor), CERE_OK);
        cere_model *model = nullptr;
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
        CHECK_EQ(put_all(executor, model, d, 1), CERE_ERR_INTERNAL);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    });
    CHECK_EQ(wait_exit(b), 0);
    CHECK_EQ(segments(dir).size(), 1);  // a still maps it
    go1.signal();
    CHECK_EQ(wait_exit(a), 0);
    CHECK_EQ(segments(dir).size(), 0);
    remove_dir(dir);
}

// A lane killed without detaching leaves its published segment with no mapper: the next lane reuses it when the
// digest verifies (and, as its last mapper, unlinks it), and fails closed when it does not.
void test_stale_segment_reused_only_when_verified() {
    const std::string dir = make_dir();
    const std::string id = device_id(dir, kArtifactA);
    {
        Pipe ready1, go1;
        const pid_t a = spawn([&] { lane(id, ready1, go1, /*crash=*/true); });
        CHECK(ready1.wait());
        go1.signal();
        CHECK_EQ(wait_exit(a), 0);
    }
    CHECK_EQ(segments(dir).size(), 1);
    {
        Pipe ready2, go2;
        const pid_t b = spawn([&] { lane(id, ready2, go2); });
        CHECK(ready2.wait());
        go2.signal();
        CHECK_EQ(wait_exit(b), 0);
    }
    CHECK_EQ(segments(dir).size(), 0);
    {
        Pipe ready1, go1;
        const pid_t a = spawn([&] { lane(id, ready1, go1, /*crash=*/true); });
        CHECK(ready1.wait());
        go1.signal();
        CHECK_EQ(wait_exit(a), 0);
    }
    const auto segs = segments(dir);
    CHECK_EQ(segs.size(), 1);
    if (segs.size() == 1) tamper(dir + "/" + segs[0]);
    const pid_t c = spawn([&] { attach_fails_at_seal(id); });
    CHECK_EQ(wait_exit(c), 0);
    CHECK_EQ(segments(dir).size(), 1);  // kept for the operator
    remove_dir(dir);
}

// The host budget counts the region only for the creator: below it the first lane is CAPACITY, while a lane that
// attaches to a published segment fits.
void test_budget_counts_the_region_once() {
    const std::string dir = make_dir();
    const std::string small = ",host_bytes=" + std::to_string(uint64_t(1) << 20);
    const pid_t first = spawn([&] {
        Desc d;
        cere_executor *executor = nullptr;
        CHECK_EQ(create_executor(device_id(dir, kArtifactA, small), &executor), CERE_OK);
        cere_model *model = nullptr;
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_CAPACITY);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    });
    CHECK_EQ(wait_exit(first), 0);
    CHECK_EQ(segments(dir).size(), 0);
    Pipe ready1, go1, ready2, go2;
    const pid_t a = spawn([&] { lane(device_id(dir, kArtifactA), ready1, go1); });
    CHECK(ready1.wait());
    const pid_t b = spawn([&] { lane(device_id(dir, kArtifactA, small), ready2, go2); });
    CHECK(ready2.wait());
    go2.signal();
    CHECK_EQ(wait_exit(b), 0);
    go1.signal();
    CHECK_EQ(wait_exit(a), 0);
    remove_dir(dir);
}

// The options come together and well formed; without them nothing is written to any directory.
void test_options() {
    const std::string dir = make_dir();
    cere_executor *executor = nullptr;
    for (const std::string &bad :
         {"toy,host_share=" + dir, "toy,artifact=" + kArtifactA, "toy,host_share=relative,artifact=" + kArtifactA,
          "toy,host_share=" + dir + ",artifact=" + std::string(63, 'a'),
          "toy,host_share=" + dir + ",artifact=" + std::string(64, 'A'),
          device_id(dir, kArtifactA) + ",artifact=" + kArtifactA, std::string("toy,host_share_keep=1"),
          device_id(dir, kArtifactA) + ",host_share_keep=0", device_id(dir, kArtifactA) + ",host_share_keep=yes",
          device_id(dir, kArtifactA) + ",host_share_keep=1,host_share_keep=1"}) {
        CHECK_EQ(create_executor(bad, &executor), CERE_ERR_ARGUMENT);
        CHECK(executor == nullptr);
    }
    Pipe ready, go;
    const pid_t a = spawn([&] { lane("toy", ready, go); });
    CHECK(ready.wait());
    go.signal();
    CHECK_EQ(wait_exit(a), 0);
    DIR *d = opendir(dir.c_str());
    int entries = 0;
    while (dirent *e = readdir(d)) entries += std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0;
    closedir(d);
    CHECK_EQ(entries, 0);
    remove_dir(dir);
}

// DM-6: a digest that cannot be computed for want of memory throws; it is never a zero digest (which a publish would
// record and every attach would then fail to verify). The allocation failure is made deterministic with an address
// space limit just above the process's size.
void test_region_digest_never_made_up() {
    std::vector<uint8_t> small(4096, 7);
    const std::array<uint8_t, 32> good = cere::host::region_digest(small.data(), small.size());
    CHECK((good != std::array<uint8_t, 32>{}));

    uint64_t pages = 0;
    {
        std::ifstream statm("/proc/self/statm");
        statm >> pages;
    }
    rlimit saved{};
    CHECK_EQ(getrlimit(RLIMIT_AS, &saved), 0);
    rlimit tight = saved;
    tight.rlim_cur = pages * uint64_t(sysconf(_SC_PAGESIZE)) + (uint64_t(64) << 20);
    if (saved.rlim_max != RLIM_INFINITY && tight.rlim_cur > saved.rlim_max) tight.rlim_cur = saved.rlim_max;
    CHECK_EQ(setrlimit(RLIMIT_AS, &tight), 0);
    bool threw = false;
    try {
        // 2^50 B is 2^24 chunks: 512 MiB of per-chunk digests, past the limit; the data is never read.
        (void)cere::host::region_digest(nullptr, uint64_t(1) << 50);
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    CHECK_EQ(setrlimit(RLIMIT_AS, &saved), 0);
    CHECK(threw);
    CHECK(cere::host::region_digest(small.data(), small.size()) == good);  // the limit left no state behind
}

// beat-sglang design note L9 (1): a create no longer zeroes the file at open (posix_fallocate allocated and zeroed
// every page on one thread): the .tmp is sparse until `populate`, which allocates every page; a filesystem too small
// for the segment is still refused at open, CERE_ERR_CAPACITY.
void test_create_allocates_at_populate() {
    const std::string dir = make_dir();
    cere::host::ShareConfig cfg;
    cfg.dir = dir;
    cfg.artifact = kArtifactA;
    const std::string name = cere::host::segment_name(cfg, "l9-test", "sparse");
    const uint64_t bytes = uint64_t(8) << 20;
    std::unique_ptr<cere::host::SharedSegment> s;
    std::string error;
    REQUIRE_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, {}, &s, &error), CERE_OK, dir);
    struct stat st{};
    CHECK_EQ(stat((dir + "/" + name + ".tmp").c_str(), &st), 0);
    CHECK(uint64_t(st.st_blocks) * 512 < bytes / 2);  // sparse: open allocated (almost) nothing
    CHECK_EQ(s->populate(4, &error), CERE_OK);
    CHECK_EQ(stat((dir + "/" + name + ".tmp").c_str(), &st), 0);
    CHECK(uint64_t(st.st_blocks) * 512 >= bytes);  // every page of the region allocated
    s.reset();
    CHECK_EQ(segments(dir).size(), 0);

    struct statvfs vfs{};
    CHECK_EQ(statvfs(dir.c_str(), &vfs), 0);
    const uint64_t too_big = uint64_t(vfs.f_blocks) * vfs.f_frsize + (uint64_t(1) << 30);
    const std::string big = cere::host::segment_name(cfg, "l9-test", "too big");
    std::unique_ptr<cere::host::SharedSegment> refused;
    CHECK_EQ(cere::host::SharedSegment::open(cfg, big, too_big, false, {}, &refused, &error), CERE_ERR_CAPACITY);
    CHECK(refused == nullptr);
    remove_dir(dir);
}

// L9 (3): the write that completes a chunk's fill bytes hashes that chunk during the pass (writes on several threads,
// out of order, a gap between the fill ranges); the published content digest is the region_digest of the bytes,
// which an attach verifies.
void test_chunks_hash_during_the_pass() {
    const std::string dir = make_dir();
    cere::host::ShareConfig cfg;
    cfg.dir = dir;
    cfg.artifact = kArtifactA;
    const std::string name = cere::host::segment_name(cfg, "l9-test", "chunks");
    constexpr uint64_t kChunk = cere::host::SharedSegment::kChunkBytes;
    const uint64_t bytes = 2 * kChunk + (uint64_t(1) << 20);
    // Two fill ranges: [0, chunk + 4096) and [chunk + 8192, bytes); the 4 KiB between them is never written.
    const cere::host::RegionRanges fill = {{kChunk + 8192, bytes - kChunk - 8192}, {0, kChunk + 4096}};
    std::unique_ptr<cere::host::SharedSegment> s;
    std::string error;
    REQUIRE_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, fill, &s, &error), CERE_OK, dir);
    CHECK_EQ(s->chunks(), 3);
    std::vector<uint8_t> want(bytes, 0);
    for (uint64_t i = 0; i < bytes; ++i) want[i] = uint8_t((i * 2654435761u) >> 13);
    for (uint64_t i = kChunk + 4096; i < kChunk + 8192; ++i) want[i] = 0;
    // The last chunk first, then the first chunk's ranges on four threads; the middle chunk is completed last.
    const uint64_t piece = uint64_t(1) << 20;
    CHECK_EQ(s->write(2 * kChunk, want.data() + 2 * kChunk, bytes - 2 * kChunk, &error), CERE_OK);
    CHECK_EQ(s->chunks_hashed(), 1);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (uint64_t at = uint64_t(t) * piece; at < kChunk; at += 4 * piece) {
                std::string e;
                CHECK_EQ(s->write(at, want.data() + at, piece, &e), CERE_OK);
            }
        });
    }
    for (std::thread &t : threads) t.join();
    CHECK_EQ(s->chunks_hashed(), 2);
    CHECK_EQ(s->write(kChunk + 8192, want.data() + kChunk + 8192, kChunk - 8192, &error), CERE_OK);
    CHECK_EQ(s->chunks_hashed(), 2);  // the middle chunk still lacks [chunk, chunk + 4096)
    CHECK_EQ(s->write(kChunk, want.data() + kChunk, 4096, &error), CERE_OK);
    CHECK_EQ(s->chunks_hashed(), 3);
    CHECK_EQ(s->publish(&error), CERE_OK);
    CHECK(std::memcmp(s->data(), want.data(), bytes) == 0);

    // An attach of the published file verifies the digest the chunk hashes made.
    std::unique_ptr<cere::host::SharedSegment> a;
    REQUIRE_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, {}, &a, &error), CERE_OK, dir);
    CHECK(!a->creating());
    CHECK_EQ(a->verify(&error), CERE_OK);
    a.reset();
    s.reset();
    remove_dir(dir);
}

// L9 (3): fill ranges that overlap or leave the region are refused; a creator whose writes do not complete the
// declared fill (here: none) still publishes the correct digest, its chunks hashed at publish.
void test_unfilled_chunks_hash_at_publish() {
    const std::string dir = make_dir();
    cere::host::ShareConfig cfg;
    cfg.dir = dir;
    cfg.artifact = kArtifactA;
    const std::string name = cere::host::segment_name(cfg, "l9-test", "unfilled");
    const uint64_t bytes = uint64_t(3) << 20;
    std::unique_ptr<cere::host::SharedSegment> s;
    std::string error;
    CHECK_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, {{0, 4096}, {2048, 4096}}, &s, &error),
             CERE_ERR_INTERNAL);
    CHECK_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, {{bytes - 10, 11}}, &s, &error),
             CERE_ERR_INTERNAL);
    REQUIRE_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, {{0, bytes}}, &s, &error), CERE_OK, dir);
    const std::vector<uint8_t> half(bytes / 2, 0x5a);
    CHECK_EQ(s->write(0, half.data(), half.size(), &error), CERE_OK);
    CHECK_EQ(s->chunks_hashed(), 0);
    CHECK_EQ(s->publish(&error), CERE_OK);
    std::vector<uint8_t> want(bytes, 0);
    std::memset(want.data(), 0x5a, half.size());
    std::unique_ptr<cere::host::SharedSegment> a;
    REQUIRE_EQ(cere::host::SharedSegment::open(cfg, name, bytes, false, {}, &a, &error), CERE_OK, dir);
    CHECK(std::memcmp(a->data(), want.data(), bytes) == 0);
    CHECK_EQ(a->verify(&error), CERE_OK);
    a.reset();
    s.reset();
    remove_dir(dir);
}

}  // namespace

int main() {
    test_create_allocates_at_populate();
    test_chunks_hash_during_the_pass();
    test_unfilled_chunks_hash_at_publish();
    test_region_digest_never_made_up();
    test_options();
    test_keep_leaves_the_published_segment();
    test_host_resident();
    test_two_processes_share_one_segment();
    test_different_artifact_gets_another_segment();
    test_tampered_segment_fails_closed();
    test_differing_put_fails_closed();
    test_stale_segment_reused_only_when_verified();
    test_budget_counts_the_region_once();
    return cere::test::finish("test_host_share");
}
