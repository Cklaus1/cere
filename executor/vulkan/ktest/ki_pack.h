// Inkling S11 / S12 harnesses (ki_forward, ki_vision): the S0 artifact's safetensors loaded tensor by tensor into
// host-visible device buffers, exact checkpoint bytes (UMA: AMD's RADV, lavapipe). TEST ONLY.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "device/inkling/ki_model.h"
#include "launch/vk_launch.h"
#include "tests/mini_json.h"

namespace kipack {

[[noreturn]] inline void die(const std::string &m) {
    std::fprintf(stderr, "ki: %s\n", m.c_str());
    std::exit(2);
}

struct Pack {
    int fd = -1;
    uint64_t data0 = 0;
    struct T {
        uint64_t off = 0, bytes = 0;
    };
    std::map<std::string, T> t;
    void open(const std::string &path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) die("cannot open " + path);
        uint64_t n = 0;
        if (pread(fd, &n, 8, 0) != 8 || n > (1u << 30)) die("bad safetensors header");
        std::string hdr(n, '\0');
        if (pread(fd, hdr.data(), n, 8) != ssize_t(n)) die("short header read");
        cere::testjson::Value v;
        if (!cere::testjson::parse(hdr, &v) || v.kind != cere::testjson::Value::kObj) die("header JSON");
        data0 = 8 + n;
        for (const auto &kv : v.o) {
            if (kv.first == "__metadata__") continue;
            const auto &o = kv.second.get("data_offsets")->a;
            t[kv.first] = T{o[0].u, o[1].u - o[0].u};
        }
    }
};

// Every loaded tensor in its own host-visible device buffer, exact bytes.
struct Weights {
    cere::vk::Context &ctx;
    Pack &pk;
    uint64_t total = 0;
    std::map<std::string, cere::vkink::TensorView> views;
    std::vector<cere::vk::Buffer> owned;
    ~Weights() {
        for (cere::vk::Buffer &b : owned) ctx.free(b);
    }
    void load(const std::string &name) {
        auto it = pk.t.find(name);
        if (it == pk.t.end()) return;  // the model's init names it
        const uint64_t n = it->second.bytes;
        cere::vk::Buffer b;
        std::string why;
        if (ctx.allocate(n + 256, b, false, &why) != CERE_OK) die("allocate " + name + ": " + why);
        if (b.map == nullptr) die("device memory is not host-visible (UMA required)");
        owned.push_back(b);
        uint64_t done = 0;
        while (done < n) {
            const ssize_t r = pread(pk.fd, static_cast<uint8_t *>(b.map) + done, std::min<uint64_t>(n - done, 1u << 28),
                                    off_t(pk.data0 + it->second.off + done));
            if (r <= 0) die("read " + name);
            done += uint64_t(r);
        }
        std::memset(static_cast<uint8_t *>(b.map) + n, 0, 256);
        posix_fadvise(pk.fd, off_t(pk.data0 + it->second.off), off_t(n), POSIX_FADV_DONTNEED);
        total += n;
        views[name] = cere::vkink::TensorView{b.addr, b.map, n};
    }
    cere::vkink::Find find() const {
        return [this](const std::string &n, cere::vkink::TensorView *out) {
            auto it = views.find(n);
            if (it == views.end()) return false;
            *out = it->second;
            return true;
        };
    }
};

}  // namespace kipack
