// VulkanAmd P3 T2: the S3 artifact's tensor table for the Vulkan device tests, built as the runtime builds the desc
// (crates/cere-runtime/src/native/upload.rs model_spec: manifest `weights` files in manifest order, each file's tensors in
// data-offset order; placement by the family rules: `.ple.ple_embedding.` HOST_PINNED, `mtp.` DEVICE with drafts else
// DISCARD, the rest DEVICE). The same rule as vk_memory_plan_test.cpp artifact_tensors, plus each tensor's file and
// absolute payload offset (for the upload). Headers only here; the payload is read by the load test.
#pragma once

#include "cere_executor.h"
#include "host/device.h"
#include "mini_json.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace vktest {

struct ArtTensor {
    cere::host::TensorPlan t;
    std::string file;      // path relative to the artifact
    uint64_t data_at = 0;  // absolute byte offset of the payload in the file
};

inline bool read_text(const std::string &path, std::string *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    *out = s.str();
    return true;
}

inline uint32_t dtype_of(const std::string &s) {
    static const std::map<std::string, uint32_t> m = {
        {"BF16", CERE_DTYPE_BF16}, {"F16", CERE_DTYPE_F16}, {"F32", CERE_DTYPE_F32}, {"F64", CERE_DTYPE_F64},
        {"F8_E4M3", CERE_DTYPE_F8_E4M3}, {"F8_E5M2", CERE_DTYPE_F8_E5M2}, {"U8", CERE_DTYPE_U8}, {"I8", CERE_DTYPE_I8},
        {"I16", CERE_DTYPE_I16}, {"I32", CERE_DTYPE_I32}, {"I64", CERE_DTYPE_I64}, {"BOOL", CERE_DTYPE_BOOL}};
    const auto it = m.find(s);
    return it == m.end() ? 0 : it->second;
}

inline bool artifact_tensors(const std::string &dir, uint32_t drafts, std::vector<ArtTensor> *out, std::string *why) {
    std::string text;
    if (!read_text(dir + "/manifest.json", &text)) {
        *why = "cannot read " + dir + "/manifest.json";
        return false;
    }
    cere::testjson::Value man;
    if (!cere::testjson::parse(text, &man) || man.get("files") == nullptr) {
        *why = "manifest.json does not parse";
        return false;
    }
    for (const cere::testjson::Value &f : man.get("files")->a) {
        const cere::testjson::Value *role = f.get("role"), *path = f.get("path");
        if (role == nullptr || path == nullptr || role->s != "weights") continue;
        std::ifstream in(dir + "/" + path->s, std::ios::binary);
        uint64_t n = 0;
        if (!in.read(reinterpret_cast<char *>(&n), 8) || n == 0 || n > (uint64_t(1) << 30)) {
            *why = "cannot read the header of " + path->s;
            return false;
        }
        std::string header(n, '\0');
        if (!in.read(header.data(), std::streamsize(n))) {
            *why = "short header in " + path->s;
            return false;
        }
        cere::testjson::Value h;
        if (!cere::testjson::parse(header, &h)) {
            *why = "header of " + path->s + " does not parse";
            return false;
        }
        std::vector<ArtTensor> entries;
        for (const auto &[name, v] : h.o) {
            if (name == "__metadata__") continue;
            const cere::testjson::Value *dt = v.get("dtype"), *shape = v.get("shape"), *off = v.get("data_offsets");
            if (dt == nullptr || shape == nullptr || off == nullptr || off->a.size() != 2) {
                *why = "tensor " + name + ": malformed header entry";
                return false;
            }
            ArtTensor a;
            a.t.name = name;
            a.t.dtype = dtype_of(dt->s);
            a.t.ndim = uint32_t(shape->a.size());
            for (uint32_t d = 0; d < a.t.ndim; ++d) a.t.shape[d] = shape->a[d].u;
            a.t.bytes = off->a[1].u - off->a[0].u;
            const bool host_table = name.find(".ple.ple_embedding.") != std::string::npos;
            const bool draft = name.rfind("mtp.", 0) == 0;
            a.t.placement = draft ? (drafts ? CERE_PLACEMENT_DEVICE : CERE_PLACEMENT_DISCARD)
                                  : host_table ? CERE_PLACEMENT_HOST_PINNED : CERE_PLACEMENT_DEVICE;
            a.file = path->s;
            a.data_at = 8 + n + off->a[0].u;
            entries.push_back(std::move(a));
        }
        std::stable_sort(entries.begin(), entries.end(),
                         [](const ArtTensor &a, const ArtTensor &b) { return a.data_at < b.data_at; });
        for (ArtTensor &e : entries) out->push_back(std::move(e));
    }
    return true;
}

inline uint64_t draft_map_count(const std::string &dir) {
    struct stat st {};
    return stat((dir + "/draft/token_map.u32le").c_str(), &st) == 0 ? uint64_t(st.st_size) / 4 : 0;
}

}  // namespace vktest
