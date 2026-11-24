// AmdAmd PAR-6 step 9: load libcere_executor_hip.so as cered does (dlopen, ABI version first) and check the device
// bring-up of the hipified CudaDevice (executor/cuda/src/device/cuda_device.cpp through gen_hip.py) on gfx1151, with
// the CUDA device's id grammar ("N[,option=value...]", N a decimal ordinal):
//   default:  "0" creates an executor (HIP_ENABLE_DEFERRED_LOADING=0 set: eager code-object loading) and destroys;
//             "7" (no such GPU), "x", "hip", "toy", "cuda:0" are CERE_ERR_UNSUPPORTED; "", "-1", "00" and an unknown
//             option are CERE_ERR_ARGUMENT;
//   --lazy:   without eager loading "0" is CERE_ERR_UNSUPPORTED (the device's require_eager_loading).
// Exit 0 on pass.
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "cere_executor.h"

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    const bool lazy = argc > 2 && std::strcmp(argv[2], "--lazy") == 0;
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto abi = reinterpret_cast<uint32_t (*)()>(dlsym(lib, "cere_executor_abi_version"));
    auto create = reinterpret_cast<cere_status (*)(cere_executor **, const cere_executor_options *)>(dlsym(lib, "cere_executor_create"));
    auto destroy = reinterpret_cast<cere_status (*)(cere_executor *)>(dlsym(lib, "cere_executor_destroy"));
    CHECK(abi && create && destroy);
    CHECK(abi() == CERE_EXECUTOR_ABI_VERSION);
    auto make = [&](const char *id, cere_executor **out) {
        cere_executor_options o{};
        o.struct_size = sizeof o; o.abi_version = CERE_EXECUTOR_ABI_VERSION; o.device_id = id;
        o.worker_epoch = 1; o.memory_budget_bytes = 1ull << 30;
        return create(out, &o);
    };
    if (lazy) {
        cere_executor *e = nullptr;
        const cere_status st = make("0", &e);
        std::printf("create(0) without eager loading = %d\n", st);
        CHECK(st == CERE_ERR_UNSUPPORTED && !e);
        std::printf("PASS\n");
        return 0;
    }
    {
        cere_executor *e = nullptr;
        const cere_status st = make("0", &e);
        std::printf("create(0) = %d\n", st);
        CHECK(st == CERE_OK && e);
        CHECK(destroy(e) == CERE_OK);
    }
    for (const char *id : {"7", "x", "hip", "toy", "cuda:0"}) {
        cere_executor *e = nullptr;
        const cere_status st = make(id, &e);
        std::printf("create(%s) = %d\n", id, st);
        CHECK(st == CERE_ERR_UNSUPPORTED);
    }
    for (const char *id : {"", "-1", "00", "0,bogus=1"}) {
        cere_executor *e = nullptr;
        const cere_status st = make(id, &e);
        std::printf("create(\"%s\") = %d\n", id, st);
        CHECK(st == CERE_ERR_ARGUMENT);
    }
    std::printf("PASS\n");
    return 0;
}
