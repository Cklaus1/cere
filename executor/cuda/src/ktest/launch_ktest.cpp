// SelfTune ST1 launch-sequence equality (selftune design note ST1, E3; executor/tests/gpu/test_graph_launches.py):
// a recorder of every CUDA graph the process instantiates or updates in place, for comparing the launch sequence of
// the forward's captured graphs (decode, verify, MTP draft) across libraries. It subscribes to CUPTI's driver-API
// callbacks (libcupti is loaded with dlopen while recording, so neither library links it) and, when an instantiate
// or an in-place update returns CUDA_SUCCESS, walks the graph it was given: per node in the graph's node order its
// kind and launch shape (a kernel: symbol, grid, block, dynamic shared memory, its programmatic stream serialization
// attribute; a copy: memory types and extent; a memset: element size, extent, value), then every edge with its
// dependency type and ports (a PDL launch is a programmatic edge). With eager launches on (mode 3, ST1d: prefill
// runs eagerly), every kernel launch, copy and memset issued to a stream that is not capturing, and every graph
// launch, is a record of its own in issue order, in the same node format (a kernel's PDL attribute is its launch
// config's). No device pointer or kernel argument is recorded, so equal forwards give equal text in any process. The
// driver calls are resolved through the runtime's cudaGetDriverEntryPointByVersion at CUDA 13.0 (no libcuda link).
// Nothing here runs unless a test starts it. cere_ktest_model_arena reads the five arena sizes of a ktest model's
// memory plan (device/memory_plan.h), so the suite can compare them per case as well (O279 D-b8).
#include "cere_executor.h"
#include "device/cuda_device.h"
#include "device/memory_plan.h"
#include "host/state.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using GetNodesFn = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
using GetEdgesFn = CUresult (*)(CUgraph, CUgraphNode *, CUgraphNode *, CUgraphEdgeData *, size_t *);
using NodeTypeFn = CUresult (*)(CUgraphNode, CUgraphNodeType *);
using KernelParamsFn = CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
using KernelAttrFn = CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue *);
using FuncNameFn = CUresult (*)(const char **, CUfunction);
using KernNameFn = CUresult (*)(const char **, CUkernel);
using MemcpyParamsFn = CUresult (*)(CUgraphNode, CUDA_MEMCPY3D *);
using MemsetParamsFn = CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS *);
using SubscribeFn = CUptiResult (*)(CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *);
using EnableCallbackFn = CUptiResult (*)(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);
using CapturingFn = CUresult (*)(CUstream, CUstreamCaptureStatus *);
using UnsubscribeFn = CUptiResult (*)(CUpti_SubscriberHandle);

// Every graph instantiation and in-place update of the driver API (the runtime's cudaGraphInstantiate* and
// cudaGraphExecUpdate reach one of these). Each one's parameter struct starts with the executable (pointer) and the
// graph, which is all the walk reads.
constexpr CUpti_CallbackId kInstantiate[] = {
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz};
constexpr CUpti_CallbackId kUpdate[] = {CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate,
                                        CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2};
struct GraphParams {
    void *exec;  // CUgraphExec * (instantiate) or CUgraphExec (update)
    CUgraph graph;
};

// The eager launches (mode 3): kernels, copies, memsets and graph launches (the stream-ordered ones with their
// _ptsz twins), whatever the runtime call that reached them.
constexpr CUpti_CallbackId kEager[] = {
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,              CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,   CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync,               CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,        CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,        CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2,        CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2,          CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy,                    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,             CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async,             CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async,            CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async,            CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,               CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz};

struct Recorder {
    std::mutex mutex;
    bool loaded = false;
    void *cupti = nullptr;
    SubscribeFn subscribe = nullptr;
    EnableCallbackFn enable = nullptr;
    UnsubscribeFn unsubscribe = nullptr;
    GetNodesFn get_nodes = nullptr;
    GetEdgesFn get_edges = nullptr;
    NodeTypeFn node_type = nullptr;
    KernelParamsFn kernel_params = nullptr;
    KernelAttrFn kernel_attr = nullptr;
    FuncNameFn func_name = nullptr;
    KernNameFn kern_name = nullptr;
    MemcpyParamsFn memcpy_params = nullptr;
    MemsetParamsFn memset_params = nullptr;
    CapturingFn capturing = nullptr;
    CUpti_SubscriberHandle subscriber = nullptr;
    bool recording = false;
    bool eager = false;  // mode 3: the eager launches too
    // the records since the last start, and the first failure of a walk (a driver call refused): read reports it
    std::vector<std::string> records;
    uint64_t instantiates = 0;
    uint64_t updates = 0;
    uint64_t launches = 0;  // eager records
    std::string text;  // the records joined (read)
    CUresult failure = CUDA_SUCCESS;
};

Recorder &recorder() {
    static Recorder r;
    return r;
}

template <typename F>
bool driver_fn(const char *symbol, F *out) {
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult found{};
    if (cudaGetDriverEntryPointByVersion(symbol, &fn, 13000, cudaEnableDefault, &found) != cudaSuccess ||
        found != cudaDriverEntryPointSuccess || fn == nullptr) {
        cudaGetLastError();
        return false;
    }
    *out = reinterpret_cast<F>(fn);
    return true;
}

template <typename F>
bool cupti_fn(void *lib, const char *symbol, F *out) {
    void *fn = dlsym(lib, symbol);
    if (fn == nullptr) return false;
    *out = reinterpret_cast<F>(fn);
    return true;
}

bool load(Recorder &r) {
    if (r.loaded) return true;
    if (r.cupti == nullptr) r.cupti = dlopen("libcupti.so.13", RTLD_NOW | RTLD_LOCAL);
    if (r.cupti == nullptr) return false;
    r.loaded = cupti_fn(r.cupti, "cuptiSubscribe", &r.subscribe) &&
               cupti_fn(r.cupti, "cuptiEnableCallback", &r.enable) &&
               cupti_fn(r.cupti, "cuptiUnsubscribe", &r.unsubscribe) && driver_fn("cuGraphGetNodes", &r.get_nodes) &&
               driver_fn("cuGraphGetEdges", &r.get_edges) && driver_fn("cuGraphNodeGetType", &r.node_type) &&
               driver_fn("cuGraphKernelNodeGetParams", &r.kernel_params) &&
               driver_fn("cuGraphKernelNodeGetAttribute", &r.kernel_attr) &&
               driver_fn("cuFuncGetName", &r.func_name) && driver_fn("cuKernelGetName", &r.kern_name) &&
               driver_fn("cuGraphMemcpyNodeGetParams", &r.memcpy_params) &&
               driver_fn("cuGraphMemsetNodeGetParams", &r.memset_params) &&
               driver_fn("cuStreamIsCapturing", &r.capturing);
    return r.loaded;
}

// printf-style append of any length (a kernel's mangled template name runs past 1 KiB).
void append(std::string &s, const char *format, ...) __attribute__((format(printf, 2, 3)));
void append(std::string &s, const char *format, ...) {
    va_list args;
    va_start(args, format);
    va_list again;
    va_copy(again, args);
    const int n = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);
    if (n > 0) {
        const size_t at = s.size();
        s.resize(at + size_t(n) + 1);
        std::vsnprintf(&s[at], size_t(n) + 1, format, again);
        s.resize(at + size_t(n));
    }
    va_end(again);
}

// One graph's text (see the file comment); CUDA_SUCCESS or the first driver call that failed.
CUresult walk(const Recorder &r, CUgraph graph, const char *op, uint64_t index, std::string &out) {
    size_t n = 0;
    CUresult st = r.get_nodes(graph, nullptr, &n);
    if (st != CUDA_SUCCESS) return st;
    std::vector<CUgraphNode> nodes(n);
    if (n != 0 && (st = r.get_nodes(graph, nodes.data(), &n)) != CUDA_SUCCESS) return st;
    size_t e = 0;
    if ((st = r.get_edges(graph, nullptr, nullptr, nullptr, &e)) != CUDA_SUCCESS) return st;
    std::vector<CUgraphNode> from(e), to(e);
    std::vector<CUgraphEdgeData> data(e);
    if (e != 0 && (st = r.get_edges(graph, from.data(), to.data(), data.data(), &e)) != CUDA_SUCCESS) return st;
    std::unordered_map<CUgraphNode, size_t> at;
    for (size_t i = 0; i < n; ++i) at.emplace(nodes[i], i);
    append(out, "record %llu %s nodes=%zu edges=%zu\n", (unsigned long long)index, op, n, e);
    for (size_t i = 0; i < n; ++i) {
        CUgraphNodeType type{};
        if ((st = r.node_type(nodes[i], &type)) != CUDA_SUCCESS) return st;
        if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
            CUDA_KERNEL_NODE_PARAMS p{};
            if ((st = r.kernel_params(nodes[i], &p)) != CUDA_SUCCESS) return st;
            const char *name = nullptr;
            st = p.func != nullptr ? r.func_name(&name, p.func) : r.kern_name(&name, p.kern);
            if (st != CUDA_SUCCESS) return st;
            CUkernelNodeAttrValue pss{};
            const long long serial =
                r.kernel_attr(nodes[i], CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, &pss) == CUDA_SUCCESS
                    ? (long long)pss.programmaticStreamSerializationAllowed
                    : -1LL;
            append(out, "%zu kernel %s grid=%u,%u,%u block=%u,%u,%u smem=%u pss=%lld\n", i, name, p.gridDimX,
                   p.gridDimY, p.gridDimZ, p.blockDimX, p.blockDimY, p.blockDimZ, p.sharedMemBytes, serial);
        } else if (type == CU_GRAPH_NODE_TYPE_MEMCPY) {
            CUDA_MEMCPY3D p{};
            if ((st = r.memcpy_params(nodes[i], &p)) != CUDA_SUCCESS) return st;
            append(out, "%zu memcpy src=%d dst=%d extent=%zux%zux%zu\n", i, int(p.srcMemoryType),
                   int(p.dstMemoryType), p.WidthInBytes, p.Height, p.Depth);
        } else if (type == CU_GRAPH_NODE_TYPE_MEMSET) {
            CUDA_MEMSET_NODE_PARAMS p{};
            if ((st = r.memset_params(nodes[i], &p)) != CUDA_SUCCESS) return st;
            append(out, "%zu memset element=%u extent=%zux%zu value=%u\n", i, p.elementSize, p.width, p.height,
                   p.value);
        } else {
            append(out, "%zu node type=%d\n", i, int(type));
        }
    }
    std::vector<std::string> edges;
    edges.reserve(e);
    for (size_t i = 0; i < e; ++i) {
        std::string line;
        append(line, "edge %zu %zu type=%u from_port=%u to_port=%u\n", at.at(from[i]), at.at(to[i]),
               unsigned(data[i].type), unsigned(data[i].from_port), unsigned(data[i].to_port));
        edges.push_back(std::move(line));
    }
    // edges in (to, from) order: the driver's edge order is not part of the launch sequence
    std::sort(edges.begin(), edges.end(), [](const std::string &a, const std::string &b) {
        size_t af = 0, at_ = 0, bf = 0, bt = 0;
        std::sscanf(a.c_str(), "edge %zu %zu", &af, &at_);
        std::sscanf(b.c_str(), "edge %zu %zu", &bf, &bt);
        return at_ != bt ? at_ < bt : af != bf ? af < bf : a < b;
    });
    for (const std::string &line : edges) out += line;
    return CUDA_SUCCESS;
}

// One eager launch's record (mode 3): a "record <index> launch nodes=1 edges=0" header and its node line in walk's
// format (a graph launch: "record <index> graph_launch nodes=0 edges=0"). *recorded false: the call went to a
// capturing stream, so the graph's walk records it. CUDA_SUCCESS or the driver call that failed.
CUresult eager(const Recorder &r, CUpti_CallbackId cbid, const void *params, uint64_t index, std::string &out,
               bool *recorded) {
    *recorded = false;
    CUstream stream = nullptr;
    CUfunction f = nullptr;
    unsigned grid[3] = {}, block[3] = {}, smem = 0;
    long long pss = 0;
    std::string node;
#define CERE_LAUNCH(P)                                                \
    {                                                                 \
        const auto &p = *static_cast<const P *>(params);              \
        f = p.f;                                                      \
        grid[0] = p.gridDimX, grid[1] = p.gridDimY, grid[2] = p.gridDimZ; \
        block[0] = p.blockDimX, block[1] = p.blockDimY, block[2] = p.blockDimZ; \
        smem = p.sharedMemBytes;                                      \
        stream = p.hStream;                                           \
        break;                                                        \
    }
#define CERE_COPY(P, SRC, DST, STREAM)                                                          \
    {                                                                                           \
        const auto &p = *static_cast<const P *>(params);                                        \
        stream = STREAM;                                                                        \
        append(node, "0 memcpy src=%d dst=%d extent=%zux1x1\n", int(SRC), int(DST), p.ByteCount); \
        break;                                                                                  \
    }
#define CERE_SET(P, ELEMENT, VALUE)                                                                          \
    {                                                                                                        \
        const auto &p = *static_cast<const P *>(params);                                                     \
        stream = p.hStream;                                                                                  \
        append(node, "0 memset element=%u extent=%zux1 value=%u\n", unsigned(ELEMENT), p.N, unsigned(p.VALUE)); \
        break;                                                                                               \
    }
    constexpr CUmemorytype kHost = CU_MEMORYTYPE_HOST, kDevice = CU_MEMORYTYPE_DEVICE, kAny = CU_MEMORYTYPE_UNIFIED;
    switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: CERE_LAUNCH(cuLaunchKernel_params)
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: CERE_LAUNCH(cuLaunchKernel_ptsz_params)
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel: CERE_LAUNCH(cuLaunchCooperativeKernel_params)
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
            CERE_LAUNCH(cuLaunchCooperativeKernel_ptsz_params)
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
            // the two parameter structs have the same layout
            const auto &p = *static_cast<const cuLaunchKernelEx_params *>(params);
            const CUlaunchConfig &c = *p.config;
            f = p.f;
            grid[0] = c.gridDimX, grid[1] = c.gridDimY, grid[2] = c.gridDimZ;
            block[0] = c.blockDimX, block[1] = c.blockDimY, block[2] = c.blockDimZ;
            smem = c.sharedMemBytes;
            stream = c.hStream;
            for (unsigned i = 0; i < c.numAttrs; ++i) {
                if (c.attrs[i].id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
                    pss = c.attrs[i].value.programmaticStreamSerializationAllowed;
                }
            }
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: CERE_COPY(cuMemcpyAsync_params, kAny, kAny, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz: CERE_COPY(cuMemcpyAsync_ptsz_params, kAny, kAny, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
            CERE_COPY(cuMemcpyHtoDAsync_v2_params, kHost, kDevice, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz:
            CERE_COPY(cuMemcpyHtoDAsync_v2_ptsz_params, kHost, kDevice, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
            CERE_COPY(cuMemcpyDtoHAsync_v2_params, kDevice, kHost, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz:
            CERE_COPY(cuMemcpyDtoHAsync_v2_ptsz_params, kDevice, kHost, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
            CERE_COPY(cuMemcpyDtoDAsync_v2_params, kDevice, kDevice, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz:
            CERE_COPY(cuMemcpyDtoDAsync_v2_ptsz_params, kDevice, kDevice, p.hStream)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: CERE_COPY(cuMemcpy_params, kAny, kAny, nullptr)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2: CERE_COPY(cuMemcpyHtoD_v2_params, kHost, kDevice, nullptr)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2: CERE_COPY(cuMemcpyDtoH_v2_params, kDevice, kHost, nullptr)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2: CERE_COPY(cuMemcpyDtoD_v2_params, kDevice, kDevice, nullptr)
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2_ptsz: {
            // the two parameter structs have the same layout
            const auto &p = *static_cast<const cuMemcpy2DAsync_v2_params *>(params);
            stream = p.hStream;
            append(node, "0 memcpy src=%d dst=%d extent=%zux%zux1\n", int(p.pCopy->srcMemoryType),
                   int(p.pCopy->dstMemoryType), p.pCopy->WidthInBytes, p.pCopy->Height);
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async: CERE_SET(cuMemsetD8Async_params, 1, uc)
        case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz: CERE_SET(cuMemsetD8Async_ptsz_params, 1, uc)
        case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async: CERE_SET(cuMemsetD16Async_params, 2, us)
        case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async_ptsz: CERE_SET(cuMemsetD16Async_ptsz_params, 2, us)
        case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async: CERE_SET(cuMemsetD32Async_params, 4, ui)
        case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz: CERE_SET(cuMemsetD32Async_ptsz_params, 4, ui)
        case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
        case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
            stream = static_cast<const cuGraphLaunch_params *>(params)->hStream;
            break;
        default:
            return CUDA_SUCCESS;
    }
#undef CERE_LAUNCH
#undef CERE_COPY
#undef CERE_SET
    CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
    CUresult st = r.capturing(stream, &capture);
    if (st != CUDA_SUCCESS) return st;
    if (capture != CU_STREAM_CAPTURE_STATUS_NONE) return CUDA_SUCCESS;
    const bool graph =
        cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch || cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
    if (f != nullptr) {
        const char *name = nullptr;
        if (r.func_name(&name, f) != CUDA_SUCCESS &&
            (st = r.kern_name(&name, reinterpret_cast<CUkernel>(f))) != CUDA_SUCCESS) {
            return st;
        }
        append(node, "0 kernel %s grid=%u,%u,%u block=%u,%u,%u smem=%u pss=%lld\n", name, grid[0], grid[1], grid[2],
               block[0], block[1], block[2], smem, pss);
    }
    append(out, "record %llu %s nodes=%d edges=0\n", (unsigned long long)index, graph ? "graph_launch" : "launch",
           graph ? 0 : 1);
    out += node;
    *recorded = true;
    return CUDA_SUCCESS;
}

void CUPTIAPI on_driver(void *, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *info) {
    const auto *cb = static_cast<const CUpti_CallbackData *>(info);
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API || cb->callbackSite != CUPTI_API_EXIT) return;
    const bool update = std::find(std::begin(kUpdate), std::end(kUpdate), cbid) != std::end(kUpdate);
    const bool instantiate =
        std::find(std::begin(kInstantiate), std::end(kInstantiate), cbid) != std::end(kInstantiate);
    const auto *ret = static_cast<const CUresult *>(cb->functionReturnValue);
    if (ret == nullptr || *ret != CUDA_SUCCESS) return;  // a refused update is followed by an instantiation
    Recorder &r = recorder();
    std::lock_guard<std::mutex> lock(r.mutex);
    if (!r.recording) return;
    if (!update && !instantiate) {
        if (!r.eager) return;
        std::string text;
        bool recorded = false;
        const CUresult st = eager(r, cbid, cb->functionParams, r.records.size(), text, &recorded);
        if (st != CUDA_SUCCESS) {
            if (r.failure == CUDA_SUCCESS) r.failure = st;
        } else if (recorded) {
            ++r.launches;
            r.records.push_back(std::move(text));
        }
        return;
    }
    const auto *p = static_cast<const GraphParams *>(cb->functionParams);
    std::string text;
    const CUresult st = walk(r, p->graph, update ? "update" : "instantiate", r.records.size(), text);
    if (st != CUDA_SUCCESS) {
        if (r.failure == CUDA_SUCCESS) r.failure = st;
        return;
    }
    ++(update ? r.updates : r.instantiates);
    r.records.push_back(std::move(text));
}

int set_callbacks(Recorder &r, uint32_t on, bool eager) {
    for (CUpti_CallbackId id : kInstantiate) {
        if (r.enable(on, r.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id) != CUPTI_SUCCESS) return CERE_ERR_INTERNAL;
    }
    for (CUpti_CallbackId id : kUpdate) {
        if (r.enable(on, r.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id) != CUPTI_SUCCESS) return CERE_ERR_INTERNAL;
    }
    for (CUpti_CallbackId id : kEager) {
        if (eager && r.enable(on, r.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id) != CUPTI_SUCCESS) {
            return CERE_ERR_INTERNAL;
        }
    }
    return CERE_OK;
}

// The ktest handle of k8_ktest.cpp (cere_ktest_model_create): this file reads its second member only.
struct K8ModelView {
    cere_executor *executor = nullptr;
    cere_model *model = nullptr;
};

}  // namespace

// cere_ktest_model_arena (SelfTune ST1, O279 D-b8): the arena sizes of the ktest model `model`'s memory plan
// (device::ModelLayout, device/memory_plan.cpp): out (host uint64 [5]) = {workspace_bytes, derived_bytes,
// session_bytes, pool_bytes, staging_bytes}. Host reads only, no CUDA call. A null handle, model or out:
// CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_model_arena(uint64_t model, uint64_t *out,
                                                                             cudaStream_t) {
    const auto *k = reinterpret_cast<const K8ModelView *>(static_cast<uintptr_t>(model));
    if (k == nullptr || k->model == nullptr || k->model->device == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const cere::device::ModelLayout &l = static_cast<const cere::device::CudaModel &>(*k->model->device).layout();
    const uint64_t values[5] = {l.workspace_bytes, l.derived_bytes, l.session_bytes, l.pool_bytes, l.staging_bytes};
    std::memcpy(out, values, sizeof(values));
    return CERE_OK;
}

// cere_ktest_graph_launches (SelfTune ST1 launch-sequence equality): mode 1 clears the records and starts recording
// graphs (loads libcupti.so.13 and subscribes; CERE_ERR_UNSUPPORTED when CUPTI or a driver entry point is unavailable,
// CERE_ERR_BUSY when another CUPTI subscriber holds the process, e.g. torch.profiler); mode 3 the same, with the eager
// launches too (ST1d); mode 0 stops and unsubscribes (the records stay); mode 2 reads: counts (host uint64 [4]) =
// {records, instantiations, in-place updates, text bytes} and, when out is not null, the first min(capacity, text
// bytes) bytes of the records' text (ASCII, one "record <i> <instantiate|update|launch|graph_launch> nodes=<n>
// edges=<e>" line, then its nodes and edges, per record, in order). A walk or eager record whose driver call failed
// makes mode 2 CERE_ERR_INTERNAL (the record is missing). Any other mode or a null counts in mode 2:
// CERE_ERR_ARGUMENT. No two calls may run concurrently.
extern "C" __attribute__((visibility("default"))) int cere_ktest_graph_launches(int64_t mode, uint8_t *out,
                                                                                uint64_t capacity, uint64_t *counts,
                                                                                cudaStream_t) {
    Recorder &r = recorder();
    if (mode == 1 || mode == 3) {
        std::lock_guard<std::mutex> lock(r.mutex);
        if (!load(r)) return CERE_ERR_UNSUPPORTED;
        if (r.subscriber == nullptr) {
            const CUptiResult st = r.subscribe(&r.subscriber, on_driver, nullptr);
            if (st != CUPTI_SUCCESS) {
                r.subscriber = nullptr;
                return st == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED ? CERE_ERR_BUSY : CERE_ERR_INTERNAL;
            }
        }
        r.records.clear();
        r.instantiates = r.updates = r.launches = 0;
        r.text.clear();
        r.failure = CUDA_SUCCESS;
        if (r.recording && r.eager && mode == 1) {
            if (const int st = set_callbacks(r, 0, true); st != CERE_OK) return st;  // a mode-3 run's eager callbacks
        }
        r.eager = mode == 3;
        const int st = set_callbacks(r, 1, r.eager);
        r.recording = st == CERE_OK;
        return st;
    }
    if (mode == 0) {
        std::lock_guard<std::mutex> lock(r.mutex);
        r.recording = false;
        if (r.subscriber == nullptr) return CERE_OK;
        const int st = set_callbacks(r, 0, r.eager);
        r.eager = false;
        const CUptiResult un = r.unsubscribe(r.subscriber);
        r.subscriber = nullptr;
        return st != CERE_OK ? st : un == CUPTI_SUCCESS ? CERE_OK : CERE_ERR_INTERNAL;
    }
    if (mode != 2 || counts == nullptr) return CERE_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(r.mutex);
    if (r.failure != CUDA_SUCCESS) return CERE_ERR_INTERNAL;
    r.text.clear();
    for (const std::string &s : r.records) r.text += s;
    counts[0] = r.records.size();
    counts[1] = r.instantiates;
    counts[2] = r.updates;
    counts[3] = r.text.size();
    if (out != nullptr) std::memcpy(out, r.text.data(), std::min<uint64_t>(capacity, r.text.size()));
    return CERE_OK;
}
