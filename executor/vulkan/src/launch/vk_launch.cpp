// VulkanAmd launch layer: see vk_launch.h. Library path: no exit / abort, no printing.
#include "vk_launch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <system_error>
#include <thread>

// The generated table (<OUT>/gen/vk_spv_table.cpp, executor/vulkan/tools/embed_spv.py). Weak: a binary that does not
// link it (the per-kernel ktests) sees an empty table and loads from its shader directory.
extern "C" {
extern const cere::vk::EmbeddedShader cere_vk_embedded_spv[] __attribute__((weak));
extern const size_t cere_vk_embedded_spv_count __attribute__((weak));
}

namespace cere::vk {

#if defined(CERE_VK_DISPATCH_PROFILE)
// TEST BUILD ONLY (build_lib.sh's forward_s3_dprof; never in the library or a ktest): the per-barrier GPU profile of
// VulkanAutotune stage 2 (vulkan-autotune design note "Stage 2"). Every command buffer gets its own query pool of
// kMax timestamps: stamp 0 at begin, one after every barrier, one before the completion marker. A segment = the
// dispatches between two stamps (their names and grids, its label) plus the barrier that ends it. The forward names
// the call's operation (cere_vk_dprof_phase, called by the taps build of vulkan_forward.cpp at every vk_forward), and
// the command buffers of a DECODE call are numbered from 0. At exit the per-(command buffer, segment, label) GPU times
// of the DECODE calls are written to $CERE_VK_DPROF_OUT (TSV: cb, seg, n, median_ms, mean_ms, p10_ms, p90_ms, label).
namespace dprof {
constexpr uint32_t kMax = 4096;
struct SlotRec {
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t n = 0;  // stamps written
    uint32_t phase = 0;
    uint32_t cb = 0;
    std::vector<std::string> labels;
    std::string cur;
};
struct Key {
    uint32_t cb, seg;
    std::string label;
    bool operator<(const Key &o) const { return std::tie(cb, seg, label) < std::tie(o.cb, o.seg, o.label); }
};
uint32_t g_phase = 0;
uint32_t g_cb = 0;
uint32_t g_step = 0;  // DECODE calls so far (cere_vk_dprof_step: the GEMV mapping rotation of k2_mgemv_vk.cpp)
std::map<std::pair<VkDevice, uint32_t>, SlotRec> g_slots;
std::map<Key, std::vector<double>> g_acc;
std::map<VkPipeline, std::string> g_names;

void dump() {
    const char *path = std::getenv("CERE_VK_DPROF_OUT");
    FILE *f = path ? std::fopen(path, "w") : stderr;
    if (!f) return;
    std::fprintf(f, "cb\tseg\tn\tmedian_ms\tmean_ms\tp10_ms\tp90_ms\tlabel\n");
    for (auto &kv : g_acc) {
        std::vector<double> &v = kv.second;
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (double x : v) sum += x;
        const size_t n = v.size();
        std::fprintf(f, "%u\t%u\t%zu\t%.5f\t%.5f\t%.5f\t%.5f\t%s\n", kv.first.cb, kv.first.seg, n, v[n / 2], sum / double(n),
                     v[n / 10], v[n * 9 / 10], kv.first.label.c_str());
    }
    if (f != stderr) std::fclose(f);
}

SlotRec &slot(VkDevice dev, uint32_t s) { return g_slots[{dev, s}]; }

void begin(VkDevice dev, VkCommandBuffer cb, uint32_t s) {
    static bool registered = false;
    if (!registered) registered = std::atexit(dump) == 0;
    SlotRec &r = slot(dev, s);
    if (!r.pool) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kMax;
        if (vkCreateQueryPool(dev, &qi, nullptr, &r.pool) != VK_SUCCESS) r.pool = VK_NULL_HANDLE;
    }
    r.n = 0;
    r.phase = g_phase;
    r.cb = g_cb++;
    r.labels.clear();
    r.cur.clear();
    if (!r.pool) return;
    vkCmdResetQueryPool(cb, r.pool, 0, kMax);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r.pool, r.n++);
}

void dispatch(VkDevice dev, uint32_t s, VkPipeline p, const std::map<std::tuple<std::string, uint32_t, uint32_t, std::vector<uint32_t>>, Pipeline> &pipes,
              uint32_t gx, uint32_t gy, uint32_t gz) {
    auto it = g_names.find(p);
    if (it == g_names.end()) {
        std::string name = "?";
        for (auto &kv : pipes)
            if (kv.second.pipe == p) name = std::get<0>(kv.first);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".spv") == 0) name.resize(name.size() - 4);
        it = g_names.emplace(p, name).first;
    }
    SlotRec &r = slot(dev, s);
    char g[48];
    std::snprintf(g, sizeof g, "(%u,%u,%u)", gx, gy, gz);
    if (!r.cur.empty()) r.cur += " ";
    r.cur += it->second + g;
}

void stamp(VkDevice dev, VkCommandBuffer cb, uint32_t s) {
    SlotRec &r = slot(dev, s);
    if (!r.pool || r.n >= kMax) return;
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r.pool, r.n++);
    r.labels.push_back(r.cur.empty() ? std::string("-") : r.cur);
    r.cur.clear();
}

void retire(VkDevice dev, uint32_t s, double period_ns, uint64_t mask) {
    SlotRec &r = slot(dev, s);
    if (!r.pool || r.phase != CERE_OP_DECODE || r.n < 2) return;
    std::vector<uint64_t> v(r.n, 0);
    if (vkGetQueryPoolResults(dev, r.pool, 0, r.n, v.size() * 8, v.data(), 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    for (uint32_t i = 0; i + 1 < r.n; ++i)
        g_acc[Key{r.cb, i, r.labels[i]}].push_back(double((v[i + 1] - v[i]) & mask) * period_ns * 1e-6);
}

void destroy(VkDevice dev) {
    for (auto &kv : g_slots)
        if (kv.first.first == dev && kv.second.pool) {
            vkDestroyQueryPool(dev, kv.second.pool, nullptr);
            kv.second.pool = VK_NULL_HANDLE;
        }
}
}  // namespace dprof
}  // namespace cere::vk

extern "C" void cere_vk_dprof_phase(uint32_t operation) {
    cere::vk::dprof::g_phase = operation;
    cere::vk::dprof::g_cb = 0;
    if (operation == CERE_OP_DECODE) ++cere::vk::dprof::g_step;
}

extern "C" uint32_t cere_vk_dprof_step() { return cere::vk::dprof::g_step; }

namespace cere::vk {
#endif

const EmbeddedShader *embedded_shaders(size_t *count) {
    // Both symbols come from the one generated translation unit: the count's presence stands for the table's.
    const bool linked = &cere_vk_embedded_spv_count != nullptr;
    *count = linked ? cere_vk_embedded_spv_count : 0;
    return linked ? cere_vk_embedded_spv : nullptr;
}

const EmbeddedShader *find_embedded_shader(const std::string &name) {
    size_t n = 0;
    const EmbeddedShader *t = embedded_shaders(&n);
    for (size_t i = 0; i < n; ++i)
        if (name == t[i].name) return &t[i];
    return nullptr;
}

cere_status status_of(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return CERE_OK;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_OUT_OF_POOL_MEMORY:
    case VK_ERROR_FRAGMENTED_POOL:
    case VK_ERROR_FRAGMENTATION:
    case VK_ERROR_TOO_MANY_OBJECTS: return CERE_ERR_CAPACITY;
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_EXTENSION_NOT_PRESENT:
    case VK_ERROR_LAYER_NOT_PRESENT:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return CERE_ERR_UNSUPPORTED;
    default: return CERE_ERR_INTERNAL;  // DEVICE_LOST, INITIALIZATION_FAILED, TIMEOUT, ...
    }
}

const char *result_name(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
    default: return "VkResult(other)";
    }
}

static void say(std::string *why, const std::string &m) {
    if (why) *why = m;
}

static bool selectable(VkPhysicalDeviceType t, bool allow_software) {
    return t == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || t == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
           (allow_software && t == VK_PHYSICAL_DEVICE_TYPE_CPU);
}

static VkResult make_instance(VkInstance *inst) {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    ai.pApplicationName = "cere-vulkan";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    return vkCreateInstance(&ici, nullptr, inst);
}

static VkResult list_selectable(VkInstance inst, bool allow_software, std::vector<VkPhysicalDevice> *out) {
    uint32_t n = 0;
    VkResult r = vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (r != VK_SUCCESS) return r;
    std::vector<VkPhysicalDevice> pds(n);
    r = vkEnumeratePhysicalDevices(inst, &n, pds.data());
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) return r;
    pds.resize(n);
    out->clear();
    for (auto d : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (selectable(p.deviceType, allow_software)) out->push_back(d);
    }
    return VK_SUCCESS;
}

cere_status Context::count_devices(uint32_t *count, bool allow_software, std::string *why) {
    if (!count) return CERE_ERR_ARGUMENT;
    *count = 0;
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = make_instance(&inst);
    if (r != VK_SUCCESS) {
        say(why, std::string("vkCreateInstance = ") + result_name(r));
        return r == VK_ERROR_INCOMPATIBLE_DRIVER ? CERE_ERR_UNSUPPORTED : status_of(r);
    }
    std::vector<VkPhysicalDevice> pds;
    r = list_selectable(inst, allow_software, &pds);
    vkDestroyInstance(inst, nullptr);
    if (r != VK_SUCCESS) {
        say(why, std::string("vkEnumeratePhysicalDevices = ") + result_name(r));
        return status_of(r);
    }
    *count = uint32_t(pds.size());
    return CERE_OK;
}

cere_status Context::create(uint32_t ordinal, std::unique_ptr<Context> *out, std::string *why, const CreateOptions &opts) {
    if (!out) return CERE_ERR_ARGUMENT;
    out->reset();
    if (opts.wait_timeout_ns == 0) {
        say(why, "wait_timeout_ns 0");
        return CERE_ERR_ARGUMENT;
    }
    if (opts.max_in_flight < 1 || opts.max_in_flight > kMaxInFlight) {
        say(why, "max_in_flight " + std::to_string(opts.max_in_flight) + " (1.." + std::to_string(kMaxInFlight) + ")");
        return CERE_ERR_ARGUMENT;
    }
    std::unique_ptr<Context> c(new Context());
    const cere_status s = c->init(ordinal, opts, why);
    if (s != CERE_OK) return s;  // c's destructor releases whatever init created
    *out = std::move(c);
    return CERE_OK;
}

cere_status Context::vk_fail(VkResult r, const std::string &what, std::string *why) {
    const std::string m = what + " = " + result_name(r);
    last_error_ = m;
    say(why, m);
    if (r == VK_ERROR_DEVICE_LOST) {
        set_fault(r, m);
        append_fault_info();
    }
    return faulted_ ? CERE_ERR_INTERNAL : status_of(r);
}

// VK_EXT_device_fault (advertised by RADV 25.2.8 on gfx1151, deviceFault = true): a description and the counts of
// address / vendor fault records the driver holds, appended to the fault reason. Diagnostics only, after a fault: it
// is not a completion signal (it reports GPU VM faults, not a job killed by the ring timeout).
void Context::append_fault_info() {
    if (!info_.device_fault || !dev_) return;
    auto fn = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(vkGetDeviceProcAddr(dev_, "vkGetDeviceFaultInfoEXT"));
    if (!fn) return;
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    const VkResult r = fn(dev_, &counts, nullptr);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        fault_reason_ += std::string("; vkGetDeviceFaultInfoEXT = ") + result_name(r);
        return;
    }
    VkDeviceFaultInfoEXT fi{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
    std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
    fi.pAddressInfos = addrs.empty() ? nullptr : addrs.data();
    fi.pVendorInfos = vendor.empty() ? nullptr : vendor.data();
    counts.vendorBinarySize = 0;
    const VkResult r2 = fn(dev_, &counts, &fi);  // VK_INCOMPLETE still fills the description
    const std::string desc = (r2 == VK_SUCCESS || r2 == VK_INCOMPLETE) ? std::string(fi.description) : std::string(result_name(r2));
    fault_reason_ += "; device_fault: \"" + desc + "\" address records " + std::to_string(counts.addressInfoCount) +
                     ", vendor records " + std::to_string(counts.vendorInfoCount);
    for (uint32_t i = 0; i < counts.addressInfoCount && i < 4; ++i) {
        char b[96];
        std::snprintf(b, sizeof b, " [type %d addr 0x%llx]", int(addrs[i].addressType), (unsigned long long)addrs[i].reportedAddress);
        fault_reason_ += b;
    }
}

void Context::set_fault(VkResult r, const std::string &why) {
    if (faulted_) return;  // the first cause is kept
    faulted_ = true;
    fault_result_ = r;
    fault_reason_ = why;
}

void Context::inject_fault_for_test(const char *why) {
    set_fault(VK_ERROR_DEVICE_LOST, std::string("injected for test: ") + (why ? why : ""));
}

cere_status Context::init(uint32_t ordinal, const CreateOptions &opts, std::string *why) {
    wait_timeout_ns_ = opts.wait_timeout_ns;
    nslots_ = opts.max_in_flight;
    VkResult r = make_instance(&inst_);
    if (r != VK_SUCCESS) {
        inst_ = VK_NULL_HANDLE;
        say(why, std::string("vkCreateInstance = ") + result_name(r));
        return r == VK_ERROR_INCOMPATIBLE_DRIVER ? CERE_ERR_UNSUPPORTED : status_of(r);
    }
    std::vector<VkPhysicalDevice> pds;
    if ((r = list_selectable(inst_, opts.allow_software, &pds)) != VK_SUCCESS)
        return vk_fail(r, "vkEnumeratePhysicalDevices", why);
    if (ordinal >= pds.size()) {
        say(why, "no Vulkan device at ordinal " + std::to_string(ordinal) + " (" + std::to_string(pds.size()) +
                     " selectable: integrated/discrete GPUs" + (opts.allow_software ? " and CPU devices)" : ")"));
        return CERE_ERR_UNSUPPORTED;
    }
    pd_ = pds[ordinal];

    VkPhysicalDeviceMaintenance3Properties p3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    p11.pNext = &p12; p12.pNext = &p13; p13.pNext = &p3;
    VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    pp.pNext = &p11;
    vkGetPhysicalDeviceProperties2(pd_, &pp);
    info_.name = pp.properties.deviceName;
    info_.driver_name = p12.driverName;
    info_.driver_info = p12.driverInfo;
    info_.ordinal = ordinal;
    info_.vendor_id = pp.properties.vendorID;
    info_.device_id = pp.properties.deviceID;
    info_.device_type = pp.properties.deviceType;
    info_.api_version = pp.properties.apiVersion;
    info_.driver_version = pp.properties.driverVersion;
    info_.subgroup_default = p11.subgroupSize;
    info_.subgroup_min = p13.minSubgroupSize;
    info_.subgroup_max = p13.maxSubgroupSize;
    info_.max_buffer = p13.maxBufferSize;
    info_.max_alloc = p3.maxMemoryAllocationSize;
    info_.timestamp_period_ns = pp.properties.limits.timestampPeriod;
    vkGetPhysicalDeviceMemoryProperties(pd_, &mp_);
    if (VK_API_VERSION_MAJOR(info_.api_version) == 1 && VK_API_VERSION_MINOR(info_.api_version) < 3) {
        say(why, info_.name + ": Vulkan 1.3 required");
        return CERE_ERR_UNSUPPORTED;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, qp.data());
    qf_ = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qp[i].timestampValidBits) { qf_ = i; break; }
    if (qf_ != UINT32_MAX && qp[qf_].timestampValidBits < 64) ts_mask_ = (uint64_t(1) << qp[qf_].timestampValidBits) - 1;
    if (qf_ == UINT32_MAX) {
        say(why, info_.name + ": no compute queue with timestamps");
        return CERE_ERR_UNSUPPORTED;
    }

    // Required features: checked before device creation so a missing one is a clear refusal, not a driver error.
    VkPhysicalDeviceVulkan13Features h13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features h12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features h11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    h11.pNext = &h12; h12.pNext = &h13;
    VkPhysicalDeviceFeatures2 hf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    hf.pNext = &h11;
    vkGetPhysicalDeviceFeatures2(pd_, &hf);
    const struct { const char *name; VkBool32 have; } need[] = {
        {"bufferDeviceAddress", h12.bufferDeviceAddress}, {"timelineSemaphore", h12.timelineSemaphore},
        {"shaderInt8", h12.shaderInt8}, {"shaderFloat16", h12.shaderFloat16},
        {"storageBuffer8BitAccess", h12.storageBuffer8BitAccess}, {"storageBuffer16BitAccess", h11.storageBuffer16BitAccess},
        {"shaderInt64", hf.features.shaderInt64}, {"shaderInt16", hf.features.shaderInt16},
        {"subgroupSizeControl", h13.subgroupSizeControl}, {"computeFullSubgroups", h13.computeFullSubgroups},
        {"synchronization2", h13.synchronization2},
        {"shaderFloat64", hf.features.shaderFloat64}, {"shaderIntegerDotProduct", h13.shaderIntegerDotProduct},
    };
    for (auto &f : need)
        if (!f.have) {
            say(why, info_.name + ": device lacks " + f.name);
            return CERE_ERR_UNSUPPORTED;
        }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.subgroupSizeControl = VK_TRUE; f13.computeFullSubgroups = VK_TRUE; f13.synchronization2 = VK_TRUE;
    f13.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE; f12.timelineSemaphore = VK_TRUE; f12.shaderInt8 = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE; f12.storageBuffer8BitAccess = VK_TRUE;
    f12.pNext = &f13;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderInt64 = VK_TRUE; f2.features.shaderInt16 = VK_TRUE; f2.features.shaderFloat64 = VK_TRUE;
    f2.pNext = &f11;
    // Optional: cooperative matrices (the large-m GEMM, prefill attention, the W4A4 MoE GEMM). Enabled when the
    // device has the extension and the feature; info().coopmat tells a kernel whether it may use them.
    uint32_t next = 0;
    if ((r = vkEnumerateDeviceExtensionProperties(pd_, nullptr, &next, nullptr)) != VK_SUCCESS)
        return vk_fail(r, "vkEnumerateDeviceExtensionProperties", why);
    std::vector<VkExtensionProperties> exts(next);
    r = vkEnumerateDeviceExtensionProperties(pd_, nullptr, &next, exts.data());
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) return vk_fail(r, "vkEnumerateDeviceExtensionProperties", why);
    exts.resize(next);
    bool have_cm = false;
    for (auto &e : exts) have_cm = have_cm || std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR hcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    if (have_cm) {
        VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        q.pNext = &hcm;
        vkGetPhysicalDeviceFeatures2(pd_, &q);
        have_cm = hcm.cooperativeMatrix == VK_TRUE;
    }
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    fcm.cooperativeMatrix = VK_TRUE;
    std::vector<const char *> ext_names;
    if (have_cm) {
        f13.pNext = &fcm;
        ext_names.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    // Optional, diagnostics only: VK_EXT_device_fault (see append_fault_info).
    bool have_df = false;
    for (auto &e : exts) have_df = have_df || std::strcmp(e.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME) == 0;
    VkPhysicalDeviceFaultFeaturesEXT hdf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    if (have_df) {
        VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        q.pNext = &hdf;
        vkGetPhysicalDeviceFeatures2(pd_, &q);
        have_df = hdf.deviceFault == VK_TRUE;
    }
    VkPhysicalDeviceFaultFeaturesEXT fdf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    fdf.deviceFault = VK_TRUE;
    if (have_df) {
        fdf.pNext = f2.pNext;
        f2.pNext = &fdf;
        ext_names.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
    }
    // Optional, diagnostics only: VK_EXT_calibrated_timestamps with the DEVICE and CLOCK_MONOTONIC domains
    // (SubmitTiming's host / GPU correlation; std::chrono::steady_clock is CLOCK_MONOTONIC on Linux).
    bool have_ct = false;
    for (auto &e : exts) have_ct = have_ct || std::strcmp(e.extensionName, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) == 0;
    if (have_ct) {
        auto domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
            vkGetInstanceProcAddr(inst_, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
        uint32_t nd = 0;
        std::vector<VkTimeDomainEXT> d;
        if (domains && domains(pd_, &nd, nullptr) == VK_SUCCESS) {
            d.resize(nd);
            if (domains(pd_, &nd, d.data()) != VK_SUCCESS) d.clear();
        }
        bool dev = false, mono = false;
        for (VkTimeDomainEXT t : d) {
            dev = dev || t == VK_TIME_DOMAIN_DEVICE_EXT;
            mono = mono || t == VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
        }
        have_ct = dev && mono;
    }
    if (have_ct) ext_names.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf_; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(ext_names.size());
    dci.ppEnabledExtensionNames = ext_names.empty() ? nullptr : ext_names.data();
    if ((r = vkCreateDevice(pd_, &dci, nullptr, &dev_)) != VK_SUCCESS) {
        dev_ = VK_NULL_HANDLE;
        return vk_fail(r, "vkCreateDevice", why);
    }
    info_.coopmat = have_cm;
    info_.device_fault = have_df;
    if (have_ct) {
        calibrate_ = reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(vkGetDeviceProcAddr(dev_, "vkGetCalibratedTimestampsEXT"));
        info_.calibrated = calibrate_ != nullptr;
    }
    info_.float64 = true;
    info_.integer_dot = true;
    vkGetDeviceQueue(dev_, qf_, 0, &q_);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qf_;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if ((r = vkCreateCommandPool(dev_, &cpi, nullptr, &pool_)) != VK_SUCCESS) {
        pool_ = VK_NULL_HANDLE;
        return vk_fail(r, "vkCreateCommandPool", why);
    }
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    for (uint32_t i = 0; i < nslots_; ++i)
        if ((r = vkAllocateCommandBuffers(dev_, &cai, &slots_[i].cb)) != VK_SUCCESS) {
            slots_[i].cb = VK_NULL_HANDLE;
            return vk_fail(r, "vkAllocateCommandBuffers", why);
        }

    VkSemaphoreTypeCreateInfo sti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE; sti.initialValue = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &sti;
    if ((r = vkCreateSemaphore(dev_, &sci, nullptr, &timeline_)) != VK_SUCCESS) {
        timeline_ = VK_NULL_HANDLE;
        return vk_fail(r, "vkCreateSemaphore", why);
    }

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = kStamps;
    for (uint32_t i = 0; i < nslots_; ++i)
        if ((r = vkCreateQueryPool(dev_, &qpi, nullptr, &slots_[i].stamps)) != VK_SUCCESS) {
            slots_[i].stamps = VK_NULL_HANDLE;
            return vk_fail(r, "vkCreateQueryPool", why);
        }
    stamp_vals_.assign(kStamps, 0);

    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if ((r = vkCreatePipelineCache(dev_, &pci, nullptr, &cache_)) != VK_SUCCESS) {
        cache_ = VK_NULL_HANDLE;
        return vk_fail(r, "vkCreatePipelineCache", why);
    }

    // Per-command-buffer GPU time (2 timestamps) and the completion marker word.
    VkQueryPoolCreateInfo fqi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    fqi.queryType = VK_QUERY_TYPE_TIMESTAMP; fqi.queryCount = 2;
    for (uint32_t i = 0; i < nslots_; ++i) {
        if ((r = vkCreateQueryPool(dev_, &fqi, nullptr, &slots_[i].frame)) != VK_SUCCESS) {
            slots_[i].frame = VK_NULL_HANDLE;
            return vk_fail(r, "vkCreateQueryPool(frame)", why);
        }
        slots_[i].marker_off = 64ull * i;  // one marker word per slot, 64 B apart (kMaxInFlight * 64 <= 256)
    }
    static_assert(kMaxInFlight * 64 <= 256, "completion marker words");
    {
        std::string w;
        const cere_status s = allocate(256, marker_, false, &w);
        if (s != CERE_OK) {
            say(why, "completion marker: " + w);
            return s;
        }
        if (!(memory_flags(marker_.memory_type) & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            free(marker_);
            say(why, "completion marker: memory type not host-coherent");
            return CERE_ERR_UNSUPPORTED;
        }
        std::random_device rd;
        nonce_salt_ = (uint64_t(rd()) << 32 | rd()) ^ uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
        nonce_salt_ &= ~uint64_t(0xFFFFFFFF);  // high half random, low half = the timeline value: never 0
    }

    if (opts.table_ring_bytes) {
        std::string w;
        if (opts.table_ring_bytes > info_.max_buffer / nslots_) {
            say(why, "table ring: " + std::to_string(opts.table_ring_bytes) + " B x " + std::to_string(nslots_) + " slots");
            return CERE_ERR_CAPACITY;
        }
        const cere_status s = allocate(opts.table_ring_bytes * nslots_, ring_, false, &w);
        if (s != CERE_OK) {
            say(why, "table ring: " + w);
            return s;
        }
        if (!(memory_flags(ring_.memory_type) & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            free(ring_);
            say(why, "table ring: memory type not host-coherent");
            return CERE_ERR_UNSUPPORTED;
        }
        ring_slot_bytes_ = opts.table_ring_bytes;
        for (uint32_t i = 0; i < nslots_; ++i) slots_[i].ring_off = opts.table_ring_bytes * i;
    }
    return CERE_OK;
}

Context::~Context() {
    // Nothing is in flight unless a wait failed or timed out (a fault); vkDeviceWaitIdle then returns once the kernel
    // driver has finished or reset the ring (DEVICE_LOST).
    if (dev_) vkDeviceWaitIdle(dev_);
#if defined(CERE_VK_DISPATCH_PROFILE)
    if (dev_) dprof::destroy(dev_);
#endif
    if (dev_) free(ring_);
    if (dev_) free(marker_);
    for (Slot &sl : slots_) {
        if (sl.frame) vkDestroyQueryPool(dev_, sl.frame, nullptr);
        if (sl.stamps) vkDestroyQueryPool(dev_, sl.stamps, nullptr);
    }
    for (auto &kv : pipes_) {
        vkDestroyPipeline(dev_, kv.second.pipe, nullptr);
        vkDestroyPipelineLayout(dev_, kv.second.layout, nullptr);
    }
    for (auto &kv : modules_) vkDestroyShaderModule(dev_, kv.second, nullptr);
    if (cache_) vkDestroyPipelineCache(dev_, cache_, nullptr);
    if (timeline_) vkDestroySemaphore(dev_, timeline_, nullptr);
    if (pool_) vkDestroyCommandPool(dev_, pool_, nullptr);
    if (dev_) vkDestroyDevice(dev_, nullptr);
    if (inst_) vkDestroyInstance(inst_, nullptr);
}

VkMemoryPropertyFlags Context::memory_flags(uint32_t memory_type) const {
    return memory_type < mp_.memoryTypeCount ? mp_.memoryTypes[memory_type].propertyFlags : 0;
}

cere_status Context::allocate(uint64_t bytes, Buffer &out, bool host_heap, std::string *why) {
    out = Buffer();
    auto refuse = [&](cere_status s, const std::string &m) {
        last_error_ = m;
        say(why, m);
        return s;
    };
    if (faulted_) return refuse(CERE_ERR_INTERNAL, "context faulted: " + fault_reason_);
    if (bytes == 0) return refuse(CERE_ERR_ARGUMENT, "buffer of 0 bytes");
    if (bytes > info_.max_buffer || bytes > info_.max_alloc)
        return refuse(CERE_ERR_CAPACITY, "buffer of " + std::to_string(bytes) + " bytes above maxBufferSize " +
                                             std::to_string(info_.max_buffer) + " / maxMemoryAllocationSize " +
                                             std::to_string(info_.max_alloc) + ": split it");
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkResult r = vkCreateBuffer(dev_, &bi, nullptr, &out.buf);
    if (r != VK_SUCCESS) {
        out = Buffer();
        return vk_fail(r, "vkCreateBuffer(" + std::to_string(bytes) + ")", why);
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev_, out.buf, &mr);
    auto undo = [&] {
        vkDestroyBuffer(dev_, out.buf, nullptr);
        out = Buffer();
    };
    if (mr.size > info_.max_alloc) {
        undo();
        return refuse(CERE_ERR_CAPACITY, "buffer needs " + std::to_string(mr.size) + " bytes > maxMemoryAllocationSize");
    }
    int type = -1;
    if (host_heap) {
        // Host-visible + coherent, not device-local; exactly those two flags first (P0-5's type 2: uncached), then any.
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (int pass = 0; pass < 2 && type < 0; ++pass)
            for (uint32_t i = 0; i < mp_.memoryTypeCount && type < 0; ++i) {
                const VkMemoryPropertyFlags f = mp_.memoryTypes[i].propertyFlags;
                if (!(mr.memoryTypeBits >> i & 1) || (f & want) != want || (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
                if (pass == 0 && f != want) continue;
                type = int(i);
            }
        if (type < 0) {
            undo();
            return refuse(CERE_ERR_UNSUPPORTED, "no host-visible, non-device-local memory type");
        }
    } else {
        // Prefer device-local + host-visible (UMA: the whole device heap is host-visible), else any host-visible.
        const VkMemoryPropertyFlags want[2] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT};
        for (auto w : want) {
            if (type >= 0) break;
            for (uint32_t i = 0; i < mp_.memoryTypeCount && type < 0; ++i)
                if ((mr.memoryTypeBits >> i & 1) && (mp_.memoryTypes[i].propertyFlags & w) == w) type = int(i);
        }
        if (type < 0) {
            undo();
            return refuse(CERE_ERR_UNSUPPORTED, "no host-visible memory type");
        }
    }
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &fi; mai.allocationSize = mr.size; mai.memoryTypeIndex = uint32_t(type);
    if ((r = vkAllocateMemory(dev_, &mai, nullptr, &out.mem)) != VK_SUCCESS) {
        out.mem = VK_NULL_HANDLE;
        undo();
        return vk_fail(r, "vkAllocateMemory(" + std::to_string(mr.size) + ")", why);
    }
    if ((r = vkBindBufferMemory(dev_, out.buf, out.mem, 0)) != VK_SUCCESS) {
        free(out);
        return vk_fail(r, "vkBindBufferMemory", why);
    }
    if ((r = vkMapMemory(dev_, out.mem, 0, VK_WHOLE_SIZE, 0, &out.map)) != VK_SUCCESS) {
        out.map = nullptr;
        free(out);
        return vk_fail(r, "vkMapMemory", why);
    }
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = out.buf;
    out.addr = vkGetBufferDeviceAddress(dev_, &da);
    out.size = bytes;
    out.memory_type = uint32_t(type);
    return CERE_OK;
}

bool Context::alloc(uint64_t bytes, Buffer &out, bool host_heap) { return allocate(bytes, out, host_heap, nullptr) == CERE_OK; }

void Context::free(Buffer &b) {
    if (!b.buf) return;  // empty, or a non-owning view (TableRegion::view)
    if (b.map) vkUnmapMemory(dev_, b.mem);
    vkDestroyBuffer(dev_, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
    b = Buffer();
}

cere_status Context::load_module(const std::string &name, VkShaderModule *mod, std::string *why) {
    auto it = modules_.find(name);
    if (it != modules_.end()) {
        *mod = it->second;
        return CERE_OK;
    }
    std::vector<uint32_t> file_words;
    const uint32_t *words = nullptr;
    size_t bytes = 0;
    if (const EmbeddedShader *e = find_embedded_shader(name)) {
        words = e->words;
        bytes = e->bytes;
    } else if (shader_dir_set_) {
        const std::string path = shader_dir_ + "/" + name;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            say(why, "SPIR-V " + name + " not embedded and not found under " + shader_dir_);
            return CERE_ERR_UNSUPPORTED;
        }
        const std::streamoff n = f.tellg();
        if (n <= 0 || n % 4 != 0) {
            say(why, "SPIR-V file " + path + " has " + std::to_string(n) + " bytes (not a positive multiple of 4)");
            return CERE_ERR_UNSUPPORTED;
        }
        file_words.resize(size_t(n) / 4);
        f.seekg(0);
        if (!f.read(reinterpret_cast<char *>(file_words.data()), n)) {
            say(why, "SPIR-V file " + path + ": read failed");
            return CERE_ERR_UNSUPPORTED;
        }
        words = file_words.data();
        bytes = size_t(n);
    } else {
        say(why, "SPIR-V " + name + " is not embedded in this binary (and no shader directory was set)");
        return CERE_ERR_UNSUPPORTED;
    }
    if (bytes < 20 || bytes % 4 != 0 || words[0] != 0x07230203u) {
        say(why, "SPIR-V " + name + ": not a SPIR-V module");
        return CERE_ERR_UNSUPPORTED;
    }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = bytes; smi.pCode = words;
    const VkResult r = vkCreateShaderModule(dev_, &smi, nullptr, mod);
    if (r != VK_SUCCESS) return vk_fail(r, "vkCreateShaderModule(" + name + ")", why);
    modules_.emplace(name, *mod);
    return CERE_OK;
}

static std::string spec_text(const PipelineSpec &s) {
    std::string t = s.spv + " push " + std::to_string(s.push_bytes) + " subgroup " + std::to_string(s.required_subgroup);
    if (!s.spec.empty()) {
        t += " spec {";
        for (size_t i = 0; i < s.spec.size(); ++i) t += (i ? "," : "") + std::to_string(s.spec[i]);
        t += "}";
    }
    return t;
}

cere_status Context::pipeline(const PipelineSpec &s, const Pipeline **out, std::string *why) {
    if (!out) return CERE_ERR_ARGUMENT;
    *out = nullptr;
    const Key key{s.spv, s.push_bytes, s.required_subgroup, s.spec};
    auto it = pipes_.find(key);
    if (it != pipes_.end()) {
        *out = &it->second;
        return CERE_OK;
    }
    if (faulted_) {
        say(why, "context faulted: " + fault_reason_);
        return CERE_ERR_INTERNAL;
    }
    if (const cere_status st = check_spec(s, why); st != CERE_OK) return st;
    VkShaderModule mod = VK_NULL_HANDLE;
    const cere_status st = load_module(s.spv, &mod, why);
    if (st != CERE_OK) return st;
    Pipeline p;
    if (const VkResult r = create_pipeline(s, mod, &p); r != VK_SUCCESS) {
        return vk_fail(r, std::string(p.layout == VK_NULL_HANDLE ? "vkCreatePipelineLayout(" : "vkCreateComputePipelines(") +
                              spec_text(s) + ")",
                       why);
    }
    *out = &pipes_.emplace(key, p).first->second;
    return CERE_OK;
}

cere_status Context::check_spec(const PipelineSpec &s, std::string *why) const {
    if (s.push_bytes > 128 || s.push_bytes % 4 != 0) {
        say(why, spec_text(s) + ": push constants must be a multiple of 4 and <= 128 bytes");
        return CERE_ERR_ARGUMENT;
    }
    if (s.required_subgroup && (s.required_subgroup < info_.subgroup_min || s.required_subgroup > info_.subgroup_max ||
                                (s.required_subgroup & (s.required_subgroup - 1)) != 0)) {
        say(why, "required subgroup size " + std::to_string(s.required_subgroup) + " unsupported (device " +
                     std::to_string(info_.subgroup_min) + ".." + std::to_string(info_.subgroup_max) + ")");
        return CERE_ERR_UNSUPPORTED;
    }
    return CERE_OK;
}

// On failure *p holds no handle; p->layout is VK_NULL_HANDLE when the layout failed, else the pipeline did.
VkResult Context::create_pipeline(const PipelineSpec &s, VkShaderModule mod, Pipeline *p) const {
    p->push_bytes = s.push_bytes;
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, s.push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = s.push_bytes ? 1 : 0; pli.pPushConstantRanges = &pcr;
    VkResult r = vkCreatePipelineLayout(dev_, &pli, nullptr, &p->layout);
    if (r != VK_SUCCESS) {
        p->layout = VK_NULL_HANDLE;
        return r;
    }
    std::vector<VkSpecializationMapEntry> me;
    for (uint32_t i = 0; i < s.spec.size(); ++i) me.push_back({i, i * 4, 4});
    VkSpecializationInfo si{uint32_t(s.spec.size()), me.data(), s.spec.size() * 4, s.spec.data()};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = s.required_subgroup;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.layout = p->layout;
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, s.required_subgroup ? &rss : nullptr,
                 s.required_subgroup ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : 0,
                 VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", s.spec.empty() ? nullptr : &si};
    // cache_ is internally synchronized (created without EXTERNALLY_SYNCHRONIZED), so threads may share it.
    r = vkCreateComputePipelines(dev_, cache_, 1, &cpi, nullptr, &p->pipe);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(dev_, p->layout, nullptr);
        p->pipe = VK_NULL_HANDLE;
    }
    return r;
}

const Pipeline &Context::pipeline(const std::string &spv, uint32_t push_bytes, uint32_t required_subgroup,
                                  const std::vector<uint32_t> &spec) {
    const Key key{spv, push_bytes, required_subgroup, spec};
    auto it = pipes_.find(key);
    if (it != pipes_.end()) {
        last_pipeline_status_ = CERE_OK;
        return it->second;
    }
    const Pipeline *p = nullptr;
    std::string why;
    last_pipeline_status_ = pipeline(PipelineSpec{spv, push_bytes, required_subgroup, spec}, &p, &why);
    if (last_pipeline_status_ == CERE_OK) return *p;
    last_error_ = why;
    return null_pipe_;
}

cere_status Context::warm(const std::vector<PipelineSpec> &specs, std::string *why) {
    // Serial: dedupe, the checks and the modules (the maps are not thread-safe), stopping at the first failure in
    // `specs` order. Parallel: the compiles of the specs before it. Serial again: insert in `specs` order, then report.
    struct Job {
        const PipelineSpec *s;
        VkShaderModule mod;
        Pipeline p;
        VkResult r;
    };
    std::vector<Job> jobs;
    std::set<Key> seen;
    cere_status first = CERE_OK;
    std::string first_why;
    for (const PipelineSpec &s : specs) {
        const Key key{s.spv, s.push_bytes, s.required_subgroup, s.spec};
        if (pipes_.count(key) != 0 || !seen.insert(key).second) continue;
        std::string w;
        VkShaderModule mod = VK_NULL_HANDLE;
        cere_status st = CERE_OK;
        if (faulted_) {
            w = "context faulted: " + fault_reason_;
            st = CERE_ERR_INTERNAL;
        } else if ((st = check_spec(s, &w)) == CERE_OK) {
            st = load_module(s.spv, &mod, &w);
        }
        if (st != CERE_OK) {
            first = st;
            first_why = spec_text(s) + ": " + w;
            break;
        }
        jobs.push_back(Job{&s, mod, Pipeline{}, VK_SUCCESS});
    }
    const size_t nthreads = std::min<size_t>(kWarmThreads, jobs.size());
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < jobs.size();) jobs[i].r = create_pipeline(*jobs[i].s, jobs[i].mod, &jobs[i].p);
    };
    std::vector<std::thread> pool;
    try {
        for (size_t t = 1; t < nthreads; ++t) pool.emplace_back(work);
    } catch (const std::system_error &) {  // no thread: the threads started (and this one) take the rest
    }
    work();
    for (std::thread &t : pool) t.join();
    // Every job precedes the serial failure (if any), so the first failed job is the first failure in `specs` order.
    // Compiled pipelines are kept even after a failure (the context destroys them); the caller fails instantiate.
    cere_status st = first;
    std::string w;
    for (Job &j : jobs) {
        const PipelineSpec &s = *j.s;
        if (j.r == VK_SUCCESS) {
            pipes_.emplace(Key{s.spv, s.push_bytes, s.required_subgroup, s.spec}, j.p);
        } else if (w.empty()) {
            const char *call = j.p.layout == VK_NULL_HANDLE ? "vkCreatePipelineLayout(" : "vkCreateComputePipelines(";
            std::string vw;
            st = vk_fail(j.r, std::string(call) + spec_text(s) + ")", &vw);
            w = spec_text(s) + ": " + vw;
        }
    }
    if (w.empty()) w = first_why;
    if (st != CERE_OK) say(why, w);
    return st;
}

bool Context::has_pipeline(const PipelineSpec &s) const {
    return pipes_.count(Key{s.spv, s.push_bytes, s.required_subgroup, s.spec}) != 0;
}

// --- recording ---------------------------------------------------------------------------------------------------

cere_status Context::report(cere_status s) {
    if (s != CERE_OK && error_hook_) error_hook_(*this, s);
    return s;
}

cere_status Context::begin() { return report(begin_impl()); }
cere_status Context::dispatch(const Pipeline &p, const void *push, uint32_t gx, uint32_t gy, uint32_t gz) {
    return report(dispatch_impl(p, push, gx, gy, gz));
}
cere_status Context::barrier() { return report(barrier_impl()); }
cere_status Context::stamp(uint32_t i) { return report(stamp_impl(i)); }
cere_status Context::submit(uint64_t *ticket) { return report(submit_impl(ticket)); }
cere_status Context::wait(uint64_t ticket, int64_t spin_from_ns) { return report(wait_impl(ticket, spin_from_ns)); }
cere_status Context::wait_all() {
    if (faulted_) return report(CERE_ERR_INTERNAL);
    return timeline_value_ == retired_value_ ? CERE_OK : report(wait_impl(timeline_value_, -1));
}
cere_status Context::submit_wait() {
    uint64_t t = 0;
    const cere_status s = submit_impl(&t);
    return report(s != CERE_OK ? s : wait_impl(t, -1));
}

uint32_t Context::in_flight() const {
    uint32_t n = 0;
    for (uint32_t i = 0; i < nslots_; ++i) n += slots_[i].busy ? 1 : 0;
    return n;
}

cere_status Context::latch(cere_status s) {
    if (rec_status_ == CERE_OK) rec_status_ = s;
    return s;
}

// The common refusals of a recording call: faulted, not recording, an earlier error in this command buffer.
cere_status Context::recording_gate() {
    if (faulted_) return CERE_ERR_INTERNAL;
    if (!recording_) return CERE_ERR_ARGUMENT;
    return rec_status_;
}

// The caller stamps of slot `slot` (one query call; an unwritten stamp reads 0).
void Context::load_stamps(uint32_t slot) {
    std::vector<uint64_t> v(2 * kStamps, 0);
    const VkResult r = vkGetQueryPoolResults(dev_, slots_[slot].stamps, 0, kStamps, v.size() * 8, v.data(), 16,
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    for (uint32_t i = 0; i < kStamps; ++i) stamp_vals_[i] = (r == VK_SUCCESS || r == VK_NOT_READY) && v[2 * i + 1] ? v[2 * i] : 0;
    stamps_read_ = true;
}

cere_status Context::begin_impl() {
    if (faulted_) return CERE_ERR_INTERNAL;
    if (recording_) {
        last_error_ = "begin() while a command buffer is open (submit first)";
        return CERE_ERR_ARGUMENT;
    }
    uint32_t slot = nslots_;
    bool in_flight = false;
    for (uint32_t i = 0; i < nslots_; ++i) {
        if (slots_[i].busy) in_flight = true;
        else if (slot == nslots_) slot = i;
    }
    if (slot == nslots_) {
        last_error_ = "begin() with all " + std::to_string(nslots_) + " command buffers in flight (wait first)";
        return CERE_ERR_ARGUMENT;
    }
    // The last retired submit's stamps live in this slot's pool, which the new command buffer resets: keep them.
    if (int32_t(slot) == stamp_slot_ && !stamps_read_) load_stamps(slot);
    Slot &sl = slots_[slot];
    VkResult r = vkResetCommandBuffer(sl.cb, 0);
    if (r != VK_SUCCESS) return vk_fail(r, "vkResetCommandBuffer", nullptr);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkBeginCommandBuffer(sl.cb, &bi)) != VK_SUCCESS) return vk_fail(r, "vkBeginCommandBuffer", nullptr);
    if (in_flight) {
        // An earlier submit may still run: this command buffer starts after all of it, its writes visible (the first
        // scope of a pipeline barrier is every command earlier in submission order, earlier command buffers included).
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(sl.cb, &di);
        ++device_commands_;
    }
    vkCmdResetQueryPool(sl.cb, sl.stamps, 0, kStamps);
    vkCmdResetQueryPool(sl.cb, sl.frame, 0, 2);
    vkCmdWriteTimestamp2(sl.cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, sl.frame, 0);
    ++device_commands_;
#if defined(CERE_VK_DISPATCH_PROFILE)
    dprof::begin(dev_, sl.cb, slot);
#endif
    cur_ = slot;
    recording_ = true;
    rec_status_ = CERE_OK;
    ring_used_ = 0;
    return CERE_OK;
}

cere_status Context::dispatch_impl(const Pipeline &p, const void *push, uint32_t gx, uint32_t gy, uint32_t gz) {
    if (const cere_status g = recording_gate(); g != CERE_OK) return g;
    if (!p.pipe) return latch(last_pipeline_status_ != CERE_OK ? last_pipeline_status_ : CERE_ERR_ARGUMENT);
    if (p.push_bytes && !push) return latch(CERE_ERR_ARGUMENT);
    const VkCommandBuffer cb = slots_[cur_].cb;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    if (p.push_bytes) vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.push_bytes, push);
    vkCmdDispatch(cb, gx, gy, gz);
    ++device_commands_;
#if defined(CERE_VK_DISPATCH_PROFILE)
    dprof::dispatch(dev_, cur_, p.pipe, pipes_, gx, gy, gz);
#endif
    return CERE_OK;
}

cere_status Context::barrier_impl() {
    if (const cere_status g = recording_gate(); g != CERE_OK) return g;
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(slots_[cur_].cb, &di);
    ++device_commands_;
#if defined(CERE_VK_DISPATCH_PROFILE)
    dprof::stamp(dev_, slots_[cur_].cb, cur_);
#endif
    return CERE_OK;
}

cere_status Context::stamp_impl(uint32_t i) {
    if (const cere_status g = recording_gate(); g != CERE_OK) return g;
    if (i >= kStamps) {
        last_error_ = "timestamp index " + std::to_string(i) + " >= " + std::to_string(kStamps);
        return latch(CERE_ERR_ARGUMENT);
    }
    vkCmdWriteTimestamp2(slots_[cur_].cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, slots_[cur_].stamps, i);
    ++device_commands_;
    return CERE_OK;
}

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

cere_status Context::submit_impl(uint64_t *ticket) {
    if (ticket) *ticket = 0;
    if (faulted_) return CERE_ERR_INTERNAL;
    if (!recording_) {
        last_error_ = "submit without begin()";
        return CERE_ERR_ARGUMENT;
    }
    recording_ = false;  // closed whatever the outcome; the next begin() resets the command buffer
    if (rec_status_ != CERE_OK) return rec_status_;  // a latched recording error: nothing is submitted
    Slot &sl = slots_[cur_];
    const uint64_t signal = timeline_value_ + 1;
    // The completion marker: after everything recorded (compute and transfer writes), write the nonce into this slot's
    // word; make it visible to the host; stamp the end. The host clears the word first, so only this submit's write can
    // match.
    const uint64_t nonce = nonce_salt_ | (signal & 0xFFFFFFFFull);
    const uint64_t expected = nonce + nonce_skew_;
    nonce_skew_ = 0;
    std::memset(static_cast<uint8_t *>(marker_.map) + sl.marker_off, 0, 8);
#if defined(CERE_VK_DISPATCH_PROFILE)
    dprof::stamp(dev_, sl.cb, cur_);
#endif
    {
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(sl.cb, &di);
        if (!omit_marker_) vkCmdUpdateBuffer(sl.cb, marker_.buf, sl.marker_off, 8, &nonce);
        VkMemoryBarrier2 hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        hb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        hb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        hb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        hb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        di.pMemoryBarriers = &hb;
        vkCmdPipelineBarrier2(sl.cb, &di);
        vkCmdWriteTimestamp2(sl.cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, sl.frame, 1);
    }
    omit_marker_ = false;
    const int64_t t0 = now_ns();
    VkResult r = vkEndCommandBuffer(sl.cb);
    if (r != VK_SUCCESS) return vk_fail(r, "vkEndCommandBuffer", nullptr);
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = sl.cb;
    VkSemaphoreSubmitInfo ssi{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    ssi.semaphore = timeline_; ssi.value = signal; ssi.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1; si.pSignalSemaphoreInfos = &ssi;
    r = vkQueueSubmit2(q_, 1, &si, VK_NULL_HANDLE);
    ++device_commands_;
    if (r != VK_SUCCESS) {  // whether anything reached the queue is unknown: fault
        set_fault(r, std::string("vkQueueSubmit2 = ") + result_name(r));
        last_error_ = fault_reason_;
        return CERE_ERR_INTERNAL;
    }
    const int64_t t1 = now_ns();
    timeline_value_ = signal;
    sl.busy = true;
    sl.ticket = signal;
    sl.nonce = nonce;
    sl.expected = expected;
    sl.host_submit_end_ns = t1;
    sl.submit_call_ms = double(t1 - t0) * 1e-6;
    if (ticket) *ticket = signal;
    return CERE_OK;
}

cere_status Context::wait_impl(uint64_t ticket, int64_t spin_from_ns) {
    if (faulted_) return CERE_ERR_INTERNAL;
    if (ticket == 0 || ticket > timeline_value_) {
        last_error_ = "wait(" + std::to_string(ticket) + "): not a submitted ticket (last " + std::to_string(timeline_value_) + ")";
        return CERE_ERR_ARGUMENT;
    }
    // Strictly in submission order: the in-flight tickets are retired_value_ + 1 .. timeline_value_.
    while (retired_value_ < ticket) {
        uint32_t slot = nslots_;
        for (uint32_t i = 0; i < nslots_; ++i)
            if (slots_[i].busy && slots_[i].ticket == retired_value_ + 1) slot = i;
        if (slot == nslots_) {
            set_fault(VK_ERROR_UNKNOWN, "launch layer: ticket " + std::to_string(retired_value_ + 1) + " not in flight");
            last_error_ = fault_reason_;
            return CERE_ERR_INTERNAL;
        }
        if (const cere_status s = retire_one(slot, slots_[slot].ticket == ticket ? spin_from_ns : -1); s != CERE_OK)
            return s;
    }
    return CERE_OK;
}

// Retires the oldest in-flight submit (slot): bounded wait, marker check, stamps, timing. A failure faults the context;
// this slot and every later in-flight one stay unretired (never reported complete).
cere_status Context::retire_one(uint32_t slot, int64_t spin_from_ns) {
    Slot &sl = slots_[slot];
    last_gpu_ms_ = -1;
    const uint64_t later = timeline_value_ - sl.ticket;  // in-flight submits behind this one
    const auto discarded = [&] {
        return later ? "; " + std::to_string(later) + " later in-flight submit(s) discarded unverified" : std::string();
    };
    const int64_t w0 = now_ns();
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1; wi.pSemaphores = &timeline_; wi.pValues = &sl.ticket;
    VkResult r = VK_TIMEOUT;
    uint64_t waited_ns = 0;  // of the bound, spent before the final blocking wait
    if (spin_from_ns >= 0) {
        // Block until spin_from_ns (a timeout there is the expected outcome), then poll for at most kMaxSpinNs.
        if (spin_from_ns > w0)
            r = vkWaitSemaphores(dev_, &wi, std::min<uint64_t>(uint64_t(spin_from_ns - w0), wait_timeout_ns_));
        const int64_t s0 = now_ns();
        while (r == VK_TIMEOUT && now_ns() - s0 < kMaxSpinNs) {
            uint64_t v = 0;
            r = vkGetSemaphoreCounterValue(dev_, timeline_, &v);
            if (r == VK_SUCCESS && v < sl.ticket) r = VK_TIMEOUT;
        }
        waited_ns = uint64_t(now_ns() - w0);
    }
    if (r == VK_TIMEOUT) r = vkWaitSemaphores(dev_, &wi, waited_ns < wait_timeout_ns_ ? wait_timeout_ns_ - waited_ns : 0);
    const int64_t w1 = now_ns();
    if (r != VK_SUCCESS) {  // VK_TIMEOUT included: the submission's completion is unknown
        set_fault(r, std::string("vkWaitSemaphores = ") + result_name(r) +
                         (r == VK_TIMEOUT ? " after " + std::to_string(wait_timeout_ns_ / 1000000) + " ms" : "") + " (ticket " +
                         std::to_string(sl.ticket) + ")" + discarded());
        if (r == VK_ERROR_DEVICE_LOST) append_fault_info();
        last_error_ = fault_reason_;
        return CERE_ERR_INTERNAL;
    }
    uint64_t seen = 0;
    std::memcpy(&seen, static_cast<const uint8_t *>(marker_.map) + sl.marker_off, 8);
    if (seen != sl.expected) {  // the wait succeeded but the command buffer did not run to its end: an aborted job
        char m[240];
        std::snprintf(m, sizeof m,
                      "completion marker missing after a successful wait (ticket %llu, expected 0x%016llx, found 0x%016llx): "
                      "the job was aborted by the driver (e.g. the amdgpu ring timeout)",
                      (unsigned long long)sl.ticket, (unsigned long long)sl.expected, (unsigned long long)seen);
        set_fault(VK_ERROR_DEVICE_LOST, m + discarded());
        if (sl.expected == sl.nonce) append_fault_info();  // a real mismatch (not the test skew): ask the driver
        last_error_ = fault_reason_;
        return CERE_ERR_INTERNAL;
    }
    sl.busy = false;
    retired_value_ = sl.ticket;
    last_nonce_ = seen;
    stamp_slot_ = int32_t(slot);
    stamps_read_ = false;
    SubmitTiming t;
    t.ticket = sl.ticket;
    t.submit_call_ms = sl.submit_call_ms;
    t.wait_call_ms = double(w1 - w0) * 1e-6;
    uint64_t ts[2] = {0, 0};
    if (vkGetQueryPoolResults(dev_, sl.frame, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        last_gpu_ms_ = double((ts[1] - ts[0]) & ts_mask_) * info_.timestamp_period_ns * 1e-6;
        t.gpu_ms = last_gpu_ms_;
        if (prev_end_ticket_ != 0 && prev_end_ticket_ + 1 == sl.ticket)
            t.gpu_gap_before_ms = double(int64_t((ts[0] - prev_end_tick_) & ts_mask_)) * info_.timestamp_period_ns * 1e-6;
        prev_end_tick_ = ts[1];
        prev_end_ticket_ = sl.ticket;
        if (calibrate_ && calibrated_timing_) {
            VkCalibratedTimestampInfoEXT ci[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT},
                                                  {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT}};
            uint64_t cal[2] = {0, 0}, dev = 0;
            if (calibrate_(dev_, 2, ci, cal, &dev) == VK_SUCCESS) {
                const auto host_ms = [&](uint64_t tick) {  // GPU tick -> CLOCK_MONOTONIC ms
                    return double(cal[1]) * 1e-6 + double(int64_t(tick - cal[0])) * info_.timestamp_period_ns * 1e-6;
                };
                t.start_after_submit_ms = host_ms(ts[0]) - double(sl.host_submit_end_ns) * 1e-6;
                t.retire_after_end_ms = double(w1) * 1e-6 - host_ms(ts[1]);
                t.calibration_dev_ms = double(dev) * 1e-6;
            }
        }
    }
#if defined(CERE_VK_DISPATCH_PROFILE)
    dprof::retire(dev_, slot, info_.timestamp_period_ns, ts_mask_);
#endif
    last_timing_ = t;
    return CERE_OK;
}

double Context::stamp_ms(uint32_t a, uint32_t b) {
    if (faulted_ || a >= kStamps || b >= kStamps || stamp_slot_ < 0) return 0;
    if (!stamps_read_) load_stamps(uint32_t(stamp_slot_));
    return double(stamp_vals_[b] - stamp_vals_[a]) * info_.timestamp_period_ns * 1e-6;
}

cere_status Context::table_alloc(uint64_t bytes, uint64_t align, TableRegion *out) {
    if (!out) return CERE_ERR_ARGUMENT;
    *out = TableRegion();
    if (faulted_) return CERE_ERR_INTERNAL;
    if (!recording_ || bytes == 0 || align == 0 || align > kMaxTableAlign || (align & (align - 1)) != 0) return CERE_ERR_ARGUMENT;
    if (!ring_.buf) {
        last_error_ = "no table ring (CreateOptions::table_ring_bytes 0)";
        return CERE_ERR_UNSUPPORTED;
    }
    const uint64_t part = ring_.addr + slots_[cur_].ring_off;  // this command buffer's part of the ring
    const uint64_t base = part + ring_used_;
    const uint64_t start = (base + align - 1) / align * align - part;  // offset whose device address is aligned
    if (start > ring_slot_bytes_ || bytes > ring_slot_bytes_ - start) {
        last_error_ = "table ring overflow: " + std::to_string(bytes) + " B at offset " + std::to_string(start) + " of " +
                      std::to_string(ring_slot_bytes_);
        return CERE_ERR_CAPACITY;
    }
    out->addr = part + start;
    out->host = static_cast<uint8_t *>(ring_.map) + slots_[cur_].ring_off + start;
    out->bytes = bytes;
    ring_used_ = start + bytes;
    return CERE_OK;
}

}  // namespace cere::vk
