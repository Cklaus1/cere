// m32n design note D10: per CUDA ordinal, how many times this process reset the device's primary context (the last
// executor created with the device-id option `release_device=1` was destroyed: cuda_device.h). State that CUDA keeps
// per context and the library caches per ordinal (a kernel's raised dynamic shared-memory limit, p3b_decode.cu
// launch) is valid only for the generation it was recorded in; after a reset it is set again.
#pragma once

#include <atomic>
#include <cstdint>

namespace cere::device {

// The ordinals whose generation is tracked; a device-id with release_device=1 needs an ordinal below it.
inline constexpr int kTrackedDevices = 16;

// The generation of `ordinal`'s primary context (0 until its first reset); `ordinal` in [0, kTrackedDevices).
inline std::atomic<uint32_t> &context_generation(int ordinal) {
    static std::atomic<uint32_t> generations[kTrackedDevices]{};
    return generations[ordinal];
}

}  // namespace cere::device
