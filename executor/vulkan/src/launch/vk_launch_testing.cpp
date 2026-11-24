// VulkanAmd ktest convenience: see vk_launch_testing.h. TEST ONLY: never linked into the executor library.
#include "launch/vk_launch_testing.h"

#include <cstdio>
#include <cstdlib>

namespace cere::vk::testing {

static void die_on_error(const Context &ctx, cere_status s) {
    std::printf("FATAL launch call returned %d: %s%s%s\n", int(s), ctx.last_error().c_str(), ctx.faulted() ? "; fault: " : "",
                ctx.faulted() ? ctx.fault_reason().c_str() : "");
    std::fflush(stdout);
    std::exit(2);
}

std::unique_ptr<Context> create_or_die(const std::string &shader_dir, uint32_t ordinal, uint32_t max_in_flight) {
    std::unique_ptr<Context> ctx;
    std::string why;
    // CERE_VK_ALLOW_SOFTWARE=1 also offers CPU devices (lavapipe): kernels without a required subgroup size run there,
    // for development hosts without the GPU (Inkling S11 G1 rows; RADV on AMD stays the gate).
    CreateOptions opts;
    const char *sw = std::getenv("CERE_VK_ALLOW_SOFTWARE");
    opts.allow_software = sw != nullptr && sw[0] == '1';
    opts.max_in_flight = max_in_flight;
    const cere_status s = Context::create(ordinal, &ctx, &why, opts);
    if (s != CERE_OK) {
        std::printf("%s %s\n", s == CERE_ERR_UNSUPPORTED ? "UNSUPPORTED" : "FATAL", why.c_str());
        std::fflush(stdout);
        std::exit(s == CERE_ERR_UNSUPPORTED ? 3 : 2);
    }
    ctx->set_shader_dir(shader_dir);
    ctx->set_error_hook_for_test(die_on_error);
    return ctx;
}

}  // namespace cere::vk::testing
