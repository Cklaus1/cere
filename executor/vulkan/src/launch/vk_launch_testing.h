// VulkanAmd ktest convenience on top of the launch layer: the P1 behaviour "any launch error ends the process" for the
// standalone test binaries. Built into its own translation unit (vk_launch_testing.cpp); the executor library never
// links it, so nothing on the library path can exit.
#pragma once

#include <memory>
#include <string>

#include "launch/vk_launch.h"

namespace cere::vk::testing {

// Context::create(ordinal) with the shader directory set (env CERE_VK_ALLOW_SOFTWARE=1: CPU devices such as lavapipe are
// selectable too); on failure prints "UNSUPPORTED <why>" (exit 3) or
// "FATAL <why>" (exit 2). Installs an error hook that prints the failing status and exits 2 on any non-OK begin /
// dispatch / barrier / stamp / submit_wait (the P1 launch layer's exit-on-error).
// max_in_flight: CreateOptions::max_in_flight (command-buffer slots).
std::unique_ptr<Context> create_or_die(const std::string &shader_dir, uint32_t ordinal = 0, uint32_t max_in_flight = 1);

}  // namespace cere::vk::testing
