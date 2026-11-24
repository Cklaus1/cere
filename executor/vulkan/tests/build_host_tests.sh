#!/bin/sh
# VulkanAmd P3 T2: build the host-only tests of the Vulkan memory plan (no GPU, no Vulkan device). From the worktree
# root, in the pinned image:
#   docker run --rm -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh executor/vulkan/tests/build_host_tests.sh
# Run (the artifact mounted read-only; only manifest.json and safetensors headers are read):
#   docker run --rm -v "$PWD":/w -w /w -v /var/lib/cere/artifacts:/artifacts:ro llama-amd:rocm7.2.4 \
#     sh -c 'build/vulkan-t2a/vk_memory_plan_test /artifacts/qwen38-flash-next-fp4local-cere-s3-dedc7dba9bc84493 \
#            configs/artifacts/qwen38-flash-next-fp4local-cere-s3.inventory.json; build/vulkan-t2a/vk_kernel_sizes_drift_test'
# Output: build/vulkan-t2a/{vk_memory_plan_test, vk_kernel_sizes_drift_test, gen/cere_model_constants.h}.
#
# vk_memory_plan_test is pure host code: the memory plan, the CUDA-free host layer pieces it reuses (host/support.cpp,
# host/kv_pool.cpp, host/sha256.cpp; headers model/layout.h, model/mxfp8_plan.h, host/device.h) and the generated
# model constants. vk_kernel_sizes_drift_test also compiles the Vulkan kernel sources and links libvulkan (no context
# is created): it checks the restated workspace table against the kernels' own sizers. The kernel sources are being
# edited by other tasks; if they do not compile, that test is reported NOT BUILT (the exit status says so) and the
# memory plan test is still built.
set -eu
cd "$(dirname "$0")/../../.."
OUT=${OUT:-build/vulkan-t2a}
mkdir -p "$OUT/gen" "$OUT/obj"
g++ --version | head -1
python3 executor/tools/gen_model_header.py executor/models/qwen38-flash-next.json "$OUT/gen/cere_model_constants.h"
CXX="g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iexecutor/include -Iexecutor/cuda/src -Iexecutor/vulkan/src -I$OUT/gen"
HOST="executor/cuda/src/host/support.cpp executor/cuda/src/host/kv_pool.cpp executor/cuda/src/host/sha256.cpp"
$CXX executor/vulkan/tests/vk_memory_plan_test.cpp executor/vulkan/src/device/vk_memory_plan.cpp $HOST \
  -o "$OUT/vk_memory_plan_test"
echo "built $OUT/vk_memory_plan_test"

# The drift test (kernel sources as executor/vulkan/build.sh compiles them: same flags plus
# -Wno-missing-field-initializers).
K=executor/vulkan/src/kernels
KCXX="$CXX -Wno-missing-field-initializers"
status=0
objs=""
for src in executor/vulkan/src/launch/vk_launch.cpp $K/k1_primitives_vk.cpp $K/k2_mgemv_vk.cpp $K/k2_gemm_vk.cpp \
           $K/k3_hc_vk.cpp $K/k4_ple_vk.cpp $K/k5_gdn_vk.cpp $K/k5_gdn_chunk_vk.cpp $K/k5_gdn_layer_vk.cpp \
           $K/k6_attention_vk.cpp $K/k7_moe_vk.cpp $K/k9_qsa_vk.cpp $K/k12_sampler_vk.cpp $K/k13s_spec_accept_vk.cpp \
           $K/k13_moe_bf16_vk.cpp; do
  o="$OUT/obj/$(basename "$src" .cpp).o"
  if $KCXX -c "$src" -o "$o"; then objs="$objs $o"; else echo "NOT BUILT: $src does not compile"; status=1; fi
done
if [ "$status" -eq 0 ] && $KCXX executor/vulkan/tests/vk_kernel_sizes_drift_test.cpp $objs -lvulkan \
     -o "$OUT/vk_kernel_sizes_drift_test"; then
  echo "built $OUT/vk_kernel_sizes_drift_test"
else
  echo "NOT BUILT: vk_kernel_sizes_drift_test"
  rm -f "$OUT/vk_kernel_sizes_drift_test"
  status=1
fi
sha256sum "$OUT"/vk_memory_plan_test "$OUT"/vk_kernel_sizes_drift_test 2>/dev/null || true
exit $status
