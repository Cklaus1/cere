#!/bin/sh
# VulkanAmd: build the Vulkan kernels and their ktests in the llama-amd:rocm7.2.4 image (glslc from shaderc 2023.8 /
# glslang 14.0.0, libvulkan-dev 1.3.275, g++ 13.3; the pinned toolchain is the image's). From the worktree root:
#   docker run --rm -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh executor/vulkan/build.sh
# Optional arguments: fragment names (build.d/<name>.sh) to build only those; env OUT overrides the output dir
# (parallel builds use one OUT each). Output: build/vulkan/spv/*.spv, build/vulkan/{k2v_test,k2mx_test} and, when hipcc exists, build/vulkan/hip_gemv_timing
# (the same-session HIP reference timing). Prints the toolchain versions and the sha256 of every output.
set -eu
cd "$(dirname "$0")/../.."
V=executor/vulkan
OUT=${OUT:-build/vulkan}
mkdir -p $OUT/spv
rm -f $OUT/.fail
glslc --version | head -1
G="glslc --target-env=vulkan1.3 -O -fshader-stage=compute"
# CXXFLAGS_EXTRA (optional, e.g. -fPIC): appended to every C++ compile, so build_lib.sh gets position-independent kernel
# and launch-layer objects it can link into libcere_executor_vulkan.so (the ktests link them unchanged).
CXX="g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -I$V/src -Iexecutor/include ${CXXFLAGS_EXTRA:-}"
# The launch layer: vk_launch_core.o is the library's (no exit, no printing); vk_launch_testing.o is the ktests'
# exit-on-error convenience; vk_launch.o (what every fragment links) is both, relocatably linked.
$CXX -c $V/src/launch/vk_launch.cpp -o $OUT/vk_launch_core.o
$CXX -c $V/src/launch/vk_launch_testing.cpp -o $OUT/vk_launch_testing.o
ld -r $OUT/vk_launch_core.o $OUT/vk_launch_testing.o -o $OUT/vk_launch.o
# One fragment per kernel family (build.d/<kernel>.sh): its shaders, objects and ktests. A fragment touches only its
# own outputs, so kernels are added without editing this file.
ONLY="$*"   # captured here: a fragment may reset the positional parameters
for f in $V/build.d/*.sh; do
  [ -z "$ONLY" ] || case " $ONLY " in *" $(basename "$f" .sh) "*) ;; *) continue ;; esac
  echo "== $f"
  . "$f"
done
# After every fragment: embed whatever SPIR-V is in $OUT/spv into one generated table (OD-4: the library's sha256
# covers its shaders), then the launch-layer test, which uses that table (and the K1 / K2 / K3 objects for its kernel rows).
mkdir -p $OUT/gen
python3 $V/tools/embed_spv.py $OUT/spv $OUT/gen/vk_spv_table.cpp
$CXX -fPIC -c $OUT/gen/vk_spv_table.cpp -o $OUT/vk_spv_table.o
LT_OBJS="$OUT/k1_primitives_vk.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o $OUT/k3_hc_vk.o"
if ls $LT_OBJS >/dev/null 2>&1; then
  $CXX $V/ktest/vk_launch_test.cpp $OUT/vk_launch_core.o $OUT/vk_spv_table.o $LT_OBJS -lvulkan -o $OUT/vk_launch_test
else
  echo "vk_launch_test: needs the k1_primitives, k2_mgemv, k2_gemm and k3_hc objects in $OUT: not built"
fi
sha256sum $V/tools/embed_spv.py $OUT/gen/vk_spv_table.cpp $OUT/vk_spv_table.o $OUT/vk_launch_core.o $( [ -f $OUT/vk_launch_test ] && echo $OUT/vk_launch_test )
