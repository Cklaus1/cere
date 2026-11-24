#!/bin/sh
# VulkanAmd P3 T2: build libcere_executor_vulkan.so in the llama-amd:rocm7.2.4 image (g++ 13.3, glslc shaderc 2023.8,
# libvulkan-dev 1.3.275). From the worktree root:
#   docker run --rm -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh executor/vulkan/build_lib.sh
# The library = the UNCHANGED host layer (executor/cuda/src/host/*.cpp + the host-side research/*.cpp, as the Makefile's
# toy library and executor/hip/build.sh) + the Vulkan device (src/device/*.cpp) + every kernel object + the launch
# layer's library object vk_launch_core.o + the embedded SPIR-V table vk_spv_table.o, -fPIC -fvisibility=hidden, the
# CUDA / HIP version script (executor/cuda/src/host/exports.map), -lvulkan. Never vk_launch_testing.o or a ktest.
# Shaders and kernel objects come from executor/vulkan/build.sh (CXXFLAGS_EXTRA=-fPIC) into $KOUT.
# Env: OUT (library dir, default build/vulkan), KOUT (kernel build, default build/vulkan-t2b/k), TOUT (tests,
# default build/vulkan-t2b), SKIP_KERNELS=1 to reuse $KOUT. Tests built: smoke, device_test, load_s3, forward_s3,
# forward_s3_static (T5), spec_s3, spec_s3_static, spec_s3_prof (T6),
# coverage_s3, coverage_s3_static (P4-F). Prints the library's sha256 and exported symbol count.
set -eu
cd "$(dirname "$0")/../.."
V=executor/vulkan
OUT=${OUT:-build/vulkan}
KOUT=${KOUT:-build/vulkan-t2b/k}
TOUT=${TOUT:-build/vulkan-t2b}
mkdir -p $OUT $TOUT/obj $KOUT/gen
CERE_MODEL=${CERE_MODEL:-qwen38-flash-next}
case "$CERE_MODEL" in
  qwen38-flash-next|inkling-2-mini) ;;
  *) echo "build_lib.sh: only the qwen38-flash-next and inkling-2-mini descriptors are built" >&2; exit 2 ;;
esac

# 1. Shaders, embedded SPIR-V table, launch layer and kernel objects (position-independent).
if [ "${SKIP_KERNELS:-0}" != 1 ]; then
  CXXFLAGS_EXTRA=-fPIC OUT=$KOUT sh $V/build.sh > $TOUT/kernels_build.log 2>&1 || { tail -30 $TOUT/kernels_build.log; exit 1; }
fi

# Inkling 2 Mini (inkling design note S11): the same host layer with the ki_* forward and the inkling device
# (src/device/inkling/), the launch layer and K2; no Qwen device, warm list or Qwen tests. ki_model.cpp is compiled
# with -ffp-contract=off as in build.d/ki_inkling.sh (its host-side arithmetic stays bit-equal to the ktests').
if [ "$CERE_MODEL" = inkling-2-mini ]; then
  IOBJS="$KOUT/vk_launch_core.o $KOUT/vk_spv_table.o $KOUT/ki_ops_vk.o $KOUT/k2_gemm_vk.o $KOUT/k2_mgemv_vk.o"
  for o in $IOBJS; do
    [ -f $o ] || { echo "build_lib.sh: missing kernel object $o" >&2; exit 1; }
    readelf -r $o | grep -q 'R_X86_64_32S\? ' && { echo "build_lib.sh: $o is not PIC" >&2; exit 1; }
  done
  GEN=$TOUT/generated
  mkdir -p $GEN
  python3 executor/tools/gen_model_header.py executor/models/$CERE_MODEL.json $GEN/cere_model_constants.h
  CXX="g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread -fPIC -fvisibility=hidden \
 -Iexecutor/include -Iexecutor/cuda/src -Iexecutor -I$V/src -I$GEN"
  HOST_SRC=$(ls executor/cuda/src/host/*.cpp executor/cuda/src/research/*.cpp | sort)
  DEV_SRC="$V/src/device/inkling/ki_model.cpp $V/src/device/inkling/vk_inkling_device.cpp"
  rm -f $TOUT/obj/*.o
  n=0
  for f in $HOST_SRC $DEV_SRC; do
    n=$((n + 1)); $CXX -ffp-contract=off -c $f -o $TOUT/obj/$(echo $f | tr / _).o & [ $((n % 8)) -ne 0 ] || wait
  done; wait
  for f in $HOST_SRC $DEV_SRC; do [ -f $TOUT/obj/$(echo $f | tr / _).o ] || { echo "build_lib.sh: compile failed: $f" >&2; exit 1; }; done
  g++ -shared -fPIC -fvisibility=hidden -pthread -Wl,--version-script=executor/cuda/src/host/exports.map -Wl,-z,defs \
    $TOUT/obj/*.o $IOBJS -lvulkan -o $OUT/libcere_executor_vulkan.so
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iexecutor/include -Iexecutor/cuda/src $V/tests/smoke.cpp \
    executor/cuda/src/host/support.cpp -ldl -o $TOUT/smoke
  echo "library: $OUT/libcere_executor_vulkan.so (inkling-2-mini)"
  sha256sum $OUT/libcere_executor_vulkan.so
  echo "exported functions: $(nm -D --defined-only $OUT/libcere_executor_vulkan.so | awk '$2=="T"' | wc -l)"
  exit 0
fi
# The Qwen library's SPIR-V table holds no Inkling module (ki_*, kiv_*: the inkling-2-mini library's). build.sh's
# shared table has them, so without this filter every Inkling shader change moved the Qwen library's bytes (main's
# build stopped reproducing its pin a04ba007 when S11 added them; with the filter it does again, D-I-EPB).
# (Same file names as build.sh's table: the object's file symbol lands in the library.)
rm -rf $KOUT/qwen && mkdir -p $KOUT/qwen/spv
for f in $KOUT/spv/*.spv; do case ${f##*/} in ki_*|kiv_*) ;; *) cp $f $KOUT/qwen/spv/ ;; esac; done
python3 $V/tools/embed_spv.py $KOUT/qwen/spv $KOUT/qwen/vk_spv_table.cpp
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -I$V/src -Iexecutor/include -fPIC \
  -c $KOUT/qwen/vk_spv_table.cpp -o $KOUT/qwen/vk_spv_table.o
KOBJS="$KOUT/vk_launch_core.o $KOUT/qwen/vk_spv_table.o"
for k in k1_primitives k2_mgemv k2_gemm k3_hc k4_ple k5_gdn k5_gdn_chunk k5_gdn_layer k5_gdn_chain k6_attention \
         k6_prologue k7_moe k9_qsa k12_sampler k13s_spec_accept k13_moe_bf16 mtp_kernels; do
  [ -f $KOUT/${k}_vk.o ] || { echo "build_lib.sh: missing kernel object $KOUT/${k}_vk.o" >&2; exit 1; }
  KOBJS="$KOBJS $KOUT/${k}_vk.o"
done
for o in $KOBJS; do readelf -r $o | grep -q 'R_X86_64_32S\? ' && { echo "build_lib.sh: $o is not PIC" >&2; exit 1; }; done

# 2. Generated headers: the model constants (gen_model_header.py, as the Makefile / HIP build) and which kernel
#    families' pipeline lists exist (vulkan_warm.cpp: a list absent at build time is skipped and reported).
GEN=$TOUT/generated
mkdir -p $GEN
python3 executor/tools/gen_model_header.py executor/models/$CERE_MODEL.json $GEN/cere_model_constants.h
W=$GEN/vk_warm_have.h
K=$V/src/kernels
have() {  # have MACRO file exact-declaration-text
  if grep -qF "$3" "$K/$2"; then echo "#define CERE_VKW_$1 1"; else echo "/* CERE_VKW_$1: absent in $2 */"; fi
}
{
  echo "// generated by executor/vulkan/build_lib.sh: the kernel pipeline lists present at build time"
  have K1 k1_primitives_vk.h 'pipeline_specs(const std::vector<int> &norm_dims, const std::vector<int> &qk_head_dims)'
  have K2G k2_gemm_vk.h 'pipeline_specs(bool bf16, bool mx'
  have K2V k2_mgemv_vk.h 'pipeline_specs(const std::vector<uint32_t> &bf16_ks, const std::vector<uint32_t> &mx_ks, bool dequant)'
  have K3 k3_hc_vk.h 'pipeline_specs(const HcShape &shape, bool gemm)'
  have K4 k4_ple_vk.h 'pipeline_specs(bool fp8_table, bool nvfp4_table, bool gemm)'
  have K5L k5_gdn_layer_vk.h 'pipeline_specs(const std::vector<uint32_t> &hidden_sizes, bool gemm, bool mx'
  have K5C k5_gdn_chain_vk.h 'std::vector<vk::PipelineSpec> pipeline_specs();'
  have K6 k6_attention_vk.h 'pipeline_specs(const Options &opt, bool bf16_kv, bool fp8_kv)'
  have K6P k6_prologue_vk.h 'std::vector<vk::PipelineSpec> pipeline_specs();'
  have K7 k7_moe_vk.h 'pipeline_specs(const MoeDims &dims, bool gemv, bool gemm, bool option_variants, bool mx'
  have K9 k9_qsa_vk.h 'pipeline_specs(const Options &opt, bool bf16_kv, bool fp8_kv, const std::vector<int> &rows_heads)'
  have K12 k12_sampler_vk.h 'pipeline_specs(int64_t max_top_k)'
  have K13 k13_moe_bf16_vk.h 'pipeline_specs(const MoeDims &dims, bool bf16_experts, bool nvfp4_experts, bool gemm)'
  have K13S k13s_spec_accept_vk.h 'std::vector<vk::PipelineSpec> pipeline_specs();'
  have MTP mtp_kernels_vk.h 'pipeline_specs(const std::vector<int> &extra_norm_dims'
} > $W
cat $W

# 3. Host layer + device objects, then the library.
CXX="g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread -fPIC -fvisibility=hidden \
 -Iexecutor/include -Iexecutor/cuda/src -I$V/src -I$GEN"
HOST_SRC=$(ls executor/cuda/src/host/*.cpp executor/cuda/src/research/*.cpp | sort)
DEV_SRC="$V/src/device/vk_memory_plan.cpp $V/src/device/vulkan_bind.cpp $V/src/device/vulkan_device.cpp $V/src/device/vulkan_warm.cpp \
 $V/src/device/vulkan_forward.cpp $V/src/device/vulkan_spec.cpp"
rm -f $TOUT/obj/*.o
n=0
for f in $HOST_SRC $DEV_SRC; do
  n=$((n + 1)); $CXX -c $f -o $TOUT/obj/$(echo $f | tr / _).o & [ $((n % 8)) -ne 0 ] || wait
done; wait
for f in $HOST_SRC $DEV_SRC; do [ -f $TOUT/obj/$(echo $f | tr / _).o ] || { echo "build_lib.sh: compile failed: $f" >&2; exit 1; }; done
g++ -shared -fPIC -fvisibility=hidden -pthread -Wl,--version-script=executor/cuda/src/host/exports.map -Wl,-z,defs \
  $TOUT/obj/*.o $KOBJS -lvulkan -o $OUT/libcere_executor_vulkan.so

# 4. Tests: the dlopen smoke, and the statically linked device test / S3 load test (same objects as the library).
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iexecutor/include -Iexecutor/cuda/src $V/tests/smoke.cpp \
  executor/cuda/src/host/support.cpp -ldl -o $TOUT/smoke
TCXX="g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread -Iexecutor/include \
 -Iexecutor/cuda/src -I$V/src -I$GEN -I$V/tests"
$TCXX $V/tests/device_test.cpp $TOUT/obj/*.o $KOBJS -lvulkan -o $TOUT/device_test
$TCXX $V/tests/load_s3.cpp $TOUT/obj/*.o $KOBJS -lvulkan -o $TOUT/load_s3
# T5: the S3 forward harness, dlopen build (any library of the ABI family) and the static test build (the device and
# forward compiled again with the per-layer taps: TEST ONLY, never in the library).
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread -Iexecutor/include \
  -Iexecutor/cuda/src -I$GEN -I$V/tests $V/tests/forward_s3.cpp executor/cuda/src/host/support.cpp -ldl -o $TOUT/forward_s3
mkdir -p $TOUT/obj_taps
TFLAGS="-DCERE_VK_FORWARD_TAPS"
$CXX $TFLAGS -c $V/src/device/vulkan_forward.cpp -o $TOUT/obj_taps/vulkan_forward.o
$CXX $TFLAGS -c $V/src/device/vulkan_device.cpp -o $TOUT/obj_taps/vulkan_device.o
SOBJS=$(ls $TOUT/obj/*.o | grep -v -e 'src_device_vulkan_forward.cpp.o' -e 'src_device_vulkan_device.cpp.o')
$TCXX $TFLAGS -DFS3_STATIC $V/tests/forward_s3.cpp $SOBJS $TOUT/obj_taps/*.o $KOBJS -lvulkan -o $TOUT/forward_s3_static
# VulkanAutotune stage 2 (DPROF=1 only): forward_s3_dprof = forward_s3_static without taps, with the launch layer and the
# forward compiled again with CERE_VK_DISPATCH_PROFILE (per-barrier GPU stamps of every DECODE command buffer, written
# to $CERE_VK_DPROF_OUT at exit). TEST ONLY, never in the library.
if [ "${DPROF:-0}" = 1 ]; then
  mkdir -p $TOUT/obj_dprof
  $CXX -DCERE_VK_DISPATCH_PROFILE -c $V/src/device/vulkan_forward.cpp -o $TOUT/obj_dprof/vulkan_forward.o
  $CXX -DCERE_VK_DISPATCH_PROFILE -c $V/src/launch/vk_launch.cpp -o $TOUT/obj_dprof/vk_launch_core.o
  $CXX -DCERE_VK_DISPATCH_PROFILE -c $V/src/kernels/k2_mgemv_vk.cpp -o $TOUT/obj_dprof/k2_mgemv_vk.o
  DOBJS=$(ls $TOUT/obj/*.o | grep -v 'src_device_vulkan_forward.cpp.o')
  $TCXX -DFS3_STATIC $V/tests/forward_s3.cpp $DOBJS $TOUT/obj_dprof/*.o \
    $(echo $KOBJS | sed -e "s#$KOUT/vk_launch_core.o##" -e "s#$KOUT/k2_mgemv_vk.o##") -lvulkan -o $TOUT/forward_s3_dprof
fi

# T6: the S3 speculation harness, dlopen and static builds (no taps), and the static build with the T5 taps objects
# (spec_s3_prof: CERE_VK_FORWARD_PROFILE=1 splits the plain / speculating prefill and the extends by op family).
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread -Iexecutor/include \
  -Iexecutor/cuda/src -I$GEN -I$V/tests $V/tests/spec_s3.cpp executor/cuda/src/host/support.cpp -ldl -o $TOUT/spec_s3
$TCXX -DFS3_STATIC $V/tests/spec_s3.cpp $TOUT/obj/*.o $KOBJS -lvulkan -o $TOUT/spec_s3_static
$TCXX $TFLAGS -DFS3_STATIC $V/tests/spec_s3.cpp $SOBJS $TOUT/obj_taps/*.o $KOBJS -lvulkan -o $TOUT/spec_s3_prof

# P4-F: the coverage harness (long context / limit, vocab masks, cancel, levers, draft windows, sampled-speculation
# distribution, long-context speculation), dlopen and static builds (no taps).
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -pthread -Iexecutor/include \
  -Iexecutor/cuda/src -I$GEN -I$V/tests $V/tests/coverage_s3.cpp executor/cuda/src/host/support.cpp -ldl -o $TOUT/coverage_s3
$TCXX -DFS3_STATIC $V/tests/coverage_s3.cpp $TOUT/obj/*.o $KOBJS -lvulkan -o $TOUT/coverage_s3_static

echo "library: $OUT/libcere_executor_vulkan.so"
sha256sum $OUT/libcere_executor_vulkan.so $KOUT/qwen/vk_spv_table.o $KOUT/vk_launch_core.o
echo "exported functions: $(nm -D --defined-only $OUT/libcere_executor_vulkan.so | awk '$2=="T"' | wc -l)"
echo "exported symbols (all kinds): $(nm -D --defined-only $OUT/libcere_executor_vulkan.so | wc -l)"
