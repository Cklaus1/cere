# K4 PLE (P2): n-gram hash, split-table gather (FP8 / NVFP4), embed, the layer forward; kernel object and k4_ple_test.
# Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined. The forward
# records K1's rmsnorm and K2 (vkk2::run: GEMV / large-m GEMM), so build those fragments into the same OUT first:
#   docker run --rm -e OUT=build/vulkan-k4 -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh executor/vulkan/build.sh k1_primitives k2_gemm k2_mgemv k4_ple
# HIP reference timing (k4_hip_timing): built when hipcc exists and K4_HIPSRC (default /src/ai-cere, a checkout mounted
# read-only) holds a gen_hip.py tree build/hip/gen with kernels/k4_ple.hip and build/hip/libcere_hipk2.a.
for o in k1_primitives_vk k2_mgemv_vk k2_gemm_vk; do
  [ -f $OUT/$o.o ] || { echo "k4_ple: $OUT/$o.o missing: build the k1_primitives, k2_mgemv and k2_gemm fragments into $OUT too"; exit 1; }
done
rm -f $OUT/.fail4
$G $V/shaders/k4_hash.comp -o $OUT/spv/k4_hash.spv || touch $OUT/.fail4 &
for nv in 0 1; do
  fmt=fp8; [ $nv = 1 ] && fmt=nvfp4
  $G -DNVFP4=$nv $V/shaders/k4_gather.comp -o $OUT/spv/k4_gather_$fmt.spv || touch $OUT/.fail4 &
  $G -DNVFP4=$nv $V/shaders/k4_embed.comp -o $OUT/spv/k4_embed_$fmt.spv || touch $OUT/.fail4 &
done
$G $V/shaders/k4_gate.comp -o $OUT/spv/k4_gate.spv || touch $OUT/.fail4 &
$G $V/shaders/k4_conv.comp -o $OUT/spv/k4_conv.spv || touch $OUT/.fail4 &
$G $V/shaders/k4_state.comp -o $OUT/spv/k4_state.spv || touch $OUT/.fail4 &
wait
[ ! -f $OUT/.fail4 ] || { echo "k4 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k4_ple_vk.cpp -o $OUT/k4_ple_vk.o
$CXX -I$V/ktest -Iexecutor/cuda/src $V/ktest/k4_ple_test.cpp $OUT/vk_launch.o $OUT/k4_ple_vk.o $OUT/k1_primitives_vk.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -pthread -o $OUT/k4_ple_test
K4_HIPSRC=${K4_HIPSRC:-/src/ai-cere}
if command -v hipcc >/dev/null 2>&1 && [ -f $K4_HIPSRC/build/hip/gen/kernels/k4_ple.hip ] && [ -f $K4_HIPSRC/build/hip/libcere_hipk2.a ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $K4_HIPSRC/executor/hip/src/cuda_compat.h -I$K4_HIPSRC/executor/include \
        -I$K4_HIPSRC/build/hip/gen -I$K4_HIPSRC/executor/hip/src -x hip $V/ktest/k4_hip_timing_test.cpp \
        $K4_HIPSRC/build/hip/gen/kernels/k4_ple.hip $K4_HIPSRC/build/hip/gen/kernels/k1_primitives.hip \
        -x none $K4_HIPSRC/build/hip/libcere_hipk2.a -lhipblaslt -o $OUT/k4_hip_timing
else
  echo "k4_ple: no hipcc or no hipified tree at $K4_HIPSRC/build/hip: HIP reference timing not built"
fi
sha256sum $V/shaders/k4_*.comp $OUT/spv/k4_*.spv $OUT/k4_ple_test $( [ -f $OUT/k4_hip_timing ] && echo $OUT/k4_hip_timing )
