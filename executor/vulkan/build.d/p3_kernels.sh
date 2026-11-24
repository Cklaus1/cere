# VulkanAmd P3 (T3): the kernels the first-light forward still lacked: vkk6::prologue (k6_prologue_vk), vkk5::chain_core
# (k5_gdn_chain_vk) and the MTP-local kernels (mtp_kernels_vk: norm_hidden, add_embedding, argmax_map), their objects and
# ktests (k6_prologue_test, k5_chain_test, mtp_kernels_test). Sourced by executor/vulkan/build.sh with V, OUT, G (glslc
# command), CXX and $OUT/vk_launch.o defined. Needs the k1_primitives, k2_mgemv, k5_gdn and k5_gdn_chunk fragments'
# outputs in the same OUT (their objects and SPIR-V: the K1 qk-norm/rope/kv-store/rmsnorm shaders, the K5 conv/decode
# rows shaders, the GDN layer object used as the chain test's reference; k2_gemm too when the layer object links it), so
# build with:
#   build.sh k1_primitives k2_gemm k2_mgemv k5_gdn k5_gdn_chunk k6_attention p3_kernels
# HIP timing (optional): T3_HIPROOT names a checkout whose build/hip holds gen/kernels/*.hip, gen/model/mtp.hip and
# libcere_hipk2.a (executor/hip/build.sh output), e.g. /src/ai-cere; unset or missing: the HIP reference binaries are not
# built (said so). They are the same test sources compiled with -DT3_HIP_TIMING (timing only, no contract rows).
for f in $OUT/k1_primitives_vk.o $OUT/k5_gdn_vk.o $OUT/k5_gdn_chunk_vk.o $OUT/k5_gdn_layer_vk.o $OUT/k2_mgemv_vk.o; do
  [ -f $f ] || { echo "p3_kernels: missing $f (build its fragment into the same OUT first)"; exit 1; }
done
$G $V/shaders/mtp_add_embedding.comp -o $OUT/spv/mtp_add_embedding.spv
$G $V/shaders/mtp_argmax_map.comp -o $OUT/spv/mtp_argmax_map.spv
$CXX -c $V/src/kernels/k6_prologue_vk.cpp -o $OUT/k6_prologue_vk.o
$CXX -c $V/src/kernels/k5_gdn_chain_vk.cpp -o $OUT/k5_gdn_chain_vk.o
$CXX -c $V/src/kernels/mtp_kernels_vk.cpp -o $OUT/mtp_kernels_vk.o
$CXX -I$V/ktest $V/ktest/k6_prologue_test.cpp $OUT/vk_launch.o $OUT/k6_prologue_vk.o $OUT/k1_primitives_vk.o -lvulkan -o $OUT/k6_prologue_test
$CXX -I$V/ktest $V/ktest/k5_chain_test.cpp $OUT/vk_launch.o $OUT/k5_gdn_chain_vk.o $OUT/k5_gdn_layer_vk.o $OUT/k5_gdn_chunk_vk.o \
     $OUT/k5_gdn_vk.o $OUT/k1_primitives_vk.o $OUT/k2_mgemv_vk.o $( [ -f $OUT/k2_gemm_vk.o ] && echo $OUT/k2_gemm_vk.o ) -lvulkan -o $OUT/k5_chain_test
$CXX -I$V/ktest $V/ktest/mtp_kernels_test.cpp $OUT/vk_launch.o $OUT/mtp_kernels_vk.o $OUT/k1_primitives_vk.o -lvulkan -o $OUT/mtp_kernels_test
T3_HB=${T3_HIPROOT:-}/build/hip
if command -v hipcc >/dev/null 2>&1 && [ -n "${T3_HIPROOT:-}" ] && [ -f $T3_HB/gen/kernels/k6_attention.hip ] && [ -f $T3_HB/libcere_hipk2.a ] \
   && [ -f $T3_HB/gen/model/mtp.hip ]; then
  T3_HIPCC="hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $T3_HIPROOT/executor/hip/src/cuda_compat.h -I$T3_HIPROOT/executor/include -I$T3_HB/gen -I$T3_HIPROOT/executor/hip/src -DT3_HIP_TIMING"
  $T3_HIPCC -x hip $V/ktest/k6_prologue_test.cpp $T3_HB/gen/kernels/k6_attention.hip $T3_HB/gen/kernels/k1_primitives.hip \
            $T3_HIPROOT/executor/hip/src/p4_attention_hip.cpp -o $OUT/hip_k6_prologue_timing
  $T3_HIPCC -x hip $V/ktest/k5_chain_test.cpp $T3_HB/gen/kernels/k5_gdn_layer.hip $T3_HB/gen/kernels/k5_gdn.hip \
            $T3_HB/gen/kernels/k5_gdn_decode.hip $T3_HB/gen/kernels/k1_primitives.hip $T3_HIPROOT/executor/hip/src/k5_gdn_chunk_hip.cpp \
            -x none $T3_HB/libcere_hipk2.a -lhipblaslt -o $OUT/hip_k5_chain_timing
  # The three MTP kernels are file-local in mtp.hip: their hipified text is extracted verbatim (one __global__ each).
  for k in norm_hidden_kernel add_embedding_kernel argmax_map_kernel; do
    sed -n "/^__global__ void $k(/,/^}/p" $T3_HB/gen/model/mtp.hip
  done > $OUT/hip_mtp_kernels.inc
  [ "$(grep -c '^__global__' $OUT/hip_mtp_kernels.inc)" = 3 ] || { echo "p3_kernels: MTP kernel extraction failed"; exit 1; }
  $T3_HIPCC -I$OUT -x hip $V/ktest/mtp_kernels_test.cpp -o $OUT/hip_mtp_kernels_timing
  sha256sum $T3_HB/gen/kernels/k6_attention.hip $T3_HB/gen/kernels/k5_gdn_layer.hip $T3_HB/gen/model/mtp.hip $OUT/hip_mtp_kernels.inc \
            $OUT/hip_k6_prologue_timing $OUT/hip_k5_chain_timing $OUT/hip_mtp_kernels_timing
else
  echo "p3_kernels: no hipcc or no HIP build tree at T3_HIPROOT: HIP reference timing not built"
fi
sha256sum $V/shaders/mtp_*.comp $OUT/spv/mtp_*.spv $V/src/kernels/k6_prologue_vk.* $V/src/kernels/k5_gdn_chain_vk.* \
          $V/src/kernels/mtp_kernels_vk.* $V/ktest/k6_prologue_test.cpp $V/ktest/k5_chain_test.cpp $V/ktest/mtp_kernels_test.cpp \
          $OUT/k6_prologue_test $OUT/k5_chain_test $OUT/mtp_kernels_test
