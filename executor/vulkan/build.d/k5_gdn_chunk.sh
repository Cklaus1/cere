# K5 GDN chunked prefill and the GDN mixer layer (P2): shaders, kernel objects, k5_chunk_test / k5_layer_test and (when
# hipcc and a HIP build tree exist) the same-session HIP reference timing.
# Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined. Needs the k5_gdn,
# k1_primitives, k2_gemm and k2_mgemv fragments' outputs in the same OUT (their objects and SPIR-V: l2norm, conv, gating,
# decode, the gated norm, the small-M GEMV and the large-m GEMM), so build with:
# build.sh k1_primitives k2_gemm k2_mgemv k5_gdn k5_gdn_chunk.
# HIP timing: K5C_HIPROOT names a checkout whose build/hip holds gen/kernels/*.hip and libcere_hipk2.a (executor/hip/build.sh
# output), e.g. /src/ai-cere; unset or missing: not built.
for st in 0:permute 1:cumsum 2:prep 3:recur 4:out 5:unpermute; do
  $G -DSTAGE=${st%%:*} $V/shaders/k5_gdn_chunk.comp -o $OUT/spv/k5_gdn_chunk_${st#*:}.spv
done
$CXX -c $V/src/kernels/k5_gdn_chunk_vk.cpp -o $OUT/k5_gdn_chunk_vk.o
$CXX $V/ktest/k5_chunk_test.cpp $OUT/vk_launch.o $OUT/k5_gdn_chunk_vk.o $OUT/k5_gdn_vk.o -lvulkan -o $OUT/k5_chunk_test
if [ -f $V/src/kernels/k5_gdn_layer_vk.cpp ]; then
  $CXX -c $V/src/kernels/k5_gdn_layer_vk.cpp -o $OUT/k5_gdn_layer_vk.o
  $CXX $V/ktest/k5_layer_test.cpp $OUT/vk_launch.o $OUT/k5_gdn_layer_vk.o $OUT/k5_gdn_chunk_vk.o $OUT/k5_gdn_vk.o \
       $OUT/k1_primitives_vk.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/k5_layer_test
fi
K5C_HB=${K5C_HIPROOT:-}/build/hip
if command -v hipcc >/dev/null 2>&1 && [ -n "${K5C_HIPROOT:-}" ] && [ -f $K5C_HB/gen/kernels/k5_gdn_layer.hip ] && [ -f $K5C_HB/libcere_hipk2.a ] \
   && [ -f $V/ktest/hip_k5_chunk_timing.cpp ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $K5C_HIPROOT/executor/hip/src/cuda_compat.h -I$K5C_HIPROOT/executor/include \
        -I$K5C_HB/gen -I$K5C_HIPROOT/executor/hip/src -x hip $V/ktest/hip_k5_chunk_timing.cpp $K5C_HB/gen/kernels/k5_gdn_layer.hip \
        $K5C_HB/gen/kernels/k5_gdn.hip $K5C_HB/gen/kernels/k5_gdn_decode.hip $K5C_HB/gen/kernels/k1_primitives.hip \
        $K5C_HIPROOT/executor/hip/src/k5_gdn_chunk_hip.cpp -x none $K5C_HB/libcere_hipk2.a -lhipblaslt -o $OUT/hip_k5_chunk_timing
else
  echo "k5_gdn_chunk: no hipcc or no HIP build tree at K5C_HIPROOT: HIP reference timing not built"
fi
sha256sum $V/shaders/k5_gdn_chunk*.comp $OUT/spv/k5_gdn_chunk_*.spv $OUT/k5_chunk_test \
          $( [ -f $OUT/k5_layer_test ] && echo $OUT/k5_layer_test ) $( [ -f $OUT/hip_k5_chunk_timing ] && echo $OUT/hip_k5_chunk_timing )
