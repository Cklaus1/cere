# K1 primitives (P2): shaders, kernel object, k1_test. Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command),
# CXX and $OUT/vk_launch.o defined. Own outputs only: $OUT/spv/k1_*.spv, $OUT/k1_primitives_vk.o, $OUT/k1_test, and (when
# hipcc exists and K1_HIP_GEN names a generated HIP tree, e.g. build/hip/gen) $OUT/k1_hip_timing, the same-method HIP reference.
$G $V/shaders/k1_embedding.comp -o $OUT/spv/k1_embedding.spv
$G -DGATED=0 $V/shaders/k1_rmsnorm.comp -o $OUT/spv/k1_rmsnorm_g0.spv
$G -DGATED=1 $V/shaders/k1_rmsnorm.comp -o $OUT/spv/k1_rmsnorm_g1.spv
$G $V/shaders/k1_sigmoid_mul.comp -o $OUT/spv/k1_sigmoid_mul.spv
$G -DTABLE=1 $V/shaders/k1_rope.comp -o $OUT/spv/k1_rope_table.spv
$G -DTABLE=0 $V/shaders/k1_rope.comp -o $OUT/spv/k1_rope.spv
$G $V/shaders/k1_top2.comp -o $OUT/spv/k1_top2.spv
$G $V/shaders/k1_kv_store.comp -o $OUT/spv/k1_kv_store.spv
$G $V/shaders/k1_kv_store_rows.comp -o $OUT/spv/k1_kv_store_rows.spv
$G $V/shaders/k1_qk_norm_rope_gate.comp -o $OUT/spv/k1_qk_norm_rope_gate.spv
$G $V/shaders/k1_copy_segments.comp -o $OUT/spv/k1_copy_segments.spv
$CXX -c $V/src/kernels/k1_primitives_vk.cpp -o $OUT/k1_primitives_vk.o
$CXX $V/ktest/k1_test.cpp $OUT/vk_launch.o $OUT/k1_primitives_vk.o -lvulkan -o $OUT/k1_test
if command -v hipcc >/dev/null 2>&1 && [ -n "${K1_HIP_GEN:-}" ] && [ -f "$K1_HIP_GEN/kernels/k1_primitives.hip" ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include executor/hip/src/cuda_compat.h -Iexecutor/include -I$K1_HIP_GEN -I$K1_HIP_GEN/../generated -DK1_HIP_TIMING \
        -x hip $V/ktest/k1_test.cpp $K1_HIP_GEN/kernels/k1_primitives.hip -o $OUT/k1_hip_timing
fi
sha256sum $V/shaders/k1_*.comp $OUT/spv/k1_*.spv $OUT/k1_test $( [ -f $OUT/k1_hip_timing ] && echo $OUT/k1_hip_timing )
