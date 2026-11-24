# K2 small-M GEMV (P1): shaders, kernel object, k2v_test / k2mx_test, the same-method HIP reference timing.
# Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined.
# Mappings "G,E,WG,U,J": the defaults of k2_mgemv_vk.cpp mapping_for_k plus the candidates k2mx_test --tune times.
MAPS="16,16,256,1,1 32,16,256,1,1 32,16,256,1,2 32,16,256,1,4 32,16,256,2,1 32,16,256,2,2 32,16,256,4,1 32,16,128,1,1 32,16,128,1,2 32,16,128,1,4 32,16,128,2,1 32,16,128,2,2 32,16,64,1,2 32,16,64,1,4 32,16,64,2,2 8,8,256,1,1 8,8,128,1,2 8,8,256,1,2 8,16,256,1,1 8,16,128,1,2 8,16,256,1,2 16,8,256,1,1 16,8,128,1,2 16,8,256,1,2"
# VulkanAutotune stage 2 experiments only: K2V_EXTRA_MAPS adds mappings (never set by a library build that ships).
MAPS="$MAPS ${K2V_EXTRA_MAPS:-}"
n=0
for mp in $MAPS; do
  set -- $(echo $mp | tr , ' ')
  for fmt in bf16 mx; do
    mx=0; [ $fmt = mx ] && mx=1
    for mr in 1 2 4 8 16; do
      $G -DMX=$mx -DMR=$mr -DG=$1 -DE=$2 -DWG=$3 -DU=$4 -DJ=$5 $V/shaders/k2_mgemv.comp -o $OUT/spv/k2_mgemv_${fmt}_g$1_e$2_wg$3_u$4_j$5_mr$mr.spv || touch $OUT/.fail &
      n=$((n + 1)); [ $((n % 8)) -ne 0 ] || wait
    done
  done
done
wait
[ ! -f $OUT/.fail ] || { echo "shader compile failed"; exit 1; }
$G $V/shaders/k2_mxfp8_dequant.comp -o $OUT/spv/k2_mxfp8_dequant.spv
$CXX -c $V/src/kernels/k2_mgemv_vk.cpp -o $OUT/k2_mgemv_vk.o
$CXX $V/ktest/k2v_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o -lvulkan -o $OUT/k2v_test
$CXX $V/ktest/k2mx_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o -lvulkan -o $OUT/k2mx_test
if command -v hipcc >/dev/null 2>&1; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -Iexecutor/include -Iexecutor/hip/src $V/ktest/hip_gemv_timing.cpp \
        executor/hip/src/k2_mgemv.cpp -o $OUT/hip_gemv_timing
fi
sha256sum $V/shaders/*.comp $OUT/spv/k2_mgemv_*_g32_e16_wg256_u4_j1_mr1.spv $OUT/spv/k2_mgemv_*_g32_e16_wg64_u1_j4_mr*.spv \
          $OUT/spv/k2_mgemv_*_g16_e8_wg256_u1_j1_mr1.spv $OUT/spv/k2_mgemv_*_g16_e8_wg128_u1_j2_mr*.spv \
          $OUT/spv/k2_mxfp8_dequant.spv $OUT/k2v_test $OUT/k2mx_test $( [ -f $OUT/hip_gemv_timing ] && echo $OUT/hip_gemv_timing )
