# K2 large-m GEMM (P2): shaders, kernel object, k2w_test / k2_test, the same-method HIP reference timing.
# Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined. k2_gemm_vk's
# run() routes row classes <= 16 to K2's small-M GEMV, so the k2_mgemv SPIR-V variants of its default mappings
# (k2_mgemv_vk.cpp mapping_for) and k2_mgemv_vk.o are built here too (same names as build.d/k2_mgemv.sh).
S2=$V/shaders
rm -f $OUT/.fail2g
n=0
k2g_bg() { "$@" || touch $OUT/.fail2g & n=$((n + 1)); [ $((n % 8)) -ne 0 ] || wait; }
for mx in 0 1; do
  fm=bf16; [ $mx = 1 ] && fm=mx
  k2g_bg $G -DMX=$mx $S2/k2_gemm.comp -o $OUT/spv/k2_gemm_$fm.spv
  k2g_bg $G -DMX=$mx $S2/k2_gemm_f32.comp -o $OUT/spv/k2_gemm_f32_$fm.spv
  k2g_bg $G -DMX=$mx -DSUBN=${K2G_SUBN:-1} $S2/k2_gemm_scan.comp -o $OUT/spv/k2_gemm_scan_$fm.spv
done
k2g_bg $G $S2/k2_gemm_probe.comp -o $OUT/spv/k2_gemm_probe.spv
# K2G_CAPTURE=1: TEST BUILD ONLY (P4-B activation capture, k2_gemm_vk.cpp CERE_VK_K2_CAPTURE); never the default build.
K2G_DEF=""
if [ "${K2G_CAPTURE:-0}" = 1 ]; then
  k2g_bg $G $S2/k2_gemm_capture.comp -o $OUT/spv/k2_gemm_capture.spv
  K2G_DEF="-DCERE_VK_K2_CAPTURE"
fi
# k2_mgemv default mappings "G E WG U J" and their row capacities (k <= 1024: m 1 / m > 1; k > 1024: m 1 / m > 1)
k2g_map() {
  for fmt in bf16 mx; do
    mx=0; [ $fmt = mx ] && mx=1
    for mr in $6; do
      k2g_bg $G -DMX=$mx -DMR=$mr -DG=$1 -DE=$2 -DWG=$3 -DU=$4 -DJ=$5 $S2/k2_mgemv.comp -o $OUT/spv/k2_mgemv_${fmt}_g$1_e$2_wg$3_u$4_j$5_mr$mr.spv
    done
  done
}
k2g_map 16 8 256 1 1 "1"
k2g_map 16 8 128 1 2 "2 4 8 16"
k2g_map 32 16 256 2 1 "1"
k2g_map 32 16 64 1 4 "2 4 8 16"
wait
[ ! -f $OUT/.fail2g ] || { echo "k2_gemm shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k2_mgemv_vk.cpp -o $OUT/k2_mgemv_vk.o
$CXX $K2G_DEF -c $V/src/kernels/k2_gemm_vk.cpp -o $OUT/k2_gemm_vk.o
$CXX $V/ktest/k2w_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/k2w_test
$CXX $V/ktest/k2_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/k2_test
$CXX $V/ktest/k2_gemm_sites_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/k2_gemm_sites_test
if command -v hipcc >/dev/null 2>&1; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -Iexecutor/include -Iexecutor/hip/src $V/ktest/hip_k2_gemm_timing.cpp \
        executor/hip/src/k2_wmma.cpp executor/hip/src/k2_mgemv.cpp -o $OUT/hip_k2_gemm_timing
else
  echo "hip_k2_gemm_timing skipped (no hipcc)"
fi
sha256sum $S2/k2_gemm*.comp $V/src/kernels/k2_gemm_vk.cpp $OUT/spv/k2_gemm_*.spv $OUT/k2_gemm_vk.o $OUT/k2w_test $OUT/k2_test $OUT/k2_gemm_sites_test \
          $( [ -f $OUT/hip_k2_gemm_timing ] && echo $OUT/hip_k2_gemm_timing )
