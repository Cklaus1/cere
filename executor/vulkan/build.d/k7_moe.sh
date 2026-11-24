# K7 NVFP4 MoE block, W4A4 (P2): shaders, kernel object, k7_test and, when the HIP reference tree is mounted, the
# same-method HIP timing harness. Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and
# $OUT/vk_launch.o defined. K7 runs K2 (vkk2::run: the small-M GEMV for row classes <= 16, the large-m GEMM above) for
# its BF16 GEMMs, so the k2_mgemv SPIR-V variants of the default mappings (k2_mgemv_vk.cpp mapping_for), the k2_gemm
# SPIR-V, k2_mgemv_vk.o and k2_gemm_vk.o are built here too (same names and flags as build.d/k2_mgemv.sh, k2_gemm.sh).
S7=$V/shaders
rm -f $OUT/.fail7
n=0
bg() { "$@" || touch $OUT/.fail7 & n=$((n + 1)); [ $((n % 8)) -ne 0 ] || wait; }
for glu in 0 1; do
  bg $G -DGLU=$glu $S7/k7_gemm.comp -o $OUT/spv/k7_gemm_glu$glu.spv
  for mr in 1 2 4 8; do
    for g in 4 8 16; do
      bg $G -DGLU=$glu -DMR=$mr -DG=$g $S7/k7_gemv.comp -o $OUT/spv/k7_gemv_glu${glu}_mr${mr}_g$g.spv
    done
  done
done
for epl in 16 32; do
  for mode in 0 1; do bg $G -DMODE=$mode -DEPL=$epl $S7/k7_route.comp -o $OUT/spv/k7_route_m${mode}_epl$epl.spv; done
done
bg $G -DMODE=2 $S7/k7_route.comp -o $OUT/spv/k7_route_m2.spv
bg $G -DMODE=3 $S7/k7_route.comp -o $OUT/spv/k7_route_m3.spv
bg $G -DMODE=0 $S7/k7_elem.comp -o $OUT/spv/k7_elem_gate.spv
bg $G -DMODE=1 $S7/k7_elem.comp -o $OUT/spv/k7_elem_silu.spv
bg $G -DMODE=2 $S7/k7_elem.comp -o $OUT/spv/k7_elem_combine.spv
bg $G $S7/k7_quant.comp -o $OUT/spv/k7_quant.spv
bg $G $S7/k7_probe.comp -o $OUT/spv/k7_probe.spv
# k2_mgemv default mappings "G,E,WG,U,J:row capacities" (k <= 1024: m 1 / m > 1; k > 1024: m 1 / m > 1)
# (a function's own positional parameters: the caller's "$@", build.sh's fragment filter, is left alone)
k7_k2map() {
  for mr in $6; do
    bg $G -DMX=0 -DMR=$mr -DG=$1 -DE=$2 -DWG=$3 -DU=$4 -DJ=$5 $V/shaders/k2_mgemv.comp -o $OUT/spv/k2_mgemv_bf16_g$1_e$2_wg$3_u$4_j$5_mr$mr.spv
  done
}
k7_k2map 16 8 256 1 1 "1"
k7_k2map 16 8 128 1 2 "2 4 8 16"
k7_k2map 32 16 256 2 1 "1"
k7_k2map 32 16 64 1 4 "2 4 8 16"
# K2 large-m GEMM (row classes >= 17 of the router / shared expert): the same SPIR-V names and flags as build.d/k2_gemm.sh
for mx in 0 1; do
  fm=bf16; [ $mx = 1 ] && fm=mx
  bg $G -DMX=$mx $V/shaders/k2_gemm.comp -o $OUT/spv/k2_gemm_$fm.spv
  bg $G -DMX=$mx $V/shaders/k2_gemm_f32.comp -o $OUT/spv/k2_gemm_f32_$fm.spv
  bg $G -DMX=$mx -DSUBN=${K2G_SUBN:-1} $V/shaders/k2_gemm_scan.comp -o $OUT/spv/k2_gemm_scan_$fm.spv
done
bg $G $V/shaders/k2_gemm_probe.comp -o $OUT/spv/k2_gemm_probe.spv
wait
[ ! -f $OUT/.fail7 ] || { echo "k7 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k2_mgemv_vk.cpp -o $OUT/k2_mgemv_vk.o
$CXX -c $V/src/kernels/k2_gemm_vk.cpp -o $OUT/k2_gemm_vk.o
$CXX -c $V/src/kernels/k7_moe_vk.cpp -o $OUT/k7_moe_vk.o
# k7_test links its own object with the VulkanAutotune knobs (-DCERE_VK_TUNE, MoeOptions); the library keeps k7_moe_vk.o
$CXX -DCERE_VK_TUNE -c $V/src/kernels/k7_moe_vk.cpp -o $OUT/k7_moe_vk_tune.o
$CXX -DCERE_VK_TUNE $V/ktest/k7_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o $OUT/k7_moe_vk_tune.o -lvulkan -o $OUT/k7_test
# Same-session HIP reference timing: needs the generated HIP K7 sources of a built HIP tree (executor/hip/build.sh),
# mounted read-only at $HIP_REF (default /src/ai-cere); skipped (with a line saying so) otherwise.
HIP_REF=${HIP_REF:-/src/ai-cere}
if command -v hipcc >/dev/null 2>&1 && [ -f $HIP_REF/build/hip/gen/kernels/k7_moe.hip ] && [ -f $HIP_REF/build/hip/libcere_hipk2.a ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $HIP_REF/executor/hip/src/cuda_compat.h -I$HIP_REF/executor/include \
        -I$HIP_REF/build/hip/gen -I$HIP_REF/executor/hip/src -x hip $V/ktest/hip_k7_timing.cpp \
        $HIP_REF/build/hip/gen/kernels/k7_moe.hip $HIP_REF/build/hip/gen/kernels/k0_nvfp4.hip \
        $HIP_REF/build/hip/gen/kernels/k1_primitives.hip -x none $HIP_REF/build/hip/libcere_hipk2.a -lhipblaslt \
        -o $OUT/hip_k7_timing
else
  echo "hip_k7_timing skipped (no hipcc or no HIP tree at $HIP_REF)"
fi
sha256sum $S7/k7_*.comp $V/src/kernels/k7_moe_vk.cpp $OUT/spv/k7_gemv_glu*_mr1_g*.spv $OUT/spv/k7_gemm_glu*.spv \
          $OUT/spv/k7_route_m*.spv $OUT/k7_test $( [ -f $OUT/hip_k7_timing ] && echo $OUT/hip_k7_timing )
