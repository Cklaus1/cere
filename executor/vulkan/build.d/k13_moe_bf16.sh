# K13 BF16 MoE of the MTP draft (P2): shaders, kernel object, k13_test and, when the HIP reference tree is mounted, the
# same-method HIP timing harness. Sourced by executor/vulkan/build.sh with V, OUT, G, CXX and $OUT/vk_launch.o defined.
# K13 reuses K7's router / grouping / element-wise shaders and K2 (vkk2::run: the small-M GEMV, the large-m GEMM); this
# fragment sorts before build.d/k2_gemm.sh, k2_mgemv.sh and k7_moe.sh, so it compiles the SPIR-V and objects it needs
# itself (same names and flags as those fragments: building them again is harmless).
S13=$V/shaders
rm -f $OUT/.fail13
n13=0
bg13() { "$@" || touch $OUT/.fail13 & n13=$((n13 + 1)); [ $((n13 % 8)) -ne 0 ] || wait; }
for glu in 0 1; do
  for fp4 in 0 1; do
    for mr in 1 8; do
      for le in 8 16; do
        bg13 $G -DGLU=$glu -DFP4=$fp4 -DMR=$mr -DLE=$le $S13/k13_gemv.comp -o $OUT/spv/k13_gemv_glu${glu}_fp4${fp4}_mr${mr}_le$le.spv
      done
    done
  done
done
for epl in 16 32; do
  for mode in 0 1; do bg13 $G -DMODE=$mode -DEPL=$epl $S13/k7_route.comp -o $OUT/spv/k7_route_m${mode}_epl$epl.spv; done
done
bg13 $G -DMODE=2 $S13/k7_route.comp -o $OUT/spv/k7_route_m2.spv
bg13 $G -DMODE=3 $S13/k7_route.comp -o $OUT/spv/k7_route_m3.spv
bg13 $G -DMODE=0 $S13/k7_elem.comp -o $OUT/spv/k7_elem_gate.spv
bg13 $G -DMODE=1 $S13/k7_elem.comp -o $OUT/spv/k7_elem_silu.spv
bg13 $G -DMODE=2 $S13/k7_elem.comp -o $OUT/spv/k7_elem_combine.spv
# k2_mgemv default mappings (k2_mgemv_vk.cpp mapping_for): "G E WG U J" and the row capacities
k13_k2map() {
  for mr in $6; do
    bg13 $G -DMX=0 -DMR=$mr -DG=$1 -DE=$2 -DWG=$3 -DU=$4 -DJ=$5 $S13/k2_mgemv.comp -o $OUT/spv/k2_mgemv_bf16_g$1_e$2_wg$3_u$4_j$5_mr$mr.spv
  done
}
k13_k2map 16 8 256 1 1 "1"
k13_k2map 16 8 128 1 2 "2 4 8 16"
k13_k2map 32 16 256 2 1 "1"
k13_k2map 32 16 64 1 4 "2 4 8 16"
# K2 large-m GEMM (row classes >= 17 of the router / shared expert): the same SPIR-V names and flags as build.d/k2_gemm.sh
for mx in 0 1; do
  fm=bf16; [ $mx = 1 ] && fm=mx
  bg13 $G -DMX=$mx $S13/k2_gemm.comp -o $OUT/spv/k2_gemm_$fm.spv
  bg13 $G -DMX=$mx $S13/k2_gemm_f32.comp -o $OUT/spv/k2_gemm_f32_$fm.spv
  bg13 $G -DMX=$mx -DSUBN=${K2G_SUBN:-1} $S13/k2_gemm_scan.comp -o $OUT/spv/k2_gemm_scan_$fm.spv
done
bg13 $G $S13/k2_gemm_probe.comp -o $OUT/spv/k2_gemm_probe.spv
wait
[ ! -f $OUT/.fail13 ] || { echo "k13 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k2_mgemv_vk.cpp -o $OUT/k2_mgemv_vk.o
$CXX -c $V/src/kernels/k2_gemm_vk.cpp -o $OUT/k2_gemm_vk.o
$CXX -c $V/src/kernels/k13_moe_bf16_vk.cpp -o $OUT/k13_moe_bf16_vk.o
$CXX $V/ktest/k13_test.cpp $OUT/vk_launch.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o $OUT/k13_moe_bf16_vk.o -lvulkan -o $OUT/k13_test
HIP_REF=${HIP_REF:-/src/ai-cere}
if command -v hipcc >/dev/null 2>&1 && [ -f $HIP_REF/build/hip/gen/kernels/k13_moe_bf16.hip ] && [ -f $HIP_REF/build/hip/libcere_hipk2.a ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $HIP_REF/executor/hip/src/cuda_compat.h -I$HIP_REF/executor/include \
        -I$HIP_REF/build/hip/gen -I$HIP_REF/executor/hip/src -x hip $V/ktest/hip_k13_timing.cpp \
        $HIP_REF/build/hip/gen/kernels/k13_moe_bf16.hip $HIP_REF/build/hip/gen/kernels/k7_moe.hip \
        $HIP_REF/build/hip/gen/kernels/k0_nvfp4.hip $HIP_REF/build/hip/gen/kernels/k1_primitives.hip \
        -x none $HIP_REF/build/hip/libcere_hipk2.a -lhipblaslt -o $OUT/hip_k13_timing
else
  echo "hip_k13_timing skipped (no hipcc or no HIP tree at $HIP_REF)"
fi
sha256sum $S13/k13_gemv.comp $V/src/kernels/k13_moe_bf16_vk.cpp $V/src/kernels/k13_moe_bf16_vk.h $OUT/spv/k13_gemv_*.spv \
          $OUT/k13_test $( [ -f $OUT/hip_k13_timing ] && echo $OUT/hip_k13_timing )
