# Inkling S11 (inkling design note): the Inkling kernels' shaders (shaders/ki_*.comp), kernel object and ki_test (G1 on
# Vulkan) and ki_forward (the G3 / G5 forward; links k2_gemm.sh's k2_mgemv_vk.o / k2_gemm_vk.o, so build k2_gemm first:
# build.sh k2_gemm ki_inkling). Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and
# $OUT/vk_launch.o defined.
# CPU references: executor/ref/inkling/*_ref.h (header-only, shared with the CUDA ktests).
rm -f $OUT/.failki
S=$V/shaders
# -O0 (glslc: the last -O flag wins): spirv-opt -O drops NoContraction from code it copies, and the ki kernels are
# bit-exact only without contraction; tools/check_nocontract.py fails the build on any unmarked float op.
ki() { out=$1; src=$2; shift 2; $G -O0 -DKI_SOFT_FMA=${KI_SOFT_FMA:-0} "$@" $S/$src -o $OUT/spv/$out.spv || touch $OUT/.failki; }
{
ki ki_selftest_e ki_selftest.comp -DMODE=0 &
ki ki_selftest_r ki_selftest.comp -DMODE=1 &
ki ki_selftest_t ki_selftest.comp -DMODE=2 &
ki ki_selftest_f ki_selftest.comp -DMODE=3 &
ki ki_quant_mx ki_quant.comp -DMODE=0 &
ki ki_quant_nv ki_quant.comp -DMODE=1 &
for st in 0 1; do ki ki_router_s$st ki_router.comp -DSTAGE=$st & done
for m in 0 1 2 3 4; do ki ki_attn_m$m ki_attn.comp -DMODE=$m & done
# D-I-EPB (a2): the tiled attention spans' fail-first mutant (ki_test only).
ki ki_attn_m4m ki_attn.comp -DMODE=4 -DMUT=1 &
ki ki_dense ki_dense.comp &
for f in 0 1; do for st in 0 1 3 4; do ki ki_moe_f${f}_s$st ki_moe.comp -DFMT=$f -DSTAGE=$st & done; done
# D-I-EPB (a1): the NVFP4 grouped stages on int8 cooperative matrices, their fail-first mutant, the layout probe.
for st in 3 4; do
  ki ki_moe_f0_s${st}c ki_moe.comp -DFMT=0 -DSTAGE=$st -DCM=1 &
  ki ki_moe_f0_s${st}m ki_moe.comp -DFMT=0 -DSTAGE=$st -DCM=1 -DMUT=1 &
done
ki ki_moe_probe ki_moe.comp -DFMT=0 -DSTAGE=6 &
ki ki_moe_comb ki_moe.comp -DFMT=0 -DSTAGE=2 &
ki ki_moe_bucket ki_moe.comp -DFMT=0 -DSTAGE=5 &
for m in 0 1 2 3 4 5 6 7 8; do ki ki_mix_m$m ki_mix.comp -DMODE=$m & done
ki ki_mix_m7m ki_mix.comp -DMODE=7 -DMUT=1 &  # D-I-EPB (a2): the tiled message gate's fail-first mutant (ki_test only)
for m in 0 1; do
  for l in 256 32 16; do ki ki_rownorm_m${m}_l$l ki_rownorm.comp -DMODE=$m -DLN=${l}u & done
  ki ki_conv_m$m ki_conv.comp -DMODE=$m &
done
# S12 vision tower (kiv_*: statistical gate, so optimized and outside the contraction check).
for m in 0 1 2 3 4 5; do
  $G -DKI_SOFT_FMA=${KI_SOFT_FMA:-0} -DMODE=$m $S/kiv_vision.comp -o $OUT/spv/kiv_vision_m$m.spv || touch $OUT/.failki &
done
wait
}
[ ! -f $OUT/.failki ] || { echo "ki shader compile failed"; exit 1; }
# The cooperative-matrix shaders (Vulkan memory model): shared-memory visibility at the barriers, not every access.
python3 $V/tools/spv_shared_barriers.py $OUT/spv/ki_moe_f0_s3c.spv $OUT/spv/ki_moe_f0_s3m.spv \
  $OUT/spv/ki_moe_f0_s4c.spv $OUT/spv/ki_moe_f0_s4m.spv $OUT/spv/ki_moe_probe.spv || { echo "ki spv pass failed"; exit 1; }
python3 $V/tools/check_nocontract.py $OUT/spv/ki_*.spv || { echo "ki shader contraction check failed"; exit 1; }
$CXX -c $V/src/kernels/ki_ops_vk.cpp -o $OUT/ki_ops_vk.o
$CXX -ffp-contract=off -I$V/ktest -Iexecutor $V/ktest/ki_test.cpp $OUT/vk_launch.o $OUT/ki_ops_vk.o -lvulkan -o $OUT/ki_test
$CXX -ffp-contract=off -Iexecutor -c $V/src/device/inkling/ki_model.cpp -o $OUT/ki_model.o
$CXX -ffp-contract=off -I$V -Iexecutor $V/ktest/ki_forward.cpp $OUT/vk_launch.o $OUT/ki_ops_vk.o $OUT/ki_model.o \
  $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/ki_forward
$CXX -ffp-contract=off -Iexecutor -c $V/src/device/inkling/ki_vision.cpp -o $OUT/ki_vision.o
$CXX -I$V -Iexecutor $V/ktest/ki_vision.cpp $OUT/vk_launch.o $OUT/ki_ops_vk.o $OUT/ki_vision.o \
  $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/ki_vision
sha256sum $S/ki_*.comp $S/kiv_*.comp $OUT/spv/ki_*.spv $OUT/spv/kiv_*.spv $OUT/ki_test $OUT/ki_forward $OUT/ki_vision
