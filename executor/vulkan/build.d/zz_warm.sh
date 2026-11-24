# VulkanAmd P3 (pipeline warm-up): vk_warm_test, the completeness test of every kernel family's pipeline_specs list (design 4.2).
# Sorted last: it needs every family's object (and SPIR-V) in the same OUT, i.e. a no-argument build.sh (or `build.sh <all fragments> zz_warm`).
# Sourced by executor/vulkan/build.sh with V, OUT, G, CXX and $OUT/vk_launch.o defined.
WARM_OBJS="k1_primitives k2_mgemv k2_gemm k3_hc k4_ple k5_gdn k5_gdn_chunk k5_gdn_layer k5_gdn_chain k6_attention k6_prologue k7_moe k9_qsa k12_sampler k13s_spec_accept k13_moe_bf16 mtp_kernels"
WARM_LINK=""
WARM_MISSING=""
for o in $WARM_OBJS; do
  if [ -f $OUT/${o}_vk.o ]; then WARM_LINK="$WARM_LINK $OUT/${o}_vk.o"; else WARM_MISSING="$WARM_MISSING ${o}_vk.o"; fi
done
if [ -n "$WARM_MISSING" ]; then
  echo "zz_warm: missing$WARM_MISSING in $OUT: vk_warm_test not built (build every fragment into the same OUT)"
else
  $CXX $V/ktest/vk_warm_test.cpp $OUT/vk_launch.o $WARM_LINK -lvulkan -o $OUT/vk_warm_test
  sha256sum $V/ktest/vk_warm_test.cpp $OUT/vk_warm_test
fi
