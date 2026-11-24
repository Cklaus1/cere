# VulkanAutotune H7 (vulkan-autotune design note): the cooperative-matrix peak microbenchmark, ktest only. Its SPIR-V
# goes to $OUT/spv-tune, NOT $OUT/spv: build.sh embeds every module of $OUT/spv into the library, whose bytes must not
# change. Run: $OUT/cm_peak_test $OUT/spv-tune. Sourced by executor/vulkan/build.sh.
# Without glslc -O: shaderc 2023.8's optimizer segfaults on this shader (the coopmat accumulator array); RADV's
# compiler optimizes the module either way.
mkdir -p $OUT/spv-tune
GT="glslc --target-env=vulkan1.3 -fshader-stage=compute"
$GT -DINT8=0 $V/shaders/cm_peak.comp -o $OUT/spv-tune/cm_peak_fp16.spv
$GT -DINT8=1 $V/shaders/cm_peak.comp -o $OUT/spv-tune/cm_peak_int8.spv
$CXX $V/ktest/cm_peak_test.cpp $OUT/vk_launch.o -lvulkan -o $OUT/cm_peak_test
sha256sum $V/shaders/cm_peak.comp $OUT/spv-tune/cm_peak_*.spv $OUT/cm_peak_test
