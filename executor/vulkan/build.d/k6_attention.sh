# K6 dense causal GQA (P2): split-key kernel (KS 64 / 128 keys per split) + combine, FP32 prefill tile kernel, gate;
# P4-A: the cooperative-matrix prefill kernel (p4att_dense), the FP32 kernel's FALLBACK variant and the device probe;
# kernel object and k6_test; when hipcc and a built HIP tree are mounted at $HIP_REF (default /src/ai-cere), the
# same-method HIP timing harness hip_k6_timing. Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and
# $OUT/vk_launch.o defined.
rm -f $OUT/.fail6
for ks in 64 128; do
  for fp8 in 0 1; do
    fmt=bf16; [ $fp8 = 1 ] && fmt=fp8
    $G -DFP8=$fp8 -DKS=$ks $V/shaders/k6_decode.comp -o $OUT/spv/k6_decode_${fmt}_ks$ks.spv || touch $OUT/.fail6 &
  done
  $G -DKS=$ks $V/shaders/k6_combine.comp -o $OUT/spv/k6_combine_ks$ks.spv || touch $OUT/.fail6 &
done
$G -DFP8=0 $V/shaders/k6_prefill.comp -o $OUT/spv/k6_prefill_bf16.spv || touch $OUT/.fail6 &
$G -DFP8=1 $V/shaders/k6_prefill.comp -o $OUT/spv/k6_prefill_fp8.spv || touch $OUT/.fail6 &
$G $V/shaders/k6_gate.comp -o $OUT/spv/k6_gate.spv || touch $OUT/.fail6 &
for fp8 in 0 1; do
  fmt=bf16; [ $fp8 = 1 ] && fmt=fp8
  $G -DFP8=$fp8 $V/shaders/p4att_dense.comp -o $OUT/spv/p4att_dense_$fmt.spv || touch $OUT/.fail6 &
  $G -DFP8=$fp8 -DFALLBACK=1 $V/shaders/k6_prefill.comp -o $OUT/spv/k6_prefill_fb_$fmt.spv || touch $OUT/.fail6 &
done
$G $V/shaders/p4att_probe.comp -o $OUT/spv/p4att_probe.spv || touch $OUT/.fail6 &
$G $V/shaders/p4att_probe2.comp -o $OUT/spv/p4att_probe2.spv || touch $OUT/.fail6 &
wait
[ ! -f $OUT/.fail6 ] || { echo "k6 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k6_attention_vk.cpp -o $OUT/k6_attention_vk.o
$CXX -I$V/ktest $V/ktest/k6_test.cpp $OUT/vk_launch.o $OUT/k6_attention_vk.o -lvulkan -o $OUT/k6_test
HIP_REF=${HIP_REF:-/src/ai-cere}
if command -v hipcc >/dev/null 2>&1 && [ -f $HIP_REF/build/hip/gen/kernels/k6_attention.hip ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $HIP_REF/executor/hip/src/cuda_compat.h -I$HIP_REF/executor/include \
        -I$HIP_REF/build/hip/gen -I$HIP_REF/executor/hip/src -x hip $V/ktest/hip_k6_timing.cpp \
        $HIP_REF/build/hip/gen/kernels/k6_attention.hip $HIP_REF/build/hip/gen/kernels/k1_primitives.hip \
        $HIP_REF/executor/hip/src/p4_attention_hip.cpp -o $OUT/hip_k6_timing
else
  echo "hip_k6_timing skipped (no hipcc or no HIP tree at $HIP_REF)"
fi
sha256sum $V/shaders/k6_*.comp $V/shaders/p4att_*.comp $V/src/kernels/k6_attention_vk.h $V/src/kernels/k6_attention_vk.cpp \
          $V/ktest/k6_test.cpp $OUT/spv/k6_*.spv $OUT/spv/p4att_*.spv $OUT/k6_test \
          $( [ -f $OUT/hip_k6_timing ] && echo $OUT/hip_k6_timing $V/ktest/hip_k6_timing.cpp )
