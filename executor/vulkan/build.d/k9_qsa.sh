# K9 QSA sparse attention (P2; P4-A: the cooperative-matrix prefill path): prep (index q, compressed keys, ring, step, key save), block logits, top-512 selection,
# sparse GQA (split tiles + combine for <= 16 rows, per-row online softmax above), kernel object, k9_prep_test,
# k9_test and, when hipcc and a built HIP tree are mounted at $HIP_REF (default /src/ai-cere), the same-method HIP
# timing harness hip_k9_timing. Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and
# $OUT/vk_launch.o defined.
S9=$V/shaders
rm -f $OUT/.fail9
n9=0
bg9() { "$@" || touch $OUT/.fail9 & n9=$((n9 + 1)); [ $((n9 % 8)) -ne 0 ] || wait; }
for m in 0 1 2 3 4 5; do bg9 $G -DMODE=$m -DAT=0 $S9/k9_prep.comp -o $OUT/spv/k9_prep_m$m.spv; done
for m in 1 2; do bg9 $G -DMODE=$m -DAT=1 $S9/k9_prep.comp -o $OUT/spv/k9_prep_m${m}_at.spv; done
for r in 1 8; do bg9 $G -DR=$r $S9/k9_logits.comp -o $OUT/spv/k9_logits_r$r.spv; done
bg9 $G $S9/k9_topk.comp -o $OUT/spv/k9_topk.spv
bg9 $G $S9/k9_combine.comp -o $OUT/spv/k9_combine.spv
for m in 0 1 2; do bg9 $G -DMODE=$m $S9/k9_kvcopy.comp -o $OUT/spv/k9_kvcopy_m$m.spv; done
for split in 1 0; do
  mode=rows; [ $split = 1 ] && mode=split
  for fp8 in 0 1; do
    fmt=bf16; [ $fp8 = 1 ] && fmt=fp8
    for pvh in 3 6 12; do
      ts=64; [ $split = 0 ] && ts=${K9_ROWS_TS:-64}
      bg9 $G -DSPLIT=$split -DTS=$ts -DFP8=$fp8 -DPVH=$pvh $S9/k9_attn.comp -o $OUT/spv/k9_attn_${mode}_${fmt}_pvh$pvh.spv
    done
  done
done
# P4-A: the cooperative-matrix prefill kernel and the FP32 rows kernel's FALLBACK variant (its exact fallback)
for fp8 in 0 1; do
  fmt=bf16; [ $fp8 = 1 ] && fmt=fp8
  bg9 $G -DFP8=$fp8 $S9/p4att_sparse.comp -o $OUT/spv/p4att_sparse_$fmt.spv
  bg9 $G -DFP8=$fp8 $S9/p4att_dense.comp -o $OUT/spv/p4att_dense_$fmt.spv   # also K6's (the same SPIR-V): the dense-prefix blocks
  bg9 $G -DSPLIT=0 -DTS=${K9_ROWS_TS:-64} -DFP8=$fp8 -DPVH=3 -DFALLBACK=1 $S9/k9_attn.comp -o $OUT/spv/k9_attn_rows_fb_$fmt.spv
done
bg9 $G $S9/p4att_route.comp -o $OUT/spv/p4att_route.spv
wait
[ ! -f $OUT/.fail9 ] || { echo "k9 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k9_qsa_vk.cpp -o $OUT/k9_qsa_vk.o
$CXX -I$V/ktest $V/ktest/k9_prep_test.cpp $OUT/vk_launch.o $OUT/k9_qsa_vk.o -lvulkan -o $OUT/k9_prep_test
$CXX -I$V/ktest $V/ktest/k9_test.cpp $OUT/vk_launch.o $OUT/k9_qsa_vk.o -lvulkan -o $OUT/k9_test
HIP_REF=${HIP_REF:-/src/ai-cere}
if command -v hipcc >/dev/null 2>&1 && [ -f $HIP_REF/build/hip/gen/kernels/k9_qsa.hip ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $HIP_REF/executor/hip/src/cuda_compat.h -I$HIP_REF/executor/include \
        -I$HIP_REF/build/hip/gen -I$HIP_REF/executor/hip/src -x hip $V/ktest/hip_k9_timing.cpp \
        $HIP_REF/build/hip/gen/kernels/k9_qsa.hip $HIP_REF/build/hip/gen/kernels/k1_primitives.hip \
        $HIP_REF/executor/hip/src/p4_attention_hip.cpp -o $OUT/hip_k9_timing
else
  echo "hip_k9_timing skipped (no hipcc or no HIP tree at $HIP_REF)"
fi
sha256sum $S9/k9_*.comp $S9/p4att_common.comp $S9/p4att_sparse.comp $S9/p4att_dense.comp $S9/p4att_route.comp $V/src/kernels/k9_qsa_vk.h $V/src/kernels/k9_qsa_vk.cpp $V/ktest/k9_prep_test.cpp \
          $V/ktest/k9_test.cpp $OUT/spv/k9_*.spv $OUT/spv/p4att_sparse_*.spv $OUT/spv/p4att_dense_*.spv $OUT/spv/p4att_route.spv $OUT/k9_prep_test $OUT/k9_test \
          $( [ -f $OUT/hip_k9_timing ] && echo $OUT/hip_k9_timing $V/ktest/hip_k9_timing.cpp )
