# K12 device sampler + K13S speculative accept/resample (P2): shaders, kernel objects, k12_test / k13s_test and, when hipcc exists and
# K12_HIP_GEN names a generated HIP tree (e.g. /src/ai-cere/build/hip/gen), the same-method HIP timing binaries k12_hip_timing /
# k13s_hip_timing. Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined.
# The sampler needs shaderFloat64 (fp64 sums for exact draws); the shared launch layer enables it.
rm -f $OUT/.fail12
S=$V/shaders
cs() { out=$1; src=$2; shift 2; $G "$@" $S/$src -o $OUT/spv/$out.spv || touch $OUT/.fail12; }
{
cs k12_process k12_process.comp -DBIAS=0 &
cs k12_bias k12_process.comp -DBIAS=1 &
cs k12_top2 k12_top2.comp &
for st in 0 1 2; do cs k12_probs_s$st k12_probs.comp -DSTAGE=$st & done
cs k12_keep_clear k12_keep.comp -DSTAGE=6 &
cs k12_keep_hist0 k12_keep.comp -DSTAGE=0 &
cs k12_keep_hist1 k12_keep.comp -DSTAGE=1 &
cs k12_keep_hist2 k12_keep.comp -DSTAGE=2 &
cs k12_keep_collect k12_keep.comp -DSTAGE=3 &
cs k12_keep_resolve k12_keep.comp -DSTAGE=4 &
cs k12_keep_masked k12_keep.comp -DSTAGE=5 &
cs k12_draw_prefix k12_draw.comp -DSTAGE=0 -DSEEDED=0 &
cs k12_draw_seeded k12_draw.comp -DSTAGE=1 -DSEEDED=1 &
cs k12_draw_u k12_draw.comp -DSTAGE=1 -DSEEDED=0 &
cs k12_finalize k12_small.comp -DSTAGE=0 &
cs k12_count_commit k12_small.comp -DSTAGE=1 &
cs k12_logprobs k12_logprobs.comp &
cs k12_selftest k12_selftest.comp &
for st in 0 1 2; do cs k13s_accept_s$st k13s_accept.comp -DSTAGE=$st & done
wait
}
[ ! -f $OUT/.fail12 ] || { echo "k12 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k12_sampler_vk.cpp -o $OUT/k12_sampler_vk.o
$CXX -c $V/src/kernels/k13s_spec_accept_vk.cpp -o $OUT/k13s_spec_accept_vk.o
REF=executor/cuda/src/host/sampler_ref.cpp
$CXX -Iexecutor/cuda/src $V/ktest/k12_test.cpp $REF $OUT/vk_launch.o $OUT/k12_sampler_vk.o -lvulkan -o $OUT/k12_test
$CXX -Iexecutor/cuda/src $V/ktest/k13s_test.cpp $REF $OUT/vk_launch.o $OUT/k12_sampler_vk.o $OUT/k13s_spec_accept_vk.o -lvulkan -o $OUT/k13s_test
if command -v hipcc >/dev/null 2>&1 && [ -n "${K12_HIP_GEN:-}" ] && [ -f "$K12_HIP_GEN/kernels/k12_sampler.hip" ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include executor/hip/src/cuda_compat.h -Iexecutor/include -I$K12_HIP_GEN -I$K12_HIP_GEN/../generated -Iexecutor/cuda/src -DK12_HIP_TIMING \
        -x hip $V/ktest/k12_test.cpp $K12_HIP_GEN/kernels/k12_sampler.hip -o $OUT/k12_hip_timing
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include executor/hip/src/cuda_compat.h -Iexecutor/include -I$K12_HIP_GEN -I$K12_HIP_GEN/../generated -Iexecutor/cuda/src -DK12_HIP_TIMING \
        -x hip $V/ktest/k13s_test.cpp $K12_HIP_GEN/kernels/k12_sampler.hip $K12_HIP_GEN/kernels/k13s_spec_accept.hip -o $OUT/k13s_hip_timing
fi
sha256sum $S/k12_*.comp $S/k13s_*.comp $OUT/spv/k12_*.spv $OUT/spv/k13s_*.spv $OUT/k12_test $OUT/k13s_test $( [ -f $OUT/k12_hip_timing ] && echo $OUT/k12_hip_timing $OUT/k13s_hip_timing )
