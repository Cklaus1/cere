# K5 GDN decode and pieces (P2): shaders, kernel object, k5_decode_test / k5_pieces_test, and (when hipcc and a
# hipified kernel tree exist) the same-method HIP reference timing.
# Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined.
# HIP timing: set HIPGEN to a gen_hip.py output tree (default build/hip/gen) holding kernels/k5_gdn_decode.hip and
# kernels/k5_gdn.hip, and HIPSRC to the checkout whose executor/hip/src/cuda_compat.h it was generated with (default .).
$G -DROWS=0 $V/shaders/k5_gdn_decode.comp -o $OUT/spv/k5_gdn_decode.spv
$G -DROWS=1 $V/shaders/k5_gdn_decode.comp -o $OUT/spv/k5_gdn_decode_rows.spv
$G -DMODE=0 $V/shaders/k5_gdn_conv.comp -o $OUT/spv/k5_gdn_conv.spv
$G -DMODE=1 $V/shaders/k5_gdn_conv.comp -o $OUT/spv/k5_gdn_conv_state.spv
$G -DMODE=2 $V/shaders/k5_gdn_conv.comp -o $OUT/spv/k5_gdn_conv_rows.spv
$G $V/shaders/k5_gdn_gating.comp -o $OUT/spv/k5_gdn_gating.spv
$G $V/shaders/k5_gdn_l2norm.comp -o $OUT/spv/k5_gdn_l2norm.spv
$CXX -c $V/src/kernels/k5_gdn_vk.cpp -o $OUT/k5_gdn_vk.o
$CXX $V/ktest/k5_decode_test.cpp $OUT/vk_launch.o $OUT/k5_gdn_vk.o -lvulkan -o $OUT/k5_decode_test
$CXX $V/ktest/k5_pieces_test.cpp $OUT/vk_launch.o $OUT/k5_gdn_vk.o -lvulkan -o $OUT/k5_pieces_test
K5_HIPGEN=${HIPGEN:-build/hip/gen}
K5_HIPSRC=${HIPSRC:-.}
if command -v hipcc >/dev/null 2>&1 && [ -f $K5_HIPGEN/kernels/k5_gdn_decode.hip ] && [ -f $K5_HIPGEN/kernels/k5_gdn.hip ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $K5_HIPSRC/executor/hip/src/cuda_compat.h -Iexecutor/include \
        -I$K5_HIPGEN -x hip $V/ktest/k5_gdn_hip_timing.cpp $K5_HIPGEN/kernels/k5_gdn_decode.hip \
        $K5_HIPGEN/kernels/k5_gdn.hip -o $OUT/k5_hip_timing
else
  echo "k5_gdn: no hipcc or no hipified tree at $K5_HIPGEN: HIP reference timing not built"
fi
sha256sum $V/shaders/k5_gdn*.comp $OUT/spv/k5_gdn*.spv $OUT/k5_decode_test $OUT/k5_pieces_test \
          $( [ -f $OUT/k5_hip_timing ] && echo $OUT/k5_hip_timing )
