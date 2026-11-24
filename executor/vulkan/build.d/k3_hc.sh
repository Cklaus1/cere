# K3 hyper-connections (P2): shaders, kernel object, k3_test, and (when hipcc and a hipified tree exist) the same-method HIP
# reference timing. Sourced by executor/vulkan/build.sh with V, OUT, G (glslc command), CXX and $OUT/vk_launch.o defined.
# The kernel calls K1's rmsnorm and K2 (vkk2::run: small-M GEMV / large-m GEMM), so build those fragments into the same OUT first:
#   docker run --rm -e OUT=build/vulkan-k3 -v "$PWD":/w -w /w llama-amd:rocm7.2.4 sh executor/vulkan/build.sh k1_primitives k2_gemm k2_mgemv k3_hc
# HIP timing (k3_hip_timing): K3_HIPSRC (default /src/ai-cere, a checkout mounted read-only) must hold a gen_hip.py tree
# build/hip/gen with kernels/k3_hc.hip and kernels/k1_primitives.hip, and build/hip/libcere_hipk2.a.
for o in k1_primitives_vk k2_mgemv_vk k2_gemm_vk; do
  [ -f $OUT/$o.o ] || { echo "k3_hc: $OUT/$o.o missing: build the k1_primitives, k2_mgemv and k2_gemm fragments into $OUT too"; exit 1; }
done
rm -f $OUT/.fail3
$G -DMODE=0 $V/shaders/k3_hc.comp -o $OUT/spv/k3_act.spv || touch $OUT/.fail3 &
$G -DMODE=1 $V/shaders/k3_hc.comp -o $OUT/spv/k3_gate.spv || touch $OUT/.fail3 &
$G -DMODE=2 $V/shaders/k3_hc.comp -o $OUT/spv/k3_combine.spv || touch $OUT/.fail3 &
wait
[ ! -f $OUT/.fail3 ] || { echo "k3 shader compile failed"; exit 1; }
$CXX -c $V/src/kernels/k3_hc_vk.cpp -o $OUT/k3_hc_vk.o
$CXX -I$V/ktest $V/ktest/k3_test.cpp $OUT/vk_launch.o $OUT/k3_hc_vk.o $OUT/k1_primitives_vk.o $OUT/k2_mgemv_vk.o $OUT/k2_gemm_vk.o -lvulkan -o $OUT/k3_test
K3_HIPSRC=${K3_HIPSRC:-/src/ai-cere}
if command -v hipcc >/dev/null 2>&1 && [ -f $K3_HIPSRC/build/hip/gen/kernels/k3_hc.hip ] && [ -f $K3_HIPSRC/build/hip/libcere_hipk2.a ]; then
  hipcc -std=c++17 -O2 --offload-arch=gfx1151 -include $K3_HIPSRC/executor/hip/src/cuda_compat.h -I$K3_HIPSRC/executor/include \
        -I$K3_HIPSRC/build/hip/gen -I$K3_HIPSRC/executor/hip/src -x hip $V/ktest/hip_k3_timing.cpp \
        $K3_HIPSRC/build/hip/gen/kernels/k3_hc.hip $K3_HIPSRC/build/hip/gen/kernels/k1_primitives.hip \
        -x none $K3_HIPSRC/build/hip/libcere_hipk2.a -lhipblaslt -o $OUT/k3_hip_timing
else
  echo "k3_hc: no hipcc or no hipified tree at $K3_HIPSRC/build/hip: HIP reference timing not built"
fi
sha256sum $V/shaders/k3_*.comp $OUT/spv/k3_*.spv $OUT/k3_test $( [ -f $OUT/k3_hip_timing ] && echo $OUT/k3_hip_timing )
