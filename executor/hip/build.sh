#!/bin/sh
# AmdAmd PAR-6: build libcere_executor_hip.so (host layer + HIP device) and the smoke test for gfx1151, in the
# llama-amd:rocm7.2.4 image with the repo at /src/ai-cere. Output: build/hip/. Run (GPU): build/hip/smoke
# $PWD/build/hip/libcere_executor_hip.so (with HIP_ENABLE_DEFERRED_LOADING=0; and --lazy without); build/hip/k1_test; build/hip/k2_test; build/hip/k2v_test; build/hip/k5_decode_test; build/hip/k5_pieces_test; build/hip/k12_test; build/hip/k13s_test; build/hip/lora_test; build/hip/k4_hash_test; build/hip/k4_embed_test; build/hip/k4_forward_test; build/hip/k5_chunk_test; build/hip/k5_layer_test; build/hip/k2w_test; build/hip/k6_test; build/hip/k9_test; build/hip/k9_prep_test; build/hip/k3_test; build/hip/k7_test; build/hip/k13_test; build/hip/k2mx_test; build/hip/k4n_test.
set -eu
# SelfTune ST4 S3a: the library is built for one model, CERE_MODEL (executor/models/<id>.json, default the Qwen
# descriptor, as in executor/CMakeLists.txt). The HIP port runs the Qwen family (model_type qwen4_exp) only, so another
# family is refused by name here, before anything is read under /src or built (tests/test_gen_model_header.py).
CERE_MODEL=${CERE_MODEL:-qwen38-flash-next}
DESCRIPTOR=$(dirname "$0")/../models/$CERE_MODEL.json
case $CERE_MODEL in *[!a-z0-9.-]*|''|[.-]*) echo "build.sh: CERE_MODEL='$CERE_MODEL' names no descriptor" >&2; exit 2;; esac
[ -f "$DESCRIPTOR" ] || { echo "build.sh: CERE_MODEL='$CERE_MODEL' names no descriptor under executor/models/" >&2; exit 2; }
FAMILY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model_type"]["value"])' "$DESCRIPTOR")
if [ "$FAMILY" != qwen4_exp ]; then
  echo "build.sh: CERE_MODEL='$CERE_MODEL': family $FAMILY is not built by the HIP port (Qwen family only)" >&2; exit 2
fi
cd /src/ai-cere
OUT=build/hip; mkdir -p $OUT
# The model's dimension constants (SelfTune D1), generated from its descriptor as executor/CMakeLists.txt does.
python3 executor/tools/gen_model_header.py executor/models/$CERE_MODEL.json $OUT/generated/cere_model_constants.h
FLAGS="-std=c++17 -O2 -Wall -Wextra -Werror -pthread -Iexecutor/include -Iexecutor/cuda/src -I$OUT/generated --offload-arch=gfx1151"
# HIP sources generated from the CUDA ones (gen_hip.py): kernels, the model forward and the device
python3 executor/hip/gen_hip.py
KFLAGS="-std=c++17 -O2 --offload-arch=gfx1151 -include executor/hip/src/cuda_compat.h -Iexecutor/include -I$OUT/gen -I$OUT/generated"
# the executor: the unchanged host layer + the hipified CudaDevice and model + every kernel (8 compiles at a time)
G=$OUT/gen; LF="$KFLAGS -fPIC -fvisibility=hidden -w -Iexecutor/hip/src -Iexecutor/cuda/src"
rm -rf $OUT/obj && mkdir -p $OUT/obj
n=0
for f in executor/cuda/src/host/*.cpp $G/device/cuda_device.cpp $G/device/memory_plan.cpp $G/model/forward.cpp \
         $G/model/model.cpp $G/model/mtp.hip $G/model/vision.cpp $G/model/fp8_weights.cpp $G/kernels/*.hip \
         executor/hip/src/k5_gdn_chunk_hip.cpp executor/hip/src/p4_attention_hip.cpp executor/hip/src/k2_variants_hip.cpp \
         executor/hip/src/k2_wmma.cpp executor/hip/src/k2_mgemv.cpp executor/hip/src/k2_gemm_hip.cpp; do
  n=$((n + 1)); hipcc $LF -x hip -c $f -o $OUT/obj/$(echo $f | tr / _).o & [ $((n % 8)) -ne 0 ] || wait
done; wait
hipcc $LF -shared -Wl,--version-script=executor/cuda/src/host/exports.map $OUT/obj/*.o -lhipblaslt -o $OUT/libcere_executor_hip.so
c++ -std=c++17 -O2 -Wall -Wextra -Werror -Iexecutor/include executor/hip/tests/smoke.cpp -ldl -o $OUT/smoke
hipcc $KFLAGS -x hip executor/hip/ktest/k1_test.cpp $OUT/gen/kernels/k1_primitives.hip -o $OUT/k1_test
hipcc $KFLAGS -x hip executor/hip/ktest/k2v_test.cpp $OUT/gen/kernels/k2_gemv.hip -o $OUT/k2v_test
hipcc $KFLAGS -x hip executor/hip/ktest/k5_decode_test.cpp $OUT/gen/kernels/k5_gdn_decode.hip -o $OUT/k5_decode_test
hipcc $KFLAGS -x hip executor/hip/ktest/k5_pieces_test.cpp $OUT/gen/kernels/k5_gdn.hip -o $OUT/k5_pieces_test
c++ -std=c++17 -O2 -Iexecutor/include -Iexecutor/cuda/src -I$OUT/generated -c executor/cuda/src/host/sampler_ref.cpp -o $OUT/sampler_ref.o
hipcc $KFLAGS -Iexecutor/cuda/src -x hip executor/hip/ktest/k12_test.cpp $OUT/gen/kernels/k12_sampler.hip -x none $OUT/sampler_ref.o -o $OUT/k12_test
hipcc $KFLAGS -Iexecutor/cuda/src -x hip executor/hip/ktest/k13s_test.cpp $OUT/gen/kernels/k13s_spec_accept.hip $OUT/gen/kernels/k12_sampler.hip -x none $OUT/sampler_ref.o -o $OUT/k13s_test
hipcc $KFLAGS -Iexecutor/cuda/src -x hip executor/hip/ktest/lora_test.cpp $OUT/gen/kernels/lora.hip -o $OUT/lora_test
hipcc $FLAGS -Iexecutor/hip/src -I$OUT/gen -c executor/hip/src/k2_gemm_hip.cpp -o $OUT/k2_gemm_hip.o
hipcc $FLAGS -Iexecutor/hip/src -c executor/hip/src/k2_wmma.cpp -o $OUT/k2_wmma.o
hipcc $FLAGS -Iexecutor/hip/src -c executor/hip/src/k2_mgemv.cpp -o $OUT/k2_mgemv.o
hipcc $KFLAGS -Iexecutor/hip/src -x hip -c executor/hip/src/k2_variants_hip.cpp -o $OUT/k2_variants.o
hipcc $KFLAGS -Iexecutor/hip/src -x hip -c $OUT/gen/kernels/lora.hip -o $OUT/lora.o
# the K2 GEMM the hipified layers link: hipBLASLt buckets + k2_wmma + the small-M GEMV + the refusing variants + LoRA's table
rm -f $OUT/libcere_hipk2.a && ar rcs $OUT/libcere_hipk2.a $OUT/k2_gemm_hip.o $OUT/k2_wmma.o $OUT/k2_mgemv.o $OUT/k2_variants.o $OUT/lora.o
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k4_hash_test.cpp $OUT/gen/kernels/k4_ple.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k4_hash_test
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k4_embed_test.cpp $OUT/gen/kernels/k4_ple.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k4_embed_test
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k4_forward_test.cpp $OUT/gen/kernels/k4_ple.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k4_forward_test
# K5 chunked prefill: hand-written (the CUDA one is tensor-core mma + cp.async); its header is generated
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k5_chunk_test.cpp executor/hip/src/k5_gdn_chunk_hip.cpp $OUT/gen/kernels/k5_gdn.hip -o $OUT/k5_chunk_test
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k5_layer_test.cpp $OUT/gen/kernels/k5_gdn_layer.hip $OUT/gen/kernels/k5_gdn.hip $OUT/gen/kernels/k5_gdn_decode.hip $OUT/gen/kernels/k1_primitives.hip executor/hip/src/k5_gdn_chunk_hip.cpp -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k5_layer_test
# K6 dense attention: hipified (<= 16 rows) + the hand-written WMMA P4 (HIP's only prefill path, gen_hip.py)
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k6_test.cpp $OUT/gen/kernels/k6_attention.hip $OUT/gen/kernels/k1_primitives.hip executor/hip/src/p4_attention_hip.cpp -o $OUT/k6_test
# K9 QSA: hipified (rocWMMA, copies, block top-k, 32 x 96-key splits)
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k9_test.cpp $OUT/gen/kernels/k9_qsa.hip $OUT/gen/kernels/k1_primitives.hip executor/hip/src/p4_attention_hip.cpp -o $OUT/k9_test
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k9_prep_test.cpp $OUT/gen/kernels/k9_qsa.hip $OUT/gen/kernels/k1_primitives.hip executor/hip/src/p4_attention_hip.cpp -o $OUT/k9_prep_test
# K3 hyper-connections: hipified (PDL no-ops, no cluster combine+norm, no small-M GEMMs: the kUnfused mix)
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k3_test.cpp $OUT/gen/kernels/k3_hc.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k3_test
# K7 NVFP4 MoE (+ K0's helpers): hipified (PDL -> plain, software E2M1 rounding, v_rcp_f32, __dp4a, OCP FP8 types)
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k7_test.cpp $OUT/gen/kernels/k7_moe.hip $OUT/gen/kernels/k0_nvfp4.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k7_test
# K13 BF16 MoE (the MTP draft's; NVFP4 W4A16 option): hipified (PDL dropped, software E2M1 -> FP16)
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k13_test.cpp $OUT/gen/kernels/k13_moe_bf16.hip $OUT/gen/kernels/k7_moe.hip $OUT/gen/kernels/k0_nvfp4.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k13_test
# K2: the hipBLASLt BF16 GEMM (hand-written: CUDA K2 wraps cuBLASLt and NVIDIA tensor-core kernels)
hipcc $FLAGS -Iexecutor/hip/src -I$OUT/gen executor/hip/ktest/k2_test.cpp executor/hip/src/k2_gemm_hip.cpp executor/hip/src/k2_wmma.cpp executor/hip/src/k2_mgemv.cpp -x none $OUT/k2_variants.o $OUT/lora.o -lhipblaslt -o $OUT/k2_test
# K2-SHIFT: the shift-invariant WMMA GEMM the plan runs for row classes > 16
hipcc $FLAGS -Iexecutor/hip/src -I$OUT/gen executor/hip/ktest/k2w_test.cpp executor/hip/src/k2_gemm_hip.cpp executor/hip/src/k2_wmma.cpp executor/hip/src/k2_mgemv.cpp -x none $OUT/k2_variants.o $OUT/lora.o -lhipblaslt -o $OUT/k2w_test
# step 14: the small-M GEMV (BF16 / MXFP8) and the MXFP8 upcast of the S3 artifact
hipcc $FLAGS -Iexecutor/hip/src -I$OUT/gen executor/hip/ktest/k2mx_test.cpp executor/hip/src/k2_gemm_hip.cpp executor/hip/src/k2_wmma.cpp executor/hip/src/k2_mgemv.cpp -x none $OUT/k2_variants.o $OUT/lora.o -lhipblaslt -o $OUT/k2mx_test
# step 14c: K4's NVFP4 PLE gather (the S3 table format)
hipcc $KFLAGS -Iexecutor/hip/src -x hip executor/hip/ktest/k4n_test.cpp $OUT/gen/kernels/k4_ple.hip $OUT/gen/kernels/k1_primitives.hip -x none $OUT/libcere_hipk2.a -lhipblaslt -o $OUT/k4n_test
echo built
