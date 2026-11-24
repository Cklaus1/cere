# Native executor build (m22 design note H3, Q11). Built only inside the pinned image by
# `scripts/live/native_dev.py build`:
#   cmake -S /src/executor -B /out -G Ninja -DCMAKE_BUILD_TYPE=Release -DCERE_IMAGE_DIGEST=sha256:<64 hex>
#         -DCERE_GIT_HEAD=<40 hex> -DCERE_GIT_DIRTY=0|1 && cmake --build /out
# Outputs in the build directory: libcere_executor.so (ABI v4: host layer + CUDA device + model forward + kernels),
# libcere_ktest.so (kernel-test entry points cere_ktest_*: kernels + ktest wrappers + the runtime the K8 entry points
# drive) and build.json.
#
# Sources are globbed, so new files under cuda/src/{device,model,kernels,ktest}/ need no edit here. A library whose
# required sources are absent (no CUDA device under cuda/src/device/, no entry point under cuda/src/ktest/) still
# configures, and its target fails the build with a message naming the missing directory; nothing is half-built.
#
# The same file is the build.json writer: the build runs it as `cmake -DCERE_WRITE_BUILD_JSON=<path> ... -P
# CMakeLists.txt` after both libraries are linked (tests/test_native_build_files.py runs that mode on the host).
# build.json (schemas/native_build.schema.json): git head + dirty flag and image digest (the -D inputs above), the
# compile/link flag strings of this file, the compiler launcher's ccache version (`ccache`, "off" without one),
# source_tree_sha256, the static libraries linked, and per output its sha256, size and NEEDED entries (readelf -d).
# source_tree_sha256 is the sha256 of the lines
# "<sha256 of file>  <path relative to executor/>\n" of every file under executor/ outside the build directory and
# outside `__pycache__` directories, sorted by path (the format of `sha256sum`). Host Python writes
# tests/gpu/__pycache__ when native_dev.py imports prefill_tactics.py, and a .pyc header holds its source's mtime, so
# the bytecode differs per checkout of one commit (build/repro-build).
cmake_minimum_required(VERSION 3.28)

# Provenance inputs; checked at configure time and again by the writer.
function(cere_require_input var pattern length)
    string(LENGTH "${${var}}" actual)
    if(NOT "${${var}}" MATCHES "${pattern}" OR NOT actual EQUAL length)
        message(FATAL_ERROR "${var}='${${var}}' must match ${pattern} with ${length} characters")
    endif()
endfunction()

function(cere_check_inputs)
    cere_require_input(CERE_IMAGE_DIGEST "^sha256:[0-9a-f]+$" 71)
    cere_require_input(CERE_GIT_HEAD "^[0-9a-f]+$" 40)
    cere_require_input(CERE_GIT_DIRTY "^[01]$" 1)
endfunction()

function(cere_json_string out value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    set(${out} "\"${value}\"" PARENT_SCOPE)
endfunction()

function(cere_json_string_array out)
    set(array "[]")
    set(index 0)
    foreach(item IN LISTS ARGN)
        cere_json_string(quoted "${item}")
        string(JSON array SET "${array}" ${index} "${quoted}")
        math(EXPR index "${index} + 1")
    endforeach()
    set(${out} "${array}" PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
    # ---- build.json writer ----
    if(NOT CERE_WRITE_BUILD_JSON)
        message(FATAL_ERROR "CERE_WRITE_BUILD_JSON: script mode only writes build.json")
    endif()
    cere_check_inputs()
    foreach(var CERE_SOURCE_DIR CERE_BINARY_DIR CERE_READELF CERE_STATIC_LIBS CERE_OUTPUTS CERE_CCACHE)
        if("${${var}}" STREQUAL "")
            message(FATAL_ERROR "${var} is required")
        endif()
    endforeach()
    if(NOT CERE_CCACHE MATCHES "^(off|[0-9]+\\.[0-9]+(\\.[0-9]+)?)$")
        message(FATAL_ERROR "CERE_CCACHE='${CERE_CCACHE}' must be off or a ccache version")
    endif()
    string(REPLACE "," ";" static_libs "${CERE_STATIC_LIBS}")
    string(REPLACE "," ";" outputs "${CERE_OUTPUTS}")

    file(GLOB_RECURSE tree LIST_DIRECTORIES false "${CERE_SOURCE_DIR}/*")
    list(SORT tree)
    set(tree_lines "")
    foreach(path IN LISTS tree)
        cmake_path(IS_PREFIX CERE_BINARY_DIR "${path}" NORMALIZE in_build)
        file(RELATIVE_PATH rel "${CERE_SOURCE_DIR}" "${path}")
        if(in_build OR "/${rel}" MATCHES "/__pycache__/")
            continue()
        endif()
        file(SHA256 "${path}" digest)
        string(APPEND tree_lines "${digest}  ${rel}\n")
    endforeach()
    string(SHA256 tree_sha256 "${tree_lines}")

    set(output_array "[]")
    set(index 0)
    foreach(path IN LISTS outputs)
        if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
            message(FATAL_ERROR "CERE_OUTPUTS: ${path} does not exist")
        endif()
        execute_process(COMMAND "${CERE_READELF}" -d --wide "${path}" OUTPUT_VARIABLE dynamic
                        RESULT_VARIABLE readelf_status ERROR_VARIABLE readelf_error)
        if(NOT readelf_status EQUAL 0)
            message(FATAL_ERROR "CERE_OUTPUTS: readelf -d ${path} failed: ${readelf_error}")
        endif()
        # CMake lists do not split inside square brackets: map them to <> before splitting into lines.
        set(needed "")
        string(REPLACE "[" "<" dynamic "${dynamic}")
        string(REPLACE "]" ">" dynamic "${dynamic}")
        string(REPLACE ";" "," dynamic "${dynamic}")
        string(REPLACE "\n" ";" dynamic_lines "${dynamic}")
        foreach(line IN LISTS dynamic_lines)
            if(line MATCHES "\\(NEEDED\\).*<(.*)>")
                list(APPEND needed "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        cmake_path(GET path FILENAME name)
        file(SHA256 "${path}" digest)
        file(SIZE "${path}" bytes)
        cere_json_string(name_json "${name}")
        cere_json_string(digest_json "${digest}")
        cere_json_string_array(needed_json ${needed})
        set(entry "{}")
        string(JSON entry SET "${entry}" file "${name_json}")
        string(JSON entry SET "${entry}" sha256 "${digest_json}")
        string(JSON entry SET "${entry}" bytes "${bytes}")
        string(JSON entry SET "${entry}" needed "${needed_json}")
        string(JSON output_array SET "${output_array}" ${index} "${entry}")
        math(EXPR index "${index} + 1")
    endforeach()

    if(CERE_GIT_DIRTY)
        set(dirty true)
    else()
        set(dirty false)
    endif()
    cere_json_string(head_json "${CERE_GIT_HEAD}")
    cere_json_string(image_json "${CERE_IMAGE_DIGEST}")
    cere_json_string(tree_json "${tree_sha256}")
    cere_json_string(cuda_json "${CERE_FLAGS_CUDA}")
    cere_json_string(cxx_json "${CERE_FLAGS_CXX}")
    cere_json_string(link_json "${CERE_FLAGS_LINK}")
    cere_json_string_array(static_json ${static_libs})
    set(flags "{}")
    string(JSON flags SET "${flags}" cuda "${cuda_json}")
    string(JSON flags SET "${flags}" cxx "${cxx_json}")
    string(JSON flags SET "${flags}" link "${link_json}")
    if(CERE_MOE_BACKEND STREQUAL "p3b")
        cere_json_string(p3b_cuda_json "${CERE_FLAGS_P3B_CUDA}")
        string(JSON flags SET "${flags}" moe_backend "\"p3b\"")
        string(JSON flags SET "${flags}" p3b_cuda "${p3b_cuda_json}")
    endif()
    if(DEFINED CERE_MODEL AND NOT CERE_MODEL STREQUAL "qwen38-flash-next")
        cere_json_string(model_json "${CERE_MODEL}")
        string(JSON flags SET "${flags}" model "${model_json}")
    endif()
    set(build "{}")
    string(JSON build SET "${build}" schema_version "\"1\"")
    string(JSON build SET "${build}" git_head "${head_json}")
    string(JSON build SET "${build}" git_dirty "${dirty}")
    string(JSON build SET "${build}" image_digest "${image_json}")
    string(JSON build SET "${build}" flags "${flags}")
    string(JSON build SET "${build}" source_tree_sha256 "${tree_json}")
    string(JSON build SET "${build}" static_libs "${static_json}")
    cere_json_string(ccache_json "${CERE_CCACHE}")
    string(JSON build SET "${build}" ccache "${ccache_json}")
    string(JSON build SET "${build}" outputs "${output_array}")
    file(WRITE "${CERE_WRITE_BUILD_JSON}.tmp" "${build}\n")
    file(RENAME "${CERE_WRITE_BUILD_JSON}.tmp" "${CERE_WRITE_BUILD_JSON}")
    return()
endif()

# ---- native build ----
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
project(cere_executor LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be Release (got '${CMAKE_BUILD_TYPE}')")
endif()
cere_check_inputs()

find_package(CUDAToolkit REQUIRED)
if(NOT CMAKE_READELF)
    message(FATAL_ERROR "readelf is required to write build.json")
endif()

# Compiler launcher: none, or one ccache for both languages (`native_dev.py build --ccache`, build/ccache, dev only).
# A hit returns the object a compile of the same command made, so the libraries are the bytes of a clean build;
# build.json records the version (`ccache`, "off" without a launcher) and lock_native.py refuses a cached build.
set(CERE_CCACHE "off")
if(CMAKE_CXX_COMPILER_LAUNCHER OR CMAKE_CUDA_COMPILER_LAUNCHER)
    if(NOT "${CMAKE_CXX_COMPILER_LAUNCHER}" STREQUAL "${CMAKE_CUDA_COMPILER_LAUNCHER}")
        message(FATAL_ERROR "the CXX and CUDA compiler launchers must be one ccache (got '${CMAKE_CXX_COMPILER_LAUNCHER}' and '${CMAKE_CUDA_COMPILER_LAUNCHER}')")
    endif()
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER_LAUNCHER} --version OUTPUT_VARIABLE launcher_version
                    RESULT_VARIABLE launcher_status ERROR_QUIET)
    if(NOT launcher_status EQUAL 0 OR NOT launcher_version MATCHES "^ccache version ([0-9]+\\.[0-9]+(\\.[0-9]+)?)\n")
        message(FATAL_ERROR "the compiler launcher '${CMAKE_CUDA_COMPILER_LAUNCHER}' is not ccache")
    endif()
    set(CERE_CCACHE "${CMAKE_MATCH_1}")
endif()

# Every compile and link flag is set here (environment CXXFLAGS/CUDAFLAGS and the per-configuration defaults are
# overridden), so build.json records the complete flag strings; the one per-TU flag, each CUDA TU's
# `-frandom-seed=<its path>` (cere_random_seeds below), is a function of the source list. Reproducible output: no build
# ID, source and build paths mapped to fixed prefixes, fixed nvcc seeds, no RPATH. Visibility: everything hidden; the
# version scripts export the ABI v4 functions from libcere_executor.so and cere_ktest_* from libcere_ktest.so, and keep
# the static NVIDIA libraries' symbols local. A single real architecture, sm_120a (GB202 arch-specific features), and
# no PTX.
set(CERE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cuda/src)
set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -std=c++17 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=executor -ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=build")
set(CMAKE_CUDA_FLAGS "-O3 -DNDEBUG -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -Xcompiler=-fvisibility-inlines-hidden -Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=executor -Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=build")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--build-id=none -Wl,--as-needed -Wl,-z,defs -Wl,--strip-all")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CUDA_FLAGS_RELEASE "")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "")
set(CMAKE_SKIP_RPATH ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(CERE_STATIC_LIB_FILES "")
foreach(lib IN ITEMS cudart_static cublasLt_static culibos)
    get_target_property(location CUDA::${lib} IMPORTED_LOCATION)
    if(NOT location)
        message(FATAL_ERROR "the CUDA toolkit has no static library for CUDA::${lib}")
    endif()
    cmake_path(GET location FILENAME name)
    list(APPEND CERE_STATIC_LIB_FILES ${name})
endforeach()

file(GLOB CERE_HOST_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/host/*.cpp)
file(GLOB CERE_DEVICE_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/device/*.cpp ${CERE_SRC}/device/*.cu)
file(GLOB CERE_MODEL_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/model/*.cpp ${CERE_SRC}/model/*.cu)
file(GLOB CERE_RESEARCH_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/research/*.cpp ${CERE_SRC}/research/*.cu)
file(GLOB CERE_KERNEL_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/kernels/*.cpp ${CERE_SRC}/kernels/*.cu)
file(GLOB CERE_KTEST_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/ktest/*.cpp ${CERE_SRC}/ktest/*.cu)
# SelfTune D1 (selftune design note ST1a): the model's dimension constants, generated from its descriptor by a
# stdlib script named literally here so lock_native.py's build_inputs counts it (and, through the descriptor path,
# every file under models/). The header is byte-deterministic (sorted, no timestamp or path) and every target
# compiles against it (cere_model_header first).
# SelfTune ST4 S3a: a library is built for one model, CERE_MODEL (the descriptor executor/models/<CERE_MODEL>.json,
# default the Qwen one), and the descriptor's model_type is the library's family (gen::kFamily). ST4 S3c: family sdar
# (DICE-8B) builds its own sources instead of the Qwen ones: kernels/sdar/, model/sdar/, device/sdar/ and
# ktest/sdar/, plus the shared host/*.cpp and the host-side research/*.cpp (no research CUDA: DICE-8B refuses
# research by name). An sdar library compiles no MoE (no P3b objects whatever CERE_MOE_BACKEND says; build.json still
# records the flag). Inkling 2 Mini (inkling design note S0): family inkling2_loop builds kernels/inkling/,
# model/inkling/, device/inkling/ and ktest/inkling/ the same way, no P3b objects, and compiles its CUDA with
# -fmad=false and its C++ with -ffp-contract=off (executor/ref/inkling/common_ref.h: every fused multiply-add is
# explicit, so a kernel is bit-equal to its CPU reference, G1). Any other family fails by name.
set(CERE_MODEL qwen38-flash-next CACHE STRING "model descriptor id (executor/models/<id>.json)")
if(NOT CERE_MODEL MATCHES "^[a-z0-9][a-z0-9.-]*$" OR NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/${CERE_MODEL}.json)
    message(FATAL_ERROR "CERE_MODEL='${CERE_MODEL}' names no descriptor under executor/models/")
endif()
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/models/${CERE_MODEL}.json CERE_MODEL_DESCRIPTOR_TEXT)
string(JSON CERE_MODEL_TYPE GET "${CERE_MODEL_DESCRIPTOR_TEXT}" model_type value)
if(CERE_MODEL_TYPE STREQUAL "sdar")
    file(GLOB CERE_DEVICE_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/device/sdar/*.cpp ${CERE_SRC}/device/sdar/*.cu)
    file(GLOB CERE_MODEL_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/model/sdar/*.cpp ${CERE_SRC}/model/sdar/*.cu)
    file(GLOB CERE_RESEARCH_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/research/*.cpp)
    file(GLOB CERE_KERNEL_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/kernels/sdar/*.cpp ${CERE_SRC}/kernels/sdar/*.cu)
    # ktest/l2_flush.cu: the cold-L2 flush of the timing entry points (ktest/sdar/sdar_timing_ktest.cpp, ST6 S2).
    file(GLOB CERE_KTEST_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/ktest/sdar/*.cpp ${CERE_SRC}/ktest/sdar/*.cu
         ${CERE_SRC}/ktest/l2_flush.cu)
elseif(CERE_MODEL_TYPE STREQUAL "inkling2_loop")
    file(GLOB CERE_DEVICE_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/device/inkling/*.cpp ${CERE_SRC}/device/inkling/*.cu)
    file(GLOB CERE_MODEL_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/model/inkling/*.cpp ${CERE_SRC}/model/inkling/*.cu)
    file(GLOB CERE_RESEARCH_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/research/*.cpp)
    # K2's GemmPlan (k2_*.cpp/.cu and lora.cu, which k2_gemm.h includes), shared with the Qwen family: S5's BF16
    # GEMMs and S7's vision linears (kernels/k3_hc_math.cuh compiles its hc epilogues for one stream here); K12's
    # sampler (S6g: sampled decode after model/inkling/logits.h process_logits).
    file(GLOB CERE_KERNEL_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/kernels/inkling/*.cpp
         ${CERE_SRC}/kernels/inkling/*.cu ${CERE_SRC}/kernels/k2_*.cpp ${CERE_SRC}/kernels/k2_*.cu
         ${CERE_SRC}/kernels/lora.cu ${CERE_SRC}/kernels/k12_sampler.cu)
    file(GLOB CERE_KTEST_SOURCES CONFIGURE_DEPENDS ${CERE_SRC}/ktest/inkling/*.cpp ${CERE_SRC}/ktest/inkling/*.cu
         ${CERE_SRC}/ktest/l2_flush.cu)
    string(APPEND CMAKE_CUDA_FLAGS " -fmad=false")
    string(APPEND CMAKE_CXX_FLAGS " -ffp-contract=off")
elseif(NOT CERE_MODEL_TYPE STREQUAL "qwen4_exp")
    message(FATAL_ERROR "CERE_MODEL='${CERE_MODEL}': family ${CERE_MODEL_TYPE} has no CUDA device")
endif()
set(CERE_MODEL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/cere_model_constants.h)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/cere_model_constants.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_model_header.py
        ${CMAKE_CURRENT_SOURCE_DIR}/models/${CERE_MODEL}.json
        ${CMAKE_CURRENT_BINARY_DIR}/generated/cere_model_constants.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_model_header.py ${CMAKE_CURRENT_SOURCE_DIR}/models/${CERE_MODEL}.json
    VERBATIM)
add_custom_target(cere_model_header DEPENDS ${CERE_MODEL_HEADER})
set(CERE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include ${CERE_SRC} ${CMAKE_CURRENT_BINARY_DIR}/generated)
# Inkling: its kernels and ktests include the shared CPU references as "inkling/<op>_ref.h" (root executor/ref/, never
# executor/ itself: scripts/native/lock_native.py cannot derive a build that reads the whole tree).
if(CERE_MODEL_TYPE STREQUAL "inkling2_loop")
    list(APPEND CERE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/ref)
endif()

# OH-08 (ops-hardening design note): test-only executor fault injection (cuda/src/fault/fault_inject.h), off by
# default. ON defines CERE_FAULT_INJECTION in every TU through the flag strings (so build.json records it and
# lock_native.py refuses the build) and compiles fault/fault_inject.cu into the runtime; OFF leaves the flags, the
# sources and both outputs exactly as without this option.
option(CERE_FAULT_INJECTION "OH-08 test-only fault injection (env CERE_FAULT_INJECT); never a release build" OFF)
if(CERE_FAULT_INJECTION)
    string(APPEND CMAKE_CXX_FLAGS " -DCERE_FAULT_INJECTION=1")
    string(APPEND CMAKE_CUDA_FLAGS " -DCERE_FAULT_INJECTION=1")
    list(APPEND CERE_DEVICE_SOURCES ${CERE_SRC}/fault/fault_inject.cu)
endif()

# SelfTune ST4 S4a (selftune design note ST4, D-S4-3): a fail-first mutant of family sdar, test-only and empty by
# default. `causal` or `fp32tie` (model/sdar/sdar_model.cpp) or `changing_k_short` (SdarOptIn, kernels/sdar/) defines
# CERE_SDAR_MUTANT_<NAME> in every TU through the flag strings (so build.json records it and lock_native.py refuses the
# build); any other name, or any mutant of a family other than sdar, fails at configure. Empty leaves the flags, the
# sources and both outputs exactly as without this option.
set(CERE_SDAR_MUTANT "" CACHE STRING
    "SelfTune ST4 S4a test-only sdar mutant: causal, fp32tie or changing_k_short; never a release build")
if(NOT CERE_SDAR_MUTANT STREQUAL "")
    if(NOT CERE_SDAR_MUTANT MATCHES "^(causal|fp32tie|changing_k_short)$")
        message(FATAL_ERROR "CERE_SDAR_MUTANT='${CERE_SDAR_MUTANT}' must be causal, fp32tie or changing_k_short")
    endif()
    if(NOT CERE_MODEL_TYPE STREQUAL "sdar")
        message(FATAL_ERROR "CERE_SDAR_MUTANT='${CERE_SDAR_MUTANT}': family ${CERE_MODEL_TYPE} has no sdar mutant")
    endif()
    string(TOUPPER "${CERE_SDAR_MUTANT}" CERE_SDAR_MUTANT_UPPER)
    string(APPEND CMAKE_CXX_FLAGS " -DCERE_SDAR_MUTANT_${CERE_SDAR_MUTANT_UPPER}=1")
    string(APPEND CMAKE_CUDA_FLAGS " -DCERE_SDAR_MUTANT_${CERE_SDAR_MUTANT_UPPER}=1")
endif()

# MoE backend (m23 design note M23-D1): `k7` (default; the output is byte-for-byte the pre-M2.3 build) or `p3b`
# (FlashInfer 0.6.17's SM120 CUTLASS fused MoE, vendored under third_party/, P3b). `p3b` defines CERE_MOE_P3B in
# every TU (the model's MoE types stay ODR-equal) and adds the cere_p3b objects to both libraries. Its sources are
# named one by one in cuda/src/p3b/ (not globbed) plus the vendored TRT-LLM sources the link needs, and the 32
# grouped-GEMM instantiations FlashInfer's generate_kernels.py emits for (SafeFP4, SafeFP4, BF16) on Sm120, written
# into the build dir at configure time. An FP4 MoE TU peaks at ~7 GB host RAM and the lab build container has
# 16 GiB, so every compile of a `p3b` build runs in a 2-job pool. The P3b TUs take FlashInfer's JIT flags
# (CERE_P3B_CUDA_OPTIONS, recorded in build.json `flags.p3b_cuda`). MoE stays a build variant: the moe_gemm lever
# slot (cere_slot_selection, ABI v5) only checks that a profile asks for the compiled backend. The K2 small-M GEMM
# (P11) and the P4 prefill attention are compiled into every library and selected per model at run time by the
# k2_gemm and prefill_attention slots (cere_model_desc.selection, cere_model_set_selection), so they have no build
# option.
set(CERE_MOE_BACKEND k7 CACHE STRING "MoE backend: k7 (default) or p3b (M23-D1)")
if(NOT CERE_MOE_BACKEND MATCHES "^(k7|p3b)$")
    message(FATAL_ERROR "CERE_MOE_BACKEND='${CERE_MOE_BACKEND}' must be k7 or p3b")
endif()
set(CERE_P3B_OBJECTS "")
set(CERE_P3B_CUDA_OPTIONS --expt-relaxed-constexpr -static-global-template-stub=false -use_fast_math
    -Xfatbin=-compress-all)
list(JOIN CERE_P3B_CUDA_OPTIONS " " CERE_FLAGS_P3B_CUDA)
if(CERE_MOE_BACKEND STREQUAL "p3b" AND NOT CERE_MODEL_TYPE MATCHES "^(sdar|inkling2_loop)$")
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS cere_p3b_fp4=2)
    set(CMAKE_JOB_POOL_COMPILE cere_p3b_fp4)
    add_compile_definitions(CERE_MOE_P3B=1)
    set(CERE_FLASHINFER ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer-0.6.17/flashinfer/data/csrc)
    set(CERE_TRTLLM_KERNELS ${CERE_FLASHINFER}/nv_internal/tensorrt_llm/kernels)
    set(CERE_P3B_SOURCES
        ${CERE_SRC}/p3b/p3b_moe.cu
        ${CERE_SRC}/p3b/p3b_runner.cu
        ${CERE_SRC}/p3b/p3b_decode.cu
        ${CERE_SRC}/p3b/p3b_gemm.cu
        ${CERE_FLASHINFER}/nv_internal/cpp/common/envUtils.cpp
        ${CERE_FLASHINFER}/nv_internal/cpp/common/logger.cpp
        ${CERE_FLASHINFER}/nv_internal/cpp/common/stringUtils.cpp
        ${CERE_FLASHINFER}/nv_internal/cpp/common/tllmException.cpp
        ${CERE_FLASHINFER}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
        ${CERE_FLASHINFER}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
        ${CERE_FLASHINFER}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu
        ${CERE_FLASHINFER}/nv_internal/tensorrt_llm/kernels/lora/lora.cpp)
    set(CERE_P3B_KTEST_SOURCES ${CERE_SRC}/p3b/p3b_ktest.cpp)
    set(CERE_P3B_INSTANCES "")
    foreach(epilogue IN ITEMS NONE FINALIZE)
        foreach(tile IN ITEMS 128,32,128 128,32,256 128,64,128 128,64,256 128,128,128 128,128,256 128,256,128
                              256,128,128)
            foreach(swap IN ITEMS false true)
                string(REPLACE "," "x" tile_name "${tile}")
                set(path ${CMAKE_CURRENT_BINARY_DIR}/p3b/inst_${epilogue}_${tile_name}_swap_${swap}.cu)
                configure_file(${CERE_SRC}/p3b/p3b_instance.cu.in ${path} @ONLY)
                list(APPEND CERE_P3B_INSTANCES ${path})
            endforeach()
        endforeach()
    endforeach()
    set(CERE_P3B_CUDA_DEFINITIONS FLASHINFER_ENABLE_F16 FLASHINFER_ENABLE_BF16 FLASHINFER_ENABLE_FP8_E4M3
        FLASHINFER_ENABLE_FP8_E5M2)
    add_library(cere_p3b OBJECT ${CERE_P3B_SOURCES} ${CERE_P3B_INSTANCES})
    add_dependencies(cere_p3b cere_model_header)
    target_include_directories(cere_p3b PRIVATE ${CERE_INCLUDES} ${CERE_FLASHINFER}/nv_internal
        ${CERE_FLASHINFER}/nv_internal/include ${CERE_FLASHINFER}/nv_internal/tensorrt_llm/cutlass_extensions/include
        ${CERE_TRTLLM_KERNELS}/cutlass_kernels/include ${CERE_TRTLLM_KERNELS}/cutlass_kernels)
    # CCCL (cuda/std/*) comes from the toolkit's include/cccl, which nvcc adds itself and g++ does not.
    set(CERE_CCCL_INCLUDE ${CUDAToolkit_INCLUDE_DIRS})
    list(TRANSFORM CERE_CCCL_INCLUDE APPEND /cccl)
    target_include_directories(cere_p3b SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CERE_CCCL_INCLUDE}
        ${CERE_FLASHINFER}/../include ${CERE_FLASHINFER} ${CERE_FLASHINFER}/../cutlass/include
        ${CERE_FLASHINFER}/../cutlass/tools/util/include)
    target_compile_definitions(cere_p3b PRIVATE COMPILE_BLACKWELL_TMA_GEMMS COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
        ENABLE_BF16 ENABLE_FP8 ENABLE_FP4 USING_OSS_CUTLASS_MOE_GEMM CUTLASS_ENABLE_GDC_FOR_SM100=1
        "$<$<COMPILE_LANGUAGE:CUDA>:${CERE_P3B_CUDA_DEFINITIONS}>")
    target_compile_options(cere_p3b PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CERE_P3B_CUDA_OPTIONS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>")
    set(CERE_P3B_OBJECTS $<TARGET_OBJECTS:cere_p3b>)
endif()

# Reproducible device code (build/repro-build): nvcc names a TU's internal-linkage entities (in the P3b grouped-GEMM
# instances: libcu++'s `cuda::std` CPOs and `ignore`) after the TU's module ID, which holds a random number unless
# `-frandom-seed` gives one. Those names land in the fatbin and .rodata, so two builds of one tree differed there while
# the SASS was identical. Each CUDA TU gets its own seed, as nvcc asks: its path relative to cuda/src/ (`../../` for a
# vendored one), or `build/` and its path relative to the build directory for a generated P3b instance.
function(cere_random_seeds base prefix)
    foreach(source IN LISTS ARGN)
        if(source MATCHES "[.]cu$")
            file(RELATIVE_PATH seed "${base}" "${source}")
            set_property(SOURCE "${source}" APPEND PROPERTY COMPILE_OPTIONS "-frandom-seed=${prefix}${seed}")
        endif()
    endforeach()
endfunction()
cere_random_seeds("${CERE_SRC}" "" ${CERE_HOST_SOURCES} ${CERE_DEVICE_SOURCES} ${CERE_MODEL_SOURCES}
    ${CERE_RESEARCH_SOURCES} ${CERE_KERNEL_SOURCES} ${CERE_KTEST_SOURCES} ${CERE_P3B_SOURCES}
    ${CERE_P3B_KTEST_SOURCES})
cere_random_seeds("${CMAKE_CURRENT_BINARY_DIR}" "build/" ${CERE_P3B_INSTANCES})

# Kernels are compiled once and linked into both libraries.
set(CERE_KERNEL_OBJECTS "")
if(CERE_KERNEL_SOURCES)
    add_library(cere_kernels OBJECT ${CERE_KERNEL_SOURCES})
    add_dependencies(cere_kernels cere_model_header)
    target_include_directories(cere_kernels PRIVATE ${CERE_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS})
    set(CERE_KERNEL_OBJECTS $<TARGET_OBJECTS:cere_kernels>)
endif()

# The runtime (host layer, CUDA device, model forward) is compiled once and linked into both libraries: the K8
# kernel-test entry points drive the ABI v4 runtime and read the forward's trace taps.
set(CERE_RUNTIME_OBJECTS "")
if(CERE_DEVICE_SOURCES)
    add_library(cere_runtime OBJECT ${CERE_HOST_SOURCES} ${CERE_DEVICE_SOURCES} ${CERE_MODEL_SOURCES}
                ${CERE_RESEARCH_SOURCES})
    add_dependencies(cere_runtime cere_model_header)
    target_include_directories(cere_runtime PRIVATE ${CERE_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS})
    set(CERE_RUNTIME_OBJECTS $<TARGET_OBJECTS:cere_runtime>)
endif()

if(CERE_DEVICE_SOURCES)
    add_library(cere_executor SHARED ${CERE_RUNTIME_OBJECTS} ${CERE_KERNEL_OBJECTS} ${CERE_P3B_OBJECTS})
    target_include_directories(cere_executor PRIVATE ${CERE_INCLUDES})
    target_link_libraries(cere_executor PRIVATE CUDA::cudart_static CUDA::cublasLt_static CUDA::culibos)
    target_link_options(cere_executor PRIVATE "-Wl,--version-script=${CERE_SRC}/host/exports.map")
    set_target_properties(cere_executor PROPERTIES CUDA_RUNTIME_LIBRARY Static
                          LINK_DEPENDS ${CERE_SRC}/host/exports.map)
else()
    add_custom_target(cere_executor ALL COMMAND ${CMAKE_COMMAND} -E echo "cere_executor: no CUDA device implementation under executor/cuda/src/device/ (*.cpp, *.cu)" COMMAND ${CMAKE_COMMAND} -E false VERBATIM)
endif()

if(CERE_KTEST_SOURCES)
    # CERE_P3B_KTEST_SOURCES: the P3b entry points (p3b build only; set beside CERE_P3B_SOURCES).
    add_library(cere_ktest SHARED ${CERE_KTEST_SOURCES} ${CERE_RUNTIME_OBJECTS} ${CERE_KERNEL_OBJECTS}
                ${CERE_P3B_OBJECTS} ${CERE_P3B_KTEST_SOURCES})
    add_dependencies(cere_ktest cere_model_header)
    target_include_directories(cere_ktest PRIVATE ${CERE_INCLUDES})
    target_link_libraries(cere_ktest PRIVATE CUDA::cudart_static CUDA::cublasLt_static CUDA::culibos)
    target_link_options(cere_ktest PRIVATE "-Wl,--version-script=${CERE_SRC}/host/ktest_exports.map")
    set_target_properties(cere_ktest PROPERTIES CUDA_RUNTIME_LIBRARY Static
                          LINK_DEPENDS ${CERE_SRC}/host/ktest_exports.map)
else()
    add_custom_target(cere_ktest ALL COMMAND ${CMAKE_COMMAND} -E echo "cere_ktest: no kernel-test entry point under executor/cuda/src/ktest/ (*.cpp, *.cu)" COMMAND ${CMAKE_COMMAND} -E false VERBATIM)
endif()

# build.json, rewritten by every build after both libraries are linked.
list(JOIN CERE_STATIC_LIB_FILES "," CERE_STATIC_LIBS_ARG)
add_custom_target(cere_build_json ALL
    COMMAND ${CMAKE_COMMAND}
        "-DCERE_WRITE_BUILD_JSON=${CMAKE_CURRENT_BINARY_DIR}/build.json"
        "-DCERE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DCERE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DCERE_GIT_HEAD=${CERE_GIT_HEAD}"
        "-DCERE_GIT_DIRTY=${CERE_GIT_DIRTY}"
        "-DCERE_IMAGE_DIGEST=${CERE_IMAGE_DIGEST}"
        "-DCERE_FLAGS_CUDA=${CMAKE_CUDA_FLAGS}"
        "-DCERE_FLAGS_CXX=${CMAKE_CXX_FLAGS}"
        "-DCERE_FLAGS_LINK=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DCERE_STATIC_LIBS=${CERE_STATIC_LIBS_ARG}"
        "-DCERE_MOE_BACKEND=${CERE_MOE_BACKEND}"
        "-DCERE_MODEL=${CERE_MODEL}"
        "-DCERE_FLAGS_P3B_CUDA=${CERE_FLAGS_P3B_CUDA}"
        "-DCERE_OUTPUTS=${CMAKE_CURRENT_BINARY_DIR}/libcere_executor.so,${CMAKE_CURRENT_BINARY_DIR}/libcere_ktest.so"
        "-DCERE_READELF=${CMAKE_READELF}"
        "-DCERE_CCACHE=${CERE_CCACHE}"
        -P ${CMAKE_CURRENT_LIST_FILE}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/build.json
    VERBATIM)
add_dependencies(cere_build_json cere_executor cere_ktest)
