// OH-08 (ops-hardening design note): test-only executor fault injection. A build configured with
// -DCERE_FAULT_INJECTION=ON (native_dev.py build --fault-injection; never a release build: lock_native.py refuses it)
// compiles fault/fault_inject.cu and defines CERE_FAULT_INJECTION in every TU; the environment variable
// CERE_FAULT_INJECT=<site>:<n> (n >= 1) then fails the n-th pass of that site in the process, once, and logs
// `cere: fault_inject site=<site> hit=<n>`. A malformed value aborts the process at the first site passed. Every other
// build compiles CERE_FAULT_POINT to nothing: no symbol, string or instruction of it reaches the library.
//
// Sites (`executor/tests/gpu/test_faults_oh08.py`, suite `faults`):
//   forward_launch     model::forward after its launches: the call fails with CAPACITY, a status that is not INTERNAL
//                      (MF-2: the session is broken all the same)
//   forward_sticky     model::forward after its launches: a trapping kernel, a sticky launch failure (DM-2)
//   seal               CudaDevice::instantiate after instantiate_model rewrote the MoE scales: INTERNAL (MF-1)
//   graph_instantiate  graph_step's cudaGraphInstantiate fails (DM-5)
//   moe_fork           an MoE block (K7 or P3b) fails with INTERNAL after its shared expert forked (KGM-2)
//   bound_fault        model::forward after its launches: the K9 bound fault word set as a kernel that clamped an
//                      out-of-range device position sets it (audit KernelsAttention F3): INTERNAL, session broken
#pragma once

#if defined(CERE_FAULT_INJECTION)
#include <cuda_runtime_api.h>

namespace cere::fault {

enum class Site : unsigned { forward_launch, forward_sticky, seal, graph_instantiate, moe_fork, bound_fault };

// True exactly once: at the n-th pass of the armed site (CERE_FAULT_INJECT=<site>:<n>).
bool fire(Site site);

// Enqueues a kernel that traps (cudaErrorLaunchFailure, sticky) on `stream`.
void trap(cudaStream_t stream);

}  // namespace cere::fault

#define CERE_FAULT_POINT(site, ...)                                 \
    do {                                                            \
        if (::cere::fault::fire(::cere::fault::Site::site)) {       \
            __VA_ARGS__;                                            \
        }                                                           \
    } while (0)
#else
#define CERE_FAULT_POINT(site, ...) \
    do {                            \
    } while (0)
#endif
