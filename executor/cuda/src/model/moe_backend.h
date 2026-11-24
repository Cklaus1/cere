// The MoE block the model calls (m23 design note M23-D1): K7 (kernels/k7_moe.h) in the default `k7` build, P3b
// (p3b/p3b_moe.h) in a `-DCERE_MOE_BACKEND=p3b` build. Same signatures; the `k7` expansion is the pre-M2.3 call.
#pragma once

#include "kernels/k7_moe.h"

#if defined(CERE_MOE_P3B)
#include "p3b/p3b_moe.h"
#define CERE_MOE_BLOCK p3b::moe_block
#define CERE_MOE_WORKSPACE_BYTES p3b::moe_workspace_bytes
#else
#define CERE_MOE_BLOCK kernels::moe_block
#define CERE_MOE_WORKSPACE_BYTES kernels::moe_workspace_bytes
#endif
