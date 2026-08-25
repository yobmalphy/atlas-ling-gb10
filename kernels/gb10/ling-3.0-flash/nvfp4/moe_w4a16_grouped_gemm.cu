// SPDX-License-Identifier: AGPL-3.0-only

// Ling 3.0 Flash uses the same 2560-hidden / 768-expert-intermediate NVFP4
// grouped-GEMM geometry as Qwen3.6-35B-A3B. Keep the complete production
// symbol family here, including the k64 transposed down-projection entry.
#include "../../qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu"
