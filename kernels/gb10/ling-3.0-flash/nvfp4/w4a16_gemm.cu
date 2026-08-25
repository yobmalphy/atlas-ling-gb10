// SPDX-License-Identifier: AGPL-3.0-only

// Ling's 2560-wide dense projections use the complete production W4A16
// symbol family, including BF16-to-FP8 staging used by the decode path.
#include "../../qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu"
