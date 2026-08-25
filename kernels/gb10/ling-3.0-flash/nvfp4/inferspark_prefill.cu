// SPDX-License-Identifier: AGPL-3.0-only

// Ling 3.0 MLA uses qk_head_dim=192.  The common prefill module is compiled
// for HDIM=256; its tensor-core tile loops use that compile-time width even
// though the runtime signature also carries head_dim.  Dispatching Ling to
// the common binary therefore reads across head boundaries.  Shadow it with
// the exact Ling width while retaining the production implementation.
#define HDIM 192
#include "../../common/inferspark_prefill.cu"
