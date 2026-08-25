// SPDX-License-Identifier: AGPL-3.0-only

// Ling 3.0 / Bailing Kimi Delta Attention decode recurrence.
// Grid: one CTA per value head. Block: one thread per value dimension.

#include <cuda_bf16.h>

// Ling's FusedRMSNormGated(head_dim, activation="sigmoid"):
//   out = rms_norm(input) * sigmoid(gate)
//
// This must not use Atlas' common gated_rms_norm kernel.  That kernel is for
// Mamba/GDN and applies SiLU(gate), which materially changes Ling's residual
// stream. Grid: one CTA per (token, head); block: one thread per head element.
extern "C" __global__ void kda_sigmoid_gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size
) {
    (void)group_size;
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = (blockDim.x + 31) >> 5;
    const unsigned long long base =
        (unsigned long long)token * hidden_size;
    const unsigned long long gate_base =
        (unsigned long long)token * gate_stride;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        const float x = (float)input[base + i];
        sum_sq += x * x;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) warp_sums[0] = value;
    }
    __syncthreads();

    const float inv_rms =
        rsqrtf(warp_sums[0] / (float)hidden_size + eps);
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        const float x = (float)input[base + i];
        const float g = (float)gate[gate_base + i];
        const float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[base + i] =
            __float2bfloat16(x * inv_rms * (float)weight[i] * sigmoid_g);
    }
}

extern "C" __global__ void __launch_bounds__(128, 1) kda_decode(
    float* __restrict__ state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ f_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_heads,
    unsigned int head_dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int column = threadIdx.x;
    if (head >= num_heads || column >= head_dim) return;

    extern __shared__ float shared[];
    float* q = shared;
    float* k = shared + head_dim;
    float* decay = shared + 2 * head_dim;

    const unsigned long long head_offset =
        (unsigned long long)head * head_dim;
    q[column] = (float)query[head_offset + column];
    k[column] = (float)key[head_offset + column];

    // The reference uses the safe lower-bound parameterization:
    // g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias)).
    const float rate = expf(a_log[head]);
    const float gate_input = rate *
        ((float)f_raw[head_offset + column] + dt_bias[head_offset + column]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate_input));
    decay[column] = expf(lower_bound * sigmoid_gate);
    __syncthreads();

    float* state_column = state +
        ((unsigned long long)head * head_dim * head_dim) + column;
    float hk = 0.0f;
#pragma unroll
    for (unsigned int row = 0; row < 128; ++row) {
        if (row < head_dim) {
            hk += decay[row] * state_column[row * head_dim] * k[row];
        }
    }

    const float beta = 1.0f /
        (1.0f + expf(-(float)beta_raw[head]));
    const float delta =
        ((float)value[head_offset + column] - hk) * beta;
    float q_dot = 0.0f;
#pragma unroll
    for (unsigned int row = 0; row < 128; ++row) {
        if (row < head_dim) {
            const float updated = decay[row] * state_column[row * head_dim]
                + k[row] * delta;
            state_column[row * head_dim] = updated;
            q_dot += updated * q[row];
        }
    }
    output[head_offset + column] =
        __float2bfloat16(q_dot * rsqrtf((float)head_dim));
}

// Compact padded MLA values [tokens, heads, qk_dim] to
// [tokens, heads, v_dim] while applying Ling's head-wise sigmoid gate.
extern "C" __global__ void ling_mla_compact_gate(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens,
    unsigned int heads,
    unsigned int input_head_dim,
    unsigned int value_dim
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = tokens * heads * value_dim;
    if (index >= total) return;
    const unsigned int value_index = index % value_dim;
    const unsigned int token_head = index / value_dim;
    const unsigned int head = token_head % heads;
    const unsigned int token = token_head / heads;
    const float gate_value = (float)gate[token * heads + head];
    const float multiplier = 1.0f / (1.0f + __expf(-gate_value));
    const unsigned int source =
        (token * heads + head) * input_head_dim + value_index;
    output[index] = __float2bfloat16((float)input[source] * multiplier);
}

extern "C" __global__ void ling_silu_mul_clamped(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int count,
    float limit
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float g = fminf(fmaxf((float)gate[index], -limit), limit);
    const float u = fminf(fmaxf((float)up[index], -limit), limit);
    output[index] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
}
