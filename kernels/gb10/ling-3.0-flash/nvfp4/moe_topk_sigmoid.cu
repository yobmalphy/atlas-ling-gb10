// SPDX-License-Identifier: AGPL-3.0-only

// Ling 3.0 grouped sigmoid router. Selection is restricted to the best four
// of eight 64-expert groups, scored by each group's two best biased scores.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32
#define NUM_GROUPS 8
#define TOP_GROUPS 4

__device__ __forceinline__ void ling_grouped_route(
    const __nv_bfloat16* gate,
    const float* bias,
    unsigned int* indices,
    float* weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor,
    float* sigmoid_scores,
    float* selection,
    float* top_values,
    unsigned int* top_indices,
    float* warp_values,
    unsigned int* warp_indices
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int actual_n = min(num_experts, (unsigned int)MAX_EXPERTS);
    const unsigned int top_k_c = min(top_k, (unsigned int)MAX_TOP_K);

    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        const float score = 1.0f / (1.0f + __expf(-(float)gate[i]));
        sigmoid_scores[i] = score;
        selection[i] = score + bias[i];
    }
    __syncthreads();

    // The Ling checkpoint contract is exactly 512 experts in 8 groups.
    if (tid == 0) {
        float group_scores[NUM_GROUPS];
        bool selected_groups[NUM_GROUPS] = {};
        const unsigned int group_size = actual_n / NUM_GROUPS;
        for (unsigned int group = 0; group < NUM_GROUPS; ++group) {
            float first = -1e30f;
            float second = -1e30f;
            for (unsigned int j = 0; j < group_size; ++j) {
                const float value = selection[group * group_size + j];
                if (value > first) {
                    second = first;
                    first = value;
                } else if (value > second) {
                    second = value;
                }
            }
            group_scores[group] = first + second;
        }
        for (unsigned int keep = 0; keep < TOP_GROUPS; ++keep) {
            unsigned int best = 0;
            float best_value = -1e30f;
            for (unsigned int group = 0; group < NUM_GROUPS; ++group) {
                if (!selected_groups[group] && group_scores[group] > best_value) {
                    best_value = group_scores[group];
                    best = group;
                }
            }
            selected_groups[best] = true;
        }
        for (unsigned int group = 0; group < NUM_GROUPS; ++group) {
            if (!selected_groups[group]) {
                for (unsigned int j = 0; j < group_size; ++j) {
                    selection[group * group_size + j] = -1e30f;
                }
            }
        }
    }
    __syncthreads();

    for (unsigned int rank = 0; rank < top_k_c; ++rank) {
        float local_value = -1e30f;
        unsigned int local_index = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            const float value = selection[i];
            if (value > local_value || (value == local_value && i < local_index)) {
                local_value = value;
                local_index = i;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffff, local_value, offset);
            const unsigned int other_index = __shfl_down_sync(0xffffffff, local_index, offset);
            if (other_value > local_value ||
                (other_value == local_value && other_index < local_index)) {
                local_value = other_value;
                local_index = other_index;
            }
        }
        if (lane == 0) {
            warp_values[warp] = local_value;
            warp_indices[warp] = local_index;
        }
        __syncthreads();
        if (tid == 0) {
            float best_value = warp_values[0];
            unsigned int best_index = warp_indices[0];
            for (unsigned int w = 1; w < 8; ++w) {
                if (warp_values[w] > best_value ||
                    (warp_values[w] == best_value && warp_indices[w] < best_index)) {
                    best_value = warp_values[w];
                    best_index = warp_indices[w];
                }
            }
            top_indices[rank] = best_index;
            selection[best_index] = -1e30f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int rank = 0; rank < top_k_c; ++rank) {
            top_values[rank] = sigmoid_scores[top_indices[rank]];
            sum += top_values[rank];
        }
        for (unsigned int rank = 0; rank < top_k_c; ++rank) {
            const float value = normalize && sum > 1e-20f
                ? top_values[rank] / sum
                : top_values[rank];
            indices[rank] = top_indices[rank];
            weights[rank] = value * scaling_factor;
        }
    }
}

#define LING_SHARED_ROUTER() \
    __shared__ float sigmoid_scores[MAX_EXPERTS]; \
    __shared__ float selection[MAX_EXPERTS]; \
    __shared__ float top_values[MAX_TOP_K]; \
    __shared__ unsigned int top_indices[MAX_TOP_K]; \
    __shared__ float warp_values[8]; \
    __shared__ unsigned int warp_indices[8]

extern "C" __global__ void moe_topk_sigmoid(
    const __nv_bfloat16* gate,
    const float* bias,
    unsigned int* indices,
    float* weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    LING_SHARED_ROUTER();
    ling_grouped_route(gate, bias, indices, weights, num_experts, top_k,
        normalize, scaling_factor, sigmoid_scores, selection, top_values,
        top_indices, warp_values, warp_indices);
}

extern "C" __global__ void moe_topk_sigmoid_batched(
    const __nv_bfloat16* gate,
    const float* bias,
    unsigned int* indices,
    float* weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    LING_SHARED_ROUTER();
    const unsigned int token = blockIdx.x;
    ling_grouped_route(gate + token * num_experts, bias,
        indices + token * top_k, weights + token * top_k, num_experts, top_k,
        normalize, scaling_factor, sigmoid_scores, selection, top_values,
        top_indices, warp_values, warp_indices);
}
