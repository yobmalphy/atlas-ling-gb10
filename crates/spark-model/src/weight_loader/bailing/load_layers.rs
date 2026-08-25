// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::qwen3_attention::{HeadGateActivation, MlaWeights};
use crate::layers::{
    DenseFfnLayer, FfnComponent, KdaLayer, KdaWeights, MoeLayer, Qwen3AttentionLayer,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_auto,
    quantize_to_nvfp4, quantized_v2,
};

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    ensure!(
        config.model_type == "bailing_hybrid",
        "expected bailing_hybrid"
    );
    let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let inv_freq = plain_inv_freq(config, gpu)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    let mut attention_idx = 0usize;

    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let ffn = if i < config.first_k_dense_replace {
            load_dense_ffn(store, gpu, &lp)?
        } else {
            load_moe_ffn(store, config, gpu, &lp, absmax, quantize, stream, false)?
        };

        let layer: Box<dyn TransformerLayer> = match config.layer_types[i] {
            LayerType::LinearAttention => Box::new(load_kda(
                store, config, gpu, &lp, input_norm, post_norm, ffn,
            )?),
            LayerType::FullAttention => {
                let kv_dtype = layer_kv_dtypes
                    .get(attention_idx)
                    .copied()
                    .unwrap_or(KvCacheDtype::Bf16);
                let layer = load_mla(
                    store,
                    config,
                    gpu,
                    &lp,
                    input_norm,
                    post_norm,
                    ffn,
                    attention_idx,
                    kv_dtype,
                    inv_freq,
                )?;
                attention_idx += 1;
                Box::new(layer)
            }
            other => anyhow::bail!("Ling layer {i}: unsupported layer type {other:?}"),
        };
        layers.push(layer);
    }
    Ok(layers)
}

fn null_dense_ffn() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

fn load_dense_ffn(store: &WeightStore, gpu: &dyn GpuBackend, lp: &str) -> Result<FfnComponent> {
    let mut layer = DenseFfnLayer::new(null_dense_ffn(), gpu)?;
    layer.set_bf16_weights(
        dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
        dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
        dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
    );
    Ok(FfnComponent::Dense(layer))
}

#[allow(clippy::too_many_arguments)]
fn load_moe_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    absmax: spark_runtime::gpu::KernelHandle,
    quantize: spark_runtime::gpu::KernelHandle,
    stream: u64,
    bf16_routed: bool,
) -> Result<FfnComponent> {
    let mlp = format!("{lp}.mlp");
    let null_expert = ExpertWeight {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
    };
    let experts = if bf16_routed {
        vec![null_expert; config.num_experts]
    } else {
        (0..config.num_experts)
            .map(|e| {
                let ep = format!("{mlp}.experts.{e}");
                Ok(ExpertWeight {
                    gate_proj: quantized_v2(store, &format!("{ep}.gate_proj"), gpu)?,
                    up_proj: quantized_v2(store, &format!("{ep}.up_proj"), gpu)?,
                    down_proj: quantized_v2(store, &format!("{ep}.down_proj"), gpu)?,
                })
            })
            .collect::<Result<Vec<_>>>()?
    };
    let shared = format!("{mlp}.shared_experts");
    let shared_gate = dense_auto(store, &format!("{shared}.gate_proj.weight"), gpu)?;
    let shared_up = dense_auto(store, &format!("{shared}.up_proj.weight"), gpu)?;
    let shared_down = dense_auto(store, &format!("{shared}.down_proj.weight"), gpu)?;
    let si = config.shared_expert_intermediate_size;
    let h = config.hidden_size;
    let shared_expert = ExpertWeight {
        gate_proj: quantize_to_nvfp4(&shared_gate, si, h, gpu, absmax, quantize, stream)?,
        up_proj: quantize_to_nvfp4(&shared_up, si, h, gpu, absmax, quantize, stream)?,
        down_proj: quantize_to_nvfp4(&shared_down, h, si, gpu, absmax, quantize, stream)?,
    };
    let weights = MoeWeights {
        gate: dense(store, &format!("{mlp}.gate.weight"))?,
        shared_expert,
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias: Some(dense(store, &format!("{mlp}.gate.expert_bias"))?),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    if bf16_routed {
        let mut gates = Vec::with_capacity(config.num_experts);
        let mut ups = Vec::with_capacity(config.num_experts);
        let mut downs = Vec::with_capacity(config.num_experts);
        for e in 0..config.num_experts {
            let ep = format!("{mlp}.experts.{e}");
            gates.push(dense_auto(store, &format!("{ep}.gate_proj.weight"), gpu)?);
            ups.push(dense_auto(store, &format!("{ep}.up_proj.weight"), gpu)?);
            downs.push(dense_auto(store, &format!("{ep}.down_proj.weight"), gpu)?);
        }
        layer.set_bf16_experts(
            &gates,
            &ups,
            &downs,
            shared_gate.weight,
            shared_up.weight,
            shared_down.weight,
            gpu,
        )?;
    } else {
        layer.set_bf16_shared_expert(shared_gate, shared_up, shared_down)?;
    }
    let layer_idx = lp
        .rsplit('.')
        .next()
        .and_then(|v| v.parse::<usize>().ok())
        .context("Ling MoE layer prefix must end in an index")?;
    layer.set_swiglu_limits(
        config
            .expert_swiglu_limit_list
            .get(layer_idx)
            .copied()
            .unwrap_or(0.0),
        config
            .share_expert_swiglu_limit_list
            .get(layer_idx)
            .copied()
            .unwrap_or(0.0),
    );
    Ok(FfnComponent::Moe(layer))
}

pub(super) fn load_mtp_body(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Box<dyn TransformerLayer>> {
    let i = config.num_hidden_layers;
    let lp = format!("model.layers.{i}");
    ensure!(
        store.contains(&format!("{lp}.eh_proj.weight")),
        "Ling MTP tensors missing"
    );
    let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let ffn = load_moe_ffn(
        store,
        config,
        gpu,
        &lp,
        absmax,
        quantize,
        gpu.default_stream(),
        true,
    )?;
    Ok(Box::new(load_mla(
        store,
        config,
        gpu,
        &lp,
        dense(store, &format!("{lp}.input_layernorm.weight"))?,
        dense(store, &format!("{lp}.post_attention_layernorm.weight"))?,
        ffn,
        config.num_attention_layers(),
        KvCacheDtype::Bf16,
        plain_inv_freq(config, gpu)?,
    )?))
}

fn load_kda(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    input_norm: DenseWeight,
    post_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<KdaLayer> {
    let ap = format!("{lp}.attention");
    let rows = config.num_attention_heads * config.head_dim;
    let kernel = config.short_conv_kernel_size;
    let combined = gpu.alloc(3 * rows * kernel * 2)?;
    for (slot, name) in ["q_conv1d", "k_conv1d", "v_conv1d"].iter().enumerate() {
        let source = dense(store, &format!("{ap}.{name}.weight"))?;
        gpu.copy_d2d(
            source.weight,
            combined.offset(slot * rows * kernel * 2),
            rows * kernel * 2,
        )?;
    }
    let weights = KdaWeights {
        q_proj: dense(store, &format!("{ap}.q_proj.weight"))?,
        k_proj: dense(store, &format!("{ap}.k_proj.weight"))?,
        v_proj: dense(store, &format!("{ap}.v_proj.weight"))?,
        f_proj: dense(store, &format!("{ap}.f_proj.weight"))?,
        g_proj: dense(store, &format!("{ap}.g_proj.weight"))?,
        b_proj: dense(store, &format!("{ap}.b_proj.weight"))?,
        conv1d: DenseWeight { weight: combined },
        a_log: dense(store, &format!("{ap}.A_log"))?,
        dt_bias: dense(store, &format!("{ap}.dt_bias"))?,
        o_norm: dense(store, &format!("{ap}.o_norm.weight"))?,
        o_proj: dense(store, &format!("{ap}.o_proj.weight"))?,
    };
    let layer_idx = lp
        .rsplit('.')
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .context("Ling KDA layer prefix must end in an index")?;
    KdaLayer::new(layer_idx, input_norm, weights, post_norm, ffn, config, gpu)
}

#[allow(clippy::too_many_arguments)]
fn load_mla(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    input_norm: DenseWeight,
    post_norm: DenseWeight,
    ffn: FfnComponent,
    attention_idx: usize,
    kv_dtype: KvCacheDtype,
    inv_freq: DevicePtr,
) -> Result<Qwen3AttentionLayer> {
    let ap = format!("{lp}.attention");
    let wkv_merged = dense_auto(store, &format!("{ap}.kv_a_proj_with_mqa.weight"), gpu)?;
    let wkv_a = DenseWeight {
        weight: wkv_merged.weight,
    };
    let wkv_a_rope = DenseWeight {
        weight: wkv_merged
            .weight
            .offset(config.kv_lora_rank * config.hidden_size * 2),
    };
    let wkv_b = dense_auto(store, &format!("{ap}.kv_b_proj.weight"), gpu)?;
    let (w_uk_t, w_uv) = split_kv_up(&wkv_b, config, gpu)?;
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let mla = MlaWeights {
        direct_q: true,
        wq_a: dense_auto(store, &format!("{ap}.q_proj.weight"), gpu)?,
        wq_a_nvfp4: None,
        wq_a_fp8: None,
        wq_b: null,
        wq_b_nvfp4: None,
        wq_b_fp8: None,
        q_a_norm: null,
        wkv_a,
        wkv_a_nvfp4: None,
        wkv_a_fp8: None,
        wkv_b,
        kv_a_norm: dense(store, &format!("{ap}.kv_a_layernorm.weight"))?,
        wkv_a_rope,
        wkv_a_merged: wkv_merged,
        wo: dense_auto(store, &format!("{ap}.dense.weight"), gpu)?,
        wo_nvfp4: None,
        wo_a: null,
        wo_a_nvfp4: None,
        wo_a_fp8: None,
        wo_b: null,
        wo_b_nvfp4: None,
        wo_b_fp8: None,
        w_uk_t,
        w_uv,
        wq_b_rope: null,
        w_qk_absorbed: null,
        w_uk_block_diag: null,
        w_uv_block_diag: null,
        yarn_inv_freq: inv_freq,
        main_inv_freq: inv_freq,
        q_lora_rank: 0,
        kv_lora_rank: config.kv_lora_rank,
        o_lora_rank: 0,
        nope: config.qk_nope_head_dim,
        rope: config.qk_rope_head_dim,
        v_dim: config.v_head_dim,
        compressor: None,
        attn_sink: DevicePtr::NULL,
    };
    let attn = AttentionWeights {
        q_proj: null,
        k_proj: null,
        v_proj: null,
        o_proj: QuantizedWeight::null(),
        q_norm: null,
        k_norm: null,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_norm,
        ffn,
        attention_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        0,
        config,
    )?;
    layer.set_dimension_overrides(
        config.qk_head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
    );
    layer.set_mla_weights(mla);
    layer.set_head_gate_weight(
        dense(store, &format!("{ap}.g_proj.weight"))?,
        HeadGateActivation::Sigmoid,
    );
    Ok(layer)
}

fn split_kv_up(
    weight: &DenseWeight,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, DenseWeight)> {
    let heads = config.num_attention_heads;
    let kv = config.kv_lora_rank;
    let nope = config.qk_nope_head_dim;
    let value = config.v_head_dim;
    let mut source = vec![0u8; heads * (nope + value) * kv * 2];
    gpu.copy_d2h(weight.weight, &mut source)?;
    let mut uk = vec![0u8; heads * kv * nope * 2];
    let mut uv = vec![0u8; heads * value * kv * 2];
    for head in 0..heads {
        for row in 0..nope {
            for latent in 0..kv {
                let src = ((head * (nope + value) + row) * kv + latent) * 2;
                let dst = ((head * kv + latent) * nope + row) * 2;
                uk[dst..dst + 2].copy_from_slice(&source[src..src + 2]);
            }
        }
        let src = (head * (nope + value) + nope) * kv * 2;
        let dst = head * value * kv * 2;
        uv[dst..dst + value * kv * 2].copy_from_slice(&source[src..src + value * kv * 2]);
    }
    let uk_ptr = gpu.alloc(uk.len())?;
    let uv_ptr = gpu.alloc(uv.len())?;
    gpu.copy_h2d(&uk, uk_ptr)?;
    gpu.copy_h2d(&uv, uv_ptr)?;
    Ok((
        DenseWeight { weight: uk_ptr },
        DenseWeight { weight: uv_ptr },
    ))
}

fn plain_inv_freq(config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let dim = config.qk_rope_head_dim;
    let values: Vec<f32> = (0..dim / 2)
        .map(|i| (config.rope_theta as f32).powf(-((2 * i) as f32) / dim as f32))
        .collect();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let ptr = gpu
        .alloc(bytes.len())
        .context("Ling RoPE inv_freq allocation")?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}
