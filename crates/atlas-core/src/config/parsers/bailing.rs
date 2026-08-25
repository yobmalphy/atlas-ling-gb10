// SPDX-License-Identifier: AGPL-3.0-only

//! InclusionAI Bailing Hybrid (Ling 3.0 Flash) config normalization.

use anyhow::{Context, Result, bail};

use crate::config::{LayerType, ModelConfig, finalize_config};

fn required_nonzero(raw: &serde_json::Value, key: &str) -> Result<usize> {
    let value = raw
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("bailing_hybrid config missing integer `{key}`"))?
        as usize;
    if value == 0 {
        bail!("bailing_hybrid config field `{key}` must be greater than zero");
    }
    Ok(value)
}

pub(crate) fn parse_bailing_hybrid(raw: &serde_json::Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("bailing_hybrid config.json is not an object")?;
    if object
        .get("q_lora_rank")
        .is_some_and(serde_json::Value::is_null)
    {
        object.insert("q_lora_rank".into(), serde_json::json!(0));
    }

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse bailing_hybrid config.json")?;
    config.weight_prefix = "model".to_string();
    config.nested_config = false;

    config.layer_group_size = required_nonzero(raw, "layer_group_size")?;
    config.short_conv_kernel_size = required_nonzero(raw, "short_conv_kernel_size")?;
    config.linear_conv_kernel_dim = config.short_conv_kernel_size;
    config.linear_num_key_heads = config.num_attention_heads;
    config.linear_key_head_dim = config.head_dim;
    config.linear_num_value_heads = config.num_attention_heads;
    config.linear_value_head_dim = config.head_dim;
    config.layer_types = (0..config.num_hidden_layers)
        .map(|layer_idx| {
            if (layer_idx + 1).is_multiple_of(config.layer_group_size) {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    config.full_attention_interval = config.layer_group_size;

    config.shared_expert_intermediate_size = config
        .num_shared_experts
        .saturating_mul(config.moe_shared_expert_intermediate_size);
    config.moe_router_groups = required_nonzero(raw, "n_group")?;
    config.moe_router_topk_groups = required_nonzero(raw, "topk_group")?;
    config.scoring_func = raw
        .get("scoring_func")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("sigmoid")
        .to_string();
    config.use_routing_bias = raw
        .get("moe_router_enable_expert_bias")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    config.mtp_num_hidden_layers = config.num_nextn_predict_layers;
    config.num_mtp_modules = config.num_nextn_predict_layers;
    config.mtp_transformer_layers = usize::from(config.num_nextn_predict_layers > 0);
    config.attn_gated = !config.gated_attention_proj_granularity_type.is_empty();

    if config.qk_head_dim != config.qk_nope_head_dim + config.qk_rope_head_dim {
        bail!(
            "bailing_hybrid qk_head_dim ({}) must equal qk_nope_head_dim ({}) + qk_rope_head_dim ({})",
            config.qk_head_dim,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
        );
    }
    if config.expert_swiglu_limit_list.len() != config.num_hidden_layers
        || config.share_expert_swiglu_limit_list.len() != config.num_hidden_layers
    {
        bail!(
            "bailing_hybrid SwiGLU limit arrays must each have num_hidden_layers ({}) entries",
            config.num_hidden_layers,
        );
    }

    finalize_config(&mut config, raw)?;
    Ok(config)
}
