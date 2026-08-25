// SPDX-License-Identifier: AGPL-3.0-only

//! InclusionAI Bailing Hybrid / Ling 3.0 weight loader.

mod load_layers;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

pub struct BailingMtpModule {
    pub body: Box<dyn TransformerLayer>,
    pub enorm: DenseWeight,
    pub hnorm: DenseWeight,
    pub eh_proj: DenseWeight,
    pub final_norm: DenseWeight,
}

pub fn load_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<BailingMtpModule>> {
    if config.num_nextn_predict_layers == 0 {
        return Ok(None);
    }
    let prefix = format!("model.layers.{}", config.num_hidden_layers);
    if !store.contains(&format!("{prefix}.eh_proj.weight")) {
        return Ok(None);
    }
    Ok(Some(BailingMtpModule {
        body: load_layers::load_mtp_body(store, config, gpu)?,
        enorm: dense(store, &format!("{prefix}.enorm.weight"))?,
        hnorm: dense(store, &format!("{prefix}.hnorm.weight"))?,
        eh_proj: dense(store, &format!("{prefix}.eh_proj.weight"))?,
        final_norm: dense(store, &format!("{prefix}.final_layernorm.weight"))?,
    }))
}

pub struct BailingWeightLoader;

impl ModelWeightLoader for BailingWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.word_embeddings.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Ling NEXTN is a recursive full transformer module, wired separately.
        Ok(None)
    }

    fn kv_layer_dims(&self, config: &ModelConfig) -> Vec<(usize, usize)> {
        let attention_layers = config
            .layer_types
            .iter()
            .filter(|kind| **kind == atlas_core::config::LayerType::FullAttention)
            .count();
        vec![(1, config.kv_lora_rank + config.qk_rope_head_dim); attention_layers]
    }
}
