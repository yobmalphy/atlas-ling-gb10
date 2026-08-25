// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi Delta Attention layer used by InclusionAI Bailing Hybrid / Ling 3.0.

use anyhow::{Result, anyhow};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::{FfnComponent, ops};
use crate::weight_map::DenseWeight;

#[derive(Debug, Clone, Copy)]
pub struct KdaWeights {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub f_proj: DenseWeight,
    pub g_proj: DenseWeight,
    pub b_proj: DenseWeight,
    pub conv1d: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub o_norm: DenseWeight,
    pub o_proj: DenseWeight,
}

pub struct KdaLayer {
    layer_idx: usize,
    input_norm: DenseWeight,
    weights: KdaWeights,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    h_state_bytes: usize,
    conv_state_bytes: usize,
    dense_gemv_k: KernelHandle,
    conv1d_l2norm_k: KernelHandle,
    kda_decode_k: KernelHandle,
    sigmoid_gated_rms_norm_k: KernelHandle,
    rms_norm_residual_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    residual_add_k: KernelHandle,
}

impl KdaLayer {
    pub fn new(
        layer_idx: usize,
        input_norm: DenseWeight,
        weights: KdaWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            config.model_type == "bailing_hybrid",
            "KDA requires bailing_hybrid"
        );
        anyhow::ensure!(
            config.no_kda_lora,
            "Atlas Ling v1 requires direct KDA f/g projections"
        );
        let heads = config.linear_num_value_heads;
        let key_dim = config.linear_key_head_dim;
        let value_dim = config.linear_value_head_dim;
        let conv_dim = 2 * heads * key_dim + heads * value_dim;
        Ok(Self {
            layer_idx,
            input_norm,
            weights,
            post_attn_norm,
            ffn,
            h_state_bytes: heads * key_dim * value_dim * 4,
            conv_state_bytes: conv_dim * config.linear_conv_kernel_dim * 4,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            conv1d_l2norm_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            kda_decode_k: gpu.kernel("kda", "kda_decode")?,
            // Ling's FusedRMSNormGated uses activation="sigmoid".  The
            // common Mamba kernel is intentionally SiLU-gated, so sharing it
            // silently changes every KDA block's output.
            sigmoid_gated_rms_norm_k: gpu.kernel("kda", "kda_sigmoid_gated_rms_norm")?,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual_vanilla")?,
            residual_add_rms_norm_k: gpu.kernel("norm", "residual_add_rms_norm_vanilla")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
        })
    }

    fn project(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        n: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemv(gpu, self.dense_gemv_k, input, weight, output, n, h, stream)
    }

    fn decode_kda(
        &self,
        state: &mut SsmLayerState,
        normed: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let config = ctx.config;
        let h = config.hidden_size as u32;
        let heads = config.linear_num_value_heads;
        let dim = config.linear_value_head_dim;
        let flat = heads * dim;
        let qkvz = ctx.buffers.ssm_deinterleaved();
        let q = qkvz;
        let k = qkvz.offset(flat * 2);
        let v = qkvz.offset(flat * 4);
        let output_gate = qkvz.offset(flat * 6);
        self.project(
            ctx.gpu,
            normed,
            &self.weights.q_proj,
            q,
            flat as u32,
            h,
            stream,
        )?;
        self.project(
            ctx.gpu,
            normed,
            &self.weights.k_proj,
            k,
            flat as u32,
            h,
            stream,
        )?;
        self.project(
            ctx.gpu,
            normed,
            &self.weights.v_proj,
            v,
            flat as u32,
            h,
            stream,
        )?;
        self.project(
            ctx.gpu,
            normed,
            &self.weights.g_proj,
            output_gate,
            flat as u32,
            h,
            stream,
        )?;

        let gate_beta = ctx.buffers.ssm_ba();
        let f_raw = gate_beta;
        let beta_raw = gate_beta.offset(flat * 2);
        self.project(
            ctx.gpu,
            normed,
            &self.weights.f_proj,
            f_raw,
            flat as u32,
            h,
            stream,
        )?;
        self.project(
            ctx.gpu,
            normed,
            &self.weights.b_proj,
            beta_raw,
            heads as u32,
            h,
            stream,
        )?;

        let conv_out = ctx.buffers.ssm_qkvz();
        ops::conv1d_update_l2norm(
            ctx.gpu,
            self.conv1d_l2norm_k,
            state.conv_state,
            q,
            &self.weights.conv1d,
            conv_out,
            (flat * 3) as u32,
            config.linear_conv_kernel_dim as u32,
            1,
            (flat * 2) as u32,
            dim as u32,
            1e-6,
            stream,
        )?;
        let k_conv = conv_out.offset(flat * 2);
        let v_conv = conv_out.offset(flat * 4);
        let kda_out = ctx.buffers.attn_output();
        KernelLaunch::new(ctx.gpu, self.kda_decode_k)
            .grid([heads as u32, 1, 1])
            .block([dim as u32, 1, 1])
            .shared_mem((3 * dim * 4) as u32)
            .arg_ptr(state.h_state)
            .arg_ptr(conv_out)
            .arg_ptr(k_conv)
            .arg_ptr(v_conv)
            .arg_ptr(f_raw)
            .arg_ptr(beta_raw)
            .arg_ptr(self.weights.a_log.weight)
            .arg_ptr(self.weights.dt_bias.weight)
            .arg_ptr(kda_out)
            .arg_u32(heads as u32)
            .arg_u32(dim as u32)
            .arg_f32(config.kda_lower_bound)
            .launch(stream)?;
        let gated = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm(
            ctx.gpu,
            self.sigmoid_gated_rms_norm_k,
            kda_out,
            output_gate,
            &self.weights.o_norm,
            gated,
            heads as u32,
            dim as u32,
            dim as u32,
            config.rms_norm_eps as f32,
            dim as u32,
            stream,
        )?;
        let projected = ctx.buffers.norm_output();
        self.project(
            ctx.gpu,
            gated,
            &self.weights.o_proj,
            projected,
            h,
            flat as u32,
            stream,
        )?;
        Ok(projected)
    }
}

impl TransformerLayer for KdaLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow!("KDA expected SsmLayerState"))?;
        let h = ctx.config.hidden_size as u32;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        let kda_out = self.decode_kda(state, normed, ctx, stream)?;
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            kda_out,
            &self.post_attn_norm,
            normed,
            residual,
            1,
            h,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        let ffn_out = self.ffn.forward(normed, ctx, stream)?;
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, ffn_out, h, stream)?;
        if let Ok(dir) = std::env::var("ATLAS_KDA_DUMP") {
            ctx.gpu.synchronize(stream)?;
            let mut bf16 = vec![0u8; h as usize * 2];
            ctx.gpu.copy_d2h(hidden, &mut bf16)?;
            let mut f32_bytes = Vec::with_capacity(h as usize * 4);
            for pair in bf16.chunks_exact(2) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]) as u32;
                f32_bytes.extend_from_slice(&f32::from_bits(bits << 16).to_le_bytes());
            }
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                std::path::Path::new(&dir).join(format!("atlas_decode_L{}.bin", self.layer_idx)),
                f32_bytes,
            )?;
        }
        Ok(())
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // KDA has a recurrent FP32 matrix and convolution window.  A
        // speculative verify processes K tokens in place, but a rejected
        // draft commits only the prefix (for K=2, token 0).  Save the state
        // after every non-final token so commit_accepted_prefix can restore
        // the exact accepted boundary.  The default trait loop omitted these
        // copies, causing the rollback code to restore zero/stale pool slots
        // and corrupt all subsequent Ling tokens after the first rejection.
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let row = t * h * 2; // hidden and residual are BF16.
            self.decode(
                hidden.offset(row),
                residual.offset(row),
                state,
                kv_cache,
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;

            if t + 1 < num_tokens {
                let ssm = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow!("KDA expected SsmLayerState"))?;
                anyhow::ensure!(
                    t < ssm.h_state_intermediates.len() && t < ssm.conv_state_intermediates.len(),
                    "KDA MTP intermediate buffers not allocated (need at least {} h/conv, have {}/{})",
                    t + 1,
                    ssm.h_state_intermediates.len(),
                    ssm.conv_state_intermediates.len(),
                );
                ctx.gpu.copy_d2d_async(
                    ssm.h_state,
                    ssm.h_state_intermediates[t],
                    self.h_state_bytes,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm.conv_state,
                    ssm.conv_state_intermediates[t],
                    self.conv_state_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    fn is_ssm_layer(&self) -> bool {
        true
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16: false,
            h_prefill_stage: None,
        }))
    }
}
