// SPDX-License-Identifier: AGPL-3.0-only

//! MLA branch of `prefill_attention_with_cache_skip`. Mistral4-style
//! 2-step prefill with the unabsorbed/MHA fused fallback path that
//! expands K/V via `wkv_b` and runs HDIM=128 FlashAttention. Extracted
//! from `cache_skip.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct CacheSkipMlaArgs {
    pub normed: DevicePtr,
    pub num_tokens: usize,
    pub n: u32,
    pub h: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub kv_dim: usize,
    pub eps: f32,
    pub bf16: usize,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// Run the cache-skip MLA prefill chain. Always returns the output
    /// pointer — caller short-circuits with `return Ok(out)`.
    pub(super) fn prefill_attention_cache_skip_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let CacheSkipMlaArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv,
            hd,
            kv_dim,
            eps,
            bf16,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_cache_skip_mla called without MLA config");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let use_tc = self.dense_gemm_tc_k.0 != 0;

        // Q: latent MLA, or Ling direct-Q.
        let q_latent = ctx.buffers.ssm_ba();
        let qg_out = ctx.buffers.qkv_output();
        let q_first_out = if mla.direct_q { qg_out } else { q_latent };
        let q_first_dim = if mla.direct_q { nq * hd } else { q_lora };
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wq_a,
                q_first_out,
                n,
                q_first_dim,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wq_a,
                q_first_out,
                n,
                q_first_dim,
                h,
                stream,
            )?;
        }
        if !mla.direct_q {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                n,
                q_lora,
                eps,
                stream,
            )?;
            if use_tc {
                ops::dense_gemm_tc(
                    ctx.gpu,
                    self.dense_gemm_tc_k,
                    q_latent,
                    &mla.wq_b,
                    qg_out,
                    n,
                    nq * hd,
                    q_lora,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    q_latent,
                    &mla.wq_b,
                    qg_out,
                    n,
                    nq * hd,
                    q_lora,
                    stream,
                )?;
            }
        }

        // KV latent + K_rope
        let kv_latent = ctx.buffers.expert_gate_out();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                n,
                kv_lora,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                n,
                kv_lora,
                h,
                stream,
            )?;
        }
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n,
            kv_lora,
            eps,
            stream,
        )?;
        let k_rope_buf = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_buf,
                n,
                mla_rope,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_buf,
                n,
                mla_rope,
                h,
                stream,
            )?;
        }

        // Q rope extract → RoPE
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            qg_out,
            q_rope_tmp,
            n,
            nq,
            hd,
            mla_nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        let rope_meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        ops::rope_yarn(
            ctx.gpu,
            if mla.direct_q {
                self.rope_yarn_interleaved_k
            } else {
                self.rope_yarn_k
            },
            q_rope_tmp,
            k_rope_buf,
            rope_meta.positions,
            n,
            nq,
            1,
            mla_rope,
            mla_rope,
            mla.yarn_inv_freq,
            super::super::helpers::yarn_rope_mscale(ctx.config),
            stream,
        )?;

        let mla_cache_dim = kv_lora + mla_rope;
        // Cache assembly (needed for decode regardless of path)
        let meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        let bs = kv_cache.block_size();
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_buf,
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs as u32,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;

        // Unabsorbed (MHA) prefill: expand K/V via wkv_b, use HDIM=128 FlashAttention
        let kv_expanded_dim = nkv * (mla_nope + mla_v_dim);
        let kv_expanded = ctx.buffers.ssm_deinterleaved();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            kv_latent,
            &mla.wkv_b,
            kv_expanded,
            n,
            kv_expanded_dim,
            kv_lora,
            stream,
        )?;
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        ops::mla_kv_assemble_batched(
            ctx.gpu,
            self.mla_kv_assemble_batched_k,
            kv_expanded,
            k_rope_buf,
            k_contiguous,
            v_contiguous,
            n,
            nkv,
            mla_nope,
            mla_v_dim,
            mla_rope,
            hd,
            nkv * (mla_nope + mla_v_dim),
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            qg_out,
            n,
            nq,
            hd,
            mla_nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        let attn_out_fb = ctx.buffers.attn_output();
        ops::prefill_attention_64(
            ctx.gpu,
            self.prefill_attn_64_k,
            qg_out,
            k_contiguous,
            v_contiguous,
            attn_out_fb,
            n,
            1,
            nq,
            nkv,
            hd,
            1.0f32 / (hd as f32).sqrt(),
            true,
            0,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("MLA flash_attn_64 fallback: {e}"))?;
        let o_input = if mla.direct_q {
            let gate = qg_out;
            let compact = ctx.buffers.expert_up_out();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                self.head_gate_weight
                    .as_ref()
                    .expect("Ling MLA requires head-wise g_proj"),
                gate,
                n,
                nq,
                h,
                stream,
            )?;
            let total = n * nq * mla_v_dim;
            spark_runtime::kernel_args::KernelLaunch::new(
                ctx.gpu,
                ctx.gpu.kernel("kda", "ling_mla_compact_gate")?,
            )
            .grid([total.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(attn_out_fb)
            .arg_ptr(gate)
            .arg_ptr(compact)
            .arg_u32(n)
            .arg_u32(nq)
            .arg_u32(hd)
            .arg_u32(mla_v_dim)
            .launch(stream)?;
            compact
        } else {
            attn_out_fb
        };
        let o_dim = if mla.direct_q {
            nq * mla_v_dim
        } else {
            nq * hd
        };

        // wo projection — output to qkv_output (norm_output aliases downstream)
        let o_out = ctx.buffers.qkv_output();
        if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                o_input,
                wo_nvfp4,
                o_out,
                n,
                h,
                o_dim,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                o_input,
                &mla.wo,
                o_out,
                n,
                h,
                o_dim,
                stream,
            )?;
        }
        Ok(o_out)
    }
}
