// SPDX-License-Identifier: AGPL-3.0-only

//! Ling 3.0 recursive NEXTN proposer backed by physical layer 42.

mod draft;

use std::any::Any;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::speculative::ProposerState;
use crate::weight_loader::bailing::BailingMtpModule;
use crate::weight_map::DenseWeight;

const BAILING_PREFILL_CHUNK: usize = 512;

struct BailingMtpPrefillScratch {
    embed: DevicePtr,
    norm_embed: DevicePtr,
    norm_hidden: DevicePtr,
    concat: DevicePtr,
    positions: DevicePtr,
    slots: DevicePtr,
}

pub struct BailingMtpState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub last_num_drafted: usize,
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for BailingMtpState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct BailingMtpHead {
    module: BailingMtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    mtp_vocab_size: u32,
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    argmax_k: KernelHandle,
    prefill_scratch: BailingMtpPrefillScratch,
}

impl BailingMtpHead {
    pub fn new(
        module: BailingMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        let cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        let num_layers = config.num_attention_layers() + 1;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: cache_dim,
            num_layers,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let blocks = max_seq_len / kv_config.block_size + 1;
        let h = config.hidden_size;
        let c = BAILING_PREFILL_CHUNK;
        let bf16 = 2usize;
        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            kv_cache: Mutex::new(PagedKvCache::new(kv_config, blocks, gpu)?),
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            prefill_scratch: BailingMtpPrefillScratch {
                embed: gpu.alloc(c * h * bf16)?,
                norm_embed: gpu.alloc(c * h * bf16)?,
                norm_hidden: gpu.alloc(c * h * bf16)?,
                concat: gpu.alloc(c * 2 * h * bf16)?,
                positions: gpu.alloc(c * 4)?,
                slots: gpu.alloc(c * 8)?,
            },
        })
    }

    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<BailingMtpState> {
        Ok(BailingMtpState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut BailingMtpState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h as usize * 2;
        let embed = ctx.buffers.ssm_qkvz();
        ctx.gpu.copy_d2d_async(
            self.embed_tokens.weight.offset(token as usize * row_bytes),
            embed,
            row_bytes,
            stream,
        )?;
        let norm_embed = ctx.buffers.ssm_deinterleaved();
        let norm_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed,
            &self.module.enorm,
            norm_embed,
            1,
            h,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.module.hnorm,
            norm_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        let concat = ctx.buffers.logits();
        ctx.gpu
            .copy_d2d_async(norm_embed, concat, row_bytes, stream)?;
        ctx.gpu
            .copy_d2d_async(norm_hidden, concat.offset(row_bytes), row_bytes, stream)?;
        let hidden = ctx.buffers.hidden_states();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            concat,
            &self.module.eh_proj,
            hidden,
            h,
            h * 2,
            stream,
        )?;

        let mut cache = self.kv_cache.lock();
        let bs = cache.block_size();
        while state.block_table.len() < state.seq_len / bs + 1 {
            state.block_table.push(cache.alloc_block()?);
        }
        let block = state.block_table[state.seq_len / bs];
        let slot = block as i64 * bs as i64 + (state.seq_len % bs) as i64;
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let meta = pack_mtp_attn_meta(
            position.saturating_sub(1) as u32,
            slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta, meta_base, stream)?;
        let attn = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        if let Some(ids) = ctx.token_ids {
            ctx.gpu.copy_h2d_async(&token.to_le_bytes(), ids, stream)?;
        }
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(attn),
            profile: ctx.profile,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut disk_ids = Vec::new();
        let mut disk_last = vec![0u32; 1];
        self.module.body.decode(
            hidden,
            ctx.buffers.residual(),
            state.body_state.as_mut(),
            &mut cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_ids,
            &mut disk_last,
            &mtp_ctx,
            stream,
        )?;
        drop(cache);

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.module.final_norm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;
        let vocab = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.lm_head,
            logits,
            vocab,
            h,
            stream,
        )?;
        let drafted = draft::select_token(
            ctx.gpu,
            self.argmax_k,
            logits,
            vocab,
            grammar_bitmask,
            stream,
        )?;
        state.seq_len += 1;
        Ok(drafted)
    }

    /// Fill physical layer 42's own MLA cache over the prompt in GEMM-sized
    /// chunks. Row i pairs shifted token t[i+1] with the base model's
    /// post-final-norm hidden[i] at RoPE position i, matching Bailing's
    /// `BailingMoeV3ForCausalLM.forward` NEXTN branch.
    pub(super) fn prefill_drafter_impl(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let state = match state.as_any_mut().downcast_mut::<BailingMtpState>() {
            Some(state) => state,
            None => return Ok(0),
        };
        if state.seq_len != 0 || prompt_tokens.len() < 2 {
            return Ok(0);
        }

        let t0 = std::time::Instant::now();
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let rows_total = prompt_tokens.len() - 1;
        let mut cache = self.kv_cache.lock();
        let bs = cache.block_size();
        let blocks_needed = (rows_total - 1) / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(cache.alloc_block()?);
        }

        let mut done = 0usize;
        while done < rows_total {
            let c = (rows_total - done).min(BAILING_PREFILL_CHUNK);
            for r in 0..c {
                let token = prompt_tokens[done + r + 1] as usize;
                ctx.gpu.copy_d2d_async(
                    self.embed_tokens.weight.offset(token * h * bf16),
                    self.prefill_scratch.embed.offset(r * h * bf16),
                    h * bf16,
                    stream,
                )?;
            }
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                self.prefill_scratch.embed,
                &self.module.enorm,
                self.prefill_scratch.norm_embed,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hiddens.offset(done * h * bf16),
                &self.module.hnorm,
                self.prefill_scratch.norm_hidden,
                c as u32,
                h as u32,
                eps,
                stream,
            )?;
            for r in 0..c {
                ops::bf16_concat(
                    ctx.gpu,
                    self.bf16_concat_k,
                    self.prefill_scratch.norm_embed.offset(r * h * bf16),
                    self.prefill_scratch.norm_hidden.offset(r * h * bf16),
                    self.prefill_scratch.concat.offset(r * 2 * h * bf16),
                    h as u32,
                    stream,
                )?;
            }
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                self.prefill_scratch.concat,
                &self.module.eh_proj,
                ctx.buffers.hidden_states(),
                c as u32,
                h as u32,
                (2 * h) as u32,
                stream,
            )?;

            let positions: Vec<u32> = (0..c).map(|r| (done + r) as u32).collect();
            let position_bytes =
                unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, c * 4) };
            ctx.gpu
                .copy_h2d_async(position_bytes, self.prefill_scratch.positions, stream)?;
            let slots: Vec<i64> = (0..c)
                .map(|r| {
                    let row = done + r;
                    state.block_table[row / bs] as i64 * bs as i64 + (row % bs) as i64
                })
                .collect();
            let slot_bytes =
                unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, c * 8) };
            ctx.gpu
                .copy_h2d_async(slot_bytes, self.prefill_scratch.slots, stream)?;

            let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
            let meta = pack_mtp_attn_meta(
                done as u32,
                slots[0],
                (done + c) as i32,
                &state.block_table,
                ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
            )?;
            ctx.gpu.copy_h2d_async(&meta, meta_base, stream)?;
            let attn = AttnMetadataDev {
                positions: self.prefill_scratch.positions,
                positions_h: self.prefill_scratch.positions,
                positions_w: self.prefill_scratch.positions,
                slot: self.prefill_scratch.slots,
                seq_len: meta_base.offset(16),
                block_table: meta_base.offset(256),
                max_blocks_per_seq: state.block_table.len() as u32,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            };
            let mtp_ctx = ForwardContext {
                buffers: ctx.buffers,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: Some(attn),
                profile: ctx.profile,
                comm: None,
                graph_capture: false,
                gdn_exact_replay: false,
                token_ids: None,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            };
            let mut disk_ids = Vec::new();
            let mut disk_last = vec![0u32; ctx.config.num_hidden_layers + 1];
            self.module.body.prefill(
                ctx.buffers.hidden_states(),
                ctx.buffers.residual(),
                c,
                state.body_state.as_mut(),
                &mut cache,
                done,
                &mut state.block_table,
                &mut disk_ids,
                &mut disk_last,
                0,
                &mtp_ctx,
                stream,
            )?;
            // The H2D source arrays are host Vecs. Keep them alive until all
            // kernels in this chunk have consumed their metadata.
            ctx.gpu.synchronize(stream)?;
            done += c;
        }

        state.seq_len = rows_total;
        tracing::info!(
            "Ling NEXTN prompt context: {} rows ({} prompt tokens) in {:.1} ms",
            rows_total,
            prompt_tokens.len(),
            t0.elapsed().as_secs_f64() * 1e3,
        );
        Ok(rows_total)
    }
}
