// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{BailingMtpHead, BailingMtpState};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};

pub(super) fn select_token(
    gpu: &dyn GpuBackend,
    argmax: KernelHandle,
    logits: DevicePtr,
    vocab: u32,
    mask: Option<&[i32]>,
    stream: u64,
) -> Result<u32> {
    if let Some(mask) = mask {
        let mut raw = vec![0u8; vocab as usize * 2];
        gpu.copy_d2h(logits, &mut raw)?;
        let mut best = (0u32, f32::NEG_INFINITY);
        for token in 0..vocab as usize {
            if token / 32 >= mask.len() || mask[token / 32] & (1 << (token % 32)) == 0 {
                continue;
            }
            let bits = u16::from_le_bytes([raw[token * 2], raw[token * 2 + 1]]);
            let value = f32::from_bits((bits as u32) << 16);
            if value > best.1 {
                best = (token as u32, value);
            }
        }
        return Ok(best.0);
    }
    let out = gpu.alloc(4)?;
    ops::argmax_bf16(gpu, argmax, logits, out, vocab, stream)?;
    let mut raw = [0u8; 4];
    gpu.copy_d2h(out, &mut raw)?;
    gpu.free(out)?;
    Ok(u32::from_le_bytes(raw))
}

impl DraftProposer for BailingMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<BailingMtpState>()
            .ok_or_else(|| anyhow::anyhow!("invalid Ling MTP proposer state"))?;
        let mut token = last_token;
        let mut hidden = target_hidden;
        let mut drafts = Vec::with_capacity(num_drafts);
        for i in 0..num_drafts {
            token = self.forward_one(
                token,
                hidden,
                position + i,
                state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            drafts.push(token);
            // Recursive NEXTN consumes the preceding layer-42 final-norm
            // output, matching SGLang's hidden_states return contract.
            hidden = ctx.buffers.norm_output();
        }
        state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<BailingMtpState>()
            .ok_or_else(|| anyhow::anyhow!("invalid Ling MTP proposer state"))?;
        state.seq_len = state
            .seq_len
            .saturating_sub(state.last_num_drafted.max(1).saturating_sub(accepted));
        Ok(())
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.prefill_drafter_impl(prompt_tokens, hiddens, state, ctx, stream)
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<BailingMtpState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<BailingMtpState>()
            .ok_or_else(|| anyhow::anyhow!("invalid Ling MTP proposer state"))?;
        if !state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&state.block_table);
            state.block_table.clear();
        }
        state.seq_len = 0;
        Ok(())
    }
}
