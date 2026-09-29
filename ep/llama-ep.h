#pragma once

// Expert parallelism across two machines for MoE models (gpt-oss).
// Each rank owns a contiguous range of experts and only computes those; partial MoE
// outputs are summed with a direct peer-to-peer exchange after every MoE block.
// Enabled by environment variables:
//   LLAMA_EP_RANK   0 or 1
//   LLAMA_EP_ADDR   host:port  (rank 0 listens on it, rank 1 connects to it)
//   LLAMA_EP_RANGE  lo-hi      (owned experts, inclusive)

#include <cstdint>

struct ggml_context;
struct ggml_tensor;
struct llama_batch;

// Mirroring (leader only, when LLAMA_EP_CTRL=host:port is set): every state-changing libllama call
// is streamed to the follower before running locally, so both ranks stay in lockstep.
enum llama_ep_op : int32_t {
    LLAMA_EP_OP_DECODE = 1, LLAMA_EP_OP_CLEAR, LLAMA_EP_OP_SEQ_RM, LLAMA_EP_OP_SEQ_CP,
    LLAMA_EP_OP_SEQ_KEEP, LLAMA_EP_OP_SEQ_ADD, LLAMA_EP_OP_SEQ_DIV, LLAMA_EP_OP_WARMUP,
};
bool llama_ep_mirroring();
void llama_ep_mirror_decode(const llama_batch & batch, int n_embd);
void llama_ep_mirror_op(llama_ep_op op, int32_t a0 = 0, int32_t a1 = 0, int32_t a2 = 0, int32_t a3 = 0);

bool llama_ep_active();
// LLAMA_EP_MODE=tp: every rank holds half of every expert (n_ff split), no expert masking
bool llama_ep_tp();
// LLAMA_EP_ATTN=split: attention heads are split across ranks too (sliced files), partial outputs summed after wo
bool llama_ep_attn_split();
// number of experts stored in the local model file (only owned ones when LLAMA_EP_SLICED=1)
int  llama_ep_n_local(int n_expert);
int  llama_ep_rank();

// zero routing weights of experts this rank does not own
ggml_tensor * llama_ep_mask_weights(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * selected_experts);
// mark non-owned expert picks as -1 so mul_mat_id / add_id skip them (local ids when sliced)
ggml_tensor * llama_ep_remap_ids(ggml_context * ctx, ggml_tensor * selected_experts);
// LLAMA_EP_VOCAB=split: each rank's output layer scores half the vocabulary
int  llama_ep_n_vocab_local(int n_vocab);
// turn this rank's half-vocab logits into full-size logits: own half exact, peer's top-K filled in, rest -inf
ggml_tensor * llama_ep_logits_gather(ggml_context * ctx, ggml_tensor * logits_local, int n_vocab);
// sum a tensor across both ranks (in place semantics: returns the summed tensor)
ggml_tensor * llama_ep_allreduce(ggml_context * ctx, ggml_tensor * t);
