#!/usr/bin/env python3
"""Write rank R's tensor-parallel half of a DeepSeek-V2(-Lite) / DeepSeek-Coder-V2-Lite GGUF (llama.cpp arch deepseek2).

Attention (MLA): each rank keeps half of the heads. Old "legacy" files store the combined attn_kv_b; it is split into
the per-head attn_k_b / attn_v_b that llama.cpp's MLA path uses (so the memory cache holds one 576-wide latent per
token instead of 16 full heads: ~0.5 GB for 16k tokens instead of ~4.5 GB), then halved by heads.
attn_q keeps this rank's heads, attn_output the matching input columns; kv_a (the shared latent) is replicated.
MoE: every routed expert keeps half of its n_ff (gate/up rows, down input columns); the two shared experts are split
one per rank; the leading dense layer keeps half of n_ff. Router, norms and token_embd are replicated; output keeps
half of the vocabulary. Down-projection columns that don't fall on a quantization block boundary are re-quantized
(Q8_0, or REQUANT=Q5_0) so they can be cut. Run with LLAMA_EP_MODE=tp LLAMA_EP_VOCAB=split; the graph sums attention and FFN outputs.
usage: slice_deepseek_tp.py IN.gguf OUT.gguf RANK [RANK0_EXPERT_NFF]
RANK0_EXPERT_NFF (default n_ff_exp/2): node1 also runs the server with one core less, so giving it a smaller share of
every expert (and proportionally of the shared experts and the dense layer) balances the two Macs.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
import gguf.quants as gq  # noqa: E402

Q = gguf.GGMLQuantizationType


def cols(t, d, n_in, lo, hi):
    """input columns [lo, hi) of a quantized/float matrix (..., rows, row_bytes) covering n_in inputs; requantize if
    the cut is not on a block boundary. returns (data, type)"""
    blk, _ = gguf.GGML_QUANT_SIZES[t.tensor_type]
    if lo % blk == 0 and hi % blk == 0:
        bpb = d.shape[-1] // (n_in // blk)
        assert bpb * (n_in // blk) == d.shape[-1]
        return d[..., lo // blk * bpb:hi // blk * bpb], t.tensor_type
    f = gq.dequantize(d, t.tensor_type)[..., lo:hi]
    # Q8_0 (near-lossless) unless REQUANT=Q5_0 (smaller, measured no faster)
    rq = getattr(Q, os.environ.get("REQUANT", "Q8_0"))
    print(f"  {t.name}: {t.tensor_type.name} cut at {lo}:{hi} is not {blk}-aligned -> {rq.name}", flush=True)
    return gq.quantize(np.ascontiguousarray(f, dtype=np.float32), rq), rq


def main():
    src, dst, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    r = gguf.GGUFReader(src)
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    assert arch == "deepseek2", arch
    kv = lambda k, d=None: r.fields[f"{arch}.{k}"].contents() if f"{arch}.{k}" in r.fields else d  # noqa: E731
    n_head = kv("attention.head_count")
    n_rot = kv("rope.dimension_count")
    kv_lora = kv("attention.kv_lora_rank")
    legacy = kv("attention.key_length_mla") is None
    d_k = kv("attention.key_length") if legacy else kv("attention.key_length_mla")        # 192 = nope 128 + rope 64
    d_v = kv("attention.value_length") if legacy else kv("attention.value_length_mla")    # 128
    nope = d_k - n_rot
    n_ff, n_ff_exp, n_shared = kv("feed_forward_length"), kv("expert_feed_forward_length"), kv("expert_shared_count")
    hq = n_head // 2
    h0, h1 = rank * hq, (rank + 1) * hq
    e0 = int(sys.argv[4]) if len(sys.argv) > 4 else n_ff_exp // 2      # rank 0's share of every expert's n_ff
    assert e0 % 32 == 0 or e0 == n_ff_exp // 2
    f0 = round(n_ff * e0 / n_ff_exp / 32) * 32                            # same proportion of the leading dense layer
    e_lo, e_hi = (0, e0) if rank == 0 else (e0, n_ff_exp)
    f_lo, f_hi = (0, f0) if rank == 0 else (f0, n_ff)
    s_lo, s_hi = (0, e0 * n_shared) if rank == 0 else (e0 * n_shared, n_ff_exp * n_shared)  # graph: n_ff_exp_local * n_shared

    override = {f"{arch}.attention.head_count": hq, f"{arch}.attention.head_count_kv": 1,
                f"{arch}.attention.key_length": kv_lora + n_rot, f"{arch}.attention.value_length": kv_lora,
                f"{arch}.feed_forward_length": f_hi - f_lo, f"{arch}.expert_feed_forward_length": e_hi - e_lo}
    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    al = r.fields.get(gguf.Keys.General.ALIGNMENT)
    if al is not None:
        w.data_alignment = al.contents()
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, override.get(field.name, field.contents()), vt, sub_type=st)
    for k, v in override.items():                                         # keys the source file didn't have
        if k not in r.fields:
            w.add_key_value(k, v, gguf.GGUFValueType.UINT32)
    if legacy:
        w.add_key_value(f"{arch}.attention.key_length_mla", d_k, gguf.GGUFValueType.UINT32)
        w.add_key_value(f"{arch}.attention.value_length_mla", d_v, gguf.GGUFValueType.UINT32)
    w.add_key_value("ep.tp_rank", rank, gguf.GGUFValueType.UINT32)

    out = []
    for t in r.tensors:
        d, n, ty = t.data, t.name, t.tensor_type
        if n.endswith("attn_kv_b.weight"):                                   # legacy: split into MLA k_b / v_b, this rank's heads
            f = gq.dequantize(d, ty).reshape(n_head, nope + d_v, kv_lora)[h0:h1]
            k_b = np.ascontiguousarray(f[:, :nope, :].transpose(0, 2, 1))      # (heads, kv_lora, nope)  = ne {nope, kv_lora, heads}
            v_b = np.ascontiguousarray(f[:, nope:, :])                         # (heads, d_v, kv_lora)   = ne {kv_lora, d_v, heads}
            base = n[:-len("attn_kv_b.weight")]
            out.append((base + "attn_k_b.weight", gq.quantize(k_b.astype(np.float32), Q.Q8_0), Q.Q8_0))
            out.append((base + "attn_v_b.weight", gq.quantize(v_b.astype(np.float32), Q.Q8_0), Q.Q8_0))
            continue
        if n.endswith(("attn_k_b.weight", "attn_v_b.weight")):              # already-MLA files: heads are the last dim
            d = d[h0:h1]
        elif n.endswith("attn_q.weight"):
            d = d[h0 * d_k:h1 * d_k]
        elif n.endswith("attn_output.weight"):
            d, ty = cols(t, d, n_head * d_v, h0 * d_v, h1 * d_v)
        elif n.endswith(("attn_q_a.weight", "attn_q_b.weight")):
            raise SystemExit(f"{n}: q-lora models (full DeepSeek-V2) not handled")
        elif n.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
            d = d[:, e_lo:e_hi]
        elif n.endswith("ffn_down_exps.weight"):
            d, ty = cols(t, d, n_ff_exp, e_lo, e_hi)
        elif n.endswith(("ffn_gate_shexp.weight", "ffn_up_shexp.weight")):
            d = d[s_lo:s_hi]
        elif n.endswith("ffn_down_shexp.weight"):
            d, ty = cols(t, d, n_ff_exp * n_shared, s_lo, s_hi)
        elif n.endswith(("ffn_gate.weight", "ffn_up.weight")):
            d = d[f_lo:f_hi]
        elif n.endswith("ffn_down.weight"):
            d, ty = cols(t, d, n_ff, f_lo, f_hi)
        elif n == "output.weight":
            nv = d.shape[0]
            d = d[rank * nv // 2:(rank + 1) * nv // 2]
        elif n.endswith(".bias") and "norm" not in n and "exp_probs" not in n:
            raise SystemExit(f"{n}: biases not handled")
        out.append((n, d, ty))
    assert any(n == "output.weight" for n, _, _ in out), "tied output not handled"
    # one merged gate+up expert tensor per layer (FUSE=0 to keep them separate): llama.cpp then runs one expert
    # matmul instead of two (one activation quantization, one thread sync) and splits the result into gate | up
    if os.environ.get("FUSE", "1") == "1":
        by = {n: i for i, (n, _, _) in enumerate(out)}
        merged, drop = [], set()
        for n, d, ty in out:
            if n.endswith("ffn_gate_exps.weight"):
                un = n.replace("ffn_gate_exps", "ffn_up_exps")
                ud, uty = out[by[un]][1], out[by[un]][2]
                assert uty == ty, (n, ty, uty)
                merged.append((n.replace("ffn_gate_exps", "ffn_gate_up_exps"), np.concatenate([d, ud], axis=1), ty))
                drop.update({n, un})
        out = [x for x in out if x[0] not in drop] + merged

    for n, d, ty in out:
        w.add_tensor_info(n, d.shape, d.dtype, d.nbytes, ty)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = 0
    for n, d, ty in out:
        w.write_tensor_data(np.ascontiguousarray(d), tensor_endianess=r.endianess)
        total += d.nbytes
    w.close()
    print(f"wrote {dst}: {total / 2**30:.2f} GiB (rank {rank}: heads {n_head} -> {hq}, expert n_ff {n_ff_exp} -> {e_hi - e_lo}, "
          f"dense n_ff {n_ff} -> {f_hi - f_lo}, {'legacy kv_b -> MLA k_b/v_b' if legacy else 'MLA file'})")


if __name__ == "__main__":
    main()
