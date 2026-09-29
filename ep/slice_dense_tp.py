#!/usr/bin/env python3
"""Write rank R's tensor-parallel half of a llama.cpp model (Megatron-style): dense (Gemma 3, Qwen2) or MoE (Mellum).

Per layer: ffn_gate/ffn_up keep n_ff rows [lo, hi); ffn_down keeps the matching input columns (block aligned);
attention keeps this rank's query heads and kv heads (attn_output keeps the matching input columns).
Norms are replicated. A half-vocabulary output.weight is added (from output.weight or the tied token_embd),
token_embd stays whole for input lookups. Run with LLAMA_EP_MODE=tp LLAMA_EP_VOCAB=split; the graph sums
attention and FFN outputs across ranks before the post-norms.
usage: slice_dense_tp.py IN.gguf OUT.gguf RANK [FF_SPLIT]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def col_slice(t, d, n_in, lo, hi):
    """keep input columns [lo, hi) of a (rows, row_bytes) quantized matrix whose rows cover n_in inputs"""
    blk, _ = gguf.GGML_QUANT_SIZES[t.tensor_type]
    assert lo % blk == 0 and hi % blk == 0, f"{t.name}: split not aligned to {blk}-wide blocks"
    bpb = d.shape[-1] // (n_in // blk)
    assert bpb * (n_in // blk) == d.shape[-1]
    return d[..., lo // blk * bpb:hi // blk * bpb]


def main():
    src, dst, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    kv = lambda k, d=None: reader.fields[f"{arch}.{k}"].contents() if f"{arch}.{k}" in reader.fields else d  # noqa: E731
    n_embd, n_head, n_head_kv = kv("embedding_length"), kv("attention.head_count"), kv("attention.head_count_kv")
    d_k = kv("attention.key_length", n_embd // n_head)      # files without key_length (e.g. Qwen2) use n_embd / n_head,
    d_v = kv("attention.value_length", n_embd // n_head)    # which would be wrong after halving the heads: write it explicitly
    n_ff = kv("feed_forward_length", 0)
    n_ff_exp = kv("expert_feed_forward_length", None)       # MoE models: every expert's n_ff is split too
    if isinstance(n_ff, list):
        n_ff = n_ff[0]
    if isinstance(n_ff_exp, list):
        assert len(set(n_ff_exp)) == 1, "per-layer expert sizes differ"
        n_ff_exp_list, n_ff_exp = True, n_ff_exp[0]
    else:
        n_ff_exp_list = False
    split = int(sys.argv[4]) if len(sys.argv) > 4 else n_ff // 2
    f_lo, f_hi = (0, split) if rank == 0 else (split, n_ff)
    if n_ff_exp:
        e_lo, e_hi = (0, n_ff_exp // 2) if rank == 0 else (n_ff_exp // 2, n_ff_exp)
    else:
        e_lo = e_hi = 0
    hq, hkv = n_head // 2, n_head_kv // 2
    assert n_head % 2 == 0 and n_head_kv % 2 == 0
    override = {f"{arch}.attention.head_count": hq, f"{arch}.attention.head_count_kv": hkv}
    if n_ff and not n_ff_exp:
        override[f"{arch}.feed_forward_length"] = f_hi - f_lo
    if n_ff_exp:
        h = e_hi - e_lo
        override[f"{arch}.expert_feed_forward_length"] = ([h] * len(reader.fields[f"{arch}.expert_feed_forward_length"].contents())
                                                          if n_ff_exp_list else h)

    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    align = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, override.get(field.name, field.contents()), vt, sub_type=st)
    writer.add_key_value("ep.tp_rank", rank, gguf.GGUFValueType.UINT32)
    for key, val in (("attention.key_length", d_k), ("attention.value_length", d_v)):
        if f"{arch}.{key}" not in reader.fields:
            writer.add_key_value(f"{arch}.{key}", val, gguf.GGUFValueType.UINT32)

    names = {t.name for t in reader.tensors}
    out = []
    for t in reader.tensors:
        d, n = t.data, t.name
        if n.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):   # (expert, n_ff_exp rows, row bytes)
            d = d[:, e_lo:e_hi]
        elif n.endswith("ffn_down_exps.weight"):                           # (expert, n_embd rows, bytes over n_ff_exp)
            d = col_slice(t, d, n_ff_exp, e_lo, e_hi)
        elif n.endswith(("ffn_gate_exps.bias", "ffn_up_exps.bias", "ffn_down_exps.bias", "ffn_gate_shexp.weight")):
            raise SystemExit(f"{n}: expert biases / shared experts not handled yet")
        elif n.endswith(("ffn_gate.weight", "ffn_up.weight")):
            d = d[f_lo:f_hi]
        elif n.endswith("ffn_down.weight"):
            d = col_slice(t, d, n_ff, f_lo, f_hi)
        elif n.endswith("attn_q.weight"):
            d = d[rank * hq * d_k:(rank + 1) * hq * d_k]
        elif n.endswith("attn_k.weight"):
            d = d[rank * hkv * d_k:(rank + 1) * hkv * d_k]
        elif n.endswith("attn_v.weight"):
            d = d[rank * hkv * d_v:(rank + 1) * hkv * d_v]
        elif n.endswith("attn_output.weight"):
            d = col_slice(t, d, n_head * d_v, rank * hq * d_v, (rank + 1) * hq * d_v)
        elif n.endswith("attn_q.bias"):
            d = d[rank * hq * d_k:(rank + 1) * hq * d_k]
        elif n.endswith("attn_k.bias"):
            d = d[rank * hkv * d_k:(rank + 1) * hkv * d_k]
        elif n.endswith("attn_v.bias"):
            d = d[rank * hkv * d_v:(rank + 1) * hkv * d_v]
        elif n.endswith("attn_output.bias"):                            # added once, after the sum
            d = d if rank == 0 else np.zeros_like(d)
        elif n.endswith(("ffn_gate.bias", "ffn_up.bias", "ffn_down.bias")):
            raise SystemExit(f"{n}: FFN biases not handled yet")
        elif n == "output.weight":
            nv = d.shape[0]
            d = d[rank * nv // 2:(rank + 1) * nv // 2]
        out.append((n, d, t.tensor_type))
        if n == "token_embd.weight" and "output.weight" not in names:  # tied: add this rank's half as the output layer
            nv = d.shape[0]
            out.append(("output.weight", d[rank * nv // 2:(rank + 1) * nv // 2], t.tensor_type))

    for n, d, ty in out:
        writer.add_tensor_info(n, d.shape, d.dtype, d.nbytes, ty)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    total = 0
    for n, d, ty in out:
        writer.write_tensor_data(np.ascontiguousarray(d), tensor_endianess=reader.endianess)
        total += d.nbytes
    writer.close()
    ff = f"expert n_ff {n_ff_exp} -> {e_hi - e_lo}" if n_ff_exp else f"n_ff {n_ff} -> {f_hi - f_lo}"
    print(f"wrote {dst}: {total / 2**30:.2f} GiB (rank {rank}: {ff}, heads {n_head}/{n_head_kv} -> {hq}/{hkv}, head dim {d_k})")


if __name__ == "__main__":
    main()
