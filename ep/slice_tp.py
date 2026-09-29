#!/usr/bin/env python3
"""Write rank R's tensor-parallel half of gpt-oss MoE experts (Megatron-style MLP split).

gate/up: keep n_ff rows [R*h, (R+1)*h) of every expert.  down: keep the matching input columns
(a contiguous byte range of each row, block aligned).  down bias is kept on rank 0 and zeroed on
rank 1 so it is added exactly once after the allreduce.  The file claims n_ff_exp = h, so
llama.cpp loads it as a normal (narrower) gpt-oss.
usage: slice_tp.py IN.gguf OUT.gguf RANK [SPLIT] [attn]
"attn" also splits attention by heads (rank R: q heads and kv heads of its half, matching attn_output input
columns, output bias on rank 0 only); run with LLAMA_EP_ATTN=split so the halves are summed after attention.
SPLIT = number of n_ff rows rank 0 gets (multiple of 32, default half); rank 1 gets the rest.
Use an uneven split when one machine is slower.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def main():
    src, dst, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    ff_key = f"{arch}.expert_feed_forward_length"
    n_ff = int(reader.fields[ff_key].contents())
    split = int(sys.argv[4]) if len(sys.argv) > 4 else n_ff // 2
    assert split % 32 == 0 and 0 < split < n_ff, "split must be a multiple of the 32-wide quant block"
    lo, hi = (0, split) if rank == 0 else (split, n_ff)
    h = hi - lo
    attn = len(sys.argv) > 5 and sys.argv[5] == "attn"
    n_head = int(reader.fields[f"{arch}.attention.head_count"].contents())
    n_head_kv = int(reader.fields[f"{arch}.attention.head_count_kv"].contents())
    d_head = int(reader.fields[f"{arch}.attention.key_length"].contents())
    hq, hkv = n_head // 2, n_head_kv // 2          # heads per rank (GQA groups stay intact: q head i uses kv head i // (n_head/n_head_kv))
    override = {ff_key: h}
    if attn:
        override[f"{arch}.attention.head_count"] = hq
        override[f"{arch}.attention.head_count_kv"] = hkv
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    align = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = override.get(field.name, field.contents())
        writer.add_key_value(field.name, val, vt, sub_type=st)
    writer.add_key_value("ep.tp_rank", rank, gguf.GGUFValueType.UINT32)

    datas = []
    for t in reader.tensors:
        d = t.data
        if t.name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
            d = d[:, lo:hi, :]                     # (expert, n_ff rows, row bytes)
        elif t.name.endswith(("ffn_gate_exps.bias", "ffn_up_exps.bias")):
            d = d[:, lo:hi]                        # (expert, n_ff)
        elif t.name.endswith("ffn_down_exps.weight"):
            bpb = d.shape[2] // (n_ff // 32)                          # bytes per 32-wide block
            assert bpb * (n_ff // 32) == d.shape[2], f"{t.name}: unexpected row size"
            d = d[:, :, lo // 32 * bpb:hi // 32 * bpb]                # (expert, n_embd rows, this rank's input blocks)
        elif t.name.endswith("ffn_down_exps.bias") and rank != 0:
            d = np.zeros_like(d)
        elif attn and t.name.endswith(("attn_q.weight", "attn_q.bias")):
            d = d[rank * hq * d_head:(rank + 1) * hq * d_head]      # rows = this rank's q heads
        elif attn and t.name.endswith(("attn_k.weight", "attn_k.bias", "attn_v.weight", "attn_v.bias")):
            d = d[rank * hkv * d_head:(rank + 1) * hkv * d_head]    # rows = this rank's kv heads
        elif attn and t.name.endswith("attn_sinks.weight"):
            d = d[rank * hq:(rank + 1) * hq]
        elif attn and t.name.endswith("attn_output.weight"):
            bpr = d.shape[1]                                        # (n_embd rows, row bytes over n_head*d_head inputs)
            assert bpr % 2 == 0
            d = d[:, rank * bpr // 2:(rank + 1) * bpr // 2]
        elif attn and t.name.endswith("attn_output.bias") and rank != 0:
            d = np.zeros_like(d)
        elif t.name == "output.weight":
            nv = d.shape[0]
            assert nv % 2 == 0
            d = d[rank * nv // 2:(rank + 1) * nv // 2]              # (vocab rows, row bytes): each rank scores half the vocab
        datas.append(d)  # memmap view; copied one tensor at a time while writing
        writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    total = 0
    for d in datas:
        writer.write_tensor_data(np.ascontiguousarray(d), tensor_endianess=reader.endianess)
        total += d.nbytes
    writer.close()
    print(f"wrote {dst}: {total / 2**30:.2f} GiB (tp rank {rank}, n_ff_exp {n_ff} -> {h}" + (f", heads {n_head}/{n_head_kv} -> {hq}/{hkv})" if attn else ")"))


if __name__ == "__main__":
    main()
