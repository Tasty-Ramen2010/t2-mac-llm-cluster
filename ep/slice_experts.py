#!/usr/bin/env python3
"""Write a GGUF that keeps only experts lo..hi of every MoE layer (for expert parallelism).

Experts are the outermost dimension of *_exps tensors, so each expert is a contiguous block and
slicing is a plain byte copy. Router tensors keep all experts (routing must see every expert).
usage: slice_experts.py IN.gguf OUT.gguf LO HI
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def main():
    src, dst, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    align = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
    writer.add_key_value("ep.expert_lo", lo, gguf.GGUFValueType.UINT32)
    writer.add_key_value("ep.expert_hi", hi, gguf.GGUFValueType.UINT32)

    datas = []
    for t in reader.tensors:
        d = t.data
        if "_exps" in t.name:
            d = d[lo:hi + 1]  # outermost numpy axis is the expert index
        datas.append(d)
        writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    total = 0
    for d in datas:
        writer.write_tensor_data(d, tensor_endianess=reader.endianess)
        total += d.nbytes
    writer.close()
    print(f"wrote {dst}: {total / 2**30:.2f} GiB of tensor data (experts {lo}-{hi})")


if __name__ == "__main__":
    main()
