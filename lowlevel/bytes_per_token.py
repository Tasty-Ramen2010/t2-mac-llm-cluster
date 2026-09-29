"""bytes each Mac must read from RAM to write one token (weights only), and the speed ceiling at a given bandwidth"""
import sys
sys.path.insert(0, "/home/node1/llama.cpp/gguf-py")
import gguf
path, bw = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 28.5
r = gguf.GGUFReader(path)
arch = r.fields["general.architecture"].contents()
g = lambda k, d=None: r.fields[f"{arch}.{k}"].contents() if f"{arch}.{k}" in r.fields else d
n_exp, n_used = g("expert_count", 0), g("expert_used_count", 0)
dense = exp = 0
for t in r.tensors:
    if t.name == "token_embd.weight":
        continue
    if "_exps" in t.name:
        exp += t.n_bytes * n_used / n_exp
    else:
        dense += t.n_bytes
tot = dense + exp
print(f"{path.split('/')[-1]:32} per token: {tot/1e6:7.0f} MB (always-read {dense/1e6:.0f} MB + experts {exp/1e6:.0f} MB) "
      f"-> ceiling {bw*1e9/tot:5.1f} tok/s at {bw} GB/s")
