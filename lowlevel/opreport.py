"""opreport.py MODEL_R0.gguf JOURNAL_DUMP [LABEL]: per-op ms/token and achieved GB/s for a decode profile (GGML_CPU_OPPROF)"""
import re, sys
from collections import defaultdict
sys.path.insert(0, "/home/node1/llama.cpp/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1])
arch = r.fields["general.architecture"].contents()
g = lambda k, d=0: r.fields[f"{arch}.{k}"].contents() if f"{arch}.{k}" in r.fields else d
n_exp, n_used = g("expert_count"), g("expert_used_count")
bytes_per_call = defaultdict(list)          # (name, type) -> bytes read per call, one entry per layer
for t in r.tensors:
    base = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
    b = t.n_bytes * (n_used / n_exp if "_exps" in t.name and n_exp else 1)
    bytes_per_call[(base, t.tensor_type.name.lower())].append(b)
rows, total, n_graph = [], None, None
for line in open(sys.argv[2]):
    m = re.search(r"opprof: total ([\d.]+) ms", line)
    if m: total = float(m.group(1)); continue
    m = re.search(r"opprof:\s+[\d.]+%\s+([\d.]+) ms\s+(\d+)\s+(.*)$", line)
    if m: rows.append((float(m.group(1)), int(m.group(2)), m.group(3).strip()))
for ms, n, key in rows:
    if key.startswith("MUL_MAT output.weight"): n_graph = n
tok = n_graph or 1
print(f"== {sys.argv[3] if len(sys.argv) > 3 else ''}  ({tok} graphs, {total / tok:.1f} ms per token in graph nodes)")
print(f"{'op':52} {'ms/tok':>7} {'share':>6} {'MB/tok':>7} {'GB/s':>6}")
for ms, n, key in rows[:22]:
    m = re.match(r"(MUL_MAT(?:_ID)?) (\S+) \[(\w+)\]", key.lower(), re.I)
    mb = gbs = ""
    if m and (m.group(2), m.group(3)) in bytes_per_call:
        per_layer = bytes_per_call[(m.group(2), m.group(3))]
        b_tok = sum(per_layer) * (n / tok) / len(per_layer)
        mb, gbs = f"{b_tok / 1e6:7.1f}", f"{b_tok / (ms / tok) / 1e6:6.1f}"
    print(f"{key[:52]:52} {ms / tok:7.2f} {100 * ms / total:5.1f}% {mb:>7} {gbs:>6}")
