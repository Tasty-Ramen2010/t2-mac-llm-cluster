# Orin (sm_87) llama.cpp optimizations for Qwen3.6-35B-A3B

Target: DRIVE AGX Orin, 28 GB unified LPDDR5 (~175 GB/s measured read ceiling), 16-SM Ampere iGPU, CUDA 11.4, llama.cpp `0bb496d`.
Apply: `cd llama.cpp && git apply orin-llama.cpp.patch`, build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` (needs cmake >= 3.18).
Everything is **opt-in by environment variable** and produces **byte-identical text** to the stock build (verified at temperature 0, with and without MTP).

## What is in the patch
| change | switch | effect |
|---|---|---|
| `mul_mat_vec_q_id_warp` (ggml-cuda/mmvq.cu): persistent warp-per-row MoE matvec for IQ3_S / IQ4_XS / Q6_K experts, fused gate+up+SwiGLU, no shared memory, no `__syncthreads` | `GGML_MMVQ_ID_WARP=1` | single-token decode **35.3 -> 40.8 tok/s (+15.4%)**; 931/931 `test-backend-ops MUL_MAT_ID` pass |
| MTP draft-vocabulary pruning (src/models/qwen35moe.cpp): the draft head computes logits for the first N token ids only, rest -1e30; verification unchanged | `LLAMA_MTP_DRAFT_VOCAB=65536` | MTP **58.3 -> 64.5 tok/s (n=2), 67.0 (n=3)**: **+15%** end to end, acceptance 86% -> 79-84% |
| fast dim-0 concat (ggml-cuda/concat.cu), no 64-bit division | on by default (`GGML_CONCAT_SLOW=1` = old) | 7x faster kernel in isolation, ~0% end to end (overlaps with matvecs in CUDA graphs) |
| per-op GPU profiler (ggml-cuda.cu) | `GGML_CUDA_PROF=1 GGML_CUDA_DISABLE_GRAPHS=1` (+`_BY_WEIGHT`, `_SKIP=N`) | table of GPU time per op, fused chains included |

Recommended server flags with the above: `--spec-type draft-mtp --spec-draft-n-max 3`. On the AGX: `agx-perf opt` (see IDE.md).
With a 256k context, `-ub 1024` instead of 512 raises long-prompt speed 665 -> 810 tok/s (+22%) for 1.2 GB less free memory (`agx-perf opt1024`).

## Findings (so the next person does not repeat the dead ends)
- Orin read ceiling is **~175 GB/s**, not the 205 GB/s on paper (`bwtest.cu`).
- Per token this model reads ~2.4 GB: Q8_0 attention/DeltaNet projections 1.3 GB (!), MoE experts 0.47 GB, Q6_K LM head 0.42 GB, shared experts and routers ~0.2 GB.
- Large Q8_0 matvecs already run at 150-157 GB/s in isolation; a tiled cp.async variant gained <= 11% (`bwtest.cu`). Not worth it.
- **What was slow was launch geometry, not bandwidth:** the stock MoE decode kernel launches one short-lived block per (expert, row) (4096-16384 blocks per layer, ~20 waves); a persistent warp-per-row kernel removes that. The LM head (248k rows = 248k blocks) has the same shape and is the next candidate (~1 ms/pass).
- GPU busy 98% in decode even though only ~40% of bandwidth is used: latency-bound chains of small kernels. Profile numbers from a serial profiler overstate ops that overlap in CUDA graphs: always A/B end to end (concat is the cautionary tale).
- No gain: CPU pinning / real-time priority, KV-cache quantization for speed, flash-attention on/off (+1.5%), `GGML_CUDA_GRAPH_OPT` (crashes decode), n-gram speculation on fresh code, Q4_K_S weights (+9% for +3 GB), 4 draft tokens.
- The first CUDA call in a process costs ~1.2 s (cuBLAS init) - hide it with a warm-up request (`agx-setup/warm.py`).
- The GPU memory is part of the 28 GB: compile jobs next to a 24 GB model make the kernel OOM killer kill Wi-Fi first. Protect daemons with `oom_score_adj=-1000`, test with a MemAvailable watchdog.
