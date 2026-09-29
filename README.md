# T2 Mac Cluster LLM — run big models across old Intel Macs, no GPU, no cloud

A toolkit for running large language models **split across two (or more) 2018-era Intel Mac minis** running Linux,
using only their CPUs and a cable between them. Built for schools and anyone with old "trash" hardware: no discrete
GPU, no cloud, no new parts required.

**What it gets you** (2× Mac mini 2018, i3-8100B, 8 GB each, over a 1 GbE cable):

| Model | tokens/sec | notes |
|---|---|---|
| DeepSeek-Coder-V2-Lite 16B (MoE) | ~22 | best coder, uses tools |
| Mellum 2 12B (MoE) | ~21 | fastest coder |
| gpt-oss 20B (MoE) | ~18 | general + web + tools |
| Gemma 3 12B (dense) | ~6 | general writing |

Started at **3.3 tok/s** with stock llama.cpp; reached **~18–22** through custom tensor-parallel code, hand-written
AVX2 kernels, and low-level system tuning. The full engineering story (10 rounds, every win and dead-end measured) is
in the sections below.

## What's here
- **Custom tensor-parallel llama.cpp** (`ep/llama-cpp-ep.patch`): both Macs work on every token at once, splitting each
  weight matrix in half and adding results over the cable. Plus new AVX2 kernels (Q5_0, grouped-query attention) and an
  optional Intel iGPU co-worker.
- **`start-ep.sh` / `stop-llm.sh`**: bring the whole split model up/down across both Macs.
- **`agent/`**: a phone-friendly web chat + a terminal client (`ai`), a live cluster monitor (`aitop`), a benchmark
  viewer (`aibench`), and a permission-gated sandbox so the model can run code and reach approved websites safely.
- **`gpu/`, `lowlevel/`**: the microbenchmarks and GPU kernels used to find the hardware's real limits.
- **`bench/`**: the coding-quality benchmark harness.

## Quickstart
1. Two T2 Intel Macs on Ubuntu with the T2 kernel, joined by an Ethernet cable (`10.10.10.1` / `10.10.10.2`) — see
   **Machines** and **What's configured** below.
2. Build our llama.cpp on both:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   git checkout 2145525            # the base commit these patches were made against
   git apply /path/to/ep/llama-cpp-ep.patch
   cmake -B build -DGGML_NATIVE=ON -DGGML_CPU_IGPU=ON && cmake --build build -j4
   ```
3. Slice a model into two halves (`ep/slice_*.py`), copy rank 1 to the second Mac, then `./start-ep.sh dscoder`.
4. Open `http://<node1>:8081` on any device, or type `ai` in a terminal.

Full setup, tuning, and the measured results for every optimization are documented below.

---

# Mac mini cluster — setup notes (2026-09-26)

## Machines

| | node1 | node2 |
|---|---|---|
| Hardware | Mac mini 8,1 (2018), i3-8100B, 7.6GB RAM, 128GB SSD, UHD 630 | same |
| OS | Ubuntu 24.04, kernel 7.1.8-1-t2-noble | same |
| Login | `node1` | `node2` |
| Tailscale IP | 100.82.180.15 | 100.109.16.15 |
| Home WiFi (Rampardhu) | 192.168.1.124 | 192.168.1.125 |
| Direct ethernet cable | IPv6 shared from node1 (`2600:1700:1060:4d1e::/64`) | gets an address from node1 |

## What's configured (survives reboot)

- **Passwordless sudo** on both (`/etc/sudoers.d/010-node*-nopasswd`).
- **SSH keys**: node1 and node2 can SSH into each other with no password.
- **Tailscale** on both, starts at boot, `--ssh` enabled.
- **WiFi on node2**: Broadcom firmware copied from node1 into `/lib/firmware/brcm/`.
- **GPU drivers** on both: i915, Mesa Vulkan, VA-API (intel-media-va-driver), OpenCL (intel-opencl-icd), `intel_gpu_top`.
- **Swap**: extra 16GB `/swapfile2` on both (in `/etc/fstab`), 18GB total.
- **No sleep**: suspend/hibernate targets masked; GNOME screen blanking and idle sleep off.
- **Firewall (node1)**: ufw on. SSH allowed on tailscale0, WiFi and the ethernet link. Samba only on tailscale0.
- **Storage share (node1)**: Samba share `storage` → `/srv/storage`, user `node1`, reachable over Tailscale only.
  - Windows: `\\100.82.180.15\storage`
  - Mac: Finder → Cmd+K → `smb://100.82.180.15/storage`
  - iPhone: Files → Connect to Server → `smb://100.82.180.15/storage`

## Running the 20B model (gpt-oss-20b, split across both nodes)

- Model: `/home/node1/models/gpt-oss-20b-MXFP4.gguf` (12.1GB)
- llama.cpp: `~/llama.cpp` on both. `build/` = CPU + RPC (use this). `build-vulkan/` = Vulkan + RPC.
- Start: `~/cluster/start-llm.sh`   Stop: `~/cluster/stop-llm.sh`

### Benchmark results (gpt-oss-20b)

| Setup | Prompt t/s | Generation t/s |
|---|---|---|
| **CPU on both nodes, 4 threads, `performance` governor (recommended)** | **21.59** | **8.61** |
| CPU on both nodes, 2 threads, `powersave` (old) | 12.06 | 5.76 |
| node2 GPU 4 layers, rest node1 CPU (`--repack 0 -lm mmap`) | 10.09 | 2.99 |
| node2 GPU 8 layers | 10.13 | 2.92 |
| node2 GPU 12 layers | 8.18 | 2.87 |
| node2 GPU 16+ layers | fails (out of GPU memory) | |

The UHD 630 shares RAM with the CPU, so it can't generate tokens faster than the CPU.

### Update 2026-09-26 evening
- **CPU governor**: both nodes now boot with `performance` (`cpu-performance.service`). Under `powersave` a node idles while waiting on its partner and throttles, which cut pipeline speed ~2-3x.
- **llama.cpp RPC servers default to `-t 2`**. Use `-t 4`. New best for gpt-oss-20b CPU split: **21.6 t/s prompt, 8.61 t/s generation**.
- **OpenVINO split (`~/projects/ovsplit`)**: cuts an OpenVINO model graph at a layer and runs each half on a different Mac over the direct cable. Verified word-for-word identical to the unsplit model (Qwen2.5-0.5B, full-precision KV). Split costs ~7% speed.
  - Limit: OpenVINO's CPU plugin copies each stage's weights into RAM (~1.2x stage size), so gpt-oss-20b needs **3** Macs with OpenVINO. On 2 Macs it swaps: 2.3 t/s.
  - node2 GPU via OpenVINO: 2.3x faster prompt processing, slower generation (shares RAM bandwidth).
- **PyTorch can't use the UHD 630** (Intel only supports Arc/Core Ultra). **OpenVINO can.** ESM-2 on the UHD 630 matches CPU speed with correct output.

### More tests (2026-09-26 night)
- **CPU+GPU on one Mac** (Qwen2.5-3B, node2): CPU alone wins everything. CPU 4 threads 57.2/13.4 t/s (prompt/gen); GPU only 19.9/5.5; best CPU+GPU split 50.2/6.1. llama.cpp's Vulkan kernels are slow on the UHD 630 and every layer needs a CPU-GPU sync. Use the GPU for separate jobs (OpenVINO prompt processing, HybriDock pose batches), not a slice of every layer.
- **Tensor split (`-sm tensor`) across Macs** works but generation drops to 2.6 t/s (syncs every layer over the network). Row split isn't supported over RPC.
- **Gemma 3 12B** (dense, 6.8GB) split across both Macs: 14.5 prompt / **3.3 gen** t/s. Slower than gpt-oss-20B (8.6) because dense models read every weight for every word; MoE reads ~1/8 of its experts.
- **Cable**: node1's Ethernet chip was dropping packets (receive ring 200). Ring raised to 511 (persistent in NetworkManager); drops stopped and cable = Tailscale speed. Cable IPs: node1 `10.10.10.1`, node2 `10.10.10.2` (persistent).
- `start-llm.sh` takes an optional model path, e.g. `~/cluster/start-llm.sh ~/models/gemma-3-12b-it-Q4_K_M.gguf`.

### Speculative decoding (2026-09-26 night)
A small draft model guesses a few words ahead; the big model checks them in one pass.
- **Gemma 12B + Gemma 1B draft, 3 guesses: 3.27 -> 3.98 words/s (+22%)**, 73% of guesses accepted. 5 guesses: 3.59.
- gpt-oss-20B + EAGLE3: 8.30 -> 6.10 (worse). MoE: each guess hits different experts, so checking isn't free; also pushed node1 into swap.
- Qwen 3B + 0.5B draft: break-even at best. Draft on the GPU: much slower (GPU per-step launch delay).
- `Q4_0` format is ~7% faster than `Q4_K_M` on these CPUs.
- `start-llm.sh gemma` uses the draft automatically.

### Two-Mac tensor parallel (2026-09-27 night): ~15 words/s
`start-llm.sh` (or `start-ep.sh`) now runs gpt-oss-20B with **both Macs working on every word at once**
over the 10.10.10.x cable, instead of passing layers back and forth.
Custom llama.cpp code in `src/llama-ep.{h,cpp}`: each Mac holds half of every expert (Megatron-style MLP split,
files made by `ep/slice_tp.py` from the q4att model), adds the halves together over a direct TCP link after each
layer (24 syncs/word, ~0.3 ms each), and scores half the vocabulary each (top-64 exchange).
node1 runs normal llama-server; libllama mirrors every decode/cache call to node2 (`llama-ep-run --follow`).

| setup | words/s |
|---|---|
| original layer pipeline (RPC) | 8.2-8.6 |
| + attention/output in Q4_0 (q4att) | 9.9 |
| expert parallel (each Mac owns 16 experts) | 11.7 -> 12.8 (repack, no mmap) |
| tensor parallel (half of every expert each) | 13.9 |
| **+ vocab split (llama-ep-run)** | **15.0-15.5** |
| llama-server via start-ep.sh (first version) | 14.7-15.7, prompt ~39 t/s |
| + lean AVX2 gemv kernel for MXFP4/Q4_0 (ggml repack.cpp, `GGML_LEAN_GEMV=0` to disable) | +7% |
| + low-latency cable: EEE off, no deep C-states, rx coalescing 1 us (`llm-lowlatency.service`) | per sync 0.32 -> 0.2 ms |
| + fp16 sync payload (auto fp32 if a value is too large; `LLAMA_EP_F16=0` to disable) | 16.0-16.3 |
| + node1 3 threads (leaves a core for OS/network), node2 4 | ~17 |
| + uneven expert split (node1 1280 / node2 1600 rows) + attention heads split too (`LLAMA_EP_ATTN=split`) | 17.3-17.7 |
| **llama-server via start-ep.sh now** | **17.5-18.1**, prompt ~51 t/s |

Requirements: CPU governor `performance` (cpu-performance.service), `llm-lowlatency.service` on both nodes, cable up, NIC rx ring 511.
Model slices: `ep/slice_tp.py gpt-oss-20b-MXFP4-q4att.gguf OUT RANK 1280 attn` (run on node2 where the q4att file lives, copy rank 0 to node1).
Benchmark: `ep-bench.sh 300` (llama-ep-run leader/follower, prints words/s). Per-op profile: add `GGML_CPU_OPPROF=1`.
SSH to node2 over the cable: `ssh node2@10.10.10.2` (Tailscale SSH sometimes wants a browser re-check).
Logs: `journalctl --user -u llm-ep -f` (node1), `journalctl --user -u ep-follower` (node2).
Gemma 12B is dense (7 GB read per word): 3.3 words/s split, 4.0 with a 1B draft model; ~15 is physically out of reach
on 2x 42 GB/s DDR4 without a much smaller model.

### Web app, monitor and benchmark viewers (2026-09-27 afternoon)
- **http://100.82.180.15:8081** is one app with three tabs (sidebar):
  - **Chat**: conversations kept in the browser, collapsible thinking, tool cards, code copy buttons, stop button,
    per-message speed; the model picker (top right) switches the whole cluster to another model (~1 min).
  - **Monitor**: web version of aitop: model, live prompt-reading / writing speed, speed graph, exact numbers for the
    last request, both Macs (CPU per core, RAM, swap, CPU/SSD temperature, fan, cable traffic, GPU, disk), and every AI
    process on the cluster (llama.cpp, Ollama & co, or any Python using PyTorch / OpenVINO / ONNX Runtime / TensorFlow).
  - **Benchmarks**: the coding benchmark table, every problem with the model's answer and the test output, and the run
    in progress live.
- One source of truth: `agent/cluster_stats.py` (inside the web backend) samples both Macs every second (node2 over one
  ssh stream) and follows llama-server's log for exact per-request timings (matched by task id). `/api/stats` feeds the
  web Monitor and `aitop`; numbers were checked against `free` and llama-server's own log.
- Terminal: `ai` (asks which model at startup; status line with model · conversation memory · temps · free RAM and
  warnings; `/models`, `/use NAME` keeps + ports the conversation, `/stats`, `/files`, `/show FILE`, `/save`,
  Ctrl-C stops an answer on the server too; `ai --model NAME "question"`), `aitop` (monitor), `aibench`
  (benchmark results: `aibench`, `aibench LABEL`, `aibench LABEL TASK`, `aibench --live`). All three work from any
  machine on Tailscale (set CLUSTER_AI_URL if needed).
- Model switching from the app runs `KEEP_AGENT=1 start-ep.sh PRESET` (keeps the web backend alive).
- The old single-page chat is saved as agent/index_classic.html.
- Conversations live on node1 (agent/chats/), every device sees the same list; each has a sandboxed workspace
  (/home/llmtools/work/chats/ID: run_shell, write_file, read_file, list_files, run_python; no network) shown in the
  📁 Files panel. Switching model mid-chat shows a measured loading bar, then "porting" (the new model pre-reads the
  conversation, including what the tools returned). Memory meter under the message box; health strip in the sidebar.

### Cooling (2026-09-27)
- **t2fanrd** (fan controller for T2 Macs) on both nodes, curve in /etc/t2fand.conf: ramps 50 C -> 80 C to full speed.
  Before it, Linux left the fans at ~2000 of 4400 rpm and the CPUs sat at 90-92 C under load (node1 hit its 100 C limit
  38 times); with it, full load runs at ~65-70 C.
- cooldown-guard (~/cluster/cooldown_guard.py) is DISABLED since 2026-09-27 evening (t2fanrd keeps them cool);
  re-enable with `systemctl --user enable --now cooldown-guard` if long jobs ever overheat.
- aitop shows CPU temperature and fan speed for both Macs.

### Chat apps (2026-09-27)
- **Web**: http://100.82.180.15:8081 — streams word by word, live tokens/s, survives phone connection drops
  (questions run as server-side jobs; the page reconnects), prompt cache pre-warmed at startup.
- **Terminal**: type `ai` on node1 or node2 (`ai "question"` for one-shot). Commands: /new /think /nodes /exit.
  Source: `agent/ai_term.py` (needs Python + rich). Point it elsewhere with CLUSTER_AI_URL / CLUSTER_LLM_URL.
- **Tools the AI has**: web_search, fetch_url, run_command (allowlist), **run_python** — sandbox: user llmtools,
  no network (own net namespace), 512 MB, 30 s, files kept in /home/llmtools/work. The model is told to test code first.
- Context is 16384 tokens (start-ep.sh `-c 16384` on both ranks), replies up to 4096 tokens.
- Transcripts: `agent/logs/chat-YYYY-MM-DD.jsonl` (every exchange). Browser script errors land in `journalctl --user -u llm-agent`.
- System monitors: `btop` (CPU/RAM/network/processes), `sudo intel_gpu_top` (iGPU), both nodes.

### Chat with web + commands (http://100.82.180.15:8081)
`~/cluster/agent/agent_server.py` gives the model 3 tools: `web_search` (DuckDuckGo), `fetch_url`, `run_command`.
- Commands run as the sandboxed user `llmtools` (no sudo, can't read /home/node1 or /srv/storage), from an allowlist, no pipes/redirects, 20s limit.
- `fetch_url` refuses private/LAN/Tailscale/localhost addresses, including after redirects.
- Page is Tailscale-only (ufw port 8081 on tailscale0). Started/stopped by start-llm.sh / stop-llm.sh.
- /srv/storage tightened to 770 (Samba unaffected).
- Page renders Markdown, LaTeX math (KaTeX), Mermaid diagrams (auto-repairs unquoted labels), code highlighting; all sanitized with DOMPurify. Libraries are served locally from agent/static (no CDN). Test: open http://100.82.180.15:8081/#demo

### AI permissions, /context, /usage, resuming chats (2026-09-27 evening)
- The AI's tools (run_shell, run_python, write/read/list files) run as user `llmtools` in a per-chat folder
  `/home/llmtools/work/chats/<id>` (limits: 1.5 GB memory, 2 GB files, 256 processes, 10 min per command).
- Internet: the firewall (`llmtools-firewall.service`, chain LLMTOOLS) lets llmtools reach only the gateway
  `agent/netgate.py` on 127.0.0.1:8899. A new website makes a permission box pop up (web card or terminal y/a/n):
  allow for this chat / always / deny. pip (pypi.org), git/curl (github.com), huggingface.co and npm work once allowed.
  Local-network addresses (LAN, the Macs, Tailscale) are always refused. "Always" sites live in `agent/permissions.json`;
  "for this chat" approvals last until the backend restarts. Decisions are saved in the conversation.
- Terminal `ai`: `/context` (how full the model's memory is), `/usage` (tokens and time for this chat and in total),
  `/permissions [remove SITE]`, startup offers recent chats, `ai -c` continues the latest one.
  Web: the same slash commands in the message box (/context /usage /permissions /continue /chats /new /models /files /help,
  suggestions pop up as you type /), 📊 Usage and 🔐 Permissions in the sidebar, click the memory meter for /context,
  "Continue <latest chat>" on the welcome screen.
- Mellum Thinking used to think in circles until it ran out of tokens (default sampling). Now it gets temp 0.6 / top_k 20 /
  presence penalty 1.0 and a 1024-token thinking cap (MODEL_SAMPLING in agent_server.py), then must answer.
- Stop / Ctrl-C: llama-server finishes the 512-token batch it is on before it stops.

- PyTorch: the AI user has CPU-only torch (2.14+cpu). `/home/llmtools/.config/pip/pip.conf` adds the CPU PyTorch index so
  `pip install torch` never pulls the 2.5 GB NVIDIA build; download.pytorch.org is part of the pypi.org permission.
- Tool output shown to the model drops pip progress lines and keeps start + end (errors / "Successfully installed").
- Sliding-window memory (start-ep.sh, SWA=full|small): gpt-oss, Mellum and Gemma use sliding-window attention. With the
  small (window-only) memory and no checkpoints, any change near the end of a chat made llama-server re-read the WHOLE chat
  (12.7k tokens = 4 min). gpt-oss and Mellum now run `--swa-full` (+0.2-0.4 GB per Mac): re-reads only the change (tested:
  9 tokens instead of 4,279). Gemma stays small on BOTH Macs (`llama-ep-run --swa-small`, new): full-size would need
  ~3.2 GB per Mac and swapped. node2's follower used to always keep full-size memory (llama_context default), which is
  why Gemma nearly filled node2 before. Leader and follower must use the same setting.
- Speed falls as a chat grows (Mellum: ~21 tok/s fresh, ~9 tok/s at 9k tokens): start a new chat for a new topic.

### Speed ablation, re-measured in one sitting (2026-09-27 night, `bench/speed/run.sh gpt-oss FILE`)
Same 816-token prompt + 200 forced words, greedy, no prompt cache, 2 runs each; each row adds one change.
| step | prompt t/s | writing t/s |
|---|---|---|
| 1 original: RPC layer pipeline, powersave, 2 threads, Tailscale (q4att file) | 11.0 | 3.3 (runs 1.7 / 4.8: powersave is erratic) |
| 2 + performance governor, 4 threads, cable | 21.3 | 7.9 |
| 3 tensor parallel (our llama-ep: split every matrix, vocab split, attention split, uneven split) | 37.2 | 15.4 |
| 4 + huge pages | 38.9 | 15.4 |
| 5 + lean AVX2 kernel | 43.5 | 15.7 |
| 6 + low-latency cable | 49.3 | 16.1 |
| 7 + fp16 sync | 55.0 | 16.1 |
| 8 + node1 3 threads (= production) | 54.6 | 16.0 |
Production chat speed is ~17-18 on short prompts; this test uses a long prompt, so every word attends to more text.
Scripts: bench/speed/{run.sh,rpc.sh,sys.sh,measure.py}; results in bench/speed/results.jsonl.

### Deeper optimization round (2026-09-27 late night)
- **PCIe ASPM was on for the Ethernet chip** (L1, exit latency up to 64 us): the kernel refuses to change it (firmware
  owns ASPM), so `llm-lowlatency.service` now clears the ASPM bits in the PCIe Link Control register with `setpci`
  (NIC 04:00.0 and its bridge 00:1c.1) and sets tx coalescing to 1 us. Ping over the cable 145 -> 49 us; one sync
  (5.8 KB each way) 150 -> 101 us. gpt-oss 16.0 -> 16.8 t/s, DeepSeek 16.7 -> 18.0 t/s (bench/speed step 9).
  Jumbo frames, busy-poll sockets, TSO/GRO off: all measured, none helped.
- Memory bandwidth (lowlevel/membw.c): ~29-34 GB/s per Mac; 2 threads already saturate it. Per-token bytes and speed
  ceilings per model: lowlevel/bytes_per_token.py (gpt-oss 31, Mellum 38, DeepSeek 35, Qwen7B 13, Gemma 8 t/s).
- llama-ep: spin-wait receive (LLAMA_EP_SPIN=0 to disable), optional Q8_0 sync payload (LLAMA_EP_Q8=1: +1.5% writing,
  +7% prompt, but answers are worded differently vs fp16 -> off by default), per-token timers (LLAMA_EP_PROFILE=1).
- llama-ep-run (node2) had no persistent thread pool (ggml created/joined threads every token): fixed; new options
  --poll, --prio, --cpumask. start-ep.sh passes EPF_ARGS (node2) / EPL_ARGS (node1) / T1 (node1 threads).
- ggml: GGML_HUGETLB=1 takes big buffers from reserved 2 MiB pages (start-ep.sh HUGETLB=1 reserves them). Off by
  default: node1 got only 60% THP coverage vs node2's 81%, but reserving failed on node1 (desktop + RAM) and gave no gain.
- **DeepSeek-Coder-V2-Lite** (16B MoE, 2.4B active): `start-ep.sh dscoder`. ep/slice_deepseek_tp.py converts the old
  combined attn_kv_b into MLA attn_k_b/attn_v_b (memory cache ~0.5 GB for 16k tokens instead of ~4.5 GB) and splits
  heads, every expert (optional uneven split), shared experts and the dense layer. 6.7 t/s original RPC -> 18.0-18.6.
  Uneven splits (more on node2) measured slower; even is best.

### Optimization round 2, continued (2026-09-28 early morning)
DeepSeek-Coder-V2-Lite, step by step (bench/speed, 816-token prompt, 200 forced words, 2-3 runs each):
| step | prompt t/s | writing t/s |
|---|---|---|
| 1 original: RPC pipeline, powersave, 2 threads, Tailscale | 24.1 | 6.7 |
| 2 + performance CPU, 4 threads, cable | 44.3 | 11.0 |
| 3 tensor parallel (split heads, experts, vocab; MLA memory) | 40.6 | 15.8 |
| 4-8 huge pages, lean kernel, low-latency cable, fp16 sync, node1 3 threads | 55.0 | 16.7 |
| 9 + PCIe ASPM off, tx coalescing 1 us | 54.9 | 18.0 |
| 10-11 node2 persistent thread pool, spin-wait sync | 54.1 | 18.1 |
| 19 + new lean AVX2 Q5_0x8 kernel (repacked; Q5_0 had none on AVX2) | 61.5 | 18.3 |
| 22 + threads pinned to cores, network IRQs on node1 cpu3 / node2 cpu0 | 61.2 | **18.8** |
Coding benchmark: DeepSeek 82% write / 70% fix, fastest per problem (19 s / 12.8 s), 20 t/s.
- New ggml kernel: ggml-cpu/repack.cpp + arch/x86/repack.cpp `block_q5_0x8` (8 rows interleaved, 5th bits transposed
  so one shift+and restores them). Checked: every repacked weight == ggml's q5_0 decoding (2.2 billion, 0 wrong),
  AVX2 == C reference bit-exactly (GGML_Q5X8_CHECK=1). GGML_REPACK_Q5_0=0 turns it off. Op 18% faster, prompt +13%.
- n-gram drafting on by default (start-ep.sh SPEC=0 to disable): 16-token match, 16-token guess. Code edits:
  DeepSeek 17.4 -> 28.4 t/s, gpt-oss 16.1 -> 26.7; fresh answers unchanged; summaries -2%.
- Pinning (PIN=0 to disable): +2-3% and much steadier speed.
- Measured, no gain: uneven expert split (more work on node2), node1 4 threads, Q8_0 -> Q5_0 shared experts,
  reserved huge pages, 8-bit sync (+1.5%, but changes wording: opt-in LLAMA_EP_Q8=1), turning the monitor off (+1-2%
  memory bandwidth). A stripped-down node2 would not help: it idles at 0.3% CPU and is the Mac that waits.
- Where the time goes now (DeepSeek, per token ~53 ms): node1 computing ~44 ms at ~20-23 GB/s with 3 threads, syncs ~3.5 ms
  (54 x 65 us), between tokens ~1.3 ms. Writing is no longer memory-bound: reading 28 MB less per token changed nothing.

### Round 3 (2026-09-28): DeepSeek tools, long-chat slowdown
- **DeepSeek-Coder-V2 could not use tools**: its chat template has none, so llama-server silently dropped them and the
  model invented outputs ("Output: 45"). templates/deepseek-coder-v2-tools.jinja (its User:/Assistant: format + Hermes
  <tool_call> JSON) is loaded by start-ep.sh for dscoder; it now calls run_python and reports real results. The agent
  also drops exact duplicate tool calls in one reply (DeepSeek repeats them).
- **New CPU attention kernel for decoding** (ggml-cpu/ops.cpp, ggml_fa_mqa_decode_*): llama.cpp handled every query
  head separately, re-reading and re-converting each cached row once per head, and kept the running V sum in f16. Now
  each block of 16 rows is converted once, all heads of a group are scored together (register accumulators), V is
  accumulated for all heads from the same rows, in f32 (for MLA, V is the first 512 values of the K row: converted once).
  Accuracy vs exact double precision: new <= 7e-4, old up to 0.32 (the old f16 accumulator was distorting attention).
  GGML_FA_MQA=0 disables, GGML_FA_MQA_CHECK=1 prints the comparison.
| writing t/s at ~0.1k / 2k / 4k / 8k context | before | after |
|---|---|---|
| DeepSeek | 21.9 / 15.7 / 11.4 / 7.0 | 22.0 / 19.1 / 16.3 / 12.2 |
| Mellum | 21.8 / 17.8 / 14.4 / 10.8 | 23.0 / 20.8 / 18.6 / 17.1 |
| gpt-oss (7k) | 18.1 / 15.8 / 12.4 / 7.7 | 18.9 / 17.9 / 16.6 / 14.3 |
| Qwen2.5-Coder 7B (7k) | 9.0 / 8.5 / 7.4 / 5.2 | 9.0 / 8.8 / 8.2 / 7.1 |

### Round 4 (2026-09-28): last stand
- Kernel microbenchmark (lowlevel/kbench.c, calls ggml's exported kernels): 1 core from cache / 3 cores from RAM:
  Q4_K 15.4 / 24.2 GB/s, lean Q4_0 19.4 / 26.2, new Q5_0 20.2 / 25.4. Kernels are within ~10% of the memory limit.
- Per-op profiler with GB/s: lowlevel/prof.sh PRESET (+ opreport.py). node1 3 vs 4 threads: no difference now.
- Thread polling (--poll) per model in start-ep.sh: 100 for gpt-oss (+2%) and DeepSeek (+4%), 50 for Mellum/Qwen/Gemma.
- node2 spins up to 20 ms for the next instruction (ep-run recv_all). No measurable gain, no harm.
- Measured, no gain: merged gate+up expert tensor (-2%), C1/C1E disabled, IRQs on cpu0 + 4 threads.
- UDP instead of TCP for syncs: 9-19 us less per sync (lowlevel/udppong.c), ~1-2% overall; not done (needs retransmits).
- Thunderbolt: both Macs have Titan Ridge + Alpine Ridge TB3 and thunderbolt_net. `tb-link.sh up` after plugging a
  Thunderbolt 3/4 cable; start-ep.sh then uses 10.30.30.x for the syncs automatically.
- Attention kernel also verified on Gemma (old error 2.98, new 8e-5) and Qwen 14B.
- Full report: https://claude.ai/artifact/2Faav3cQcg4yS6RLHrQ1Qw

### Round 5 (2026-09-28): using the Intel UHD 630 GPU
Tools in cluster/gpu/ (OpenCL, Intel's NEO driver 23.43, zero-copy buffers over host RAM).
- GPU alone reads RAM at ~25 GB/s, CPU alone ~30 GB/s, **both at once only 22-25 GB/s in total** (gpubw.c). Tried:
  GPU-owned vs zero-copy memory, 1-4 CPU threads, a lighter GPU stream, and the driver's cache-policy switches
  (NEOReadDebugKeys=1 OverrideStatelessMocsIndex=0/1/2, ForceAllResourcesUncached=1, confirmed applied): no change.
  The limit is the RAM controller itself (~30 GB/s real of 42.7 GB/s DDR4-2666 dual channel), not coherency/snooping.
  So the GPU cannot speed up writing (memory-bound): anything it reads, the CPU loses.
- The memory controller is locked by Apple's firmware (MCHBAR 0xfed10000, MC_LOCK 0x50FC = 0x8f): timings and refresh
  cannot be changed from Linux. Both channels populated, 2667 MT/s.
- GPU real peak (flops.c): 324 G fp32 / 594 G fp16 multiply-adds/s (CPU peak ~230 G fp32). Dispatch round trip ~35 us.
- GPU attention kernel for DeepSeek MLA decoding (gpu/attn*.cl, 8 versions): v1 2.0 ms/layer at 8k rows, v7 1.33 ms
  (= the CPU): the big fixes were one 16-byte load instead of vload_half8 (the compiler split it into 8 loads) and
  subgroup broadcasts. Integrated as an iGPU co-worker (ggml-cpu/igpu.cpp + ops.cpp, GGML_IGPU_ATTN=<fraction>,
  GGML_IGPU_ATTN_CHECK=1 compares with the CPU: max diff 9e-5). Careful measurement at 8.5k context (3x2 runs):
  12.97 t/s with it vs 13.74 without (-5.6%) -> off by default (start-ep.sh IGPU=1 turns it on).
- CPU prompt GEMM (cpugemm.c): 130-160 GMAC/s on 4 cores (int8), so a GPU GEMM would need 250+ to matter.

### Round 6 (2026-09-28): GPU for prompt processing, and the T2/RAM ceiling
Goal: use the UHD 630 for prompt processing (compute-bound, unlike generation), where it has more raw FLOPS.
- GPU real GEMM throughput is the wall. Measured (cluster/gpu/):
  - Register-only fp16 loop peak (flops.c): 594 GMAC/s -- not reachable from memory.
  - Best-case DENSE fp16 GEMM reading from RAM (f16bench.c): ~4 GMAC/s.
  - Our int4 (block_q4_0x8) GPU GEMM (gemm.cl, 2 structures): ~6 GMAC/s, register spilling (8x8 accumulators
    overflow the Gen9 GT2 thread's GRF; 168 hw threads total).
  - llama.cpp AVX2 CPU GEMM (cpugemm.c): 36 GMAC/s per core, ~145 across 4 cores.
  The UHD 630 (Gen9) has NO matrix engine (Intel DPAS/XMX is Gen12+/Xe only), so it is ~5-25x slower than the CPU at
  GEMM. It cannot speed up prompt processing either. GPU offload stays off by default.
- T2 / RAM controller: the memory controller (MCHBAR 0xfed10000) is LOCKED by Apple firmware at boot
  (MC_LOCK 0x50FC = 0x8f); timings/refresh are read-only from Linux, changeable only by altering signed firmware the
  T2 verifies each boot (bricks the Mac on error). RAM already at full rated 2667 MT/s, dual channel. The WiFi fix is
  only Broadcom's firmware blob in /lib/firmware/brcm/ (a PCIe device); it does not touch the T2 (Apple signed
  bridgeOS, not reprogrammable from the OS). None of this changes the ~30 GB/s memory wall.
- Net: generation speed is fixed by RAM bandwidth (locked); prompt speed is best on the CPU. Remaining levers for more
  speed: a faster Mac-to-Mac link (Thunderbolt, tb-link.sh) and/or more Macs.

### Round 7 (2026-09-28): custom GPU kernels, run-on-GPU-only, how close to peak
Answering "can custom kernels approach the GPU peak, and can we run on GPU only?" -- tested, not assumed.
- GPU compute peak (register loop, flops.c): 594 GMAC/s fp16.
- Custom GEMM, four generations (cluster/gpu/): naive 1 -> register-tiled 2 -> shared-memory tiled 27 ->
  K-major + double-buffered (best-gemm.cl) ~50 GMAC/s. Plateaus at 50 across sizes = 8.4% of peak. Naive versions were
  slow from UNCOALESCED access (work-item strided by K); tiling fixed it. 50 is the ceiling because Gen9 (UHD 630) has
  no matrix engine (XMX/DPAS is Gen12+/Arc). Still ~3x slower than the CPU's 145 GMAC/s (4 cores).
- Run on GPU only (generation = GEMV): GPU Q4_0 GEMV (gemvbench.c) does 8 GB/s vs CPU ~25 -> GPU-only generation ~3x
  SLOWER, and even a perfect GEMV only equals the memory ceiling the CPU already hits.
- So the GPU cannot help generation, and for prompt GEMM would at best add ~50 to the CPU's 145 (1.34x) if perfectly
  overlapped, while stealing memory + thermal headroom. GPU stays off; kernels kept for a future Gen12+/Arc/M-series GPU.
- The NEO driver (23.43) is NOT the problem: it hit 594 GMAC/s on the register loop (near hardware peak). The wall is
  the missing matrix hardware.

### Round 8 (2026-09-28): GPU GEMM to 10.8% of peak
Pushed the custom fp16 GEMM from 50 to 64 GMAC/s = 10.8% of the 595 GMAC/s hardware peak (gpu/best-gemm.cl, sgemm7).
Correctness vs exact double precision: rel error 7e-7. The wins, in order found:
- Coalesced tiling (shared memory) : 4 -> 50 GMAC/s (naive kernels strided by K = ~1 GB/s uncoalesced).
- SIMD8 (compiler auto-picks it; forcing SIMD16/32 spills the 64-reg 8x8 tile -> 4 GMAC/s).
- Double buffering the global->SLM load.
- **Wide SLM reads**: each 8-half fragment read as ONE 16-byte uint4 + unpack, not 8 half-rate scalar reads: 50 -> 64.
- Best config: BM128 BN128 BK32/64, 8x8 micro-tile, half SLM, double-buffered.
- Dead ends measured: register prefetch (spills, 2 GMAC/s), float SLM (halves occupancy, 45), 4x4/8x4 tiles (29-43),
  256-GRF flag (no change).
Perspective: 64 GMAC/s is real and near the documented Gen9-GT2 GEMM ceiling (no matrix engine), but still < the CPU's
145 (4 cores). During prompt (compute-bound) CPU+GPU could stack to ~200 (=1.4x prompt) if overlapped; generation
(memory-bound) still cannot use the GPU. So this is a proven, reusable kernel (great on a future Gen12+/Arc/M-series
GPU) rather than a win on these exact 2018 Macs. Tools: gpu/sgemmbench.c, best-gemm.cl, vcheck.c.

### Round 9 (2026-09-28): the true GPU peak (why 500 GMAC/s is impossible here)
Rigorous peak test (gpu/peak.c) + the silicon spec settle what the ceiling actually is:
- UHD 630 = 24 EUs x 16 fp32 flop/clock x 1.05 GHz = ~403 GFLOP fp32 = **~200 GMAC/s fp32, ~400 GMAC/s fp16**.
  (Intel's published UHD 630 figure is ~430-460 GFLOP fp32 = ~215-230 GMAC.)
- The "585/595" and even peak.c's "503" are SYNTHETIC-LOOP ARTIFACTS: the compiler simplifies the fake loop, so the
  number printed is above the chip's real ALU throughput. They are not an achievable ceiling.
- A correct GEMM accumulates in fp32 (fp16 accumulation over K=2048 loses too much precision), so its ceiling is the
  ~200 GMAC/s fp32 peak. Our best kernel: **64 GMAC/s = 32% of the fp32 peak** -- a legitimately good iGPU GEMM
  efficiency (published Gen9 GEMM is typically 30-50%). fp16-accumulate (v8) was tested and is SLOWER (52), proving the
  kernel is SLM/issue-bound, not FMA-bound -- so more compute tricks cannot help.
- Therefore 500 GMAC/s is ~2.5x the chip's entire fp16 arithmetic capacity: physically impossible, not a tuning gap.
  "Within 10% of the REAL peak" (~200 fp32) would be ~180; the practical Gen9 GEMM ceiling is ~100-120. Reaching that
  from 64 needs eliminating SLM traffic (block-read register GEMM) -- possible future work, ~1.5-2x at most, still far
  under 500 and still < the CPU's 145.

### Round 10 (2026-09-28): where a token's time really goes, integer path, the honest levers
- Integer/Q4 path: Gen9 UHD 630 has NO hardware dot-product (DP4A is Gen11+); int MACs measure ~= fp32. So running
  Q4/Q5 does not unlock a higher GPU compute ceiling. (Q4/Q5 DOES help by reading fewer bytes -- that is why generation
  uses it -- but the GPU still cannot add memory bandwidth: CPU+GPU share one ~30 GB/s controller.)
- Per-token generation breakdown (opprof, DeepSeek): the biggest single cost is MAP_CUSTOM1 = the Mac-to-Mac allreduce
  over the 1GbE cable (~12% of time in production, ~40% under the profiler). The matmuls run at 20-24 GB/s (near their
  ~26 standalone). So generation is ~69% of its 2-Mac memory-floor ceiling (~22 of ~32 t/s); the main non-memory cost
  is the network sync.
- Therefore the real generation multipliers on THIS hardware: Thunderbolt (cuts the ~5.7ms/token sync to <1ms -> ~1.14x)
  and a 3rd Mac (each reads 2/3 of the weights -> ~1.4x). Together ~1.6x -> ~35 t/s. Both are the user's own plan.
- "Overlap read with compute" (prefetch): real for hiding latency, but generation is bandwidth-bound -- the bus is
  already ~75-90% busy during matmuls, so prefetch adds little. The actual idle is the network allreduce, which prefetch
  cannot help; Thunderbolt / a fused-eager allreduce is the fix.
- CPU+GPU stacking: impossible for generation (shared memory bus), but real for PROMPT (compute-bound): CPU 145 + GPU 64
  = ~1.4x prompt. That is the one place the iGPU genuinely helps and is worth wiring in.

### The memory hierarchy: RAM vs SSD vs Ethernet (measured 2026-09-29)
The whole cluster design is about keeping weights on the fastest path and only shipping the small pieces over the slow one.
| path | measured speed | role |
|---|---|---|
| RAM (per Mac) | ~26 GB/s | holds this Mac's HALF of the weights; read every token |
| NVMe SSD (Apple AP0128M) | ~1.8 GB/s | 14x slower than RAM -- only for models too big for RAM |
| Ethernet cable | 0.125 GB/s (1 GbE) | carries ONLY the 4-8 KB activation sums per layer, never weights |
| Thunderbolt (planned) | ~1-2 GB/s | same role as Ethernet, ~10x faster |

**Why weights never go over Ethernet:** each Mac streams its half from its own RAM (26 GB/s) and the cable only combines
the small per-layer results (allreduce). Shipping weights over 125 MB/s Ethernet would cap generation at ~0.15 tok/s;
streaming them from local RAM is what makes ~22 tok/s possible.

**SSD streaming (MMAP=1 in start-ep.sh):** for a model that FITS in RAM, streaming weights from SSD is a ~5x slowdown
(0.41 GB/token / 1.8 GB/s = 228 ms/token = ~4-5 tok/s vs ~21 from RAM), so it is off by default. Its real use is
**capacity, not speed**: it lets you run a model BIGGER than combined RAM. `MMAP=1 ./start-ep.sh <model>` memory-maps the
weights so the OS streams the needed ones from SSD on demand and caches the hot ones. MoE models suit this best (only a
few experts per token are read), so a ~30B-class MoE can run on 16 GB of RAM at reduced speed -- "runs at all" on trash
hardware instead of "impossible". The right next step is a hot-experts-in-RAM / cold-experts-on-SSD cache with prefetch
overlapping SSD reads with compute (SSD latency CAN be hidden behind compute, unlike RAM bandwidth).

**GPU-from-RAM + CPU-from-SSD in parallel** (summing bandwidth): measured envelope is ~28 GB/s vs 26 RAM-only (~8%),
and SSD DMA still consumes RAM write bandwidth, so the real gain is smaller than that. Not worth the complexity for
RAM-fitting models; the win is capacity (above), not throughput.

### Things we learned the hard way

- Use the **CPU build** as the client. The Vulkan-built client also grabs the local GPU.
- Default `--repack 1` copies CPU weights into RAM (can't be dropped, only swapped). Use `--repack 0 -lm mmap` when one node holds most of the model.
- Start RPC servers with `systemd-run --user` (or fully detached). Backgrounded over SSH they die.
- Every node must run the **same llama.cpp commit** (RPC protocol must match). Current commit: see `git -C ~/llama.cpp log -1`.
- The RPC server has no authentication. Only bind it to Tailscale or loopback addresses.

## Next steps

- **Jetson AGX Xavier 16GB**: build llama.cpp with CUDA at the same commit (2145525), run `ggml-rpc-server`, add it to `--rpc`. Check JetPack version first (`cat /etc/nv_tegra_release`). It has no Thunderbolt; link it over its RJ45 gigabit port, or its front USB-C port (shows up as a USB network link, Jetson side 192.168.55.1).
- **Thunderbolt link node1 ↔ node2**: both have the `thunderbolt_net` driver and two Thunderbolt controllers. Needs a Thunderbolt 3 cable plugged in, then it shows up as a network interface.
- **Bluetooth keyboard/mouse test** on node1 — not done yet.

## Security to-dos

- Set strong passwords for login, sudo, and Samba before this touches a school network (the dev setup used trivial ones). Samba: `sudo smbpasswd <user>`.
- Tailscale admin console: disable key expiry for node1/node2; add an `ssh` ACL block if you want `tailscale ssh`.
