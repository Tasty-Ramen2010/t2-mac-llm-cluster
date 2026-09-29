#!/bin/bash
# gpt-oss-20B, tensor-parallel across node1 + node2 (both work on every word at once).
# Chat page / API: http://100.82.180.15:8080 (Tailscale only). Tools page: http://100.82.180.15:8081
# REASONING=medium/high for deeper thinking (default low).
set -e
NODE2=node2@10.10.10.2  # plain ssh over the cable (Tailscale SSH may ask for a browser re-check)
# tensor-parallel link: Thunderbolt (tb-link.sh up) when it answers, else the Ethernet cable
EPNET=10.10.10.1
ping -c1 -W1 10.30.30.2 >/dev/null 2>&1 && EPNET=10.30.30.1
# usage: start-ep.sh          -> gpt-oss-20B (~17.5 words/s)
#        start-ep.sh gemma    -> Gemma 3 12B QAT Q4_0 (~5.9 words/s)
#        start-ep.sh mellum | mellum-think | qcoder14 | qcoder7 | dscoder -> coding-model candidates (see bench/)
case "${1:-gpt-oss}" in
  mellum|mellum-think|qcoder14|qcoder7|dscoder)   # coding-model candidates (dense / MoE tensor parallel, split with ep/slice_dense_tp.py)
         EP="LLAMA_EP_ADDR=$EPNET:50060 LLAMA_EP_MODE=tp LLAMA_EP_VOCAB=split"
         M0=/home/node1/models/$1-tp-r0.gguf; M1=/home/node2/models/$1-tp-r1.gguf; EXTRA=()
         # DeepSeek-Coder-V2's own template has no tools (llama-server dropped them): use ours with Hermes-style tool calls
         [ "${1#ds}" != "$1" ] && EXTRA=(--jinja --chat-template-file /home/node1/cluster/templates/deepseek-coder-v2-tools.jinja)
         # iGPU attention co-worker (ggml-cpu/igpu.cpp), IGPU=1 to enable: once a chat holds 5000+ tokens the UHD 630 takes 30% of
         # the attention rows. Measured 3x2 runs at 8.5k: 12.97 vs 13.74 t/s without it (-5.6%), so it is off by default.
         [ "${1#ds}" != "$1" ] && [ "${IGPU:-0}" = 1 ] && EPX="GGML_IGPU_ATTN=0.3 GGML_IGPU_ATTN_MIN=5000 ${EPX:-}" ;;
  gemma) EP="LLAMA_EP_ADDR=$EPNET:50060 LLAMA_EP_MODE=tp LLAMA_EP_VOCAB=split"
         M0=/home/node1/models/gemma12qat-tp-r0.gguf; M1=/home/node2/models/gemma12qat-tp-r1.gguf; EXTRA=()
         SWA=small ;;   # full-size memory for Gemma's 40 sliding-window layers would need ~3.2 GB per Mac (it swaps)
  *)     EP="LLAMA_EP_ADDR=$EPNET:50060 LLAMA_EP_MODE=tp LLAMA_EP_VOCAB=split LLAMA_EP_ATTN=split"
         M0=/home/node1/models/gptoss-tpa1280-r0.gguf; M1=/home/node2/models/gptoss-tpa1280-r1.gguf
         EXTRA=(--reasoning-effort "${REASONING:-low}") ;;
esac
EP="$EP ${EPX:-}"
# thread polling per model (measured): 100 helps gpt-oss (+2%) and DeepSeek (+4%), costs Mellum 3% / Qwen 1%
case "${1:-gpt-oss}" in mellum*|qcoder*|gemma) POLL_DEF=50 ;; *) POLL_DEF=100 ;; esac
# Sliding-window models (gpt-oss, Mellum, Gemma): "full" keeps full-size memory for those layers so a chat that changes
# near its end only re-reads the change (otherwise, with no checkpoints, llama-server re-reads the whole chat).
# Costs ~0.2-0.4 GB per Mac for gpt-oss/Mellum. Leader and follower must agree. Models without SWA ignore it.
if [ "${SWA:-full}" = full ]; then SWA_L=(--swa-full); SWA_F=""; else SWA_L=(); SWA_F="--swa-small"; fi
# MMAP=1: memory-map the weights instead of loading them into RAM with --repack. For a model that FITS in RAM this is
# slower (no repacked kernels), but it lets you run a model BIGGER than combined RAM: the OS streams the needed weights
# from the NVMe SSD (~1.8 GB/s here) on demand and caches the hot ones. MoE models stream best (few experts per token).
# Expect ~4-6 tok/s when weights come from SSD vs ~20 from RAM -- the point is "runs at all", not speed.
if [ "${MMAP:-0}" = 1 ]; then RP_L=(--no-repack); RP_F=""; else RP_L=(--repack); RP_F="--repack --no-mmap"; fi   # follower (ep-run) defaults to no-repack+mmap when given neither
# n-gram drafting (SPEC=0 to disable): when the answer repeats text already in the chat (editing / reprinting code), the
# next tokens are guessed from it and checked in one pass. Needs a 16-token match, guesses 16 (defaults 12/48 cost MoE
# models ~9% on summaries). gpt-oss: code edits 16 -> 27 t/s, fresh text unchanged, summaries -2%;
# fresh text ~unchanged. The model checks every guessed token, so answers keep the same quality.
[ "${SPEC:-1}" = 1 ] && SPEC_L=(--spec-type ngram-map-k --spec-ngram-map-k-size-n 16 --spec-ngram-map-k-size-m 16) || SPEC_L=()
# pin each worker thread to its own core (PIN=0 to disable): node1's 3 decode threads on cpu 0-2 (cpu 3 keeps the network
# interrupts, see llm-lowlatency.service, plus the web server), prompt threads unpinned; node2's 4 threads on cpu 0-3.
# +2-3% writing speed. --poll: how long worker threads spin between operations instead of sleeping (per model, see POLL_DEF).
if [ "${PIN:-1}" = 1 ]; then
  PIN_L=(--cpu-mask $(printf 0x%x $(( (1 << ${T1:-3}) - 1 ))) --cpu-strict 1 --cpu-mask-batch 0xf --cpu-strict-batch 0 --poll ${POLL:-$POLL_DEF}); PIN_F="--cpumask 0,1,2,3 --poll ${POLL:-$POLL_DEF}"
else PIN_L=(); PIN_F=""; fi

KEEP_AGENT="${KEEP_AGENT:-}" "$(dirname "$0")/stop-llm.sh" >/dev/null 2>&1 || true   # KEEP_AGENT=1: switching models from the web app
systemctl --user stop llm-ep 2>/dev/null || true

# Reserved 2 MiB pages for the weights (HUGETLB=1; off by default: node1 lacks the free RAM to reserve them all, no gain): free caches, defragment, then reserve one page per 2 MiB of
# this Mac's slice. ggml takes its big weight buffers from them (GGML_HUGETLB=1), so they can't end up in 4 KiB pages
# on a fragmented Mac (node1 got only ~60% transparent huge pages) and can't be swapped out. stop-llm.sh gives them back.
if [ "${HUGETLB:-0}" = 1 ]; then
  reserve='sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; echo $1 > /proc/sys/vm/nr_hugepages'
  sudo sh -c "$reserve" _ $(( $(stat -c %s $M0) / 2097152 + 32 ))
  ssh "$NODE2" "sudo sh -c '$reserve' _ \$(( \$(stat -c %s $M1) / 2097152 + 32 ))"
  EP="$EP GGML_HUGETLB=1"
fi

# GPU clock pinned to its maximum while a model runs (no ramp-up from 350 MHz per word); stop-llm.sh releases it
gpumax='for c in /sys/class/drm/card[0-9]; do [ -f $c/gt_min_freq_mhz ] && cat $c/gt_RP0_freq_mhz > $c/gt_min_freq_mhz; done'
sudo sh -c "$gpumax" 2>/dev/null; ssh "$NODE2" "sudo sh -c '$gpumax'" 2>/dev/null

# follower: replays node1's calls on its half of the model
ssh "$NODE2" "systemctl --user stop ep-follower 2>/dev/null; systemctl --user reset-failed ep-follower 2>/dev/null; \
  systemd-run --user --unit=ep-follower --collect --setenv=LLAMA_EP_RANK=1 $(for kv in $EP; do printf -- '--setenv=%s ' "$kv"; done) \
  /home/node2/llama.cpp/build/bin/llama-ep-run -m $M1 --ctrl $EPNET:50061 --follow -t 4 -c 16384 $RP_F $SWA_F $PIN_F ${EPF_ARGS:-} >/dev/null"

# leader: normal llama-server; libllama mirrors every decode/memory call to the follower.
systemctl --user reset-failed llm-ep 2>/dev/null || true
systemd-run --user --unit=llm-ep --collect --setenv=LLAMA_EP_RANK=0 $(for kv in $EP; do printf -- '--setenv=%s ' "$kv"; done) \
  --setenv=LLAMA_EP_CTRL=$EPNET:50061 \
  /home/node1/llama.cpp/build/bin/llama-server -m $M0 -ngl 0 -t ${T1:-3} -tb 4 \
  -c 16384 -b 512 -ub 512 -np 1 --ctx-checkpoints 0 --cache-ram 0 --no-warmup --metrics -lm none "${RP_L[@]}" "${SWA_L[@]}" "${SPEC_L[@]}" "${PIN_L[@]}" ${EPL_ARGS:-} \
  "${EXTRA[@]}" --host 100.82.180.15 --port 8080 >/dev/null

sudo ufw allow in on tailscale0 to any port 8080 proto tcp comment 'llama-server via tailscale only' >/dev/null
if [ -z "${KEEP_AGENT:-}" ]; then
  systemctl --user reset-failed llm-agent 2>/dev/null || true
  systemd-run --user --unit=llm-agent --collect --working-directory=/home/node1/cluster/agent /usr/bin/python3 /home/node1/cluster/agent/agent_server.py >/dev/null
fi
echo "Plain chat: http://100.82.180.15:8080   |   Chat with web + commands: http://100.82.180.15:8081"
echo "Loading (about a minute). Follow with: journalctl --user -u llm-ep -f"
