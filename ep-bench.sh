#!/bin/bash
# ep-bench.sh [N] [extra env...]: TP benchmark with llama-ep-run (leader node1, follower node2). Prints output tail + t/s.
N=${1:-128}; shift
ENVS="LLAMA_EP_ADDR=10.10.10.1:50060 LLAMA_EP_MODE=tp LLAMA_EP_VOCAB=split LLAMA_EP_ATTN=split $*"
ssh node2@10.10.10.2 "systemctl --user stop ep-follower 2>/dev/null; systemctl --user reset-failed ep-follower 2>/dev/null; \
  systemd-run --user --unit=ep-follower --collect --setenv=LLAMA_EP_RANK=1 $(for kv in $ENVS; do printf -- '--setenv=%s ' "$kv"; done) \
  /home/node2/llama.cpp/build/bin/llama-ep-run -m ${M1:-/home/node2/models/gptoss-tpa1280-r1.gguf} --ctrl 10.10.10.1:50061 --follow -t ${T2:-4} -c 4096 --repack --no-mmap" >/dev/null
env LLAMA_EP_RANK=0 $ENVS ${PERF:-} /home/node1/llama.cpp/build/bin/llama-ep-run -m ${M0:-/home/node1/models/gptoss-tpa1280-r0.gguf} \
  --ctrl 10.10.10.1:50061 -t ${T1:-3} -n $N --repack --no-mmap ${EXTRA:-} -p "${PROMPT:-Write a detailed essay about the history of computing.}" 2>&1 \
  | if [ -n "$SHOW" ]; then cat; else grep -E "ep-run:|llama-ep profile|opprof"; fi 
