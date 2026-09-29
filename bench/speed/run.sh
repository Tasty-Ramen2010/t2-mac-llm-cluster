#!/bin/bash
# run.sh PRESET ORIGINAL.gguf : speed ablation for one model. Each step adds one improvement on top of the previous one.
P=$1; ORIG=$2
cd /home/node1/cluster/bench/speed
S=./sys.sh
restore() { $S gov performance; $S net fast; $S thp madvise; }
trap restore EXIT
tp() {   # tp STAGE-NAME  (env: EPX, T1)
  KEEP_AGENT=1 T1=${T1:-3} EPX="$EPX" /home/node1/cluster/start-ep.sh $P >/dev/null 2>&1
  sleep 3; python3 measure.py $P "$1"
}
if [ -n "$ORIG" ]; then
  # 1. original setup: layer pipeline over RPC, default power saving, RPC's default 2 threads, over Tailscale, stock kernels
  $S gov powersave; $S net default; $S thp never
  ./rpc.sh "$ORIG" 2 tailscale; python3 measure.py $P "1 original (RPC pipeline)"
  # 2. tuned pipeline: performance governor, 4 threads, direct cable
  $S gov performance
  ./rpc.sh "$ORIG" 4 cable; python3 measure.py $P "2 + performance CPU, 4 thr, cable"
  KEEP_AGENT=1 /home/node1/cluster/stop-llm.sh >/dev/null 2>&1
fi
$S gov performance; $S net default; $S thp never
EPX="GGML_LEAN_GEMV=0 LLAMA_EP_F16=0" T1=4 tp "3 tensor parallel (both Macs)"
$S thp madvise
EPX="GGML_LEAN_GEMV=0 LLAMA_EP_F16=0" T1=4 tp "4 + huge pages"
EPX="LLAMA_EP_F16=0" T1=4 tp "5 + lean AVX2 kernel"
$S net old
EPX="LLAMA_EP_F16=0" T1=4 tp "6 + low-latency cable"
EPX="" T1=4 tp "7 + fp16 sync"
EPX="" T1=3 tp "8 + node1 3 threads"
$S net fast
EPX="" T1=3 tp "9 + PCIe ASPM off, tx 1us"
