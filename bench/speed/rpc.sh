#!/bin/bash
# stock kernels (GGML_LEAN_GEMV=0)
# rpc.sh MODEL.gguf THREADS tailscale|cable : the ORIGINAL setup - llama.cpp layer pipeline over RPC (no web backend change)
set -e
MODEL=$1; T=$2; NET=$3
[ "$NET" = cable ] && N2=10.10.10.2 || N2=100.109.16.15
KEEP_AGENT=1 /home/node1/cluster/stop-llm.sh >/dev/null 2>&1 || true
ssh node2@10.10.10.2 "systemctl --user stop rpc-cpu 2>/dev/null; systemctl --user reset-failed rpc-cpu 2>/dev/null; systemd-run --user --unit=rpc-cpu --collect --setenv=GGML_LEAN_GEMV=0 ~/llama.cpp/build/bin/ggml-rpc-server -H $N2 -p 50052 -t $T" >/dev/null
systemctl --user reset-failed rpc-cpu-local llm-server 2>/dev/null || true
systemd-run --user --unit=rpc-cpu-local --collect --setenv=GGML_LEAN_GEMV=0 /home/node1/llama.cpp/build/bin/ggml-rpc-server -H 127.0.0.1 -p 50051 -t $T >/dev/null
sleep 2
systemd-run --user --unit=llm-server --collect --setenv=GGML_LEAN_GEMV=0 /home/node1/llama.cpp/build/bin/llama-server -m "$MODEL" \
  --rpc 127.0.0.1:50051,$N2:50052 -ngl 999 -c 4096 --host 100.82.180.15 --port 8080 >/dev/null
