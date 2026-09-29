#!/bin/bash
# Start gpt-oss-20b split across node1 + node2 (CPU on both), served by llama-server on node1.
# Chat page / API: http://100.82.180.15:8080 (Tailscale only)
set -e

# usage: start-llm.sh            -> gpt-oss-20B on both Macs, tensor parallel (~17.5 words/s)   [= start-ep.sh]
#        start-llm.sh gemma      -> Gemma 3 12B QAT on both Macs (~5.9 words/s)                  [= start-ep.sh gemma]
#        start-llm.sh file.gguf  -> any other model, old layer-pipeline mode over RPC (slow, for experiments)
EXTRA=()
case "${1:-}" in
  ""|gpt-oss) exec "$(dirname "$0")/start-ep.sh" ;;
  gemma) exec "$(dirname "$0")/start-ep.sh" gemma ;;
  *) MODEL="$1" ;;
esac
NODE2=node2@10.10.10.2

ssh "$NODE2" 'systemctl --user reset-failed rpc-cpu 2>/dev/null; systemd-run --user --unit=rpc-cpu --collect ~/llama.cpp/build/bin/ggml-rpc-server -H 100.109.16.15 -p 50052 -t 4'

systemctl --user reset-failed rpc-cpu-local llm-server 2>/dev/null || true
systemd-run --user --unit=rpc-cpu-local --collect /home/node1/llama.cpp/build/bin/ggml-rpc-server -H 127.0.0.1 -p 50051 -t 4
sleep 2

sudo ufw allow in on tailscale0 to any port 8080 proto tcp comment 'llama-server via tailscale only' >/dev/null

systemd-run --user --unit=llm-server --collect /home/node1/llama.cpp/build/bin/llama-server \
  -m "$MODEL" --rpc 127.0.0.1:50051,100.109.16.15:50052 -ngl 999 "${EXTRA[@]}" \
  --host 100.82.180.15 --port 8080

systemctl --user reset-failed llm-agent 2>/dev/null || true
systemd-run --user --unit=llm-agent --collect --working-directory=/home/node1/cluster/agent /usr/bin/python3 /home/node1/cluster/agent/agent_server.py >/dev/null
echo "Plain chat: http://100.82.180.15:8080   |   Chat with web + commands: http://100.82.180.15:8081"
echo "Starting. Model load takes a few minutes. Follow with: journalctl --user -u llm-server -f"
