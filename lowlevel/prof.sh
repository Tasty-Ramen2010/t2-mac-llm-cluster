#!/bin/bash
# prof.sh PRESET [PROMPT_CHARS] : decode profile on both Macs (300 tokens), per-op report for node1 and node2
P=$1; C=${2:-40}
SPEC=0 KEEP_AGENT=1 EPX="GGML_CPU_OPPROF=1 LLAMA_EP_PROFILE=1 ${EPX:-}" /home/node1/cluster/start-ep.sh $P >/dev/null 2>&1
until curl -s -m3 http://100.82.180.15:8080/health | grep -q ok; do sleep 3; done
python3 -c "
import json, urllib.request
src = open('/home/node1/cluster/agent/agent_server.py').read()[:$C]
b = {'prompt': src + '\n# Summary:\n', 'n_predict': 300, 'ignore_eos': True, 'temperature': 0, 'cache_prompt': False}
t = json.load(urllib.request.urlopen(urllib.request.Request('http://100.82.180.15:8080/completion', json.dumps(b).encode(), {'Content-Type': 'application/json'}), timeout=1800))['timings']
print('decode', round(t['predicted_per_second'], 2), 't/s at', t['prompt_n'], 'prompt tokens')"
T0=$(date +%s); systemctl --user stop llm-ep; sleep 3
D=/tmp/claude-1000/prof; mkdir -p $D
journalctl --user -u llm-ep --since "@$T0" --no-pager -o cat > $D/$P-node1.txt
ssh node2@10.10.10.2 "journalctl --user -u ep-follower --since '-2min' --no-pager -o cat" > $D/$P-node2.txt
M0=$(ls /home/node1/models/${P/gpt-oss/gptoss-tpa1280}*-r0.gguf 2>/dev/null | head -1); [ "$P" = gpt-oss ] && M0=/home/node1/models/gptoss-tpa1280-r0.gguf
python3 /home/node1/cluster/lowlevel/opreport.py $M0 $D/$P-node1.txt "$P node1"
grep "llama-ep profile" $D/$P-node1.txt; grep "llama-ep profile" $D/$P-node2.txt | sed 's/^/node2: /'
