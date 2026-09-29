#!/bin/bash
# all.sh: the whole speed ablation (about 1.5-2 h). Model offline meanwhile; Mellum Thinking is restored at the end.
cd /home/node1/cluster/bench/speed
O=/home/node1/models/orig
./run.sh gpt-oss ""                                                   # original RPC numbers for gpt-oss: README history
./run.sh mellum   $O/Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf
./run.sh qcoder7  $O/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
./run.sh gemma    $O/gemma-3-12b-it-qat-q4_0_s.gguf
./run.sh qcoder14 $O/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf
KEEP_AGENT=1 /home/node1/cluster/start-ep.sh mellum-think >/dev/null 2>&1
echo ALL-DONE
