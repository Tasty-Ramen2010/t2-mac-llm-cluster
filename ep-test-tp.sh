#!/bin/bash
# usage: ep-test-tp.sh N_TOKENS "prompt" [extra ep-run args]   (tensor-parallel experts, both nodes)
N=${1:-128}; P=${2:-Write a short paragraph about the history of the bicycle.}; X=${3:-}
ssh node2@100.109.16.15 "systemctl --user stop ep-follower 2>/dev/null; systemctl --user reset-failed ep-follower 2>/dev/null; systemd-run --user --unit=ep-follower --collect --setenv=LLAMA_EP_RANK=1 --setenv=LLAMA_EP_ADDR=10.10.10.1:50060 --setenv=LLAMA_EP_MODE=tp $( [ -n "$EP_PROFILE" ] && echo --setenv=LLAMA_EP_PROFILE=1 ) /home/node2/llama.cpp/build/bin/llama-ep-run -m /home/node2/models/gptoss-q4att-tp1.gguf --ctrl 10.10.10.1:50061 -t 4 $X >/dev/null"
env $( [ -n "$EP_PROFILE" ] && echo LLAMA_EP_PROFILE=1 ) LLAMA_EP_RANK=0 LLAMA_EP_ADDR=10.10.10.1:50060 LLAMA_EP_MODE=tp /home/node1/llama.cpp/build/bin/llama-ep-run -m /home/node1/models/gptoss-q4att-tp0.gguf --ctrl 10.10.10.1:50061 -t 4 -n "$N" -p "$P" $X 2> /tmp/ep-leader.err
grep -E "ep-run: prompt|profile" /tmp/ep-leader.err
