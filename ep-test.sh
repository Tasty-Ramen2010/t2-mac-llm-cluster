#!/bin/bash
# usage: ep-test.sh MODEL_BASENAME N_TOKENS "prompt"
M=$1; N=${2:-64}; P=${3:-Explain in two sentences why the sky is blue.}
ssh node2@100.109.16.15 "systemctl --user stop ep-follower 2>/dev/null; systemctl --user reset-failed ep-follower 2>/dev/null; systemd-run --user --unit=ep-follower --collect --setenv=LLAMA_EP_RANK=1 --setenv=LLAMA_EP_ADDR=10.10.10.1:50060 --setenv=LLAMA_EP_RANGE=16-31 $( [ -n "$EP_PROFILE" ] && echo --setenv=LLAMA_EP_PROFILE=1 ) /home/node2/llama.cpp/build/bin/llama-ep-run -m /home/node2/models/$M --ctrl 10.10.10.1:50061 -t 4 >/dev/null"
$( [ -n "$EP_PROFILE" ] && echo env LLAMA_EP_PROFILE=1 ) LLAMA_EP_RANK=0 LLAMA_EP_ADDR=10.10.10.1:50060 LLAMA_EP_RANGE=0-15 /home/node1/llama.cpp/build/bin/llama-ep-run -m /home/node1/models/$M --ctrl 10.10.10.1:50061 -t 4 -n "$N" -p "$P" 2> /tmp/ep-leader.err
grep -E "ep-run:|llama-ep:" /tmp/ep-leader.err
