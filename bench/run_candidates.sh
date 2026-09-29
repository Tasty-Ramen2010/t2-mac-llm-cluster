#!/bin/bash
# benchmark every coding-model candidate in turn (each one tensor-parallel on both Macs), then restore gpt-oss
cd /home/node1/cluster/bench
while systemctl --user is-active -q bench-gptoss; do sleep 30; done      # wait for the gpt-oss runs
for m in ${MODELS:-mellum mellum-think qcoder7 qcoder14}; do
  echo "=== $m $(date +%T)"
  KEEP_AGENT=1 /home/node1/cluster/start-ep.sh $m >/dev/null 2>&1      # KEEP_AGENT: the web app stays up
  for i in $(seq 120); do curl -sf http://100.82.180.15:8080/health >/dev/null && break; sleep 5; done
  # sanity: a trivial question must get a sensible answer, otherwise skip the (long) benchmark
  ans=$(curl -s http://100.82.180.15:8080/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"What is 7 times 8? Reply with just the number."}],"max_tokens":400,"temperature":0}' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[-80:])' 2>&1)
  echo "sanity answer: $ans"
  if ! echo "$ans" | grep -q 56; then echo "SKIP $m: failed sanity check"; continue; fi
  python3 bench_code.py $m > results/$m.log 2>&1
  tail -1 results/$m.log
done
KEEP_AGENT=1 /home/node1/cluster/start-ep.sh ${RESTORE:-} >/dev/null 2>&1          # back to gpt-oss (or RESTORE=preset) for chatting
echo "=== all done $(date +%T)"
