#!/bin/bash
# slice the coding-model candidates for two-Mac tensor parallel; rank 0 stays on node1, rank 1 goes to node2
set -e
cd /home/node1/models
declare -A SRC=([mellum]=Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf [mellum-think]=Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf
                [qcoder7]=Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf [qcoder14]=Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf)
for name in mellum qcoder7 mellum-think qcoder14; do
  for r in 0 1; do
    [ -f $name-tp-r$r.gguf ] || nice -n 15 ionice -c3 python3 /home/node1/cluster/ep/slice_dense_tp.py ${SRC[$name]} $name-tp-r$r.gguf $r
  done
  scp -q $name-tp-r1.gguf node2@10.10.10.2:/home/node2/models/ && rm $name-tp-r1.gguf
  echo "$name ready"
done
