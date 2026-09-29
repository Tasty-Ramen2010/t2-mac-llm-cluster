#!/bin/bash
# measure generation speed when weights are in RAM (warm) vs streamed from SSD (cold page cache), single Mac, mmap.
M=/home/node1/models/mellum-tp-r0.gguf   # ~4GB half, fits in 8GB with room
BIN=/home/node1/llama.cpp/build/bin/llama-cli
run(){ LLAMA_EP_ADDR=solo $BIN -m $M -p "Write a long story about a robot learning to paint." -n 80 -t 3 -c 512 --no-warmup $1 2>/dev/null | tail -3; }
echo "--- WARM (weights cached in RAM) ---"; sync; run "--mmap 1" >/dev/null 2>&1; LLAMA_EP_ADDR=solo $BIN -m $M -p "Story:" -n 60 -t 3 -c 512 2>&1 | grep -iE "eval time|tokens per second" | head -4
echo "--- COLD (drop caches, stream from SSD via mmap) ---"; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
LLAMA_EP_ADDR=solo $BIN -m $M -p "Story:" -n 60 -t 3 -c 512 2>&1 | grep -iE "eval time|tokens per second" | head -4
