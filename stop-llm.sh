#!/bin/bash
# Stop the chat servers first, then the model workers on both nodes (stopping workers first crashes the server).
systemctl --user stop llm-server llm-ep 2>/dev/null
[ -z "${KEEP_AGENT:-}" ] && systemctl --user stop llm-agent 2>/dev/null
systemctl --user stop rpc-cpu-local 2>/dev/null
ssh node2@10.10.10.2 'systemctl --user stop rpc-cpu rpc-vulkan ep-follower 2>/dev/null'
sudo sh -c 'echo 0 > /proc/sys/vm/nr_hugepages' 2>/dev/null                           # reserved weight pages (start-ep.sh)
ssh node2@10.10.10.2 "sudo sh -c 'echo 0 > /proc/sys/vm/nr_hugepages'" 2>/dev/null
gpumin='for c in /sys/class/drm/card[0-9]; do [ -f $c/gt_min_freq_mhz ] && cat $c/gt_RPn_freq_mhz > $c/gt_min_freq_mhz; done'
sudo sh -c "$gpumin" 2>/dev/null; ssh node2@10.10.10.2 "sudo sh -c '$gpumin'" 2>/dev/null   # GPU clock back to normal
echo "Stopped."
