#!/bin/bash
# sys.sh gov performance|powersave | net fast|old|default | thp madvise|never   -> applied on BOTH Macs
# (speed-ablation helper; "fast"/"madvise"/"performance" are the normal settings)
set -e
case "$1 $2" in
  "gov performance"|"gov powersave") C="for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $2 > \$g; done" ;;
  "net fast")    C="ethtool --set-eee enp4s0 eee off || true; ethtool -C enp4s0 rx-usecs 1 rx-frames 1 tx-usecs 1 tx-frames 1 || true; for d in 04:00.0 00:1c.1; do v=\$(setpci -s \$d CAP_EXP+10.w); setpci -s \$d CAP_EXP+10.w=\$(printf %04x \$((0x\$v & ~3))); done; for s in /sys/devices/system/cpu/cpu*/cpuidle/state[3-9]; do echo 1 > \$s/disable; done" ;;
  "net old")     C="ethtool --set-eee enp4s0 eee off || true; ethtool -C enp4s0 rx-usecs 1 rx-frames 1 tx-usecs 72 tx-frames 53 || true; for d in 04:00.0 00:1c.1; do v=\$(setpci -s \$d CAP_EXP+10.w); setpci -s \$d CAP_EXP+10.w=\$(printf %04x \$((0x\$v | 2))); done; for s in /sys/devices/system/cpu/cpu*/cpuidle/state[3-9]; do echo 1 > \$s/disable; done" ;;
  "net default") C="ethtool --set-eee enp4s0 eee on || true; ethtool -C enp4s0 rx-usecs 20 rx-frames 5 tx-usecs 72 tx-frames 53 || true; for d in 04:00.0 00:1c.1; do v=\$(setpci -s \$d CAP_EXP+10.w); setpci -s \$d CAP_EXP+10.w=\$(printf %04x \$((0x\$v | 2))); done; for s in /sys/devices/system/cpu/cpu*/cpuidle/state[3-9]; do echo 0 > \$s/disable; done" ;;
  "thp madvise"|"thp never") C="echo $2 > /sys/kernel/mm/transparent_hugepage/enabled" ;;
  *) echo "usage: sys.sh gov|net|thp VALUE"; exit 1 ;;
esac
sudo sh -c "$C" >/dev/null 2>&1
ssh node2@10.10.10.2 "sudo sh -c '$C'" >/dev/null 2>&1
echo "$1 = $2 on both"
