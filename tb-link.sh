#!/bin/bash
# tb-link.sh [up|test|down]: direct Thunderbolt 3 networking between node1 and node2 (plug a Thunderbolt 3/4 cable,
# the one with the lightning bolt, into any Thunderbolt port of each Mac). thunderbolt_net makes a network interface
# on each side; we give it 10.30.30.1 / 10.30.30.2. start-ep.sh then sends the tensor-parallel syncs over it
# (10-20 Gbit/s instead of the 1 Gbit/s Ethernet cable). The Ethernet cable stays for ssh/control.
set -e
N2=node2@10.10.10.2
tbif() { ls /sys/class/net | grep -E '^thunderbolt|^tb' | head -1; }
case "${1:-up}" in
up)
  for H in local node2; do
    C='sudo modprobe thunderbolt_net; for i in $(seq 20); do IF=$(ls /sys/class/net | grep -E "^thunderbolt|^tb" | head -1); [ -n "$IF" ] && break; sleep 1; done
       [ -z "$IF" ] && { echo "no Thunderbolt network interface: is a Thunderbolt cable connected?"; exit 1; }
       sudo ip addr flush dev $IF; sudo ip addr add ADDR/24 dev $IF; sudo ip link set $IF up mtu 65520 2>/dev/null || sudo ip link set $IF up
       echo "$(hostname): $IF up with ADDR, $(cat /sys/class/net/$IF/mtu) byte packets"'
    if [ $H = local ]; then bash -c "${C//ADDR/10.30.30.1}"; else ssh $N2 "${C//ADDR/10.30.30.2}"; fi
  done
  IF=$(tbif)   # node1's firewall: let node2 reach the tensor-parallel ports over Thunderbolt (like the Ethernet cable)
  sudo ufw allow in on $IF from 10.30.30.2 to any port 50060:50061 proto tcp comment 'tensor-parallel links from node2 (thunderbolt)' >/dev/null
  sleep 2; ping -c 200 -i 0.002 -q 10.30.30.2 | tail -1 ;;
test)
  L=/home/node1/cluster/lowlevel
  for n in 4104 5768 2097152; do
    ssh $N2 "timeout 60 /tmp/pingpong server 50073 $n spin" & sleep 0.5
    timeout 60 $L/pingpong client 10.30.30.2 50073 $n spin | sed 's/^/thunderbolt: /' || true; wait
  done ;;
down)
  IF=$(tbif); [ -n "$IF" ] && sudo ip link set $IF down; ssh $N2 'IF=$(ls /sys/class/net | grep -E "^thunderbolt|^tb" | head -1); [ -n "$IF" ] && sudo ip link set $IF down' ;;
esac
