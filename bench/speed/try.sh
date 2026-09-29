#!/bin/bash
# try.sh PRESET LABEL : restart with the given EPX / EPF_ARGS / EPL_ARGS / T1 env and measure
P=$1; L=$2
KEEP_AGENT=1 T1=${T1:-3} EPX="${EPX:-}" EPF_ARGS="${EPF_ARGS:-}" EPL_ARGS="${EPL_ARGS:-}" /home/node1/cluster/start-ep.sh $P >/dev/null 2>&1
sleep 3; cd /home/node1/cluster/bench/speed && python3 measure.py $P "$L"
