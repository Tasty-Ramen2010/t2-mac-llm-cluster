#!/bin/bash
# Cluster IDE: planner = the AGX (Qwen3.6-35B-A3B on its GPU), workers = Maple replicas on the Macs, login proxy for the public URL.
#   planner : this Mac's old model port 100.82.180.15:8080 now forwards to the AGX (PLANNER=host:port, default: over the cable)
#   workers : Maple on this Mac (127.0.0.1:8090) and, if NODE2 is set, on node2 -- behind a sticky router on 127.0.0.1:8095
#   login   : agent/authproxy.py on 127.0.0.1:8082 (put `tailscale funnel` on it); run `python3 agent/authproxy.py init` once first
# Undo with ./stop-ide.sh, then start-maple.sh / start-ep.sh as before.
set -e
H="$(cd "$(dirname "$0")" && pwd)"
PLANNER="${PLANNER:-10.10.13.1:8080}"
NODE2="${NODE2:-}"                       # e.g. NODE2=node2@100.109.16.15 (needs working ssh from this Mac)
NODE2_IP="${NODE2_IP:-100.109.16.15}"
NP=${NP:-2}; CTX=${CTX:-65536}          # Maple: 64k shared (-kvu) = 32k per conversation when both slots are busy
FLAGS="-ngl 0 -t 3 -tb 3 -c $CTX -kvu -b 512 -ub 512 -np $NP --cpu-mask 0x7 --cpu-strict 1 --cpu-mask-batch 0x7 --cpu-strict-batch 1 --poll 100 --ctx-checkpoints 0 --cache-ram 0 --no-warmup --metrics --jinja -fa on -ctk q8_0 -ctv q8_0"

systemctl --user stop ide-auth llm-planner llm-router llm-maple-fwd llm-server llm-ep llm-maple 2>/dev/null || true
systemctl --user reset-failed ide-auth llm-planner llm-router llm-maple-fwd llm-maple 2>/dev/null || true

systemd-run --user --unit=llm-planner --collect /usr/bin/python3 "$H/router/tcpfwd.py" \
  --listen 100.82.180.15:8080 --to "$PLANNER" >/dev/null
systemd-run --user --unit=llm-maple --collect /home/node1/llama.cpp/build/bin/llama-server \
  -m /home/node1/models/maple-rf16.gguf $FLAGS --host 127.0.0.1 --port 8090 >/dev/null
BACKENDS="--backend 127.0.0.1:8090"
if [ -n "$NODE2" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE2" "systemctl --user stop maple-replica 2>/dev/null; systemctl --user reset-failed maple-replica 2>/dev/null; \
    systemd-run --user --unit=maple-replica --collect /home/node2/llama.cpp/build/bin/llama-server \
    -m /home/node2/models/maple-rf16.gguf $FLAGS --host $NODE2_IP --port 8080 >/dev/null" \
    && BACKENDS="$BACKENDS --backend $NODE2_IP:8080" || echo "node2 not started (ssh failed); running one worker"
fi
systemd-run --user --unit=llm-router --collect /usr/bin/python3 "$H/router/maple_router.py" \
  --listen 127.0.0.1:8095 $BACKENDS --slots $NP >/dev/null
# Maple router reachable on the tailnet (for the minions MCP server on other machines)
systemd-run --user --unit=llm-maple-fwd --collect -p Restart=always /usr/bin/python3 "$H/router/tcpfwd.py" \
  --listen 100.82.180.15:8095 --to 127.0.0.1:8095 >/dev/null
sudo ufw allow in on tailscale0 to any port 8095 proto tcp comment 'maple router via tailscale only' >/dev/null 2>&1 || true
if [ -f "$HOME/.config/ide/auth.json" ]; then
  systemd-run --user --unit=ide-auth --collect -p Restart=always -p RestartSec=3 /usr/bin/python3 "$H/agent/authproxy.py" serve >/dev/null
  # public address for off-tailnet use (school etc.): Cloudflare quick tunnel to the login proxy. Left running if already up, so the address stays the same.
  if [ -x "$HOME/.local/bin/cloudflared" ] && ! systemctl --user is-active --quiet ide-tunnel; then
    systemd-run --user --unit=ide-tunnel --collect -p Restart=always -p RestartSec=5 "$HOME/.local/bin/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8082 >/dev/null
    sleep 10
  fi
  URL=$(journalctl --user -u ide-tunnel --no-pager 2>/dev/null | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1)
  [ -n "$URL" ] && { echo "$URL" > "$HOME/.config/ide/public_url"; echo "public address: $URL   (install: curl -sL $URL/i | python3)"; }
else
  echo "login proxy NOT started: run  python3 $H/agent/authproxy.py init  then start-ide.sh again"
fi
sudo ufw allow in on tailscale0 to any port 8080 proto tcp comment 'llama-server via tailscale only' >/dev/null 2>&1 || true
systemctl --user restart llm-agent          # loads agent/ide_plugin.py (delegate_code, git_clone)
echo "IDE up. planner -> $PLANNER via http://100.82.180.15:8080 | workers: ${BACKENDS//--backend /} | agent http://100.82.180.15:8081"
echo "Terminal:  ai   (anywhere: curl -sL <public address>/i | python3   then just: ide)"
