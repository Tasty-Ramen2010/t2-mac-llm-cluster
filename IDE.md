# Cluster IDE: a terminal coding agent on the AGX + Macs

A coding team you open from any terminal. A big **planner** model (Qwen3.6-35B-A3B on the Jetson-class AGX GPU) plans, reads
and tests; fast **worker** models (Maple, ternary MoE, one replica per Mac) write the code it delegates. Work happens in a
sandboxed per-conversation workspace on the cluster, so the machine you type on stays clean.

```
 your terminal / friend's PC ──HTTPS+login──▶ Tailscale Funnel ──▶ authproxy (127.0.0.1:8082)
                                                                        │ passphrase + authenticator code, allowlisted API
                                                                        ▼
                                                              agent_server (node1:8081)  +  ide_plugin.py
                                                               │ planner calls              │ delegate_code
                                                               ▼                            ▼
                                       node1:8080 ─forward─▶ AGX llama-server        router 127.0.0.1:8095
                                       (Qwen3.6-35B-A3B, CUDA)                        ├─ Maple on node1 (CPU)
                                                                                      └─ Maple on node2 (CPU)
```

## What is new (all additive; the existing app is unchanged)
| file | purpose |
|---|---|
| `agent/ide_plugin.py` | adds two tools to the agent: `delegate_code` (task → Maple → code, optionally written straight to a workspace file) and `git_clone` (private GitHub via a server-side read-only token). Loaded by 4 lines at the end of `agent_server.py`. |
| `agent/authproxy.py` | login proxy for the public address. Stdlib only. Passphrase (scrypt) **and** TOTP code, session tokens (12 h), per-IP and global lockouts, replay protection, allowlist of API paths (no model switching, benchmarks or web page). |
| `agent/ide_client.py`, `agent/ide_bootstrap.py`, `build_pyz.sh` | the client: `build_pyz.sh` bundles the client, the `ai` terminal UI and `rich` into one `agent/ide.pyz`; the proxy serves it at `/ide.pyz` and the installer at `/i`. Needs only Python 3.8+. |
| `minions/minions_mcp.py` | MCP server for Claude Code (see below). |
| `router/tcpfwd.py` | tiny TCP forwarder (planner port → AGX; also used on the AGX to expose its model server on the cable and Tailscale only). |
| `start-ide.sh` / `stop-ide.sh` | bring the IDE services up/down on node1 (planner forward, Maple replica(s), router, login proxy, agent restart). |

## Quick start (what you actually type)
- **Anywhere, first time:** `curl -sL https://<public address>/i | python3`  → installs the short command **`ide`** (one self-contained file, no pip, works behind school/corporate networks that only allow normal HTTPS to Cloudflare). **Afterwards just type `ide`.** Login = passphrase + authenticator code.
- The public address is a Cloudflare quick tunnel (`~/.config/ide/public_url` on node1; `journalctl --user -u ide-tunnel | grep trycloudflare`). It stays the same while the tunnel runs; if node1 reboots or the tunnel restarts, run `./start-ide.sh` and use the new address it prints.
- On a network that inspects HTTPS and breaks certificate checks: `ide --insecure`. Behind a proxy: the client follows the system/`HTTPS_PROXY` settings. `ide --update` fetches a newer client; `ide --logout` forgets the login.
- From Claude Code: see **Claude Code integration (MCP)** below.

## Context windows and speed (AGX planner)
Switch on the AGX with `ssh agx /mnt/persistent/data/agx-setup/agx-profile fast|long|max` (restart takes ~40 s; no argument shows the current one):

| profile | context | conversation slots | KV cache | notes |
|---|---|---|---|---|
| `fast` | 64k | 2 | f16 | biggest prompt batch (prompt ~985 tok/s) |
| `long` | 128k | 1 | 8-bit | ~24.5 GB used |
| `max` (default) | **256k** | 1 | 4-bit (+ MTP draft) | ~24-25 GB used, 667 tok/s prompt at 41k depth, 32 tok/s decode at 41k depth, 55 tok/s shallow |

Generation uses MTP speculative decoding (~61 tok/s on code, 2 drafted tokens, 93% accepted). Memory is the hard limit (28 GB shared by CPU+GPU): the 256k limit comes from KV cache + attention buffers + the MTP draft context, which is why the long profiles use quantised caches and a smaller batch. Never experiment without a memory watchdog: when the kernel runs out of memory it kills Wi-Fi/Tailscale first.
Maple workers run 64k unified context (32k per busy conversation) with an 8-bit KV cache.

## Claude Code integration (MCP): `minions/minions_mcp.py`
Lets Claude Code (or any MCP client) be the orchestrator and hand work to the minions. Pure standard library.
- Install: `claude mcp add minions --scope user -- python3 /path/to/minions_mcp.py` (needs the Mac to be on the tailnet; endpoints in `~/.config/minions/config.json`).
- Tools: `minion_status`, `minion_set_mode`, `minion_ask` (big AGX model), `minion_code` (Maple worker, optional `write_to` local file), `minion_agent` (full cluster agent: plans, delegates, runs and tests in the sandbox, returns answer + files).
- **Modes** (`minion_set_mode` or `python3 minions_mcp.py mode <m>`): `off` = Claude works alone, `big` = only the big model, `big+maple` = everything.

## One-time setup
1. **AGX serves the planner** (already done on the AGX, boot-persistent): llama-server on 127.0.0.1, forwarded to the cable
   (10.10.13.1:8080) and its Tailscale address.
2. **Login** (on node1): `python3 ~/cluster/agent/authproxy.py init` — passphrase + authenticator secret. A short passphrase needs `--allow-short`; the authenticator code is always required, which is what keeps a short passphrase safe on a public address. Non-interactive: `init --passphrase P --allow-short`.
3. **GitHub access** (for private repos): the server uses node1's `gh auth login` (token never reaches the model); or a *fine-grained read-only token* in `~/.config/ide/github_token` (preferred, least privilege): read-only
   (Contents: read, Metadata: read), limited to the repos you want, then on node1:
   `mkdir -p ~/.config/ide && chmod 700 ~/.config/ide && read -rs T && printf %s "$T" > ~/.config/ide/github_token && chmod 600 ~/.config/ide/github_token`.
   The token never leaves node1: not the friend's PC, not the model's sandbox (clones run on the server, outside the sandbox).
4. `./start-ide.sh` (set `NODE2=node2@100.109.16.15` once ssh to node2 works for a second Maple).
5. **Public address**: `start-ide.sh` starts a Cloudflare quick tunnel (`~/.local/bin/cloudflared`, no account) to the login proxy and prints the address. (Tailscale Funnel also works but its hostname is published in public certificate logs; a named Cloudflare tunnel on your own domain gives a permanent address.)

## Using it
- On node1 / any tailnet device: `ai` (existing terminal UI; the planner is now the AGX 35B).
- From anywhere: `curl -fsSL https://YOUR-HOST/ide/ide_client.py -o ide_client.py && python3 ide_client.py --url https://YOUR-HOST`
  (first run asks passphrase + code; later runs reuse the token for 12 h; `--logout` forgets it).
- Ask for work in plain language. The planner splits it, calls `delegate_code` for the pieces (each Maple call shows its
  speed), runs the code and fixes it. "Clone owner/private-repo and run its tests" uses `git_clone`.

## Security notes
- The agent can execute code, so the public door is deliberately narrow: nothing is forwarded without a valid session; the
  allowlist blocks `/api/switch`, `/api/port`, benchmarks and the web page; the code runs as the unprivileged `llmtools` user
  with the existing network gate (new websites need your approval) and resource limits.
- Wrong passphrase **or** code counts as one failure; 5 failures per address in 15 min, or 30 per hour overall, lock logins.
  Each authenticator code works once. `python3 authproxy.py logout-all` ends every session.
- The tunnel makes the login page reachable from the internet. Stop it with `systemctl --user stop ide-tunnel` when not needed. The random Cloudflare address is hard to guess but not secret forever, so the login, not the address, is what protects you.
- The GitHub token is read-only and repo-scoped; revoke it on GitHub if a machine is lost.

## Known limits (v1)
- node2's Maple replica waits for ssh from node1 to node2 (its Tailscale SSH needs a one-time browser approval).
- Code runs in the sandbox on node1 (CPU, 7 GB RAM). GPU jobs on the AGX are a next step.
- The terminal UI needs a POSIX terminal (Linux, macOS, WSL).
