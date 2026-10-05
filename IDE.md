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
| `agent/ide_client.py` | the client you run anywhere: logs in, fetches the matching `ai` terminal UI from the server and runs it through a local authenticated shim. Needs Python 3.8+ and `rich`. |
| `router/tcpfwd.py` | tiny TCP forwarder (planner port → AGX; also used on the AGX to expose its model server on the cable and Tailscale only). |
| `start-ide.sh` / `stop-ide.sh` | bring the IDE services up/down on node1 (planner forward, Maple replica(s), router, login proxy, agent restart). |

## One-time setup
1. **AGX serves the planner** (already done on the AGX, boot-persistent): llama-server on 127.0.0.1, forwarded to the cable
   (10.10.13.1:8080) and its Tailscale address.
2. **Login** (on node1): `python3 ~/cluster/agent/authproxy.py init` — choose a passphrase (12+ chars) and add the printed
   secret to an authenticator app.
3. **GitHub access** (optional, for private repos): create a *fine-grained personal access token*, read-only
   (Contents: read, Metadata: read), limited to the repos you want, then on node1:
   `mkdir -p ~/.config/ide && chmod 700 ~/.config/ide && read -rs T && printf %s "$T" > ~/.config/ide/github_token && chmod 600 ~/.config/ide/github_token`.
   The token never leaves node1: not the friend's PC, not the model's sandbox (clones run on the server, outside the sandbox).
4. `./start-ide.sh` (set `NODE2=node2@100.109.16.15` once ssh to node2 works for a second Maple).
5. **Public address**: in the Tailscale admin console (personal tailnet) enable HTTPS certificates and allow Funnel for node1, then
   `tailscale funnel --bg --https=443 http://127.0.0.1:8082`. The URL printed is the one you give the client.

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
- Funnel makes the login page reachable from the internet. If you stop using remote access, run `tailscale funnel reset`.
- The GitHub token is read-only and repo-scoped; revoke it on GitHub if a machine is lost.

## Known limits (v1)
- node2's Maple replica waits for ssh from node1 to node2 (its Tailscale SSH needs a one-time browser approval).
- Code runs in the sandbox on node1 (CPU, 7 GB RAM). GPU jobs on the AGX are a next step.
- The terminal UI needs a POSIX terminal (Linux, macOS, WSL).
