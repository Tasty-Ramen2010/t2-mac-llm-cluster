#!/usr/bin/env python3
"""minions: an MCP server that lets Claude Code (or any MCP client) delegate work to your local cluster.

  big       the AGX planner model (Qwen3.6-35B-A3B, ~60 tokens/s, long context)
  maple     the fast Maple worker models on the Macs (code writers)
  agent     the full cluster IDE agent: planner + workers + a sandboxed workspace that runs and tests code

Modes (switch with the minion_set_mode tool, or `python3 minions_mcp.py mode big+maple`):
  off         Claude works alone; every minion tool refuses and says so
  big         only the big model (minion_ask, minion_agent without Maple)
  big+maple   everything

Add to Claude Code:   claude mcp add minions --scope user -- python3 /path/to/minions_mcp.py
Config: ~/.config/minions/config.json  (mode, big_url, maple_url, agent_url); env MINIONS_<KEY> overrides.
Pure standard library, speaks MCP over stdio (newline-delimited JSON-RPC).
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

CONF_FILE = Path(os.environ.get("MINIONS_CONFIG", str(Path.home() / ".config" / "minions" / "config.json")))
DEFAULTS = {"mode": "big+maple",
            "big_url": "http://100.92.90.4:8080",          # AGX planner (OpenAI-compatible llama-server) on the tailnet
            "maple_url": "http://100.82.180.15:8095",      # Maple router on node1 (tailnet)
            "agent_url": "http://100.82.180.15:8081",      # cluster IDE agent on node1 (tailnet)
            "agx_ssh": "root@100.92.90.4",                 # to switch the AGX context profile (ssh key auth, port 2223)
            "agx_ssh_port": "2223"}
MODES = ("off", "big", "big+maple")
SERVER_INFO = {"name": "minions", "version": "1.0.0"}
INSTRUCTIONS = ("Local 'minion' models you can delegate to to save time and tokens. Use minion_code for self-contained code "
                "pieces (functions, modules, tests, boilerplate), minion_ask for bulk reading/summarising/second opinions, and "
                "minion_agent for a whole self-contained sub-task that should be built AND tested in the cluster's sandbox. "
                "Always review and test what minions return; they are fast but weaker than you. Check minion_status first if a "
                "tool reports a connection problem; if the mode is 'off', do the work yourself.")


def cfg():
    c = dict(DEFAULTS)
    try:
        c.update(json.loads(CONF_FILE.read_text()))
    except (OSError, ValueError):
        pass
    for k in c:
        v = os.environ.get("MINIONS_" + k.upper())
        if v:
            c[k] = v
    return c


def save_mode(mode):
    CONF_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        c = json.loads(CONF_FILE.read_text())
    except (OSError, ValueError):
        c = {}
    c["mode"] = mode
    CONF_FILE.write_text(json.dumps(c, indent=2))


def log(*a):
    print("[minions]", *a, file=sys.stderr, flush=True)


# ---------- HTTP helpers ----------

def http_json(url, body=None, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json", "User-Agent": "minions-mcp/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base, messages, max_tokens=1500, extra=None, timeout=900):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6, **(extra or {})}
    t0 = time.time()
    data = http_json(base.rstrip("/") + "/v1/chat/completions", body, timeout)
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    t = data.get("timings") or {}
    toks = (data.get("usage") or {}).get("completion_tokens", t.get("predicted_n", 0))
    tps = t.get("predicted_per_second") or (toks / max(time.time() - t0, 1e-6))
    return text, f"[{toks} tokens, {time.time() - t0:.0f}s, {tps:.0f} tok/s]"


def need(mode_ok, tool):
    c = cfg()
    if c["mode"] not in mode_ok:
        return (f"Minions are in mode '{c['mode']}', which does not allow {tool}. "
                f"Do this part yourself, or switch with minion_set_mode ({' / '.join(mode_ok)}).")
    return None


# ---------- tools ----------

def t_status(a):
    c = cfg()
    n_ctx, slots = _props(c)
    prof = {65536: "fast (64k x2)", 131072: "long (128k)", 262144: "max (256k)"}.get(n_ctx * max(slots, 1), f"custom ({n_ctx * max(slots, 1)} total)")
    out = [f"mode: {c['mode']}",
           f"AGX context profile: {prof}  [per conversation {n_ctx} tokens, {slots} slot(s)]",
           "recommended: Claude Code orchestrating -> fast or long; the 35B orchestrating (minion_agent / `ide`) -> max. Switch with minion_set_profile (~40 s)."]
    for name, base, path in (("big (AGX planner)", c["big_url"], "/v1/models"), ("maple (workers)", c["maple_url"], "/__router"),
                             ("agent (cluster IDE)", c["agent_url"], "/api/model")):
        try:
            t0 = time.time()
            d = http_json(base.rstrip("/") + path, timeout=8)
            extra = ""
            if "data" in d:
                extra = " model=" + ",".join(m["id"] for m in d["data"])
            elif "name" in d:
                extra = f" model={d.get('name')} ctx={d.get('n_ctx')}"
            elif isinstance(d, dict):
                extra = " " + json.dumps(d)[:120]
            out.append(f"{name}: up ({(time.time() - t0) * 1000:.0f} ms){extra}  {base}")
        except Exception as e:
            out.append(f"{name}: DOWN  {base}  ({getattr(e, 'reason', e)})")
    return "\n".join(out)


PROFILES = {"fast": "64k context, 2 parallel conversations (best when Claude Code orchestrates and fires several minion calls)",
            "long": "128k context, 1 conversation (Claude Code orchestrating with big documents)",
            "max": "256k context, 1 conversation (best when the 35B itself orchestrates with Maple workers: minion_agent / the `ide` terminal)"}


def _props(c):
    try:
        d = http_json(c["big_url"].rstrip("/") + "/props", timeout=6)
        g = d.get("default_generation_settings") or {}
        return g.get("n_ctx") or 0, d.get("total_slots") or 0
    except Exception:
        return 0, 0


def t_set_profile(a):
    p = (a.get("profile") or "").strip().lower()
    if p not in PROFILES:
        return "profile must be one of: " + "; ".join(f"{k} = {v}" for k, v in PROFILES.items())
    c = cfg()
    import subprocess
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
                        "-p", str(c["agx_ssh_port"]), c["agx_ssh"], f"sh /mnt/persistent/data/agx-setup/agx-profile {p}"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return f"could not reach the AGX over ssh: {(r.stderr or r.stdout).strip()[-200:]}"
    deadline = time.time() + 180
    time.sleep(8)
    while time.time() < deadline:
        try:
            if http_json(c["big_url"].rstrip("/") + "/health", timeout=4).get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(4)
    n_ctx, slots = _props(c)
    return f"AGX profile is now '{p}': {PROFILES[p]}. Server reports n_ctx per conversation = {n_ctx}, slots = {slots}."


def t_set_mode(a):
    m = (a.get("mode") or "").strip().lower()
    if m not in MODES:
        return f"mode must be one of: {', '.join(MODES)}"
    save_mode(m)
    return f"minion mode is now '{m}'." + {"off": " Claude works alone.", "big": " Only the big AGX model is used.",
                                          "big+maple": " Big model and Maple workers are both available."}[m]


def t_ask(a):
    refuse = need(("big", "big+maple"), "minion_ask")
    if refuse:
        return refuse
    c = cfg()
    msgs = ([{"role": "system", "content": a["system"]}] if a.get("system") else []) + [{"role": "user", "content": a["prompt"]}]
    extra = {} if a.get("think") else {"chat_template_kwargs": {"enable_thinking": False}}
    text, tag = chat(c["big_url"], msgs, int(a.get("max_tokens", 1500)), extra)
    return f"{tag}\n{text}"


def _safe_rel(p):
    p = Path(p)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path must be relative and stay inside the working directory: {p}")
    return p


def t_code(a):
    refuse = need(("big+maple",), "minion_code")
    if refuse:
        return refuse
    c = cfg()
    ctx = []
    for f in a.get("files") or []:
        try:
            txt = Path(f).read_text(errors="replace")[:12000]
        except OSError as e:
            txt = f"(could not read: {e})"
        ctx.append(f"--- file: {f} ---\n{txt}")
    lang = a.get("language") or ""
    user = a["task"] + ("\n\nExisting files for reference:\n" + "\n".join(ctx) if ctx else "")
    msgs = [{"role": "system", "content": "You are a fast, careful coding assistant working for a lead engineer. Reply with the complete "
                                          "requested code in ONE fenced code block" + (f" ({lang})" if lang else "") +
                                          ", then at most three short bullet notes. No introduction."},
            {"role": "user", "content": user}]
    text, tag = chat(c["maple_url"], msgs, int(a.get("max_tokens", 3000)),
                     {"top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 1.0, "reasoning_budget_tokens": 1024,
                      "reasoning_budget_message": "\n\nI've thought enough; now I'll write the code.\n"})
    m = re.search(r"```[A-Za-z0-9_+.-]*\n(.*?)```", text, re.S)
    if a.get("write_to") and m:
        dest = _safe_rel(a["write_to"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        code = m.group(1).rstrip("\n") + "\n"
        dest.write_text(code)
        head = "\n".join(code.splitlines()[:20])
        return f"{tag} wrote {code.count(chr(10))} lines to {dest}. Review it before relying on it.\n--- first lines ---\n{head}"
    return f"{tag}\n{text}"


def t_agent(a):
    refuse = need(("big", "big+maple"), "minion_agent")
    if refuse:
        return refuse
    c = cfg()
    base = c["agent_url"].rstrip("/")
    task = a["task"]
    if c["mode"] == "big" or a.get("workers") is False:
        task += "\n\n[Note: do not use delegate_code; write all code yourself.]"
    cid = "mcp" + uuid.uuid4().hex[:12]
    job = http_json(base + "/api/chat_start", {"messages": [{"role": "user", "content": task}], "chat_id": cid, "title": "MCP task"}, 30)["id"]
    deadline = time.time() + int(a.get("max_wait_s", 900))
    answer, tools, err = "", [], ""
    req = urllib.request.Request(f"{base}/api/chat_events?id={job}&from=0", headers={"User-Agent": "minions-mcp/1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                if time.time() > deadline:
                    err = "timed out waiting for the agent (it may still be running; files so far are listed below)"
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                t = ev.get("type")
                if t == "tool":
                    tools.append(f"{ev.get('tool')} ({ev.get('seconds', 0)}s)")
                elif t == "done":
                    answer = ev.get("answer", "")
                    break
                elif t == "error":
                    err = ev.get("error", "agent error")
                    break
    except Exception as e:
        err = f"lost connection to the agent: {getattr(e, 'reason', e)}"
    lines = [f"agent chat {cid} ({len(tools)} tool calls: {', '.join(tools[:14])}{'...' if len(tools) > 14 else ''})"]
    if err:
        lines.append("PROBLEM: " + err)
    lines.append("--- answer ---\n" + (answer or "(none)"))
    try:
        files = [f for f in http_json(f"{base}/api/files/{cid}", timeout=20) if not f.get("dir")]
        lines.append("--- files in the cluster workspace (" + cid + ") ---")
        budget = 40000
        for f in files[:30]:
            lines.append(f"{f['path']}  ({f['size']} bytes)")
            if a.get("include_files", True) and f["size"] <= 12000 and budget > 0:
                try:
                    with urllib.request.urlopen(f"{base}/api/files/{cid}/{urllib.parse.quote(f['path'])}", timeout=20) as r:
                        txt = r.read().decode("utf-8", "replace")
                    budget -= len(txt)
                    lines.append("```\n" + txt + "\n```")
                except Exception:
                    pass
    except Exception:
        pass
    return "\n".join(lines)


TOOLS = {
    "minion_status": (t_status, "Show the minion mode and whether the big model, Maple workers and cluster agent are reachable.",
                      {"type": "object", "properties": {}}),
    "minion_set_mode": (t_set_mode, "Switch minion mode: 'off' (Claude alone), 'big' (only the AGX big model), 'big+maple' (big model and Maple workers).",
                        {"type": "object", "properties": {"mode": {"type": "string", "enum": list(MODES)}}, "required": ["mode"]}),
    "minion_set_profile": (t_set_profile, "Switch the AGX big model's context profile (restarts it, ~40 s): 'fast' = 64k x 2 conversations, 'long' = 128k, "
                                          "'max' = 256k. Use fast/long when Claude Code is the orchestrator, max when the 35B orchestrates the Maple minions.",
                           {"type": "object", "properties": {"profile": {"type": "string", "enum": list(PROFILES)}}, "required": ["profile"]}),
    "minion_ask": (t_ask, "Ask the big local model (Qwen3.6-35B-A3B on the AGX, ~60 tok/s). Good for bulk summarising, drafting, "
                          "explaining code, brainstorming, second opinions. Not as strong as you: verify important answers.",
                   {"type": "object", "properties": {"prompt": {"type": "string"}, "system": {"type": "string"},
                                                     "max_tokens": {"type": "integer", "default": 1500},
                                                     "think": {"type": "boolean", "description": "let it reason first (slower)"}},
                    "required": ["prompt"]}),
    "minion_code": (t_code, "Delegate ONE self-contained coding task to a fast Maple worker and get code back. Give a precise spec "
                            "(file name, function signatures, behaviour, constraints). With write_to the code is saved to that relative "
                            "path in the current directory and only a short summary comes back (saves your tokens). Review the result.",
                    {"type": "object", "properties": {"task": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}, "description": "local files to show the worker as context"},
                                                      "write_to": {"type": "string", "description": "relative path to save the code to"},
                                                      "language": {"type": "string"}, "max_tokens": {"type": "integer", "default": 3000}},
                     "required": ["task"]}),
    "minion_agent": (t_agent, "Hand a whole self-contained sub-task to the cluster IDE agent: the 35B planner plans, Maple workers write code, "
                              "and it runs/tests everything in a sandboxed workspace on the cluster (no access to your local files; describe "
                              "the task fully). Returns its answer plus the files it produced. Can take minutes.",
                     {"type": "object", "properties": {"task": {"type": "string"}, "workers": {"type": "boolean", "default": True, "description": "allow Maple workers"},
                                                       "max_wait_s": {"type": "integer", "default": 900}, "include_files": {"type": "boolean", "default": True}},
                      "required": ["task"]}),
}


# ---------- MCP over stdio ----------

def send(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def handle(msg):
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method == "initialize":
        want = params.get("protocolVersion") or "2025-06-18"
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": want if want in ("2024-11-05", "2025-03-26", "2025-06-18") else "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{"name": n, "description": d, "inputSchema": s} for n, (_, d, s) in TOOLS.items()]}})
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in TOOLS:
            return send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}})
        try:
            text, is_err = TOOLS[name][0](args), False
        except urllib.error.URLError as e:
            text, is_err = f"Cannot reach the cluster ({getattr(e, 'reason', e)}). Check minion_status; are you on the tailnet?", True
        except Exception as e:
            text, is_err = f"{type(e).__name__}: {e}", True
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": is_err}})
    elif mid is not None and method and not method.startswith("notifications/"):
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}})


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "mode":
        print(t_set_mode({"mode": sys.argv[2] if len(sys.argv) > 2 else ""}))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "status":
        print(t_status({}))
        return
    log("ready, mode =", cfg()["mode"])
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle(json.loads(line))
        except Exception as e:
            log("bad message:", e)


if __name__ == "__main__":
    main()
