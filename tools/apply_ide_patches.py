#!/usr/bin/env python3
"""Apply (idempotently) the small edits the IDE needs in files that other work also touches. Safe to re-run.
   usage: python3 tools/apply_ide_patches.py   (run from anywhere; paths are relative to the repo root)
Edits:
  agent/agent_server.py  - load agent/ide_plugin.py; sandbox limits from env (3 GB / 1024 procs / 4 GB files / 30 min);
                           NO_PROXY for loopback so sandbox programs can reach their own servers on 127.0.0.1:30000-30999
  agent/netgate.py       - "open": true in agent/permissions.json lets the sandbox reach any PUBLIC website without asking
                           (private/LAN/Tailscale/loopback addresses stay blocked, always)
  llmtools-firewall      - sandbox user may use loopback TCP ports 30000-30999 (its own servers); installed copy: /usr/local/sbin/llmtools-firewall
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AG = ROOT / "agent"


def patch(path, edits):
    s = path.read_text()
    changed = False
    for marker, old, new in edits:
        if marker in s:
            continue
        if old not in s:
            print(f"  ! {path.name}: expected text for '{marker}' not found (file changed?); skipped")
            continue
        s = s.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(s)
    print(f"{path.name}: {'patched' if changed else 'already up to date'}")


patch(AG / "agent_server.py", [
    ("ide_plugin", 'if __name__ == "__main__":', '''# Optional IDE plugin (planner + Maple workers, GPU jobs, GitHub push). Remove these lines to disable. See ../IDE.md
try:
    import sys as _sys
    import ide_plugin
    ide_plugin.install(_sys.modules[__name__])
except ImportError:
    pass

if __name__ == "__main__":'''),
    ("SANDBOX_AS", 'MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))', '''MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
# sandbox limits (override with env): address space 3 GB, 1024 processes, 4 GB per file, commands up to 30 minutes
SANDBOX_AS = int(os.environ.get("SANDBOX_AS", str(3 * 1024 ** 3)))
SANDBOX_NPROC = int(os.environ.get("SANDBOX_NPROC", "1024"))
SANDBOX_FSIZE = int(os.environ.get("SANDBOX_FSIZE", str(4 * 1024 ** 3)))
SANDBOX_MAX_TIMEOUT = int(os.environ.get("SANDBOX_MAX_TIMEOUT", "1800"))'''),
    ("--as={SANDBOX_AS}", '"prlimit", "--as=1610612736", "--nproc=256", "--fsize=2147483648", "--",',
     'f"prlimit", f"--as={SANDBOX_AS}", f"--nproc={SANDBOX_NPROC}", f"--fsize={SANDBOX_FSIZE}", "--",'),
    ("SANDBOX_MAX_TIMEOUT), ", "timeout = max(5, min(int(timeout or 120), 600))", "timeout = max(5, min(int(timeout or 120), SANDBOX_MAX_TIMEOUT))") if False else
    ("min(int(timeout or 120), SANDBOX_MAX_TIMEOUT)", "timeout = max(5, min(int(timeout or 120), 600))", "timeout = max(5, min(int(timeout or 120), SANDBOX_MAX_TIMEOUT))"),
    ("NO_PROXY=127.0.0.1", '        net += ["PIP_USER=1"',
     '        net += ["NO_PROXY=127.0.0.1,localhost,::1", "no_proxy=127.0.0.1,localhost,::1"]      # loopback servers are reached directly\n        net += ["PIP_USER=1"'),
])

patch(AG / "netgate.py", [
    ("self.open", '        self.always = set(json.loads(PERMS_FILE.read_text()).get("always", [])) if PERMS_FILE.exists() else set()',
     '''        cfg = json.loads(PERMS_FILE.read_text()) if PERMS_FILE.exists() else {}
        self.always = set(cfg.get("always", []))
        self.open = bool(cfg.get("open"))              # open internet: every public site is allowed without asking'''),
    ('"open": self.open', 'PERMS_FILE.write_text(json.dumps({"always": sorted(self.always)}, indent=1))',
     'PERMS_FILE.write_text(json.dumps({"always": sorted(self.always), "open": self.open}, indent=1))'),
    ("return self.open or", "            return site in self.always or site in self.chat_grants.get(chat, set())",
     "            return self.open or site in self.always or site in self.chat_grants.get(chat, set())"),
])

fw = ROOT / "llmtools-firewall"
if fw.exists():
    patch(fw, [("30000:30999", "if [ $T = iptables ]; then $T -A LLMTOOLS -o lo -d 127.0.0.1 -p tcp --dport 8899 -j ACCEPT; fi",
                "if [ $T = iptables ]; then\n    $T -A LLMTOOLS -o lo -d 127.0.0.1 -p tcp --dport 8899 -j ACCEPT\n"
                "    $T -A LLMTOOLS -o lo -s 127.0.0.1 -d 127.0.0.1 -p tcp --dport 30000:30999 -j ACCEPT\n"
                "    $T -A LLMTOOLS -o lo -s 127.0.0.1 -d 127.0.0.1 -p tcp --sport 30000:30999 -j ACCEPT\n  fi")])
    print("  (copy it to /usr/local/sbin/llmtools-firewall and `sudo systemctl restart llmtools-firewall` to apply)")

pf = AG / "permissions.json"
cfg = json.loads(pf.read_text()) if pf.exists() else {}
if "--open" in os.sys.argv:
    cfg["open"] = True
    pf.write_text(json.dumps(cfg, indent=1))
    print("permissions.json: open internet ON")
else:
    print("permissions.json: open =", cfg.get("open", False), "(run with --open to turn open internet on)")
