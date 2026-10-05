#!/usr/bin/env python3
"""Installer for the cluster IDE terminal client (served by the server itself at /i).

    curl -sL @@BASE@@/i | python3

Downloads one self-contained file (~/.ide/ide.pyz, includes everything it needs), creates the short command `ide`,
remembers this server, and starts it. After that, just type:  ide
(add --insecure only on a network that inspects HTTPS with its own certificate and breaks normal verification)
"""
import json
import os
import ssl
import sys
import urllib.request

BASE = "@@BASE@@"
home = os.path.expanduser("~/.ide")
os.makedirs(home, exist_ok=True)
pyz = os.path.join(home, "ide.pyz")
ctx = ssl._create_unverified_context() if "--insecure" in sys.argv else None

print("Downloading the IDE client ...")
req = urllib.request.Request(BASE + "/ide.pyz", headers={"User-Agent": "Mozilla/5.0 (ide installer)"})
try:
    data = urllib.request.urlopen(req, timeout=120, context=ctx).read()
except Exception as e:
    sys.exit(f"Could not download from {BASE}: {e}")
with open(pyz, "wb") as f:
    f.write(data)

cfg = os.path.join(home, "session.json")
try:
    c = json.load(open(cfg))
except (OSError, ValueError):
    c = {}
c["url"] = BASE
with open(cfg, "w") as f:
    json.dump(c, f)
os.chmod(cfg, 0o600)

bindir = os.path.expanduser("~/.local/bin")
os.makedirs(bindir, exist_ok=True)
launcher = os.path.join(bindir, "ide")
with open(launcher, "w") as f:
    f.write('#!/bin/sh\nexec python3 "$HOME/.ide/ide.pyz" "$@"\n')
os.chmod(launcher, 0o755)
print("Installed. From now on just run:  ide")
if bindir not in os.environ.get("PATH", "").split(":"):
    print('  (first add it to your PATH:  export PATH="$HOME/.local/bin:$PATH")')

try:                                   # we were piped from curl: take the keyboard back for the login prompts
    os.dup2(os.open("/dev/tty", os.O_RDWR), 0)
except OSError:
    pass
extra = [a for a in sys.argv[1:] if a != "-"]
os.execv(sys.executable, [sys.executable, pyz] + extra)
