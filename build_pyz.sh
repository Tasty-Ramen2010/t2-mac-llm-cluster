#!/bin/bash
# Build agent/ide.pyz: the IDE client + the terminal UI + the `rich` library in ONE file (needs only Python 3.8+ to run).
set -e
H="$(cd "$(dirname "$0")" && pwd)"
B=$(mktemp -d)
python3 -m venv "$B/venv" >/dev/null && "$B/venv/bin/pip" install -q --disable-pip-version-check rich >/dev/null
mkdir "$B/app"
cp -r "$B"/venv/lib/python3*/site-packages/{rich,markdown_it,mdurl,pygments} "$B/app/"
find "$B/app" -name __pycache__ -prune -exec rm -rf {} +
cp "$H/agent/ai_term.py" "$B/app/ai_term.py"
cp "$H/agent/ide_client.py" "$B/app/__main__.py"
python3 -m zipapp "$B/app" -o "$H/agent/ide.pyz" -p '/usr/bin/env python3' -c
rm -rf "$B"
ls -la "$H/agent/ide.pyz" | awk '{print "built ide.pyz", $5/1e6 " MB"}'
