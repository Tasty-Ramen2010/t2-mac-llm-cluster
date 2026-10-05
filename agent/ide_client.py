#!/usr/bin/env python3
"""ide: the terminal coding agent. Works from any machine with Python 3.8+ (Linux, macOS, WSL) and the `rich` package.

It logs in to your cluster IDE (passphrase + authenticator code), then runs the cluster's own `ai` terminal UI against it:
a planner model (Qwen3.6-35B-A3B) that delegates code to fast worker models, with a sandboxed workspace, files, tests and
private GitHub clones. Nothing is stored on this machine except a login token (~/.ide/session.json, expires in 12 h).

first time:   curl -fsSL https://YOUR-HOST/ide/ide_client.py -o ide_client.py && python3 ide_client.py --url https://YOUR-HOST
afterwards:   python3 ide_client.py            (--logout to forget the token, --url to change server)
"""
import argparse
import getpass
import http.client
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path.home() / ".ide"
CONF = HOME / "session.json"


def load():
    try:
        return json.loads(CONF.read_text())
    except (OSError, ValueError):
        return {}


def save(d):
    HOME.mkdir(parents=True, exist_ok=True)
    os.chmod(HOME, 0o700)
    CONF.write_text(json.dumps(d))
    os.chmod(CONF, 0o600)


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {"error": str(e)}


def login(url):
    print(f"Logging in to {url}")
    for attempt in range(3):
        passphrase = getpass.getpass("Passphrase: ")
        code = input("Authenticator code (6 digits): ").strip().replace(" ", "")
        status, resp = post(url, "/auth/login", {"passphrase": passphrase, "code": code})
        if status == 200:
            return {"url": url, "token": resp["token"], "expires": time.time() + resp.get("expires_in", 43200) - 60}
        print("  ->", resp.get("error", status))
        if status == 429:
            break
    sys.exit("Login failed.")


class Shim(BaseHTTPRequestHandler):
    """Local plain-HTTP face for the terminal UI: forwards every request to the server and adds the login token."""
    protocol_version = "HTTP/1.0"
    remote = None          # (scheme, host, port)
    token = ""

    def log_message(self, *a):
        pass

    def forward(self):
        scheme, host, port = self.remote
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(n) if n else None
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        try:
            conn = cls(host, port, timeout=1800)
            headers = {"Authorization": "Bearer " + self.token}
            if body is not None:
                headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
            conn.request(self.command, self.path, body, headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            msg = json.dumps({"error": f"cannot reach the server: {e}"}).encode()
            self.send_response(502); self.send_header("Content-Length", str(len(msg))); self.end_headers(); self.wfile.write(msg)
            return
        self.send_response(resp.status)
        for k in ("Content-Type", "Content-Length", "Cache-Control", "Content-Disposition"):
            v = resp.getheader(k)
            if v:
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()

    do_GET = do_POST = forward


def main():
    ap = argparse.ArgumentParser(description="terminal coding agent for your cluster")
    ap.add_argument("--url", help="server address, e.g. https://agx.tailxxxx.ts.net")
    ap.add_argument("--logout", action="store_true", help="forget the saved login")
    args, rest = ap.parse_known_args()
    cfg = load()
    if args.logout:
        if cfg.get("url") and cfg.get("token"):
            try:
                req = urllib.request.Request(cfg["url"] + "/auth/logout", b"{}", {"Authorization": "Bearer " + cfg["token"], "Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).read()
            except (OSError, urllib.error.URLError):
                pass
        CONF.unlink(missing_ok=True)
        print("Logged out.")
        return
    url = (args.url or os.environ.get("IDE_URL") or cfg.get("url") or "").rstrip("/")
    if not url:
        sys.exit("Give the server address once:  python3 ide_client.py --url https://YOUR-HOST")
    if cfg.get("url") != url or cfg.get("expires", 0) < time.time() or not cfg.get("token"):
        cfg = login(url)
        save(cfg)
    try:
        import rich  # noqa: F401
    except ImportError:
        sys.exit("The terminal UI needs the 'rich' package:  python3 -m pip install --user rich")

    # fetch the matching terminal UI from the server (needs the login), so client and server always agree
    req = urllib.request.Request(url + "/ide/ai_term.py", headers={"Authorization": "Bearer " + cfg["token"]})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            CONF.unlink(missing_ok=True)
            sys.exit("Login expired. Run again to log in.")
        sys.exit(f"Could not download the terminal UI: HTTP {e.code}")
    HOME.mkdir(parents=True, exist_ok=True)
    mod_path = HOME / "ai_term_remote.py"
    mod_path.write_bytes(code)

    u = urllib.parse.urlsplit(url)
    Shim.remote = (u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80))
    Shim.token = cfg["token"]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Shim)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["CLUSTER_AI_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"

    spec = importlib.util.spec_from_file_location("ai_term_remote", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["ide"] + rest
    spec.loader.exec_module(mod)
    try:
        mod.main()
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
