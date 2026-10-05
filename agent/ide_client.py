#!/usr/bin/env python3
"""ide: the terminal coding agent. One self-contained file (ide.pyz) from your server; needs only Python 3.8+.

It logs in to your cluster IDE (passphrase + authenticator code), then runs the cluster's `ai` terminal UI against it:
a planner model (Qwen3.6-35B-A3B on the AGX GPU) that delegates code to fast worker models, with a sandboxed workspace on
the cluster, files, tests and private GitHub clones. Nothing is stored here except a login token (~/.ide/session.json, 12 h).

install:   curl -sL https://YOUR-HOST/i | python3          (creates the short command `ide`)
use:       ide            ide --logout            ide --update            ide --url https://OTHER-HOST
networks that break HTTPS checks (school/corporate inspection):  ide --insecure   (last resort)
"""
import argparse
import getpass
import importlib.util
import json
import os
import ssl
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
UA = "Mozilla/5.0 (X11; Linux x86_64) ide-client/2"
CTX = None          # set to an unverified context by --insecure
OPENER = None       # built in main(): honours the system / environment proxy settings


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


def make_opener():
    handlers = [urllib.request.ProxyHandler(urllib.request.getproxies())]      # system proxy (PAC-less), env vars
    if CTX is not None:
        handlers.append(urllib.request.HTTPSHandler(context=CTX))
    return urllib.request.build_opener(*handlers)


def open_url(req, timeout=30):
    global OPENER
    if OPENER is None:
        OPENER = make_opener()
    return OPENER.open(req, timeout=timeout)


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json", "User-Agent": UA})
    try:
        with open_url(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {"error": str(e)}
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"Cannot reach {url}: {getattr(e, 'reason', e)}\n(if this network blocks it or inspects HTTPS, try --insecure; otherwise check the address)")


def login(url):
    print(f"Logging in to {url}")
    for _ in range(3):
        passphrase = os.environ.get("IDE_PASSPHRASE") or getpass.getpass("Passphrase: ")      # env vars: scripted logins/tests
        code = (os.environ.get("IDE_CODE") or input("Authenticator code (6 digits): ")).strip().replace(" ", "")
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
    base = ""
    token = ""

    def log_message(self, *a):
        pass

    def forward(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(n) if n else None
        headers = {"Authorization": "Bearer " + self.token, "User-Agent": UA}
        if body is not None:
            headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
        req = urllib.request.Request(self.base + self.path, data=body, headers=headers, method=self.command)
        try:
            resp = open_url(req, timeout=1800)
        except urllib.error.HTTPError as e:
            resp = e                                   # an HTTP error is still a response to relay
        except (urllib.error.URLError, OSError) as e:
            msg = json.dumps({"error": f"cannot reach the server: {getattr(e, 'reason', e)}"}).encode()
            self.send_response(502); self.send_header("Content-Length", str(len(msg))); self.end_headers(); self.wfile.write(msg)
            return
        code = getattr(resp, "status", None) or resp.code
        self.send_response(code)
        for k in ("Content-Type", "Content-Length", "Cache-Control", "Content-Disposition"):
            v = resp.headers.get(k)
            if v:
                self.send_header(k, v)
        self.end_headers()
        read = getattr(resp, "read1", None) or resp.read
        try:
            while True:
                chunk = read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, ValueError):
            pass
        finally:
            resp.close()

    do_GET = do_POST = forward


def OPENER_RESET():
    global OPENER
    OPENER = None


def take_keyboard_back():
    """Started via `curl | python3`? stdin is the pipe; point it at the terminal so prompts work."""
    if not sys.stdin.isatty():
        try:
            os.dup2(os.open("/dev/tty", os.O_RDWR), 0)
            sys.stdin = open(0, closefd=False)
        except OSError:
            pass


def update(url):
    pyz = Path(sys.argv[0]).resolve() if sys.argv[0].endswith(".pyz") else HOME / "ide.pyz"
    req = urllib.request.Request(url + "/ide.pyz", headers={"User-Agent": UA})
    with open_url(req, timeout=120) as r:
        data = r.read()
    pyz.write_bytes(data)
    print(f"Updated {pyz} ({len(data) // 1024} KB).")


def main():
    global CTX
    ap = argparse.ArgumentParser(description="terminal coding agent for your cluster")
    ap.add_argument("--url", help="server address, e.g. https://xxxx.trycloudflare.com")
    ap.add_argument("--logout", action="store_true", help="forget the saved login")
    ap.add_argument("--update", action="store_true", help="download the newest client")
    ap.add_argument("--insecure", action="store_true", help="skip HTTPS certificate checks (last resort on inspected networks)")
    args, rest = ap.parse_known_args()
    if args.insecure or os.environ.get("IDE_INSECURE"):
        CTX = ssl._create_unverified_context()
    OPENER_RESET()
    take_keyboard_back()
    cfg = load()
    url = (args.url or os.environ.get("IDE_URL") or cfg.get("url") or "").rstrip("/")
    if args.logout:
        if url and cfg.get("token"):
            try:
                open_url(urllib.request.Request(url + "/auth/logout", b"{}", {"Authorization": "Bearer " + cfg["token"], "Content-Type": "application/json", "User-Agent": UA}), 10).read()
            except (OSError, urllib.error.URLError):
                pass
        cfg.pop("token", None)
        save(cfg)
        print("Logged out.")
        return
    if not url:
        sys.exit("No server saved yet. Install with:  curl -sL https://YOUR-HOST/i | python3")
    if args.update:
        return update(url)
    if cfg.get("url") != url or cfg.get("expires", 0) < time.time() or not cfg.get("token"):
        cfg = login(url)
        save(cfg)

    u = urllib.parse.urlsplit(url)
    Shim.base, Shim.token = url, cfg["token"]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Shim)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["CLUSTER_AI_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        import rich  # noqa: F401
        import ai_term                               # bundled in ide.pyz (reads CLUSTER_AI_URL when imported)
    except ImportError:
        sys.exit("The terminal UI needs the 'rich' package and ai_term.py (use the installer: curl -sL https://YOUR-HOST/i | python3)")
    sys.argv = ["ide"] + rest
    try:
        ai_term.main()
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
