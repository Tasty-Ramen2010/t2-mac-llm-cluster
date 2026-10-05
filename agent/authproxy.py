#!/usr/bin/env python3
"""Login proxy in front of the agent for the public (Tailscale Funnel) address. Standard library only.

Nothing gets through without: a passphrase AND a 6-digit authenticator (TOTP) code, then a random session token.
Only a short allowlist of agent API paths is forwarded; model switching, benchmarks, the web page etc. are blocked.

  authproxy.py init            choose the passphrase, create the authenticator secret (run once, on the server)
  authproxy.py serve           listen on 127.0.0.1:8082 (point `tailscale funnel` at it)
  authproxy.py status          show whether it is configured and how many sessions are active
  authproxy.py logout-all      invalidate every session

Env: IDE_HOME (default ~/.config/ide), IDE_UPSTREAM (default 100.82.180.15:8081), IDE_LISTEN (default 127.0.0.1:8082).
"""
import base64
import getpass
import hashlib
import hmac
import http.client
import json
import os
import secrets
import struct
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path(os.environ.get("IDE_HOME", str(Path.home() / ".config" / "ide")))
AUTH_FILE = HOME / "auth.json"
SESS_FILE = HOME / "sessions.json"
UPSTREAM = os.environ.get("IDE_UPSTREAM", "100.82.180.15:8081")
LISTEN = os.environ.get("IDE_LISTEN", "127.0.0.1:8082")
AGENT_DIR = Path(__file__).resolve().parent
SESSION_SECONDS = 12 * 3600
MAX_BODY = 2 * 1024 * 1024
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32, maxmem=64 * 1024 * 1024)

ALLOW_GET = ("/api/chat_events", "/api/model", "/api/models", "/api/chats", "/api/files/", "/api/permissions",
             "/api/usage", "/api/context/", "/api/stats", "/api/jobs")
ALLOW_POST = ("/api/chat_start", "/api/chat_cancel", "/api/permission", "/api/permissions/revoke", "/api/chats/")
PUBLIC_FILES = {"/ide/ide_client.py": "ide_client.py", "/ide.pyz": "ide.pyz"}   # the app itself holds no secrets; the API needs a login
PRIVATE_FILES = {"/ide/ai_term.py": "ai_term.py"}

LOCK = threading.Lock()
FAILS = {}            # ip -> [timestamps]
GLOBAL_FAILS = []     # timestamps
SESSIONS = {}         # sha256(token) -> expiry
LAST_TOTP = [0]


# ---------- crypto helpers ----------

def hash_pass(passphrase, salt):
    return hashlib.scrypt(passphrase.encode(), salt=salt, **SCRYPT)


def totp(secret_b32, counter):
    key = base64.b32decode(secret_b32)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = digest[-1] & 0x0F
    return f"{(struct.unpack('>I', digest[o:o + 4])[0] & 0x7FFFFFFF) % 1000000:06d}"


def match_totp(secret_b32, code):
    """Return the matching 30 s step (current +-1) that has not been used before, else None. Does not consume it."""
    if not (isinstance(code, str) and code.isdigit() and len(code) == 6):
        return None
    now = int(time.time() // 30)
    hit = None
    for c in (now - 1, now, now + 1):
        if hmac.compare_digest(totp(secret_b32, c), code) and c > LAST_TOTP[0]:
            hit = c
    return hit


def load_auth():
    try:
        return json.loads(AUTH_FILE.read_text())
    except (OSError, ValueError):
        return None


def save_sessions():
    HOME.mkdir(parents=True, exist_ok=True)
    tmp = SESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(SESSIONS))
    os.chmod(tmp, 0o600)
    tmp.replace(SESS_FILE)


def load_sessions():
    try:
        now = time.time()
        for k, exp in json.loads(SESS_FILE.read_text()).items():
            if exp > now:
                SESSIONS[k] = exp
    except (OSError, ValueError):
        pass


def new_session():
    token = secrets.token_urlsafe(32)
    with LOCK:
        now = time.time()
        for k in [k for k, e in SESSIONS.items() if e < now]:
            del SESSIONS[k]
        SESSIONS[hashlib.sha256(token.encode()).hexdigest()] = now + SESSION_SECONDS
        save_sessions()
    return token


def valid_session(token):
    if not token:
        return False
    k = hashlib.sha256(token.encode()).hexdigest()
    with LOCK:
        exp = SESSIONS.get(k)
        if exp and exp > time.time():
            return True
        SESSIONS.pop(k, None)
        return False


# ---------- rate limiting ----------

def locked_out(ip):
    now = time.time()
    with LOCK:
        recent = [t for t in FAILS.get(ip, []) if now - t < 900]
        FAILS[ip] = recent
        g = [t for t in GLOBAL_FAILS if now - t < 3600]
        GLOBAL_FAILS[:] = g
        return len(recent) >= 5 or len(g) >= 30


def record_fail(ip):
    with LOCK:
        FAILS.setdefault(ip, []).append(time.time())
        GLOBAL_FAILS.append(time.time())


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # standard HTTP for Cloudflare/Funnel; every response closes the connection
    server_version = "ide-auth"
    sys_version = ""

    def log_message(self, fmt, *a):
        pass

    def client_ip(self):
        peer = self.client_address[0]
        xff = self.headers.get("X-Forwarded-For", "")
        if peer in ("127.0.0.1", "::1") and xff:
            return xff.split(",")[-1].strip()     # last hop = the address the local proxy (Funnel) actually saw
        return peer

    def send_json(self, code, obj, extra=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def bearer(self):
        h = self.headers.get("Authorization", "")
        return h[7:].strip() if h.lower().startswith("bearer ") else ""

    def path_only(self):
        return urllib.parse.urlsplit(self.path).path

    # --- routes ---
    def do_GET(self):
        p = self.path_only()
        if p == "/healthz":
            return self.send_json(200, {"ok": True})
        if p == "/i":                                   # one-line installer: curl -sL https://HOST/i | python3
            host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or ""
            base = "https://" + host.split(",")[0].strip()
            try:
                data = (AGENT_DIR / "ide_bootstrap.py").read_text().replace("@@BASE@@", base).encode()
            except OSError:
                return self.send_json(404, {"error": "not found"})
            return self.send_bytes(data, "text/x-python; charset=utf-8")
        if p in PUBLIC_FILES or (p in PRIVATE_FILES and valid_session(self.bearer())):
            f = AGENT_DIR / {**PUBLIC_FILES, **PRIVATE_FILES}[p]
            try:
                data = f.read_bytes()
            except OSError:
                return self.send_json(404, {"error": "not found"})
            return self.send_bytes(data, "application/octet-stream" if p.endswith(".pyz") else "text/x-python; charset=utf-8")
        self.proxy("GET", ALLOW_GET)

    def do_POST(self):
        p = self.path_only()
        if p == "/auth/login":
            return self.login()
        if p == "/auth/logout":
            token = self.bearer()
            with LOCK:
                SESSIONS.pop(hashlib.sha256(token.encode()).hexdigest(), None)
                save_sessions()
            return self.send_json(200, {"ok": True})
        self.proxy("POST", ALLOW_POST)

    def read_body(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            return None
        return self.rfile.read(n) if n else b""

    def login(self):
        ip = self.client_ip()
        if locked_out(ip):
            return self.send_json(429, {"error": "too many failed attempts; try again later"}, {"Retry-After": "900"})
        body = self.read_body()
        try:
            req = json.loads(body or b"{}")
            passphrase, code = str(req.get("passphrase", "")), str(req.get("code", ""))
        except (ValueError, AttributeError):
            return self.send_json(400, {"error": "bad request"})
        auth = load_auth()
        if not auth:
            return self.send_json(503, {"error": "login is not configured on the server (run: authproxy.py init)"})
        # always evaluate both factors, so timing does not reveal which one was wrong
        ok_pass = hmac.compare_digest(hash_pass(passphrase, base64.b64decode(auth["salt"])), base64.b64decode(auth["hash"]))
        step = match_totp(auth["totp"], code)
        ok_code = step is not None
        if not (ok_pass and ok_code):
            record_fail(ip)
            print(f"login failed from {ip}", file=sys.stderr, flush=True)
            time.sleep(1.0)
            return self.send_json(401, {"error": "invalid passphrase or code"})
        with LOCK:
            LAST_TOTP[0] = max(LAST_TOTP[0], step)        # a code works once, and only after a fully successful login
        print(f"login ok from {ip}", file=sys.stderr, flush=True)
        return self.send_json(200, {"token": new_session(), "expires_in": SESSION_SECONDS})

    def proxy(self, method, allow):
        if not valid_session(self.bearer()):
            return self.send_json(401, {"error": "login required"}, {"WWW-Authenticate": "Bearer"})
        p = self.path_only()
        if not any(p == a or (a.endswith("/") and p.startswith(a)) or (not a.endswith("/") and p.startswith(a)) for a in allow):
            return self.send_json(403, {"error": "this action is not available remotely"})
        if ".." in p:
            return self.send_json(400, {"error": "bad path"})
        body = self.read_body() if method == "POST" else None
        if method == "POST" and body is None:
            return self.send_json(413, {"error": "request too large"})
        host, _, port = UPSTREAM.partition(":")
        try:
            conn = http.client.HTTPConnection(host, int(port or 80), timeout=1800)
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")} if body is not None else {}
            conn.request(method, self.path, body, headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            return self.send_json(502, {"error": f"agent unreachable: {e}"})
        self.send_response(resp.status)
        has_len = resp.getheader("Content-Length") is not None
        for k in ("Content-Type", "Content-Length", "Cache-Control", "Content-Disposition"):
            v = resp.getheader(k)
            if v:
                self.send_header(k, v)
        if not has_len:
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                if has_len:
                    self.wfile.write(chunk)
                else:
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            if not has_len:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()


# ---------- CLI ----------

def cmd_init(argv=()):
    """init [--passphrase P] [--allow-short]: the second factor (authenticator code) is always required, which is what
    keeps a short passphrase safe; --allow-short lets you choose one under 12 characters."""
    HOME.mkdir(parents=True, exist_ok=True)
    os.chmod(HOME, 0o700)
    a = None
    argv = list(argv)
    if "--passphrase" in argv:
        a = argv[argv.index("--passphrase") + 1]
    allow_short = "--allow-short" in argv
    if a is None:
        if AUTH_FILE.exists() and input("Login is already configured. Replace it? [y/N] ").strip().lower() != "y":
            return
        while True:
            a = getpass.getpass("Choose a passphrase (12+ characters): ")
            if len(a) < 12 and not allow_short:
                print("Too short."); continue
            if a != getpass.getpass("Again: "):
                print("Does not match."); continue
            break
    elif len(a) < 12 and not allow_short:
        sys.exit("passphrase too short (use --allow-short; the authenticator code still protects the login)")
    salt = os.urandom(16)
    secret = base64.b32encode(os.urandom(20)).decode().rstrip("=")
    AUTH_FILE.write_text(json.dumps({"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(hash_pass(a, salt)).decode(),
                                     "totp": secret}))
    os.chmod(AUTH_FILE, 0o600)
    SESSIONS.clear()
    save_sessions()
    uri = "otpauth://totp/Cluster%20IDE:owner?secret=" + secret + "&issuer=Cluster%20IDE"
    print("\nAdd this to your authenticator app (Google Authenticator, Authy, 1Password...):")
    print("  secret (type it in manually):", secret)
    print("  or open this link / make a QR of it:", uri)
    print("\nLogin needs the passphrase you just chose AND the 6-digit code from the app.")


def serve():
    host, _, port = LISTEN.partition(":")
    load_sessions()
    if not load_auth():
        print("not configured: run `authproxy.py init` first", file=sys.stderr)
        sys.exit(2)
    print(f"ide-auth listening on {LISTEN}, upstream {UPSTREAM}", flush=True)
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "init":
        cmd_init(sys.argv[2:])
    elif cmd == "serve":
        serve()
    elif cmd == "status":
        load_sessions()
        print("configured:", bool(load_auth()), "| active sessions:", len(SESSIONS), "| upstream:", UPSTREAM)
    elif cmd == "logout-all":
        SESSIONS.clear(); save_sessions(); print("all sessions invalidated")
    else:
        print(__doc__)
