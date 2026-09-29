"""netgate: the AI workspace's only way to the internet (an HTTP/HTTPS proxy on 127.0.0.1:8899).

The firewall lets the sandbox user (llmtools) reach nothing except this proxy. Every connection names a site; the proxy
lets it through if the user approved that site (for this conversation, or always), otherwise it asks the user through
the chat (a permission box) and waits for the answer. Sites that resolve to private/local addresses (the LAN, the Macs,
Tailscale) are refused no matter what. The requesting conversation is identified by the proxy username, which the
backend sets to the job id in the sandbox's HTTP(S)_PROXY variables.
"""
import base64
import ipaddress
import json
import select
import socket
import threading
import time
from pathlib import Path

PORT = 8899
PERMS_FILE = Path(__file__).parent / "permissions.json"
# sites that are really one service spread over several domains: approving the first covers them all
GROUPS = {
    "pypi.org": ["pypi.org", "pythonhosted.org", "download.pytorch.org"],   # pip also checks the CPU-only PyTorch index
    "github.com": ["github.com", "githubusercontent.com", "githubassets.com"],
    "huggingface.co": ["huggingface.co", "hf.co"],
    "npmjs.org": ["npmjs.org", "npmjs.com"],
}
ALLOWED_PORTS = {80, 443}


def site_of(host):
    """the 'site' a user approves: example.com covers www.example.com, api.example.com, ..."""
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return host                                     # a bare IP address is its own site
    except ValueError:
        pass
    for site, members in GROUPS.items():
        if any(host == m or host.endswith("." + m) for m in members):
            return site
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in ("co", "com", "ac", "gov", "org", "net", "edu"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _public(host, port):
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None, "can't find that website"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or ip in ipaddress.ip_network("100.64.0.0/10")):
            return None, "that address is on the local network (always blocked)"
    return infos[0], None


class Gate:
    def __init__(self, ask):
        """ask(token, site, host) -> "chat" | "always" | "deny"  (blocks until the user answers)"""
        self.ask, self.lock = ask, threading.Lock()
        self.always = set(json.loads(PERMS_FILE.read_text()).get("always", [])) if PERMS_FILE.exists() else set()
        self.chat_grants = {}                                   # chat id -> {sites}
        self.notes = {}                                         # token -> blocked-connection notes for the AI

    def save(self):
        PERMS_FILE.write_text(json.dumps({"always": sorted(self.always)}, indent=1))

    def grant(self, chat, site, scope):
        with self.lock:
            if scope == "always":
                self.always.add(site)
                self.save()
            else:
                self.chat_grants.setdefault(chat, set()).add(site)

    def revoke(self, site):
        with self.lock:
            self.always.discard(site)
            for g in self.chat_grants.values():
                g.discard(site)
            self.save()

    def allowed(self, chat, site):
        with self.lock:
            return site in self.always or site in self.chat_grants.get(chat, set())

    # ------------------------------------------------------------------ proxy
    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", PORT))
        srv.listen(64)
        while True:
            c, _ = srv.accept()
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()

    def _refuse(self, c, code, why):
        body = f"netgate: {why}\n".encode()
        try:
            c.sendall(f"HTTP/1.1 {code} {'Forbidden' if code == 403 else 'Bad Gateway'}\r\nContent-Type: text/plain\r\n"
                      f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        except OSError:
            pass
        c.close()

    def _handle(self, c):
        c.settimeout(30)
        try:
            data = b""
            while b"\r\n\r\n" not in data and len(data) < 65536:
                chunk = c.recv(8192)
                if not chunk:
                    return c.close()
                data += chunk
            head, _, rest = data.partition(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, target, version = (lines[0].split(" ") + ["", "", ""])[:3]
            headers = [(l.split(":", 1)[0].strip(), l.split(":", 1)[1].strip()) for l in lines[1:] if ":" in l]
            token = ""
            for k, v in headers:
                if k.lower() == "proxy-authorization" and v.lower().startswith("basic "):
                    try:
                        token = base64.b64decode(v[6:]).decode().split(":", 1)[0]
                    except ValueError:
                        pass
            if method == "CONNECT":
                host, _, port = target.rpartition(":")
                port = int(port or 443)
            else:
                if not target.startswith("http://"):
                    return self._refuse(c, 403, "only http(s) requests go through here")
                hp, _, path = target[7:].partition("/")
                host, _, port = hp.partition(":")
                port = int(port or 80)
            host = host.strip("[]")
            if port not in ALLOWED_PORTS:
                return self._refuse(c, 403, f"port {port} is not allowed (only 80 and 443)")
            site = site_of(host)
            chat = self.chat_of(token)
            if chat is None:
                return self._refuse(c, 403, "unknown requester")
            info, why = _public(host, port)             # local/private addresses: refused without asking anyone
            if not info:
                self.note(token, f"{host}: {why}")
                return self._refuse(c, 403, why)
            if not self.allowed(chat, site):
                decision = self.ask(token, site, host)
                if decision not in ("chat", "always"):
                    self.note(token, f"the user did not allow access to {site}")
                    return self._refuse(c, 403, f"the user did not allow access to {site}")
            up = socket.create_connection(info[4][:2], timeout=20)
            if method == "CONNECT":
                c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                if rest:
                    up.sendall(rest)
            else:
                keep = [f"{k}: {v}" for k, v in headers if k.lower() not in ("proxy-authorization", "proxy-connection", "connection")]
                up.sendall((f"{method} /{path} {version}\r\n" + "\r\n".join(keep) + "\r\nConnection: close\r\n\r\n").encode("latin-1") + rest)
            self._relay(c, up)
        except (OSError, ValueError) as e:
            self._refuse(c, 502, f"connection failed: {e}")

    def _relay(self, a, b):
        a.settimeout(None); b.settimeout(None)
        socks, last = [a, b], time.time()
        try:
            while time.time() - last < 300:
                r, _, _ = select.select(socks, [], [], 5)
                for s in r:
                    d = s.recv(65536)
                    if not d:
                        return
                    (b if s is a else a).sendall(d)
                    last = time.time()
        except OSError:
            pass
        finally:
            a.close(); b.close()

    def note(self, token, text):
        with self.lock:
            n = self.notes.setdefault(token, [])
            if text not in n:
                n.append(text)

    def take_notes(self, token):
        with self.lock:
            return self.notes.pop(token, [])

    def chat_of(self, token):          # set by the backend: job id -> conversation id
        return None
