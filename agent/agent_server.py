#!/usr/bin/env python3
"""Chat app + tool loop for the local LLM: web search, web pages, and a sandboxed per-conversation workspace
(shell, files, Python), plus live stats, model switching, benchmarks and server-side conversations.

Safety: commands run as the unprivileged `llmtools` user from an allowlist, without a shell;
fetch_url refuses private/internal addresses (also after redirects); this server binds to Tailscale only.
"""
import html
import http.client
import ipaddress
import re
import json
import shlex
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import date
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cluster_stats
import netgate

LLM = "http://100.82.180.15:8080/v1/chat/completions"
BIND = ("100.82.180.15", 8081)
UA = "Mozilla/5.0 (X11; Linux x86_64) HomeClusterAgent/1.0"
MAX_STEPS = 8

ALLOWED = {"ls", "cat", "head", "tail", "wc", "grep", "sort", "uniq", "cut", "df", "du", "free", "uptime",
           "date", "cal", "echo", "pwd", "whoami", "uname", "hostname", "ps", "nproc", "lscpu", "lsblk",
           "ip", "ping", "which", "file", "stat", "expr", "seq", "sensors", "tr", "basename", "dirname"}
FORBIDDEN_CHARS = set(";|&<>`$(){}\\\n")

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url", "description": "Download a public web page and return its readable text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a bash command in this conversation's private workspace folder (Linux). Use it to run programs "
                       "(python3 main.py, gcc, make...), install Python packages (pip install NAME), download files (curl, wget, "
                       "git clone), inspect the computer (df -h, free -h, ps), move or delete files. Internet access goes through a "
                       "permission gateway: the first time a website is used the user is asked to allow it; if they deny it, the "
                       "command fails with 'netgate: ... did not allow'. Output is returned.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                                                        "timeout": {"type": "integer", "description": "seconds, default 120, max 600"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file in the workspace (folders are created as needed). The user can see and "
                       "download the files in the app's Files panel.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "relative path, e.g. src/app.py"},
                                                        "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files in the workspace (or a sub-folder of it), with sizes.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "sub-folder, default ."}}}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run Python 3 code in a sandbox and get its output (stdout, stderr, exit code). Use it to test and "
                       "debug code before showing it, and to calculate things. Runs in the workspace folder; packages you pip install "
                       "can be imported; internet only via the permission gateway (like run_shell); no GUI; 120 s limit. "
                       "numpy is available. Files you write stay in the working folder between runs. If the program calls "
                       "input(), give the typed answers in 'stdin', one per line.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "complete Python program"},
            "stdin": {"type": "string", "description": "optional text fed to the program's input(), one answer per line"}},
            "required": ["code"]}}},
]

SYSTEM = (f"You are a helpful assistant running on a small home computer cluster. Today is {date.today():%B %d, %Y}. "
          "You can search the web and read web pages. You also have a private workspace folder for this conversation: "
          "create files with write_file, look at them with list_files and read_file, and run anything there with run_shell "
          "(bash) or run_python. From the workspace you can install Python packages (pip install) and download things "
          "(curl, wget, git clone). PyTorch (CPU), numpy, scikit-learn and matplotlib are already installed; there is no GPU. "
          "Each new website needs the user's permission, which they give in a permission box, so "
          "say briefly why you need a site before using it. "
          "When you write a program, test it with run_python first (give sample answers in stdin if it uses input()), "
          "read the output, fix any errors and test again; only then show the final, working code. "
          "Use tools when they help answer accurately; cite the URLs you used. Keep answers concise. "
          "Your replies are rendered as Markdown (bold, italics, headings, lists, tables, code blocks). "
          "Write math with LaTeX: $...$ inline and $$...$$ for display. "
          "For diagrams, flowcharts or illustrations, use a ```mermaid code block.")


# ---------- tools ----------

def _public_host(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified or ip in ipaddress.ip_network("100.64.0.0/10")):
            return False
    return True


def _check_url(url):
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("only http(s) URLs are allowed")
    if not _public_host(p.hostname):
        raise ValueError("refusing to access a private or internal address")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirect)


def _get(url, data=None, limit=3_000_000):
    _check_url(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with _opener.open(req, timeout=15) as r:
        return r.read(limit).decode(r.headers.get_content_charset() or "utf-8", "replace")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_data(self, d):
        if not self.skip:
            self.parts.append(d)


def fetch_url(url):
    t = _Text()
    t.feed(_get(url))
    text = "\n".join(line.strip() for line in "".join(t.parts).splitlines() if line.strip())
    return text[:6000] + ("\n[...truncated]" if len(text) > 6000 else "")


class _DDG(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results, self.cur, self.field = [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "a" and "result__a" in cls:
            href = a.get("href", "")
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            self.cur = {"title": "", "url": q.get("uddg", [href])[0], "snippet": ""}
            self.results.append(self.cur)
            self.field = "title"
        elif "result__snippet" in cls and self.cur is not None:
            self.field = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self.field = None

    def handle_data(self, d):
        if self.cur is not None and self.field:
            self.cur[self.field] += d


def web_search(query):
    p = _DDG()
    p.feed(_get("https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": query}).encode()))
    res = [r for r in p.results if r["url"].startswith("http")][:6]
    if not res:
        return "No results."
    return "\n\n".join(f"{i+1}. {r['title'].strip()}\n{r['url']}\n{r['snippet'].strip()}" for i, r in enumerate(res))


def run_command(command):
    if any(c in FORBIDDEN_CHARS for c in command):
        return "Refused: pipes, redirects, and other shell features are not allowed. Run one plain command."
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"Refused: {e}"
    if not argv or argv[0] not in ALLOWED:
        return f"Refused: '{argv[0] if argv else ''}' is not an allowed program. Allowed: {', '.join(sorted(ALLOWED))}"
    try:
        p = subprocess.run(["sudo", "-n", "-u", "llmtools", "--", "env", "-C", "/home/llmtools/work", "timeout", "20", *argv],
                           capture_output=True, text=True, timeout=25,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "Command timed out."
    out = (p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else ""))[:8000]
    return out or f"(no output, exit code {p.returncode})"


WORK = Path("/home/llmtools/work/chats")


def sandbox(argv, ws, stdin="", timeout=60, token=None):
    """run argv as the unprivileged llmtools user in workspace ws, with resource limits. With a token (a chat job) it
    can reach the internet, but only through netgate (the firewall allows llmtools nothing else), which asks the user
    before any new website. Without a token it gets no network at all."""
    net = []
    if token:
        proxy = f"http://{token}:x@127.0.0.1:{netgate.PORT}"
        net = [f"{k}={proxy}" for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY")]
        net += ["PIP_USER=1", "PIP_BREAK_SYSTEM_PACKAGES=1", "PIP_DISABLE_PIP_VERSION_CHECK=1", "PIP_NO_INPUT=1", "PIP_PROGRESS_BAR=off", "PIP_NO_WARN_SCRIPT_LOCATION=1"]
    try:
        p = subprocess.run(["sudo", "-n"] + ([] if token else ["unshare", "--net", "--"]) +
                           ["setpriv", "--reuid=llmtools", "--regid=llmtools", "--init-groups", "--",
                            "env", "-i", "-C", str(ws), "HOME=/home/llmtools", "PATH=/home/llmtools/.local/bin:/usr/local/bin:/usr/bin:/bin",
                            "LANG=C.UTF-8", "OPENBLAS_NUM_THREADS=1", "MPLBACKEND=Agg", "TERM=dumb", *net,
                            "prlimit", "--as=1610612736", "--nproc=256", "--fsize=2147483648", "--",
                            "timeout", str(timeout), *argv],
                           input=stdin or "", capture_output=True, text=True, timeout=timeout + 15)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def workspace(chat_id):
    cid = chat_id if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", chat_id or "") else "scratch"
    ws = WORK / cid
    if not ws.is_dir():
        sandbox(["mkdir", "-p", str(ws)], "/home/llmtools/work", timeout=10)
    return ws


def _inside(ws, rel):
    """resolve a relative path and refuse anything outside the workspace"""
    p = (ws / (rel or ".")).resolve()
    if p != ws.resolve() and not p.is_relative_to(ws.resolve()):
        raise ValueError("paths must stay inside the workspace")
    return p


def clip(text, n=1500):
    """short version of a tool result for the page: the start and the end (where errors and 'Successfully installed' are)"""
    return text if len(text) <= n else text[:n // 3] + f"\n… ({len(text) - n} characters skipped) …\n" + text[-(n - n // 3):]


PIP_NOISE = re.compile(r"^\s*(Requirement already satisfied|Collecting |Downloading |Using cached |Obtaining |"
                       r"Installing collected packages|Attempting uninstall|Found existing installation|Uninstalling |"
                       r"Successfully uninstalled|Building wheel|Created wheel|Stored in directory|Preparing metadata|"
                       r"Getting requirements|Installing build dependencies|Looking in indexes|\S+ \S+/\S+ [\d.]+ [kMG]B/s)")


def squeeze(text):
    """drop pip's progress chatter (it can fill the model's memory with thousands of useless tokens)"""
    lines = text.split("\n")
    keep = [l for l in lines if not PIP_NOISE.match(l)]
    if len(keep) < len(lines):
        keep.insert(0, f"[{len(lines) - len(keep)} lines of pip progress hidden]")
    return "\n".join(keep)


def _fmt(rc, out, err, t0):
    out, err = squeeze(out), squeeze(err)
    txt = clip(out, 5000) + (("\n[stderr]\n" + clip(err, 2500)) if err.strip() else "")
    status = "timed out" if rc == 124 else f"exit code {rc}"
    return f"{txt.rstrip() or '(no output)'}\n[{status}, {time.time() - t0:.1f}s]"


def run_shell(command, chat_id, timeout=120, token=None):
    t0, ws = time.time(), workspace(chat_id)
    timeout = max(5, min(int(timeout or 120), 600))
    return _fmt(*sandbox(["bash", "-c", command], ws, timeout=timeout, token=token), t0)


def write_file(path, content, chat_id):
    ws = workspace(chat_id)
    p = _inside(ws, path)
    code = "import os,sys; p=sys.argv[1]; os.makedirs(os.path.dirname(p) or '.', exist_ok=True); open(p,'w').write(sys.stdin.read())"
    rc, out, err = sandbox(["python3", "-c", code, str(p)], ws, stdin=content, timeout=20)
    return f"wrote {p.relative_to(ws)} ({len(content.encode())} bytes)" if rc == 0 else f"could not write: {err[-500:]}"


def read_file(path, chat_id):
    ws = workspace(chat_id)
    p = _inside(ws, path)
    rc, out, err = sandbox(["head", "-c", "12000", str(p)], ws, timeout=10)
    return out if rc == 0 else f"could not read: {err[-300:]}"


def list_files(path, chat_id):
    ws = workspace(chat_id)
    p = _inside(ws, path or ".")
    rc, out, err = sandbox(["find", str(p), "-maxdepth", "4", "-not", "-path", "*/.*", "-printf", "%y %10s  %P\n"], ws, timeout=10)
    rows = sorted(l for l in out.splitlines() if l.strip() and not l.endswith("  "))
    return "\n".join(("dir " if l[0] == "d" else "    ") + l[2:] for l in rows)[:6000] or "(empty workspace)"


def run_python(code, stdin="", chat_id=None, token=None):
    if chat_id is not None:                    # in the conversation's workspace (files persist there; pip packages too)
        t0 = time.time()
        return _fmt(*sandbox(["python3", "-c", code], workspace(chat_id), stdin=stdin, timeout=120, token=token), t0)
    return run_python_isolated(code, stdin)


def run_python_isolated(code, stdin=""):
    """sandbox: separate user, no network namespace, memory/process/file-size limits, 30 s timeout"""
    t0 = time.time()
    try:
        p = subprocess.run(["sudo", "-n", "unshare", "--net", "--", "setpriv", "--reuid=llmtools", "--regid=llmtools", "--init-groups", "--",
                            "env", "-i", "-C", "/home/llmtools/work", "HOME=/home/llmtools", "PATH=/usr/bin:/bin", "MPLBACKEND=Agg",
                            "OPENBLAS_NUM_THREADS=1", "prlimit", "--as=536870912", "--nproc=64", "--fsize=52428800", "--",
                            "timeout", "30", "python3", "-I", "-c", code],
                           input=stdin or "", capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        return "Timed out after 30 s."
    out = p.stdout[:6000] + (("\n[stderr]\n" + p.stderr[-4000:]) if p.stderr.strip() else "")
    status = "timed out after 30 s" if p.returncode == 124 else f"exit code {p.returncode}"
    return f"{out.rstrip() or '(no output)'}\n[{status}, {time.time() - t0:.1f}s]"


def call_tool(name, args, chat_id=None, token=None):
    try:
        if name == "run_shell":
            return run_shell(args["command"], chat_id, args.get("timeout", 120), token)
        if name == "write_file":
            return write_file(args["path"], args.get("content", ""), chat_id)
        if name == "read_file":
            return read_file(args["path"], chat_id)
        if name == "list_files":
            return list_files(args.get("path", "."), chat_id)
        if name == "web_search":
            return web_search(args["query"])
        if name == "fetch_url":
            return fetch_url(args["url"])
        if name == "run_command":
            return run_command(args["command"])
        if name == "run_python":
            return run_python(args["code"], args.get("stdin", ""), chat_id or "scratch", token)
        return f"Unknown tool {name}"
    except Exception as e:
        return f"Tool error: {e}"


# ---------- agent loop ----------

def llm(messages):
    body = json.dumps({"messages": messages, "tools": TOOLS, "max_tokens": 1500}).encode()
    req = urllib.request.Request(LLM, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]


def chat(history):
    messages = [{"role": "system", "content": SYSTEM}] + history
    events = []
    for _ in range(MAX_STEPS):
        msg = llm(messages)
        calls = msg.get("tool_calls") or []
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_content")})
        if not calls:
            return msg.get("content") or "", events
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            t = time.time()
            result = call_tool(name, args)
            events.append({"tool": name, "args": args, "result": clip(result), "seconds": round(time.time() - t, 1)})
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
    return "(stopped after too many tool steps)", events


# Mellum Thinking (a Qwen3-style thinking model) loops ("We should also mention...") with llama-server's default
# sampling and can think until it runs out of tokens. Use its recommended thinking settings, a presence penalty
# against repeating itself, and a cap on thinking after which it has to answer.
MODEL_SAMPLING = {
    "mellum-think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 1.0,
                     "reasoning_budget_tokens": 1024,
                     "reasoning_budget_message": "\n\nI've thought enough; now I'll write the answer.\n"},
}


AFTER_TOOL_BUDGET = 300   # after a tool result the plan already exists; it only needs to read the result, not re-plan


def sampling(messages=()):
    f = model_info().get("file", "")
    p = dict(next((v for k, v in MODEL_SAMPLING.items() if f.startswith(k + "-")), {}))
    if "reasoning_budget_tokens" in p and messages and messages[-1].get("role") == "tool":
        p["reasoning_budget_tokens"] = min(p["reasoning_budget_tokens"], AFTER_TOOL_BUDGET)
    return p


def llm_stream(messages, emit, should_stop=lambda: False, on_open=lambda r: None):
    """One streamed LLM call. Forwards answer/reasoning text through emit() as it arrives and returns the
    assembled message (content, reasoning_content, tool_calls) plus the server's timings."""
    body = json.dumps({"messages": messages, "tools": TOOLS, "max_tokens": 6144, "stream": True, **sampling(messages)}).encode()
    content, reasoning, calls, timings = [], [], {}, None
    # open the connection ourselves (not urlopen): Stop can then cut it even while llama-server is still reading the prompt
    u = urllib.parse.urlparse(LLM)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    conn.request("POST", u.path, body, {"Content-Type": "application/json"})
    on_open(conn)
    with conn.getresponse() as r:
        for raw in list_iter(r):
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if should_stop():
                break                      # closing the stream makes llama-server stop generating
            chunk = json.loads(data)
            timings = chunk.get("timings") or timings
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                    emit({"type": "token", "text": d["content"]})
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    emit({"type": "thinking", "text": d["reasoning_content"]})
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["function"]["name"] += f.get("name") or ""
                    c["function"]["arguments"] += f.get("arguments") or ""
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, timings


def list_iter(r):
    """iterate the response lines; a connection cut by Stop simply ends the iteration"""
    while True:
        try:
            raw = r.readline()
        except (OSError, ValueError, AttributeError):
            return
        if not raw:
            return
        yield raw


def chat_stream(history, emit, should_stop=lambda: False, chat_id=None, on_open=lambda r: None, token=None):
    messages = [{"role": "system", "content": SYSTEM}] + history
    for _ in range(MAX_STEPS):
        try:
            msg, t = llm_stream(messages, emit, should_stop, on_open)
        except (OSError, ValueError, http.client.HTTPException):
            if not should_stop():
                raise
            msg, t = {"role": "assistant", "content": ""}, None
        if should_stop():
            emit({"type": "done", "answer": (msg.get("content") or "") + "\n\n*(stopped)*", "stopped": True})
            return
        if t:
            emit({"type": "stats", "tokens": t.get("predicted_n", 0), "tps": round(t.get("predicted_per_second", 0), 1),
                  "prompt_tokens": t.get("prompt_n", 0), "prompt_tps": round(t.get("prompt_per_second", 0), 1),
                  # how full this conversation's memory is: everything the model holds after this reply
                  "context": t.get("cache_n", 0) + t.get("prompt_n", 0) + t.get("predicted_n", 0),
                  "n_ctx": model_info().get("n_ctx", 0)})
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        seen, uniq = set(), []                  # some models (DeepSeek-Coder-V2) repeat the identical call: run it once
        for c in calls:
            key = (c["function"]["name"], c["function"].get("arguments"))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        if len(uniq) < len(calls):
            calls = msg["tool_calls"] = uniq
        if not calls:
            emit({"type": "done", "answer": msg.get("content") or ""})
            return
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            emit({"type": "tool_start", "tool": name, "args": args})
            t0 = time.time()
            result = call_tool(name, args, chat_id, token)
            if token and name in ("run_shell", "run_python"):
                blocked = GATE.take_notes(token)        # tell the model plainly why a download failed
                if blocked:
                    result += "\n[internet gateway: " + "; ".join(blocked) + "]"
            emit({"type": "tool", "tool": name, "args": args, "result": clip(result), "seconds": round(time.time() - t0, 1)})
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
    emit({"type": "done", "answer": "(stopped after too many tool steps)"})


# ---------- web ----------

PAGE = (Path(__file__).parent / "index.html").read_text()
STATIC = (Path(__file__).parent / "static").resolve()
TYPES = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/chat_events"):
            return self._events()
        if self.path == "/api/model":
            return self._send(200, json.dumps(model_info()), "application/json")
        if self.path == "/api/chats":
            return self._send(200, json.dumps(chat_list()), "application/json")
        if self.path == "/api/switch":
            return self._send(200, json.dumps(switch_state()), "application/json")
        if self.path == "/api/port":
            return self._send(200, json.dumps(PORT), "application/json")
        if self.path == "/api/permissions":
            return self._send(200, json.dumps(permissions_view()), "application/json")
        if self.path.startswith("/api/usage"):
            cid = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("chat", [""])[0]
            return self._send(200, json.dumps(usage_report(cid)), "application/json")
        if self.path.startswith("/api/context/"):
            return self._send(200, json.dumps(context_report(self.path[len("/api/context/"):])), "application/json")
        if self.path.startswith("/api/files/"):
            rest = urllib.parse.unquote(self.path[len("/api/files/"):].split("?")[0])
            cid, _, rel = rest.partition("/")
            if not rel:
                return self._send(200, json.dumps(files_list(cid)), "application/json")
            p = file_get(cid, rel)
            if not p:
                return self._send(404, "not found", "text/plain")
            data = p.read_bytes()[:20_000_000]
            try:
                data.decode("utf-8"); ctype = "text/plain; charset=utf-8"
            except UnicodeDecodeError:
                ctype = "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if "download=1" in self.path or ctype.startswith("application/"):
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
            self.end_headers()
            return self.wfile.write(data)
        if self.path.startswith("/api/chats/"):
            c = chat_load(self.path[len("/api/chats/"):])
            if c and c.get("pending") and c["pending"]["job"] not in JOBS:
                c.pop("pending")                        # its job is gone (server restarted)
            return self._send(200 if c else 404, json.dumps(c or {"error": "no such chat"}), "application/json")
        if self.path == "/api/stats":
            return self._send(200, json.dumps(STATS.snapshot()), "application/json")
        if self.path == "/api/models":
            return self._send(200, json.dumps(model_list()), "application/json")
        if self.path.startswith("/api/bench_live"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            res = bench_live(qs.get("label", [""])[0], int(qs.get("offset", ["0"])[0] or 0))
            return self._send(200, json.dumps(res), "application/json")
        if self.path == "/api/bench":
            return self._send(200, json.dumps(bench_status()), "application/json")
        if self.path.startswith("/api/bench/"):
            d = bench_detail(urllib.parse.unquote(self.path[len("/api/bench/"):]))
            return self._send(200 if d else 404, json.dumps(d or {"error": "no such run"}), "application/json")
        if self.path == "/api/jobs":  # for monitors: questions in the last hour, newest last
            with JOBS_LOCK:
                jobs = sorted(JOBS.items(), key=lambda kv: kv[1].created)
                info = [{"id": k, "created": j.created, "done": j.done, "events": len(j.events),
                         "question": (j.record["history"][-1]["content"] if j.record["history"] else "")[:200]} for k, j in jobs]
            return self._send(200, json.dumps(info), "application/json")
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path.startswith("/static/"):
            f = (STATIC / urllib.parse.unquote(self.path[len("/static/"):].split("?")[0])).resolve()
            if f.is_file() and f.is_relative_to(STATIC) and f.suffix in TYPES:
                b = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", TYPES[f.suffix])
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                return self.wfile.write(b)
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/clientlog":  # script errors reported by the browser page
            n = min(int(self.headers.get("Content-Length", 0)), 10_000)
            print("browser error:", self.rfile.read(n).decode("utf-8", "replace"), flush=True)
            return self._send(204, "", "text/plain")
        if self.path.startswith("/api/chat_cancel"):
            job = JOBS.get(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("id", [""])[0])
            if job:
                job.cancel()
            return self._send(200 if job else 404, json.dumps({"ok": bool(job)}), "application/json")
        if self.path in ("/api/permission", "/api/permissions/revoke"):
            n = min(int(self.headers.get("Content-Length", 0)), 10_000)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            if self.path == "/api/permission":
                ok = permission_decide(str(body.get("id", "")), str(body.get("decision", "deny")))
            else:
                GATE.revoke(str(body.get("site", ""))); ok = True
            return self._send(200 if ok else 404, json.dumps({"ok": ok}), "application/json")
        if self.path == "/api/port":
            n = min(int(self.headers.get("Content-Length", 0)), 10_000)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            code, res = port_chat(str(body.get("chat_id", "")))
            return self._send(code, json.dumps(res), "application/json")
        if self.path == "/api/switch":
            n = min(int(self.headers.get("Content-Length", 0)), 10_000)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            code, res = switch_model(str(body.get("preset", "")), bool(body.get("force")))
            return self._send(code, json.dumps(res), "application/json")
        if self.path == "/api/chat_start":
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_000_000:
                return self._send(413, "too large", "text/plain")
            try:
                body = json.loads(self.rfile.read(n))
                history = [{"role": m["role"], "content": str(m["content"])} for m in body["messages"] if m.get("role") in ("user", "assistant")]
            except Exception as e:
                return self._send(400, json.dumps({"error": html.escape(str(e))}), "application/json")
            jid = start_job(history, self.client_address[0], body.get("chat_id"), body.get("title", ""))
            return self._send(200, json.dumps({"id": jid}), "application/json")
        if self.path == "/api/chat_stream":
            return self._stream()
        if self.path != "/api/chat":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n > 1_000_000:
            return self._send(413, "too large", "text/plain")
        try:
            history = json.loads(self.rfile.read(n))["messages"]
            history = [{"role": m["role"], "content": str(m["content"])} for m in history if m.get("role") in ("user", "assistant")]
            answer, events = chat(history)
            self._send(200, json.dumps({"answer": answer, "events": events}), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": html.escape(str(e))}), "application/json")

    def _events(self):
        # stream a job's events from index `from` as NDJSON; heartbeat every 2 s so idle phones keep the connection
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        job = JOBS.get(qs.get("id", [""])[0])
        if not job:
            return self._send(404, json.dumps({"error": "unknown or expired job"}), "application/json")
        seen = int(qs.get("from", ["0"])[0])
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            while True:
                evs, done = job.wait(seen, 2.0)
                for ev in evs:
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                seen += len(evs)
                if not evs:
                    self.wfile.write(b'{"type": "ping"}\n')
                self.wfile.flush()
                if done and seen >= len(job.events):
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass  # page will reconnect with ?from=

    def _stream(self):
        # newline-delimited JSON events, flushed as they happen (HTTP/1.0: the body ends when the connection closes)
        n = int(self.headers.get("Content-Length", 0))
        if n > 1_000_000:
            return self._send(413, "too large", "text/plain")
        try:
            history = json.loads(self.rfile.read(n))["messages"]
            history = [{"role": m["role"], "content": str(m["content"])} for m in history if m.get("role") in ("user", "assistant")]
        except Exception as e:
            return self._send(400, json.dumps({"error": html.escape(str(e))}), "application/json")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "client": self.client_address[0], "history": history,
                  "tools": [], "answer": None, "stats": []}

        def emit(ev):
            if ev["type"] == "tool":
                record["tools"].append({k: ev[k] for k in ("tool", "args", "result", "seconds")})
            elif ev["type"] == "stats":
                record["stats"].append(ev)
            elif ev["type"] in ("done", "error"):
                record["answer"] = ev.get("answer", ev.get("error"))
            self.wfile.write((json.dumps(ev) + "\n").encode())
            self.wfile.flush()
        try:
            chat_stream(history, emit)
        except (BrokenPipeError, ConnectionResetError):
            record["answer"] = record["answer"] or "(browser disconnected)"
        except Exception as e:
            try:
                emit({"type": "error", "error": html.escape(str(e))})
            except OSError:
                record["answer"] = f"(error: {e})"
        finally:
            log_exchange(record)

    def do_PUT(self):
        if not self.path.startswith("/api/chats/"):
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n > 5_000_000:
            return self._send(413, "too large", "text/plain")
        try:
            c = json.loads(self.rfile.read(n))
            c["id"] = self.path[len("/api/chats/"):]
            c = {"id": c["id"], "title": str(c.get("title", ""))[:80], "messages": list(c.get("messages", [])),
                 "created": c.get("created", time.time())}
        except (ValueError, TypeError) as e:
            return self._send(400, json.dumps({"error": str(e)}), "application/json")
        with CHAT_LOCK:
            ok = chat_save(c)
        return self._send(200 if ok else 400, json.dumps({"ok": ok}), "application/json")

    def do_DELETE(self):
        cid = self.path[len("/api/chats/"):] if self.path.startswith("/api/chats/") else ""
        p = _chat_path(cid)
        if p and p.exists():
            p.unlink()
        if p and (WORK / cid).is_dir():
            sandbox(["rm", "-rf", str(WORK / cid)], "/home/llmtools/work", timeout=30)
        return self._send(200 if p else 404, json.dumps({"ok": bool(p)}), "application/json")

    def log_message(self, fmt, *args):
        print(f"{self.client_address[0]} {fmt % args}", flush=True)


# ---------- resumable chat jobs (phones drop connections: the job keeps running, the page reconnects) ----------

import threading
import uuid

JOBS = {}
JOBS_LOCK = threading.Lock()


class Job:
    def __init__(self, history, client, chat_id=None, jid=None):
        self.chat_id, self.jid = chat_id, jid
        self.events, self.done, self.cond, self.cancelled, self.resp = [], False, threading.Condition(), False, None
        self.created = time.time()
        self.record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "client": client, "history": history,
                       "tools": [], "answer": None, "stats": []}
        threading.Thread(target=self._run, args=(history,), daemon=True).start()

    def emit(self, ev):
        r = self.record
        if ev["type"] == "tool":
            r["tools"].append({k: ev[k] for k in ("tool", "args", "result", "seconds")})
        elif ev["type"] == "stats":
            r["stats"].append(ev)
        elif ev["type"] in ("done", "error"):
            r["answer"] = ev.get("answer", ev.get("error"))
        ev.setdefault("ts", round(time.time(), 2))
        with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    def _run(self, history):
        try:
            chat_stream(history, self.emit, lambda: self.cancelled, self.chat_id or "scratch", self._opened, self.jid)
        except Exception as e:
            self.emit({"type": "error", "error": html.escape(str(e))})
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()
            self.record["model"] = model_info().get("name", "")
            self.record["chat_id"] = self.chat_id
            log_exchange(self.record)
            if self.chat_id:                            # the server saves the answer itself: complete on every device
                with CHAT_LOCK:
                    c = chat_load(self.chat_id)
                    if c:
                        msg = answer_from_events(self.events, self.created, model_info().get("name", ""))
                        if msg["content"].strip():
                            c["messages"].append(msg)
                        elif c["messages"] and c["messages"][-1]["role"] == "user":
                            c["messages"].pop()             # never leave an unanswered question
                        c.pop("pending", None)
                        chat_save(c)

    def _opened(self, r):
        self.resp = r
        if self.cancelled:
            self.cancel()

    def cancel(self):
        """Stop: flag it, and cut the live connection to llama-server so it aborts even mid prompt-reading"""
        self.cancelled = True
        r = self.resp
        if r is not None:
            try:
                r.sock.shutdown(socket.SHUT_RDWR)      # r is the HTTPConnection
            except (AttributeError, OSError):
                pass

    def wait(self, seen, timeout):
        """events after index `seen` (waits up to timeout for new ones); also whether the job has finished"""
        with self.cond:
            if len(self.events) <= seen and not self.done:
                self.cond.wait(timeout)
            return self.events[seen:], self.done


def start_job(history, client, chat_id=None, title=""):
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if time.time() - j.created > 3600]:
            del JOBS[k]
        jid = uuid.uuid4().hex
    if chat_id and _chat_path(chat_id) and history:
        with CHAT_LOCK:                                 # record the question (and that it's being answered) right away
            c = chat_load(chat_id) or {"id": chat_id, "title": (title or history[-1]["content"])[:60], "messages": [], "created": time.time()}
            c["messages"].append({"role": "user", "content": history[-1]["content"]})
            c["pending"] = {"job": jid, "started": time.time()}
            chat_save(c)
            history = history_for_model(c)
    with JOBS_LOCK:
        JOBS[jid] = Job(history, client, chat_id if _chat_path(chat_id) else None, jid)
    return jid


# ---------- which model is loaded (for the page, `ai` and `aitop`) ----------

MODEL_NAMES = [  # file name fragment -> (friendly name, how it runs)
    ("gptoss", "gpt-oss-20B", "MoE 21B, 3.6B active"),
    ("mellum-think", "Mellum 2 12B Thinking", "MoE 12B, 2.5B active"),
    ("mellum", "Mellum 2 12B Instruct", "MoE 12B, 2.5B active"),
    ("qcoder14", "Qwen2.5-Coder 14B", "dense 14B"),
    ("qcoder7", "Qwen2.5-Coder 7B", "dense 7B"),
    ("gemma12", "Gemma 3 12B QAT", "dense 12B"),
]


def model_info():
    return STATS.ai.model or {"up": False, "name": "offline", "kind": "", "file": "", "n_ctx": 0}


# ---------- models you can switch to (all run split over both Macs) ----------

PRESETS = [  # preset, name, what it is good at, benchmark label, slice file stem
    ("gpt-oss", "gpt-oss-20B", "best all-rounder: chat, web research, tools, debugging", "gptoss20b-low", "gptoss-tpa1280"),
    ("mellum-think", "Mellum 2 12B Thinking", "coding: writes and debugs code, fast", "mellum-think", "mellum-think-tp"),
    ("mellum", "Mellum 2 12B Instruct", "coding: fastest, best at writing new code, weak at debugging", "mellum", "mellum-tp"),
    ("qcoder7", "Qwen2.5-Coder 7B", "coding: short, to-the-point answers (dense, slower)", "qcoder7", "qcoder7-tp"),
    ("qcoder14", "Qwen2.5-Coder 14B", "coding: bigger Qwen coder (dense, slowest)", "qcoder14", "qcoder14-tp"),
    ("gemma", "Gemma 3 12B QAT", "general chat, good writing (dense, slow)", "", "gemma12qat-tp"),
    ("dscoder", "DeepSeek-Coder-V2-Lite", "coding: DeepSeek's 16B MoE coder (2.4B active)", "dscoder", "dscoder-tp"),
]
BENCH = Path.home() / "cluster" / "bench" / "results"
SWITCH = {"busy": False, "target": None, "started": 0.0, "error": ""}


def bench_summary():
    rows = {}
    try:
        for line in (BENCH / "summary.tsv").read_text().splitlines():
            f = line.split("\t")
            if not f or f[0] == "smoke" or len(f) < 7:
                continue
            num = lambda pat, txt: [float(x) for x in re.findall(pat, txt)]  # noqa: E731
            a, b = re.search(r"(\d+)% \((\d+)/(\d+)\)", f[1]), re.search(r"(\d+)% \((\d+)/(\d+)\)", f[2])
            rows[f[0]] = {"label": f[0], "write_pct": int(a[1]) if a else None, "write": f"{a[2]}/{a[3]}" if a else "-",
                          "fix_pct": int(b[1]) if b else None, "fix": f"{b[2]}/{b[3]}" if b else "-",
                          "write_avg_s": num(r"([\d.]+)s", f[3])[0] if num(r"([\d.]+)s", f[3]) else None,
                          "write_avg_tokens": int(num(r"([\d.]+) tok", f[3])[0]) if num(r"([\d.]+) tok", f[3]) else None,
                          "fix_avg_s": num(r"([\d.]+)s", f[4])[0] if num(r"([\d.]+)s", f[4]) else None,
                          "gen_tps": num(r"([\d.]+) t/s", f[5])[0] if num(r"([\d.]+) t/s", f[5]) else None,
                          "total_min": num(r"([\d.]+) min", f[6])[0] if num(r"([\d.]+) min", f[6]) else None}
    except OSError:
        pass
    return rows


def switch_state():
    st = dict(SWITCH)
    if st.get("busy") and st.get("phase") == "loading" and st.get("total_bytes"):
        rss = 0
        for n in STATS.snapshot()["nodes"]:
            for p in n.get("ai_processes", []):
                if p.get("model_file") in st.get("files", []):
                    rss += p["ram_mb"] * 1024 * 1024
        st["progress"] = round(min(0.95, rss / st["total_bytes"]), 3)
        SWITCH["progress"] = max(SWITCH.get("progress", 0), st["progress"])
        st["progress"] = SWITCH["progress"]
    st["elapsed"] = round(time.time() - st.get("started", time.time()), 1) if st.get("busy") else 0
    return st


def model_list():
    have2 = set()
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "node2@10.10.10.2", "ls /home/node2/models"],
                             capture_output=True, text=True, timeout=8).stdout.split()
        have2 = set(out)
    except Exception:
        pass
    have1 = set(p.name for p in (Path.home() / "models").glob("*.gguf"))
    scores, cur = bench_summary(), model_info().get("file", "")
    out = []
    for preset, name, good, label, stem in PRESETS:
        r0, r1 = f"{stem}-r0.gguf", f"{stem}-r1.gguf"            # node1 half, node2 half
        out.append({"preset": preset, "name": name, "good_for": good, "available": r0 in have1 and r1 in have2,
                    "loaded": cur == r0, "bench": scores.get(label)})
    return {"models": out, "switching": switch_state()}


def switch_model(preset, force=False):
    if SWITCH["busy"]:
        return 409, {"error": "already switching, give it a minute"}
    if preset not in [p[0] for p in PRESETS]:
        return 400, {"error": "unknown model"}
    bench_running = subprocess.run(["pgrep", "-f", "python3 bench_code.py"], capture_output=True).returncode == 0
    with JOBS_LOCK:                           # someone's question still being answered (stopped ones don't count)
        answering = any(not j.done and not j.cancelled for j in JOBS.values())
    if (bench_running or answering) and not force:
        what = "a benchmark is running" if bench_running else "someone's question is being answered right now"
        return 409, {"error": f"the AI is busy ({what}); switch anyway with force"}
    stem = next(p[4] for p in PRESETS if p[0] == preset)
    size0 = (Path.home() / "models" / f"{stem}-r0.gguf").stat().st_size if (Path.home() / "models" / f"{stem}-r0.gguf").exists() else 0
    try:
        size1 = int(subprocess.run(["ssh", "-o", "BatchMode=yes", "node2@10.10.10.2", f"stat -c %s /home/node2/models/{stem}-r1.gguf"],
                                   capture_output=True, text=True, timeout=8).stdout.strip() or 0)
    except (subprocess.SubprocessError, ValueError):
        size1 = 0
    name = next(p[1] for p in PRESETS if p[0] == preset)
    SWITCH.update(busy=True, target=preset, name=name, started=time.time(), error="", phase="stopping", progress=0.0,
                  files=[f"{stem}-r0.gguf", f"{stem}-r1.gguf"], total_bytes=size0 + size1)

    def run():
        try:
            subprocess.run(["systemd-run", "--user", "--unit=llm-switch", "--collect", "--wait", "-q", "--setenv=KEEP_AGENT=1",
                            "/home/node1/cluster/start-ep.sh", preset], capture_output=True, timeout=120)
            SWITCH["phase"] = "loading"
            for _ in range(180):                      # wait until the new model answers
                try:
                    urllib.request.urlopen(LLM.replace("/v1/chat/completions", "/health"), timeout=3).read()
                    break
                except Exception:
                    time.sleep(1)
            SWITCH.update(phase="warming", progress=0.97)
            STATS.ai.refresh_model()
            warm_cache(wait=True)
            SWITCH.update(phase="ready", progress=1.0)
        except Exception as e:
            SWITCH["error"] = str(e)
        finally:
            SWITCH["busy"] = False
    threading.Thread(target=run, daemon=True).start()
    return 200, {"ok": True, "switching_to": preset}


def bench_status():
    running = None
    for pid in subprocess.run(["pgrep", "-f", "python3 bench_code.py"], capture_output=True, text=True).stdout.split():
        try:
            args = open(f"/proc/{pid}/cmdline").read().split("\0")
            label = args[args.index("bench_code.py") + 1]
        except (OSError, ValueError, IndexError):
            continue
        log = BENCH / f"{label}.log"
        lines = log.read_text().splitlines() if log.exists() else []
        done = [l for l in lines if re.match(r"^[AB] +\d+/\d+", l)]
        running = {"label": label, "done": len(done), "passed": sum(" PASS " in l for l in done), "total": 50,
                   "recent": lines[-15:]}
    runs = []
    for f in sorted(BENCH.glob("*.json")):
        if f.stem == "smoke":
            continue
        try:
            res = json.load(open(f))["results"]
        except (OSError, ValueError):
            continue
        runs.append({"label": f.stem, "tasks": len(res), "passed": sum(r["pass"] for r in res)})
    return {"summary": bench_summary(), "runs": runs, "running": running,
            "queue_active": subprocess.run(["systemctl", "--user", "is-active", "-q", "bench-cand"]).returncode == 0}


def files_list(chat_id):
    ws = WORK / chat_id if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", chat_id or "") else None
    if not ws or not ws.is_dir():
        return []
    out = []
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"path": str(rel), "dir": p.is_dir(), "size": st.st_size, "mtime": st.st_mtime})
    return out[:500]


def file_get(chat_id, rel):
    ws = WORK / chat_id if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", chat_id or "") else None
    if not ws:
        return None
    p = (ws / rel).resolve()
    if not p.is_relative_to(ws.resolve()) or not p.is_file():
        return None
    return p


# ---------- porting a conversation into a newly loaded model (pre-reading it, so the next reply starts at once) ----------

PORT = {"busy": False, "chat": None, "tokens": 0, "error": ""}


def port_chat(chat_id):
    c = chat_load(chat_id)
    if not c or not c.get("messages"):
        return 400, {"error": "empty or unknown conversation"}
    if PORT["busy"] or SWITCH["busy"]:
        return 409, {"error": "busy"}
    msgs = [{"role": "system", "content": SYSTEM}] + history_for_model(c)
    PORT.update(busy=True, chat=chat_id, error="", started=time.time())

    def run():
        try:
            body = json.dumps({"messages": msgs + [{"role": "user", "content": "."}], "tools": TOOLS, "max_tokens": 1}).encode()
            with urllib.request.urlopen(urllib.request.Request(LLM, body, {"Content-Type": "application/json"}), timeout=900) as r:
                PORT["tokens"] = json.load(r).get("timings", {}).get("prompt_n", 0)
        except Exception as e:
            PORT["error"] = str(e)
        finally:
            PORT["busy"] = False
    threading.Thread(target=run, daemon=True).start()
    return 200, {"ok": True}


def bench_live(label, offset):
    """new events from a benchmark's live feed since byte `offset` (default: the run in progress, else the newest feed)"""
    feeds = sorted(BENCH.glob("*.live.jsonl"), key=lambda f: f.stat().st_mtime)
    if not label:
        run = bench_status()["running"]
        label = run["label"] if run else (feeds[-1].name[:-len(".live.jsonl")] if feeds else "")
    f = BENCH / f"{Path(label).name}.live.jsonl"
    if not label or not f.exists():
        return {"label": label, "events": [], "offset": 0, "running": False}
    with open(f, "rb") as fh:
        size = fh.seek(0, 2)
        if offset > size:
            offset = 0                       # the feed was restarted
        fh.seek(offset)
        chunk = fh.read(min(size - offset, 2_000_000))
    end = chunk.rfind(b"\n") + 1            # only whole lines
    events = [json.loads(l) for l in chunk[:end].splitlines() if l.strip()]
    running = subprocess.run(["pgrep", "-f", f"python3 bench_code.py {label}"], capture_output=True).returncode == 0
    return {"label": label, "events": events, "offset": offset + end, "running": running}


def bench_detail(label):
    f = BENCH / f"{Path(label).name}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return {"label": d["label"], "args": d.get("args", {}), "results": d["results"]}


# ---------- conversations, stored on the server so every device sees the same list ----------

CHATDIR = Path(__file__).parent / "chats"
CHAT_LOCK = threading.Lock()


def _chat_path(cid):
    return CHATDIR / f"{cid}.json" if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", cid or "") else None


def chat_load(cid):
    p = _chat_path(cid)
    try:
        return json.loads(p.read_text()) if p and p.exists() else None
    except (OSError, ValueError):
        return None


def chat_save(c):
    p = _chat_path(c.get("id"))
    if not p:
        return False
    CHATDIR.mkdir(exist_ok=True)
    c["updated"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, ensure_ascii=False))
    tmp.replace(p)                                      # atomic: readers never see half a file
    return True


def chat_list():
    out = []
    for p in CHATDIR.glob("*.json") if CHATDIR.exists() else []:
        try:
            c = json.loads(p.read_text())
            out.append({"id": c["id"], "title": c.get("title", ""), "updated": c.get("updated", 0),
                        "n": len(c.get("messages", [])), "pending": c.get("pending")})
        except (OSError, ValueError, KeyError):
            pass
    return sorted(out, key=lambda c: -c["updated"])


def history_for_model(c):
    """the conversation as the model sees it: earlier answers include a short record of the tools they used and what
    came back, so a newly loaded model knows what actually happened (not just the final wording)"""
    out = []
    for m in c.get("messages", []):
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            notes = []
            for st in m.get("steps", []):
                if st.get("perm"):
                    notes.append(f"[website {st['perm'].get('site')}: {st['perm'].get('decision')} by the user]")
                t = st.get("tool")
                if t:
                    a = t.get("args") or {}
                    what = a.get("command") or a.get("query") or a.get("url") or a.get("path") or ("Python program" if a.get("code") else "")
                    res = (t.get("result") or "").strip().replace("\n", " ⏎ ")
                    notes.append(f"[{t.get('tool')}: {what[:120]} → {res[:300]}]")
            content = ("\n".join(notes) + "\n\n" if notes else "") + (m.get("content") or "")
            out.append({"role": "assistant", "content": content})
    return out


def answer_from_events(events, started, model):
    """rebuild the assistant message (thinking, tool steps, answer, stats) from a job's events, like the page does"""
    m = {"role": "assistant", "content": "", "steps": [], "model": model}
    think, t_think = "", None
    for ev in events:
        t = ev["type"]
        if t == "thinking":
            think += ev["text"]; t_think = t_think or ev.get("ts")
            continue
        if think and t in ("token", "tool_start", "done", "error"):
            m["steps"].append({"think": think, "thinkSecs": max(1, round((ev.get("ts") or 0) - (t_think or 0)))}); think, t_think = "", None
        if t == "token":
            m["content"] += ev["text"]
        elif t == "tool_start":
            if m["content"].strip():
                m["steps"].append({"note": m["content"]})
            m["content"] = ""
        elif t == "tool":
            m["steps"].append({"tool": {k: ev.get(k) for k in ("tool", "args", "result", "seconds")}})
        elif t == "permission_result":
            m["steps"].append({"perm": {k: ev.get(k) for k in ("site", "decision", "note")}})
        elif t == "stats":
            m["stats"] = {k: ev.get(k) for k in ("tokens", "tps", "prompt_tokens", "prompt_tps", "context", "n_ctx")}
        elif t == "done":
            m["content"] = ev.get("answer") or m["content"]
        elif t == "error":
            m["content"] += f"\n\n**Error:** {ev.get('error')}"
    m["seconds"] = f"{time.time() - started:.1f}"
    return m


# ---------- website permissions for the workspace (asked through the chat, enforced by netgate + firewall) ----------

PERM_PENDING, PERM_LOCK = {}, threading.Lock()


def permission_ask(token, site, host):
    """called by netgate (in its connection thread) when the AI's code wants a website nobody approved yet"""
    job = JOBS.get(token)
    if not job or job.done:
        return "deny"
    key = (job.chat_id or "scratch", site)
    with PERM_LOCK:                                   # pip opens several connections at once: ask only once per site
        p = next((x for x in PERM_PENDING.values() if x["key"] == key and not x["event"].is_set()), None)
        if not p:
            pid = uuid.uuid4().hex[:12]
            p = {"id": pid, "key": key, "site": site, "host": host, "job": token, "event": threading.Event(), "decision": None,
                 "time": time.time()}
            PERM_PENDING[pid] = p
            job.emit({"type": "permission", "id": pid, "site": site, "host": host})
    t0 = time.time()
    while not p["event"].wait(1):
        if job.cancelled or time.time() - t0 > 180:  # no answer in 3 minutes, or the question was stopped: deny
            permission_decide(p["id"], "deny", "no answer" if not job.cancelled else "stopped")
    return p["decision"]


def permission_decide(pid, decision, note=""):
    with PERM_LOCK:
        p = PERM_PENDING.get(pid)
        if not p or p["event"].is_set():
            return False
        p["decision"] = decision if decision in ("chat", "always") else "deny"
        if p["decision"] != "deny":
            GATE.grant(p["key"][0], p["site"], p["decision"])
        p["event"].set()
    job = JOBS.get(p["job"])
    if job:
        job.emit({"type": "permission_result", "id": pid, "site": p["site"], "decision": p["decision"], "note": note})
    return True


def usage_report(chat_id=""):
    """replies, tokens read/written and AI time: today, last 7 days, all time, by model (from the transcripts), plus
    everything the AI server did since the backend started (from its own log: includes benchmarks)"""
    recs = []
    for f in sorted(LOGDIR.glob("chat-*.jsonl")) if LOGDIR.exists() else []:
        for line in f.open():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    now = time.time()

    def when(r):
        try:
            return time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))
        except (KeyError, ValueError):
            return 0

    def agg(sel):
        out = {"replies": 0, "tokens_written": 0, "tokens_read": 0, "ai_seconds": 0.0, "tool_runs": 0}
        for r in sel:
            st = r.get("stats") or []
            if not st:
                continue
            out["replies"] += 1
            out["tool_runs"] += len(r.get("tools") or [])
            for x in st:
                out["tokens_written"] += x.get("tokens", 0) or 0
                out["tokens_read"] += x.get("prompt_tokens", 0) or 0
                if x.get("tps"):
                    out["ai_seconds"] += (x.get("tokens", 0) or 0) / x["tps"]
                if x.get("prompt_tps"):
                    out["ai_seconds"] += (x.get("prompt_tokens", 0) or 0) / x["prompt_tps"]
        out["ai_seconds"] = round(out["ai_seconds"])
        out["avg_write_tps"] = round(sum((x.get("tokens", 0) or 0) for r in sel for x in (r.get("stats") or [])) /
                                     max(1e-9, sum((x.get("tokens", 0) or 0) / x["tps"] for r in sel for x in (r.get("stats") or []) if x.get("tps"))), 1)
        return out
    midnight = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
    by_model = {}
    for r in recs:
        by_model.setdefault(r.get("model") or "(before model tracking)", []).append(r)
    res = {"today": agg([r for r in recs if when(r) >= midnight]), "week": agg([r for r in recs if when(r) >= now - 7 * 86400]),
           "all": agg(recs), "by_model": {m: agg(v) for m, v in by_model.items()},
           "since": min((r["time"] for r in recs if r.get("time")), default="")}
    if chat_id:
        res["chat"] = agg([r for r in recs if r.get("chat_id") == chat_id])
    with STATS.ai.log.lock:
        t = dict(STATS.ai.log.totals)
    res["server"] = {"replies": t["requests"], "tokens_read": t["prompt_tokens"], "tokens_written": t["gen_tokens"],
                     "ai_seconds": round((t["prompt_ms"] + t["gen_ms"]) / 1000)}
    return res


def _count_tokens(text):
    try:
        req = urllib.request.Request(LLM.replace("/v1/chat/completions", "/tokenize"), json.dumps({"content": text}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return len(json.load(r).get("tokens", []))
    except Exception:
        return len(text) // 4                          # rough fallback while the server is busy or offline


def context_report(chat_id):
    """what fills this conversation's memory, counted with the loaded model's own tokenizer"""
    c = chat_load(chat_id) or {"messages": []}
    parts = [{"what": "instructions (system prompt)", "tokens": _count_tokens(SYSTEM)},
             {"what": f"tool descriptions ({len(TOOLS)} tools)", "tokens": _count_tokens(json.dumps(TOOLS))}]
    for i, m in enumerate(history_for_model(c), 1):
        prev = m["content"].strip().replace("\n", " ")
        parts.append({"what": f"{'you' if m['role'] == 'user' else 'AI'} #{(i + 1) // 2}: {prev[:60]}{'…' if len(prev) > 60 else ''}",
                      "tokens": _count_tokens(m["content"]) + 5, "role": m["role"]})
    last = next((m.get("stats") for m in reversed(c["messages"]) if m.get("role") == "assistant" and (m.get("stats") or {}).get("context")), None)
    n_ctx = model_info().get("n_ctx", 0) or (last or {}).get("n_ctx", 0)
    total = sum(p["tokens"] for p in parts)
    return {"parts": parts, "estimate": total, "measured": (last or {}).get("context"), "n_ctx": n_ctx,
            "messages": len(c["messages"]), "model": model_info().get("name", "")}


def permissions_view():
    with PERM_LOCK:
        pending = [{"id": p["id"], "site": p["site"], "host": p["host"], "chat": p["key"][0]} for p in PERM_PENDING.values()
                   if not p["event"].is_set()]
    return {"always": sorted(GATE.always), "pending": pending}


LOGDIR = Path(__file__).parent / "logs"


def log_exchange(record):
    """keep a transcript of every exchange (the page only holds the chat in the browser)"""
    try:
        LOGDIR.mkdir(exist_ok=True)
        with open(LOGDIR / f"chat-{time.strftime('%Y-%m-%d')}.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"transcript log failed: {e}", flush=True)


def warm_cache(wait=False):
    """Once the model is up, process the fixed system prompt + tool list, so llama-server keeps it cached and the first
    real question only has to read the user's own words (it re-warms if something else evicted it)."""
    import threading

    def run():
        health = LLM.replace("/v1/chat/completions", "/health")
        for _ in range(600):
            try:
                urllib.request.urlopen(health, timeout=5).read()
                break
            except Exception:
                time.sleep(2)
        try:
            body = json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}],
                               "tools": TOOLS, "max_tokens": 1}).encode()
            req = urllib.request.Request(LLM, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                t = json.load(r).get("timings", {})
            print(f"warmed prompt cache: {t.get('prompt_n')} tokens in {t.get('prompt_ms', 0) / 1000:.1f}s", flush=True)
        except Exception as e:
            print(f"cache warm-up failed: {e}", flush=True)
    if wait:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    print(f"agent chat on http://{BIND[0]}:{BIND[1]}", flush=True)
    STATS = cluster_stats.Collector()
    GATE = netgate.Gate(permission_ask)
    GATE.chat_of = lambda token: (JOBS[token].chat_id or "scratch") if token in JOBS else None
    threading.Thread(target=GATE.serve, daemon=True).start()
    warm_cache()
    ThreadingHTTPServer(BIND, Handler).serve_forever()
