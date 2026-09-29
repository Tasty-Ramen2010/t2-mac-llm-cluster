#!/usr/bin/env python3
"""ai: terminal client for the local AI on the Mac minis (whatever model is loaded on node1 + node2).

Talks to the same backend as the web app: web search, web pages, and a sandboxed workspace per conversation (shell,
files, Python). Streams the model's thinking and answer live and shows the speed. Conversations are saved on the
server (they also appear in the web app's sidebar), so switching models keeps the conversation and ports it over.
Needs only Python 3 + the `rich` package.

usage: ai                          interactive chat (asks which model, then offers your recent conversations)
       ai -c                       continue the most recent conversation
       ai "question"               ask once and exit
       ai --model NAME "question"  switch model first (names: /models)
commands inside: /new /models /use NAME /chats /open N /context /usage /permissions /stats /monitor /bench /files /show FILE
                 /save /think /nodes /help /exit
"""
import json
import os
import readline
import signal
import socket
import sys
import termios
import time
import urllib.error
import urllib.request
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

AGENT = os.environ.get("CLUSTER_AI_URL", "http://100.82.180.15:8081")
HIST = os.path.expanduser("~/.ai_history")
console = Console(highlight=False)
TTY = sys.stdin.isatty()

BANNER = r"""
  ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗      █████╗ ██╗
 ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██╔══██╗██║
 ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝    ███████║██║
 ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗    ██╔══██║██║
 ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║    ██║  ██║██║
  ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝"""


# ------------------------------------------------------------------ terminal input that behaves

def ask(prompt_markup="[bold bright_magenta]❯[/] "):
    """read a line with full line editing. The coloured prompt is handed to readline wrapped in \\001..\\002, so readline
    knows its real width: backspace, arrows, history and long wrapped lines never eat into the prompt or the screen.
    Anything typed while the AI was busy is thrown away first."""
    if TTY:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    with console.capture() as cap:
        console.print(prompt_markup, end="")
    ansi = cap.get()
    safe = ""
    i = 0
    while i < len(ansi):                               # mark every escape sequence as zero-width for readline
        if ansi[i] == "\x1b":
            j = ansi.find("m", i)
            safe += "\x01" + ansi[i:j + 1] + "\x02"
            i = j + 1
        else:
            safe += ansi[i]
            i += 1
    return input(safe)


class Quiet:
    """while the AI is working: don't echo keystrokes into the live display (they are discarded before the next prompt),
    and keep Ctrl-C from flushing half-written output"""

    def __enter__(self):
        self.old = termios.tcgetattr(sys.stdin) if TTY else None
        if self.old:
            new = termios.tcgetattr(sys.stdin)
            new[3] &= ~termios.ECHO
            new[3] |= termios.NOFLSH             # Ctrl-C must not make the terminal throw away output mid-character
            termios.tcsetattr(sys.stdin, termios.TCSANOW, new)
        return self

    def __exit__(self, *exc):
        if self.old:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old)


# ------------------------------------------------------------------ backend

def get_json(path, timeout=5):
    with urllib.request.urlopen(AGENT + path, timeout=timeout) as r:
        return json.load(r)


def post_json(path, body, timeout=30):
    req = urllib.request.Request(AGENT + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {"error": str(e)}


def gradient(text, start=(0, 200, 255), end=(190, 70, 255)):
    lines = text.strip("\n").split("\n")
    width = max(len(l) for l in lines)
    out = Text()
    for li, line in enumerate(lines):
        for i, ch in enumerate(line):
            t = i / max(width - 1, 1)
            r, g, b = (int(start[k] + (end[k] - start[k]) * t) for k in range(3))
            out.append(ch, style=f"bold rgb({r},{g},{b})")
        if li < len(lines) - 1:
            out.append("\n")
    return out


def node_status():
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold"); t.add_column()
    try:
        m = get_json("/api/model")
        name, kind, n_ctx = m["name"], m["kind"], m["n_ctx"]
        up = "[green]● online[/]" if m["up"] else "[red]● offline[/]  (start it with ~/cluster/start-llm.sh)"
        tools = "[green]● online[/]  web search · web pages · workspace: shell, files, python (sandboxed, no internet)"
    except Exception:
        name, kind, n_ctx, up, tools = "?", "", "?", "[red]● offline[/]  (start it with ~/cluster/start-llm.sh)", "[red]● offline[/]"
    t.add_row("model", f"[bold]{name}[/]  [dim]({kind}, {n_ctx} token memory)[/]")
    t.add_row("cluster", f"{up}  node1 + node2 · tensor parallel over a 1 Gb cable")
    t.add_row("tools", tools)
    t.add_row("hardware", "2 × Mac mini 2018 · i3-8100B · 8 GB · Ubuntu 24.04")
    return t


def fence_safe_tail(md, max_lines):
    """last lines of a growing markdown answer, reopening a code block if the cut lands inside one"""
    lines = md.split("\n")
    if len(lines) <= max_lines:
        return md
    cut = len(lines) - max_lines
    fence = None
    for l in lines[:cut]:
        s = l.strip()
        if s.startswith("```"):
            fence = None if fence is not None else (s[3:].strip() or "text")
    tail = "\n".join(lines[cut:])
    return (f"```{fence}\n" + tail) if fence is not None else tail


# ------------------------------------------------------------------ one question

TOOL_TITLES = {"web_search": "🔎 web search", "fetch_url": "🌐 read page", "run_shell": "💻 shell", "run_python": "🐍 python",
               "write_file": "📝 write file", "read_file": "📄 read file", "list_files": "📂 list files", "run_command": "💻 command"}


def tool_body(tool, a):
    if tool == "run_python":
        return Syntax(a.get("code", ""), "python", theme="monokai", word_wrap=True)
    if tool == "run_shell":
        return Syntax(a.get("command", ""), "bash", theme="monokai", word_wrap=True)
    if tool == "write_file":
        path, content = a.get("path", ""), a.get("content", "")
        lines = content.split("\n")
        shown = "\n".join(lines[:40]) + (f"\n… ({len(lines) - 40} more lines)" if len(lines) > 40 else "")
        lexer = Syntax.guess_lexer(path, shown) if path else "text"
        return Group(Text(path, style="bold"), Syntax(shown, lexer, theme="monokai", word_wrap=True))
    return Text(a.get("query") or a.get("url") or a.get("path") or a.get("command") or json.dumps(a)[:300])


class Turn:
    def __init__(self, session, question, show_thinking):
        self.s, self.q, self.show_thinking = session, question, show_thinking
        self.phase, self.tool = "reading", ""
        self.answer, self.thinking = "", ""
        self.stamps, self.n_tok, self.t0 = deque(), 0, time.time()
        self.stats, self.total_tok, self.done, self.error, self.jid = None, 0, False, None, None
        self.last_cmd, self.in_prompt, self.quiet, self.live = "", False, None, None

    def rate(self):
        now = time.time()
        while self.stamps and now - self.stamps[0] > 2:
            self.stamps.popleft()
        if len(self.stamps) < 3:
            return 0.0
        return (len(self.stamps) - 1) / max(self.stamps[-1] - self.stamps[0], 1e-6)

    def view(self):
        h = console.size.height
        parts = []
        if self.thinking and (self.phase == "thinking" or self.show_thinking) and not self.answer:
            parts.append(Panel(Text(self.thinking[-600:].replace("\n", " "), style="italic grey50"), title="[grey50]💭 thinking[/]",
                               border_style="grey23", title_align="left", height=6))
        if self.answer:
            parts.append(Markdown(fence_safe_tail(self.answer, max(h - 8, 5)), code_theme="monokai"))
        r = self.rate()
        label = {"reading": "reading", "thinking": "thinking", "writing": "writing", "tool": f"running {TOOL_TITLES.get(self.tool, self.tool)}",
                 "reconnecting": "connection dropped, reconnecting"}[self.phase]
        meter = f"[bold cyan]{r:5.1f}[/] tok/s · {self.n_tok} tokens · " if self.phase in ("thinking", "writing") else ""
        parts.append(Spinner("dots", text=Text.from_markup(f" [magenta]{label}[/] · {meter}{time.time() - self.t0:.0f}s"
                                                           f"   [grey42](Ctrl-C to stop)[/]", style="grey70"), style="magenta"))
        return Group(*parts)

    def on_event(self, ev, live):
        t = ev["type"]
        if t in ("thinking", "token"):
            self.n_tok += 1
            self.stamps.append(time.time())
            if t == "thinking":
                self.thinking += ev["text"]
                if self.phase != "writing":
                    self.phase = "thinking"
            else:
                self.phase = "writing"
                self.answer += ev["text"]
        elif t == "stats":
            self.stats = ev
            self.total_tok += ev.get("tokens", 0)
            self.s.context, self.s.n_ctx = ev.get("context", self.s.context), ev.get("n_ctx") or self.s.n_ctx
        elif t == "permission":
            self.ask_permission(ev)
        elif t == "permission_result":
            ok = ev.get("decision") in ("chat", "always")
            note = {"chat": "allowed for this conversation", "always": "always allowed", "deny": "denied"}.get(ev.get("decision"), ev.get("decision"))
            live.console.print(f"  {'[green]✓' if ok else '[red]✗'} {ev.get('site')}: {note}[/]" + (f" [grey50]({ev['note']})[/]" if ev.get("note") else ""))
        elif t == "tool_start":
            a = ev.get("args") or {}
            self.last_cmd = a.get("command") or ("Python program" if a.get("code") else "")
            if self.answer.strip():                    # text written before the tool call: keep it on screen
                live.console.print(Markdown(self.answer, code_theme="monokai"))
            self.answer, self.thinking = "", ""
            self.phase, self.tool = "tool", ev["tool"]
            live.console.print(Panel(tool_body(ev["tool"], ev.get("args") or {}), title=f"[bold yellow]{TOOL_TITLES.get(ev['tool'], ev['tool'])}[/]",
                                     title_align="left", border_style="yellow"))
        elif t == "tool":
            res = ev.get("result") or ""
            short = res if len(res) < 1200 else res[:1200] + " …"
            live.console.print(Panel(Text(short, style="grey70"), title=f"[green]result[/] [dim]{ev.get('seconds')}s[/]",
                                     title_align="left", border_style="green"))
            self.phase, self.stamps, self.n_tok = "reading", deque(), 0
        elif t == "done":
            if ev.get("answer"):
                self.answer = ev["answer"]
            self.done = True
        elif t == "error":
            self.error, self.done = ev.get("error", "error"), True

    def ask_permission(self, ev):
        """pause the live display, ask the user about a website, answer the backend"""
        self.live.stop()
        self.quiet.__exit__()
        console.print(Panel(Text.assemble(("The AI wants to connect to ", ""), (ev["site"], "bold"),
                                          (f"  ({ev['host']})" if ev.get("host") != ev["site"] else "", "grey62"),
                                          (f"\nfor: {self.last_cmd[:140]}" if self.last_cmd else "", "grey62")),
                            title="[bold yellow]🔐 permission[/]", border_style="yellow", title_align="left"))
        decision = "deny"
        self.in_prompt = True
        try:
            while True:
                a = ask("[bold]allow?[/] [grey62][y] for this conversation · [a] always · [n] no[/] ").strip().lower()
                if a in ("y", "yes", "a", "always", "n", "no", ""):
                    decision = {"y": "chat", "yes": "chat", "a": "always", "always": "always"}.get(a, "deny")
                    break
        except (KeyboardInterrupt, EOFError):
            decision = "deny"
        finally:
            self.in_prompt = False
        post_json("/api/permission", {"id": ev["id"], "decision": decision}, timeout=10)
        self.quiet.__enter__()
        self.live.start()

    def run(self):
        code, res = post_json("/api/chat_start", {"chat_id": self.s.chat_id, "title": self.s.title or self.q[:60],
                                                   "messages": [{"role": "user", "content": self.q}]})
        if code != 200:
            raise OSError(res.get("error", f"HTTP {code}"))
        self.jid, seen, deadline = res["id"], 0, time.time() + 1800
        self.stop_requested, self.resp = False, None

        def on_ctrl_c(signum, frame):
            if self.in_prompt:                     # answering the permission box: Ctrl-C means "no"
                raise KeyboardInterrupt
            # don't raise mid-write (that can split a character on screen): flag it and cut the event stream instead
            self.stop_requested = True
            try:
                self.resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
        old_handler = signal.signal(signal.SIGINT, on_ctrl_c)
        try:
            self.quiet = Quiet()
            with self.quiet, Live(console=console, get_renderable=self.view, refresh_per_second=12, transient=True) as live:
                self.live = live
                while not self.done and not self.stop_requested and time.time() < deadline:
                    try:
                        with urllib.request.urlopen(f"{AGENT}/api/chat_events?id={self.jid}&from={seen}", timeout=60) as r:
                            self.resp = r
                            for raw in r:
                                if self.stop_requested:
                                    break
                                ev = json.loads(raw)
                                if ev["type"] == "ping":
                                    continue
                                seen += 1
                                if self.phase == "reconnecting":
                                    self.phase = "reading"
                                self.on_event(ev, live)
                                if self.done:
                                    break
                    except urllib.error.HTTPError as e:
                        if e.code == 404:
                            self.error, self.done = "the server lost this question (was it restarted?)", True
                    except (OSError, ValueError):
                        if not self.stop_requested:
                            self.phase = "reconnecting"
                            time.sleep(1.5)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        stopped = self.stop_requested
        if stopped:                                # really stop it on the server, so the Macs don't keep working
            post_json(f"/api/chat_cancel?id={self.jid}", {}, timeout=5)
        if self.answer.strip():
            console.print(Markdown(self.answer.replace("\n\n*(stopped)*", ""), code_theme="monokai"))
        self.s.title = self.s.title or self.q[:60]
        if stopped:
            console.print("[yellow]■ stopped[/] [grey50](the AI finishes the bit it was reading in a few seconds; anything you ask next just waits for it)[/]")
        elif self.error:
            console.print(f"[bold red]error:[/] {self.error}")
        elif self.stats:
            s = self.stats
            console.print(f"[grey50]⚡ [bold cyan]{s['tps']}[/] tok/s · {self.total_tok} tokens · read {s['prompt_tokens']} new prompt "
                          f"tokens at {s['prompt_tps']}/s · {time.time() - self.t0:.1f}s total[/]")


# ------------------------------------------------------------------ models

def models():
    return get_json("/api/models", timeout=20)["models"]


def model_table(ms, numbered=False):
    t = Table(box=None, padding=(0, 2), header_style="bold grey62")
    cols = (["#"] if numbered else []) + ["", "name", "model", "write code", "fix bugs", "speed"]
    for c in cols:
        t.add_column(c, no_wrap=True)
    for i, m in enumerate(ms, 1):
        b = m["bench"] or {}
        row = [str(i)] if numbered else []
        row += ["[green]●[/]" if m["loaded"] else ("[grey42]○[/]" if m["available"] else "[red]✗[/]"),
                f"[bold]{m['preset']}[/]", m["name"], f"{b['write_pct']}%" if b else "–", f"{b['fix_pct']}%" if b else "–",
                f"{b['gen_tps']} tok/s" if b else "–"]
        t.add_row(*row)
    return t


def bar(frac, width=34):
    n = round(max(0.0, min(1.0, frac)) * width)
    return f"[bright_magenta]{'█' * n}[/][grey30]{'·' * (width - n)}[/]"


def wait_switch():
    """live loading bar while the cluster switches model (the same progress the web app shows)"""
    with Quiet(), Live(console=console, refresh_per_second=6, transient=True) as live:
        while True:
            try:
                sw = get_json("/api/switch")
            except OSError:
                time.sleep(1); continue
            if not sw.get("busy"):
                return sw
            phase = {"stopping": "stopping the current model", "loading": "loading the weights into both Macs' memory",
                     "warming": "warming up", "ready": "ready"}.get(sw.get("phase"), sw.get("phase", ""))
            live.update(Text.from_markup(f"  [bold]switching to {sw.get('name', sw.get('target'))}[/]  {bar(sw.get('progress', 0))} "
                                         f"[bold]{round(100 * sw.get('progress', 0))}%[/]  [grey62]{phase} · {sw.get('elapsed', 0):.0f}s[/]"))
            time.sleep(0.5)


def wait_port():
    """loading bar while the new model pre-reads this conversation"""
    with Quiet(), Live(console=console, refresh_per_second=6, transient=True) as live:
        time.sleep(0.5)
        while True:
            try:
                p = get_json("/api/port")
                a = get_json("/api/stats")["ai"]
            except OSError:
                time.sleep(1); continue
            if not p.get("busy"):
                return p
            done, total = a.get("prompt_done", 0), a.get("prompt_total", 0)
            live.update(Text.from_markup(f"  [bold]porting this conversation[/]  {bar(done / total if total else 0.02)} "
                                         f"[bold]{done:,}/{total:,}[/] tokens  [grey62]the new model is reading it[/]"))
            time.sleep(0.5)


def use_model(preset, session=None, quiet_if_loaded=True):
    """switch the cluster to another model (loading bar), then port the conversation (second bar)"""
    ms = models()
    m = next((x for x in ms if x["preset"] == preset), None)
    if not m:
        console.print(f"[red]unknown model '{preset}'.[/] names: {', '.join(x['preset'] for x in ms)}")
        return False
    if m["loaded"]:
        if not quiet_if_loaded:
            console.print(f"[grey50]{m['name']} is already running[/]")
        return True
    if not m["available"]:
        console.print(f"[red]{m['name']} isn't installed on both Macs[/]")
        return False
    code, res = post_json("/api/switch", {"preset": preset})
    if code == 409 and "force" in res.get("error", ""):
        if ask(f"[yellow]{res['error']}. Switch anyway? (y/n)[/] ").strip().lower().startswith("y"):
            code, res = post_json("/api/switch", {"preset": preset, "force": True})
        else:
            return False
    if code != 200:
        console.print(f"[red]can't switch:[/] {res.get('error')}")
        return False
    t0 = time.time()
    sw = wait_switch()
    if sw.get("error"):
        console.print(f"[red]switch failed:[/] {sw['error']}")
        return False
    console.print(f"[green]✓[/] now running [bold]{m['name']}[/] [dim]· loaded in {time.time() - t0:.0f}s[/]")
    if session and session.has_messages():
        code, res = post_json("/api/port", {"chat_id": session.chat_id})
        if code == 200:
            p = wait_port()
            console.print(f"[green]✓[/] conversation ported [dim]({p.get('tokens', 0):,} tokens){' · ' + p['error'] if p.get('error') else ''}[/]")
    return True


class Session:
    """one conversation, stored on the server (also visible in the web app)"""

    def __init__(self):
        self.new()

    def new(self):
        self.chat_id = "t" + format(int(time.time() * 1000), "x")[-10:]
        self.title, self.context, self.n_ctx = "", 0, 0

    def has_messages(self):
        try:
            return bool(get_json(f"/api/chats/{self.chat_id}").get("messages"))
        except OSError:
            return False


HELP = """[bold]commands[/]
  [cyan]/new[/]       start a new conversation
  [cyan]/models[/]    list the models (with benchmark scores)
  [cyan]/use NAME[/]  switch the cluster to another model; this conversation comes along (e.g. /use mellum-think)
  [cyan]/think[/]     keep showing the model's thinking while it answers (toggle)
  [cyan]/stats[/]     both Macs (CPU, memory, temperature, fan, disk) and the AI's speed
  [cyan]/files[/]     the files the AI made in this conversation's workspace · [cyan]/show FILE[/] to read one
  [cyan]/save[/]      save this conversation as a Markdown file (/save name.md)
  [cyan]/chats[/]     all saved conversations (also the web app's) · [cyan]/open N[/] continues one here
  [cyan]/context[/]   what fills this conversation's memory (and how full it is)
  [cyan]/usage[/]     replies, tokens and AI time: this conversation, today, 7 days, all time, by model
  [cyan]/permissions[/] websites the AI may use · [cyan]/permissions remove SITE[/]
  [cyan]/monitor[/]   the full live monitor (aitop) · press q to come back
  [cyan]/bench[/]     coding-benchmark results · [cyan]/bench -r[/] watch a running benchmark live · [cyan]/bench LABEL[/] details
  [cyan]/nodes[/]     show cluster status
  [cyan]/exit[/]      quit  (or Ctrl-D)
Ctrl-C stops an answer (or clears the line). Ask anything: it can search the web, read pages, and create, read and run
files in its own sandboxed workspace. Your conversations also appear in the web app."""


# ------------------------------------------------------------------ quality of life

def status_line(session):
    """one dim line above the prompt: model, conversation memory, both Macs; plus warnings when something needs care"""
    try:
        st = get_json("/api/stats", timeout=3)
    except OSError:
        return "[red]● can't reach the AI backend on node1[/] [grey50](is it running? ~/cluster/start-llm.sh)[/]"
    a, parts, warns = st["ai"], [], []
    m = a.get("model") or {}
    parts.append(f"[bold grey70]{m.get('name')}[/]" if a.get("up") else "[red]AI offline[/]")
    if not a.get("up"):
        warns.append("the AI server isn't running: start it with ~/cluster/start-llm.sh (or it's switching models)")
    n_ctx = session.n_ctx or m.get("n_ctx") or 0
    if n_ctx and session.context:
        pct = session.context / n_ctx
        col = "green" if pct < 0.7 else "yellow" if pct < 0.9 else "red"
        parts.append(f"memory [{col}]{session.context / 1000:.1f}k/{n_ctx // 1000}k ({pct:.0%})[/]")
        if pct >= 0.8:
            warns.append("this conversation's memory is getting full; /new starts fresh (otherwise the start may get cut off)")
    for n in st["nodes"]:
        if not n.get("online"):
            warns.append(f"{n['name']} isn't responding (the AI needs both Macs)")
            continue
        t, free = n["temp_cpu"], n["ram"]["available_mb"]
        tc = "green" if t < 80 else "yellow" if t < 90 else "red"
        parts.append(f"{n['name']} [{tc}]{t}°C[/] {free / 1024:.1f}G free")
        if t >= 90:
            warns.append(f"{n['name']} is running hot ({t} °C): give it air, or take a break")
        if free < 300:
            warns.append(f"{n['name']} is almost out of memory ({free} MB free): avoid starting other big programs")
        if n["disk"]["free_gb"] < 5:
            warns.append(f"{n['name']}'s disk is almost full ({n['disk']['free_gb']} GB free)")
    if a.get("phase") in ("reading", "writing"):
        parts.append("[yellow]AI busy[/]")
    return "[grey50]" + " · ".join(parts) + "[/]" + "".join(f"\n[yellow]⚠ {w}[/]" for w in warns)


def show_stats():
    st = get_json("/api/stats")
    t = Table(box=None, padding=(0, 2), header_style="bold grey62")
    for c in ("", "CPU", "memory used", "free", "swap", "CPU temp", "fan", "SSD", "cable ↓ / ↑", "disk free", "up"):
        t.add_column(c, no_wrap=True)
    for n in st["nodes"]:
        if not n.get("online"):
            t.add_row(n["name"], "[red]offline[/]"); continue
        rate = lambda b: f"{b / 1e6:.1f} MB/s" if b >= 1e5 else f"{b / 1e3:.0f} KB/s"  # noqa: E731
        tc = "green" if n["temp_cpu"] < 80 else "yellow" if n["temp_cpu"] < 90 else "red"
        t.add_row(f"[bold]{n['name']}[/]", f"{n['cpu_pct']:.0f}%", f"{n['ram']['used_mb'] / 1024:.1f} GB", f"{n['ram']['available_mb'] / 1024:.1f} GB",
                  f"{n['swap']['used_mb'] / 1024:.1f} GB", f"[{tc}]{n['temp_cpu']} °C[/]", f"{n['fan_rpm']} rpm", f"{n['temp_ssd']} °C",
                  f"{rate(n['net']['rx_bps'])} / {rate(n['net']['tx_bps'])}", f"{n['disk']['free_gb']} GB", f"{n['uptime_s'] / 3600:.1f} h")
    console.print(t)
    a = st["ai"]
    l, tt = a.get("last"), a.get("totals") or {}
    console.print(f"[bold]AI[/] {(a.get('model') or {}).get('name')} · {a.get('phase')}"
                  + (f" · last reply: read {l['prompt_tokens']} tokens @ {l['prompt_tps']} tok/s, wrote {l['gen_tokens']} @ {l['gen_tps']} tok/s" if l else "")
                  + (f" · {tt.get('requests', 0)} replies since the backend started, avg {tt.get('avg_gen_tps', 0)} tok/s" if tt else ""))
    for n in st["nodes"]:
        for p in n.get("ai_processes", []):
            console.print(f"  [grey50]{n['name']}: {p['program']} ({p['framework']}) {p['model'] or ''} · {p['ram_mb'] / 1024:.1f} GB · CPU {p['cpu_pct']}%[/]")


def show_files(session):
    try:
        files = get_json(f"/api/files/{session.chat_id}")
    except OSError:
        files = []
    if not files:
        console.print("[grey50]no files yet in this conversation's workspace (ask the AI to create or run something)[/]")
        return
    t = Table(box=None, padding=(0, 2), header_style="bold grey62")
    t.add_column("file"); t.add_column("size", justify="right")
    for f in files:
        t.add_row(("📂 " if f["dir"] else "📄 ") + f["path"], "" if f["dir"] else (f"{f['size'] / 1000:.1f} KB" if f["size"] >= 1000 else f"{f['size']} B"))
    console.print(t)
    console.print("[grey50]read one with [cyan]/show FILE[/] · also in the web app's 📁 Files panel[/]")


def show_file(session, path):
    try:
        with urllib.request.urlopen(f"{AGENT}/api/files/{session.chat_id}/{urllib.request.quote(path)}", timeout=10) as r:
            data = r.read()
    except OSError:
        console.print(f"[red]no file {path} in this conversation's workspace[/] [grey50](see /files)[/]")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        console.print(f"[grey50]{path} is a binary file ({len(data)} bytes)[/]")
        return
    console.print(Panel(Syntax(text, Syntax.guess_lexer(path, text), theme="monokai", line_numbers=True, word_wrap=True),
                        title=f"[bold]{path}[/]", title_align="left"))


def save_chat(session, path):
    try:
        c = get_json(f"/api/chats/{session.chat_id}")
    except OSError:
        console.print("[grey50]nothing to save yet[/]")
        return
    out = [f"# {c.get('title', 'conversation')}\n"]
    for m in c["messages"]:
        if m["role"] == "user":
            out.append(f"## You\n\n{m['content']}\n")
        else:
            out.append(f"## {m.get('model') or 'AI'}\n")
            for st in m.get("steps", []):
                if st.get("tool"):
                    tl = st["tool"]
                    a = tl.get("args") or {}
                    out.append(f"*{tl.get('tool')}: {a.get('command') or a.get('query') or a.get('url') or a.get('path') or 'Python program'}*\n\n```\n{(tl.get('result') or '')[:2000]}\n```\n")
            out.append(f"{m['content']}\n")
    path = path or f"ai-chat-{session.chat_id}.md"
    with open(path, "w") as f:
        f.write("\n".join(out))
    console.print(f"[green]✓[/] saved to [bold]{os.path.abspath(path)}[/]")


def list_chats():
    try:
        cs = get_json("/api/chats")
    except OSError:
        cs = []
    if not cs:
        console.print("[grey50]no saved conversations yet[/]")
        return cs
    t = Table(box=None, padding=(0, 2), header_style="bold grey62")
    for c in ("#", "conversation", "messages", "last active"):
        t.add_column(c, no_wrap=True)
    for i, c in enumerate(cs[:30], 1):
        t.add_row(str(i), ("● " if c.get("pending") else "") + c["title"][:70], str(c["n"]), time.strftime("%b %d %H:%M", time.localtime(c["updated"])))
    console.print(t)
    console.print("[grey50]continue one with [cyan]/open N[/] · these are the same conversations the web app shows[/]")
    return cs


def open_chat(session, which):
    try:
        cs = get_json("/api/chats")
    except OSError as e:
        console.print(f"[red]can't list conversations:[/] {e}"); return
    c = cs[int(which) - 1] if which.isdigit() and 1 <= int(which) <= len(cs) else next((x for x in cs if x["id"] == which), None)
    if not c:
        console.print("[grey50]no such conversation · see /chats[/]"); return
    full = get_json(f"/api/chats/{c['id']}")
    session.chat_id, session.title, session.context, session.n_ctx = c["id"], full.get("title", ""), 0, 0
    console.print(Rule(f"[bold]{full.get('title', '')}[/]", style="magenta"))
    for m in full.get("messages", []):
        if m["role"] == "user":
            console.print(Text.assemble(("❯ ", "bold bright_magenta"), (m["content"], "bold")))
        else:
            for st in m.get("steps", []):
                if st.get("perm"):
                    pm = st["perm"]
                    ok = pm.get("decision") in ("chat", "always")
                    console.print(f"  {'[green]✓' if ok else '[red]✗'} 🔐 {pm.get('site')}: {pm.get('decision')}[/]")
                tl = st.get("tool")
                if tl:
                    a = tl.get("args") or {}
                    console.print(f"[yellow]  {TOOL_TITLES.get(tl.get('tool'), tl.get('tool'))}[/] [grey62]{(a.get('command') or a.get('query') or a.get('url') or a.get('path') or 'Python program')[:100]}[/]")
            console.print(Markdown(m.get("content") or "", code_theme="monokai"))
            if m.get("stats", {}).get("context"):
                session.context, session.n_ctx = m["stats"]["context"], m["stats"].get("n_ctx", 0)
            console.print(f"[grey42]  — {m.get('model', '')}[/]")
    console.print(Rule("[grey50]continue below · replies are saved to this conversation[/]", style="grey30"))


def show_context(session):
    try:
        c = get_json(f"/api/context/{session.chat_id}", timeout=30)
    except OSError as e:
        console.print(f"[red]can't read the context:[/] {e}"); return
    n_ctx, used = c["n_ctx"] or 1, c["measured"] or c["estimate"]
    pct = used / n_ctx
    col = "green" if pct < 0.7 else "yellow" if pct < 0.9 else "red"
    console.print(f"[bold]conversation memory[/] ({c['model']}): [{col}]{used:,} of {n_ctx:,} tokens · {pct:.0%}[/]  {bar(pct, 40)}")
    console.print(f"[grey50]{c['messages']} messages · {'measured by the model after its last reply' if c['measured'] else 'estimated'}; "
                  f"when it's full the start of the conversation gets cut off (/new starts fresh)[/]")
    t = Table(box=None, padding=(0, 2), header_style="bold grey62")
    t.add_column("tokens", justify="right"); t.add_column("what")
    parts = c["parts"]
    shown = parts if len(parts) <= 14 else parts[:4] + [{"what": f"… {len(parts) - 12} more messages …", "tokens": sum(p["tokens"] for p in parts[4:-8])}] + parts[-8:]
    for p in shown:
        t.add_row(f"{p['tokens']:,}", p["what"])
    console.print(t)


def show_usage(session):
    try:
        u = get_json(f"/api/usage?chat={session.chat_id}", timeout=20)
    except OSError as e:
        console.print(f"[red]can't read usage:[/] {e}"); return
    hm = lambda s: f"{s // 3600}h {s % 3600 // 60}m" if s >= 3600 else f"{s // 60}m {s % 60}s"  # noqa: E731
    t = Table(box=None, padding=(0, 2), header_style="bold grey62", title="usage (chats from the web app and ai)", title_style="bold")
    for c in ("", "replies", "tokens written", "tokens read", "AI time", "tool runs", "avg writing speed"):
        t.add_column(c, justify="right" if c else "left", no_wrap=True)
    rows = [("this conversation", u.get("chat")), ("today", u["today"]), ("last 7 days", u["week"]), ("all time", u["all"])]
    for name, a in rows:
        if a:
            t.add_row(name, str(a["replies"]), f"{a['tokens_written']:,}", f"{a['tokens_read']:,}", hm(a["ai_seconds"]), str(a["tool_runs"]),
                      f"{a['avg_write_tps']} tok/s" if a["replies"] else "–")
    console.print(t)
    if len(u["by_model"]) > 1 or "(before" not in next(iter(u["by_model"]), "("):
        m = Table(box=None, padding=(0, 2), header_style="bold grey62")
        for c in ("by model", "replies", "tokens written", "AI time"):
            m.add_column(c, no_wrap=True)
        for name, a in sorted(u["by_model"].items(), key=lambda kv: -kv[1]["replies"]):
            m.add_row(name, str(a["replies"]), f"{a['tokens_written']:,}", hm(a["ai_seconds"]))
        console.print(m)
    s = u["server"]
    console.print(f"[grey50]since the backend last started, the AI server did {s['replies']} replies in total (including benchmarks and "
                  f"other programs): {s['tokens_written']:,} tokens written, {s['tokens_read']:,} read · counting since {u['since'] or 'now'}[/]")


def show_permissions(arg=""):
    if arg.startswith("remove "):
        site = arg[7:].strip()
        post_json("/api/permissions/revoke", {"site": site})
        console.print(f"[green]✓[/] {site} removed: the AI will have to ask again")
        return
    p = get_json("/api/permissions")
    console.print("[bold]websites the AI may always use[/] (from its workspace: pip, curl, git, python):")
    for s in p["always"] or []:
        console.print(f"  [green]✓[/] {s}")
    if not p["always"]:
        console.print("  [grey50](none yet; approvals 'for this conversation' last until the backend restarts)[/]")
    console.print("[grey50]remove one with [cyan]/permissions remove SITE[/] · local-network addresses are always blocked[/]")


def recent_chats(limit=5):
    try:
        return get_json("/api/chats")[:limit]
    except OSError:
        return []


def run_tool(script, args=()):
    """run aitop / aibench inside ai (they come back here when you quit them)"""
    import subprocess
    here = os.path.dirname(os.path.realpath(__file__))
    try:
        subprocess.run([sys.executable, os.path.join(here, script), *args])
    except KeyboardInterrupt:
        pass


def choose_model(session):
    """pick the model when `ai` starts (Enter keeps the one that's running)"""
    try:
        ms = [m for m in models() if m["available"]]
    except OSError as e:
        console.print(f"[red]can't reach the AI at {AGENT}:[/] {e}")
        return
    cur = next((m for m in ms if m["loaded"]), None)
    console.print(model_table(ms, numbered=True))
    while True:
        try:
            pick = ask(f"[bold]choose a model[/] [grey62](1-{len(ms)} or a name · Enter keeps {cur['name'] if cur else 'the current one'})[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not pick:
            return
        m = ms[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(ms) else next((x for x in ms if x["preset"] == pick.lower()), None)
        if m:
            use_model(m["preset"], session)
            return
        console.print(f"[grey50]type a number from 1 to {len(ms)}, a name like mellum-think, or just Enter[/]")


def main():
    readline.set_history_length(1000)
    try:
        readline.read_history_file(HIST)
    except OSError:
        pass
    session, show_thinking = Session(), False
    args = sys.argv[1:]
    resume = bool(args) and args[0] in ("-c", "--continue")          # ai -c: continue the latest conversation
    if resume:
        args = args[1:]
    if len(args) >= 2 and args[0] in ("--model", "-m"):           # ai --model mellum-think "question"
        try:
            use_model(args[1])
        except OSError as e:
            console.print(f"[red]can't reach the AI:[/] {e}")
        args = args[2:]
    one_shot = " ".join(args).strip()
    if not one_shot:
        console.print(gradient(BANNER))
        console.print(Panel(node_status(), title="[bold]local AI · no cloud · no tokens burned[/]", border_style="bright_magenta",
                            title_align="left", padding=(0, 1)))
        choose_model(session)
        if resume:
            rc = recent_chats(1)
            open_chat(session, rc[0]["id"]) if rc else console.print("[grey50]no earlier conversation to continue[/]")
        else:
            rc = recent_chats(5)
            if rc:
                console.print("[bold]continue a conversation?[/]")
                for i, c in enumerate(rc, 1):
                    console.print(f"  [cyan]{i}[/]  {c['title'][:70]}  [grey50]{c['n']} messages · {time.strftime('%b %d %H:%M', time.localtime(c['updated']))}[/]")
                try:
                    pick = ask(f"[grey62](1-{len(rc)} · Enter starts a new one)[/] ").strip()
                except (EOFError, KeyboardInterrupt):
                    pick = ""
                if pick.isdigit() and 1 <= int(pick) <= len(rc):
                    open_chat(session, rc[int(pick) - 1]["id"])
        console.print("[grey50]type [cyan]/help[/] for all commands · [cyan]/chats[/] your conversations · [cyan]/monitor[/] live stats · "
                      "Ctrl-C stops an answer · Ctrl-D quits[/]\n")
    while True:
        if one_shot:
            q = one_shot
        else:
            console.print(status_line(session))
            try:
                q = ask().strip()
            except EOFError:
                console.print("\n[grey50]bye[/]")
                break
            except KeyboardInterrupt:
                console.print("[grey42]^C (Ctrl-D or /exit to quit)[/]")
                continue
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        try:
            if q == "/help":
                console.print(HELP)
            elif q == "/new":
                session.new(); console.print(Rule("[grey50]new conversation[/]", style="grey30"))
            elif q == "/think":
                show_thinking = not show_thinking; console.print(f"[grey50]show thinking: {'on' if show_thinking else 'off'}[/]")
            elif q == "/nodes":
                console.print(node_status())
            elif q == "/stats":
                show_stats()
            elif q == "/files":
                show_files(session)
            elif q.startswith("/show"):
                name = q[5:].strip()
                show_file(session, name) if name else console.print("[grey50]usage: /show FILE   (see /files)[/]")
            elif q.startswith("/save"):
                save_chat(session, q[5:].strip())
            elif q == "/chats":
                list_chats()
            elif q == "/context":
                show_context(session)
            elif q == "/usage":
                show_usage(session)
            elif q.startswith("/permissions"):
                show_permissions(q[12:].strip())
            elif q.startswith("/open"):
                arg = q[5:].strip()
                open_chat(session, arg) if arg else console.print("[grey50]usage: /open N   (see /chats)[/]")
            elif q == "/monitor":
                run_tool("aitop.py")
            elif q.startswith("/bench"):
                run_tool("aibench.py", q[6:].split())
            elif q == "/models":
                console.print(model_table(models()))
                console.print("[grey50]switch with [cyan]/use NAME[/] (e.g. /use mellum-think) · this conversation comes along[/]")
            elif q.startswith("/use"):
                name = q[4:].strip().lower()
                if not name:
                    console.print("[grey50]usage: /use NAME   (see /models)[/]")
                else:
                    use_model(name, session, quiet_if_loaded=False)
            elif q.startswith("/"):
                console.print(f"[grey50]unknown command {q.split()[0]} · /help lists them[/]")
            else:
                if session.n_ctx and session.context / session.n_ctx >= 0.9 and not one_shot:
                    ans = ask(f"[yellow]this conversation's memory is {session.context / session.n_ctx:.0%} full; the start may get cut off. "
                              f"Start a new conversation first? (Y/n)[/] ").strip().lower()
                    if not ans.startswith("n"):
                        session.new(); console.print(Rule("[grey50]new conversation[/]", style="grey30"))
                Turn(session, q, show_thinking).run()
        except KeyboardInterrupt:
            console.print("[yellow]■ stopped[/]")
        except OSError as e:
            console.print(f"[bold red]can't reach the AI at {AGENT}[/] ({e}). Is it running? ~/cluster/start-llm.sh")
        if one_shot:
            break
        console.print()
    try:
        readline.write_history_file(HIST)
    except OSError:
        pass


if __name__ == "__main__":
    main()
