#!/usr/bin/env python3
"""aitop: live terminal monitor for the local AI cluster (the terminal twin of the web app's Monitor tab).

Reads the same /api/stats as the web app, so both always show identical numbers: both Macs (CPU per core, RAM, swap,
temperatures, fan, cable traffic, GPU, disk), every AI process running on them, the loaded model, live prompt-reading
and writing speed, exact numbers for the last request (from llama-server's own log), and the words appearing.
Works from any machine that can reach the cluster over Tailscale.

usage: aitop            (q or Ctrl-C to quit)
"""
import json
import os
import select
import sys
import termios
import threading
import time
import tty
import urllib.request

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

AGENT = os.environ.get("CLUSTER_AI_URL", "http://100.82.180.15:8081")
console = Console()
STATE = {"stats": None, "error": None}


def fetch_loop():
    while True:
        try:
            with urllib.request.urlopen(f"{AGENT}/api/stats", timeout=3) as r:
                STATE["stats"], STATE["error"] = json.load(r), None
        except Exception as e:
            STATE["error"] = str(e)
        time.sleep(0.5)


class Words:
    """the newest question asked through the web app or `ai`, followed live"""

    def __init__(self):
        self.question, self.thinking, self.answer, self.tools, self.job, self.live = "", "", "", [], None, False

    def follow(self):
        while True:
            try:
                jobs = json.load(urllib.request.urlopen(f"{AGENT}/api/jobs", timeout=3))
                newest = jobs[-1] if jobs else None
                self.live = bool(newest and not newest["done"])
                if newest and newest["id"] != self.job:
                    self.job, self.question, self.thinking, self.answer, self.tools = newest["id"], newest["question"], "", "", []
                    self.live = not newest["done"]
                    with urllib.request.urlopen(f"{AGENT}/api/chat_events?id={self.job}&from=0", timeout=900) as r:
                        for raw in r:
                            ev = json.loads(raw)
                            if ev["type"] == "thinking":
                                self.thinking += ev["text"]
                            elif ev["type"] == "token":
                                self.answer += ev["text"]
                            elif ev["type"] == "tool_start":
                                a = ev["args"]
                                self.tools.append(f"{ev['tool']}: {a.get('command') or a.get('query') or a.get('url') or 'Python program'}")
                                self.answer, self.thinking = "", ""
            except Exception:
                pass
            time.sleep(1)


# ------------------------------------------------------------------ drawing helpers

def heat(p):
    return "green" if p < 50 else "yellow" if p < 80 else "red"


def temp_color(t):
    return "green" if t < 80 else "yellow" if t < 90 else "red"


def bar(frac, width, color):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    part = " ▏▎▍▌▋▊▉"[int((frac * width - full) * 8)] if full < width else ""
    t = Text("█" * full + part, style=color)
    t.append("·" * (width - full - (1 if part else 0)), style="grey23")
    return t


def rate(b):
    return f"{b / 1e6:5.1f} MB/s" if b >= 1e5 else f"{b / 1e3:5.0f} KB/s"


def gb(mb):
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def graph(values, width, height, vmax, lo, hi):
    vals = list(values)[-width:]
    vals = [0.0] * (width - len(vals)) + vals
    rows = []
    for row in range(height, 0, -1):
        line = Text()
        for v in vals:
            lvl = v / vmax * height if vmax else 0
            ch = "█" if lvl >= row else (" ▁▂▃▄▅▆▇"[int((lvl - (row - 1)) * 8)] if lvl > row - 1 else " ")
            line.append(ch, style=hi if row > height / 2 else lo)
        rows.append(line)
    return rows


def node_panel(n, width):
    if not n.get("online"):
        return Panel(Text("offline: no data from this Mac", style="red"), title=f"[bold]{n['name']}[/]", border_style="red")
    w = max(width - 24, 8)
    t = Table.grid(padding=(0, 1))
    t.add_column(width=6, style="bold", no_wrap=True); t.add_column(no_wrap=True); t.add_column(justify="right", width=6, no_wrap=True)
    for i, c in enumerate(n["cores"]):
        t.add_row(f"cpu{i}", bar(c / 100, w, heat(c)), Text(f"{c:3d}%", style=heat(c)))
    ram = n["ram"]
    t.add_row("ram", bar(ram["used_mb"] / ram["total_mb"], w, "magenta"), Text(gb(ram["used_mb"]), style="magenta"))
    t.add_row("", Text(f"{gb(ram['available_mb'])} free of {gb(ram['total_mb'])} · swap {gb(n['swap']['used_mb'])} used", style="grey62"), "")
    fan = f"   fan {n['fan_rpm']} rpm ({round(100 * n['fan_rpm'] / n['fan_max']) if n['fan_max'] else 0}%)"
    t.add_row("temp", Text.assemble((f"CPU {n['temp_cpu']} °C", f"bold {temp_color(n['temp_cpu'])}"), (f"   SSD {n['temp_ssd']} °C", "grey70"), (fan, "grey70")), "")
    t.add_row("cable", Text(f"↓ {rate(n['net']['rx_bps'])}   ↑ {rate(n['net']['tx_bps'])}", style="cyan"), "")
    t.add_row("gpu", Text("idle" if n["gpu_mhz"] <= 350 else f"{n['gpu_mhz']} MHz", style="blue"), "")
    t.add_row("disk", Text(f"{n['disk']['free_gb']} of {n['disk']['total_gb']} GB free · load {n['load']:.2f}", style="grey70"), "")
    return Panel(t, title=f"[bold]{n['name']}[/] [dim]{n['role']}[/]", border_style="bright_blue", title_align="left")


def ai_panel(a, width):
    m = a.get("model") or {}
    if not a.get("up"):
        return Panel(Text("AI server offline  (start it: ~/cluster/start-llm.sh)", style="red"), title="[bold]AI[/]", border_style="red")
    w = max(width - 16, 20)
    if a["phase"] == "reading":
        rt = f"{a['prefill_tps']:5.1f}" if a["prompt_done"] else "  ..."
        big = Group(Text.assemble(("📖 reading prompt  ", "bold yellow"), (rt, "bold bright_yellow"), (" tok/s  ", "yellow"),
                                  (f"{a['prompt_done']}/{a['prompt_total']} tokens" + (f" (+{a['cached_tokens']} cached)" if a["cached_tokens"] else ""), "grey70")),
                    bar(a["prompt_done"] / a["prompt_total"] if a["prompt_total"] else 0, w, "yellow"))
    elif a["phase"] == "writing":
        big = Text.assemble(("✍️  writing  ", "bold bright_cyan"), (f"{a['gen_tps']:5.1f}", "bold bright_cyan"), (" tok/s  ", "cyan"),
                            (f"{a['gen_tokens']} tokens · prompt was read at {a['prefill_tps']:.1f} tok/s", "grey70"))
    else:
        big = Text("💤 idle, waiting for a question", style="grey58")
    g = Table.grid(); g.add_column(width=10, style="grey58", no_wrap=True); g.add_column(no_wrap=True, overflow="crop")
    top = max(max(a["hist"]["gen"]), 20.0)
    for i, row in enumerate(graph(a["hist"]["gen"], w, 4, top, "cyan", "bright_cyan")):
        g.add_row("writing" if i == 0 else (f"{top:.0f} t/s" if i == 1 else ""), row)
    topp = max(max(a["hist"]["prefill"]), 60.0)
    for i, row in enumerate(graph(a["hist"]["prefill"], w, 2, topp, "yellow", "bright_yellow")):
        g.add_row("prefill" if i == 0 else f"{topp:.0f} t/s", row)
    l, tt = a.get("last"), a.get("totals") or {}
    last = Text("last request: ", style="grey58")
    if l:
        last.append(f"read {l['prompt_tokens']} tokens @ {l['prompt_tps']} t/s", style="yellow")
        last.append("  ·  ", style="grey42")
        last.append(f"wrote {l['gen_tokens']} tokens @ {l['gen_tps']} t/s", style="cyan")
        last.append(f"  ·  {l['seconds']}s  ", style="grey58")
        last.append("exact (llama-server)" if l["exact"] else "estimated", style="green" if l["exact"] else "grey50")
    else:
        last.append("none yet", style="grey42")
    tot = Text(f"since start: {tt.get('requests', 0)} requests · {tt.get('prompt_tokens', 0):,} tokens read · "
               f"{tt.get('gen_tokens', 0):,} written · avg {tt.get('avg_gen_tps', 0)} tok/s writing", style="grey50")
    info = Text(f"{m.get('file', '')} · memory {m.get('n_ctx', 0):,} tokens", style="grey42")
    title = f"[bold]AI  {m.get('name', '?')}[/]" + (f" [grey62]· {m['kind']}[/]" if m.get("kind") else "")
    return Panel(Group(big, Text(""), g, last, tot, info), title=title, border_style="bright_magenta", title_align="left")


def procs_panel(nodes):
    t = Table(expand=True, box=None, padding=(0, 1), header_style="bold grey62")
    for c in ("Mac", "program", "framework", "model", "RAM", "CPU"):
        t.add_column(c, no_wrap=True)
    n_rows = 0
    for n in nodes:
        for p in n.get("ai_processes", []):
            t.add_row(n["name"], p["program"], p["framework"], p["model"] or "–", gb(p["ram_mb"]), f"{p['cpu_pct']}%")
            n_rows += 1
    if not n_rows:
        t.add_row("–", "no AI processes running", "", "", "", "")
    return Panel(t, title="[bold]AI processes on the cluster[/] [dim](CPU 100% = one core)[/]", border_style="grey42", title_align="left")


def words_panel(words, height, busy):
    body = Text()
    if busy and not words.live:
        body.append("The AI is busy with a request that didn't come from the chat apps (e.g. a benchmark),\n"
                    "so its words aren't shown here. The speed above is live.\n\n", style="yellow")
        if words.question:
            body.append("last chat question:\n", style="grey50")
    if words.question:
        body.append("❯ " + words.question[:150] + "\n", style="bold bright_magenta")
    for t in words.tools[-3:]:
        body.append("🔧 " + t.replace("\n", " ")[:120] + "\n", style="yellow")
    if words.thinking and not words.answer:
        body.append(words.thinking[-600:].replace("\n", " "), style="italic grey50")
    if words.answer:
        body.append(words.answer[-1500:], style="white")
    lines = body.split("\n")
    body = Text("\n").join(lines[-max(height - 2, 3):])
    return Panel(body, title="[bold]words appearing[/] [dim](questions from the web app or `ai`)[/]", border_style="grey42", title_align="left")


def render(words):
    width, height = console.size
    s = STATE["stats"]
    lay = Layout()
    if not s:
        lay.update(Panel(Text(f"connecting to {AGENT} …  {STATE['error'] or ''}", style="yellow"), title="aitop"))
        return lay
    nodes, a = s["nodes"], s["ai"]
    n_cores = max([len(n.get("cores", [])) for n in nodes] + [4])
    n_procs = sum(len(n.get("ai_processes", [])) for n in nodes) or 1
    warns = []
    for n in nodes:
        if not n.get("online"):
            warns.append(f"{n['name']} offline")
        else:
            if n["temp_cpu"] >= 90: warns.append(f"{n['name']} hot {n['temp_cpu']}°C")
            if n["ram"]["available_mb"] < 300: warns.append(f"{n['name']} low memory")
            if n["disk"]["free_gb"] < 5: warns.append(f"{n['name']} disk almost full")
    if not a.get("up"):
        warns.append("AI server offline")
    guard = ("⚠ " + " · ".join(warns)) if warns else "✓ all healthy"
    lay.split_column(Layout(name="head", size=1), Layout(name="nodes", size=n_cores + 8), Layout(name="ai", size=16),
                     Layout(name="procs", size=n_procs + 3), Layout(name="words"))
    lay["head"].update(Text.assemble((" aitop ", "bold black on bright_magenta"), ("  local AI cluster monitor · ", "grey70"),
                                     (s["time"], "bold"), ("   q to quit   ", "grey42"), (guard, "bold yellow" if warns else "green")))
    lay["nodes"].split_row(Layout(node_panel(nodes[0], width // 2)), Layout(node_panel(nodes[1], width // 2)))
    lay["ai"].update(ai_panel(a, width))
    lay["procs"].update(procs_panel(nodes))
    used = 1 + n_cores + 8 + 16 + n_procs + 3
    lay["words"].update(words_panel(words, max(height - used, 4), a.get("phase") in ("reading", "writing")))
    return lay


def main():
    words = Words()
    threading.Thread(target=fetch_loop, daemon=True).start()
    threading.Thread(target=words.follow, daemon=True).start()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    try:
        if old:
            tty.setcbreak(fd)
        with Live(get_renderable=lambda: render(words), console=console, screen=True, refresh_per_second=6):
            while True:
                if old and select.select([sys.stdin], [], [], 0.2)[0]:
                    if sys.stdin.read(1).lower() == "q":
                        break
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
