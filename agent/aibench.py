#!/usr/bin/env python3
"""aibench: coding-benchmark results in the terminal (same data as the web app's Benchmarks tab).

usage: aibench                 summary of every model + the run in progress
       aibench LABEL           every problem for one model (e.g. aibench mellum-think)
       aibench LABEL TASK      the model's answer and the test output for one problem (e.g. aibench mellum HumanEval/82)
       aibench --live          follow the benchmark that is running right now (progress)
       aibench -r              follow it in detail: every question, the model's thinking and answer as it is written,
                               each Python run, and the graded result   (also: --responses-detailed [LABEL])
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

AGENT = os.environ.get("CLUSTER_AI_URL", "http://100.82.180.15:8081")
LABELS = {"gptoss20b-low": "gpt-oss-20B (low reasoning)", "gptoss20b-medium": "gpt-oss-20B (medium reasoning)",
          "mellum": "Mellum 2 12B Instruct", "mellum-think": "Mellum 2 12B Thinking", "qcoder7": "Qwen2.5-Coder 7B", "qcoder14": "Qwen2.5-Coder 14B"}
console = Console()


def get(path):
    with urllib.request.urlopen(AGENT + path, timeout=15) as r:
        return json.load(r)


def meter(pct, width=16, color="blue"):
    if pct is None:
        return Text("–")
    n = round(pct / 100 * width)
    return Text.assemble(("█" * n, color), ("·" * (width - n), "grey30"), (f" {pct}%", "bold"))


def running_panel(run):
    done, total = run["done"], run["total"]
    n = round(done / total * 40)
    head = Text.assemble(("▶ ", "bold magenta"), (LABELS.get(run["label"], run["label"]), "bold"),
                         (f"   {'█' * n}{'·' * (40 - n)} {done}/{total} · {run['passed']} passed", "magenta"))
    body = Text("\n").join(Text(l, style="green" if " PASS " in l else "red" if " fail " in l else "grey70") for l in run["recent"][-12:])
    return Panel(Text.assemble(head, "\n\n", body), title="[bold]running now[/]", border_style="magenta", title_align="left")


def summary():
    b = get("/api/bench")
    if b["running"]:
        console.print(running_panel(b["running"]))
    t = Table(title="Coding benchmark · every answer graded by running hidden tests", title_style="bold", header_style="bold grey62")
    t.add_column("model", no_wrap=True); t.add_column("label", style="grey50", no_wrap=True)
    t.add_column("write code (40)", no_wrap=True); t.add_column("fix bugs with tools (10)", no_wrap=True)
    t.add_column("speed", justify="right", no_wrap=True); t.add_column("avg time / problem", justify="right", no_wrap=True)
    rows = sorted(b["summary"].values(), key=lambda r: -((r["write_pct"] or 0) + (r["fix_pct"] or 0)))
    for r in rows:
        t.add_row(LABELS.get(r["label"], r["label"]), r["label"], meter(r["write_pct"]), meter(r["fix_pct"], color="green"),
                  f"{r['gen_tps']} tok/s", f"{r['write_avg_s']}s write · {r['fix_avg_s']}s fix")
    console.print(t)
    console.print("[grey50]details: [cyan]aibench LABEL[/] · one problem: [cyan]aibench LABEL TASK[/] · follow a run: [cyan]aibench --live[/][/]")


def detail(label):
    d = get("/api/bench/" + urllib.parse.quote(label))
    t = Table(title=f"{LABELS.get(label, label)}: every problem", title_style="bold", header_style="bold grey62")
    for c in ("", "task", "part", "time", "tokens", "tool runs"):
        t.add_column(c, no_wrap=True)
    for r in d["results"]:
        t.add_row(Text("✓", style="bold green") if r["pass"] else Text("✗", style="bold red"), r["id"],
                  "write code" if r["part"] == "A" else "fix bug", f"{r['seconds']}s", str(r.get("gen_tokens", "–")), str(r.get("tool_calls", "–")))
    console.print(t)
    p = sum(r["pass"] for r in d["results"])
    console.print(f"[bold]{p}/{len(d['results'])} passed[/] · see one: [cyan]aibench {label} TASK[/]")


def one(label, task):
    d = get("/api/bench/" + urllib.parse.quote(label))
    r = next((r for r in d["results"] if r["id"].lower() == task.lower()), None)
    if not r:
        console.print(f"[red]no task {task} in {label}[/]"); return
    console.print(Panel(Markdown(r["answer"] or "(empty)", code_theme="monokai"),
                        title=f"[bold]{r['id']}[/] · {'[green]PASS[/]' if r['pass'] else '[red]FAIL[/]'} · {r['seconds']}s", title_align="left"))
    console.print(Panel(Text(r["log"] or ("all tests passed" if r["pass"] else "(no output)"), style="grey70"), title="test output", title_align="left"))


def live():
    with Live(console=console, refresh_per_second=2) as lv:
        while True:
            b = get("/api/bench")
            lv.update(running_panel(b["running"]) if b["running"] else Text("no benchmark running right now (Ctrl-C to quit)", style="grey50"))
            time.sleep(2)


class Item:
    def __init__(self, q):
        self.q, self.thinking, self.answer, self.tools, self.tps, self.t0 = q, "", "", [], None, time.time()


def question_panel(q):
    part = "write code" if q["part"] == "A" else "fix the bug (with tools)"
    return Panel(Text(q["text"].strip()[:1800], style="white"), border_style="bright_blue", title_align="left",
                 title=f"[bold]{q['n']}/{q['of']}  {q['id']}[/] [grey62]· {part}[/]")


def tool_panel(tl):
    body = Text.assemble((tl["code"].strip()[-900:], "yellow"), ("\n→ ", "grey50"),
                         (tl["output"].strip()[-500:] or "(no output)", "grey70"), (f"   [exit {tl['exit']}]", "grey50"))
    return Panel(body, title="[bold yellow]🐍 ran Python[/]", border_style="yellow", title_align="left")


def live_view(it, height):
    parts = [question_panel(it.q)]
    for tl in it.tools[-2:]:
        parts.append(tool_panel(tl))
    if it.thinking and not it.answer:
        parts.append(Panel(Text(it.thinking[-500:].replace("\n", " "), style="italic grey50"), title="[grey50]💭 thinking[/]",
                           border_style="grey30", title_align="left", height=6))
    if it.answer:
        lines = it.answer.split("\n")
        keep = max(height - 22, 8)
        cut, tail = lines[:-keep], "\n".join(lines[-keep:])
        if sum(l.strip().startswith("```") for l in cut) % 2:      # the cut landed inside a code block: reopen it
            tail = "```python\n" + tail
        parts.append(Panel(Markdown(tail, code_theme="monokai"), title="[bold cyan]✍️ answer (live)[/]", border_style="cyan", title_align="left"))
    secs = time.time() - it.t0
    parts.append(Text(f"  {secs:.0f}s on this problem" + (f" · {it.tps} tok/s" if it.tps else ""), style="grey50"))
    return Group(*parts)


def responses(label=""):
    offset, it, caught_up = 0, None, False
    console.print("[grey50]following the benchmark live · Ctrl-C to stop[/]")
    with Live(console=console, refresh_per_second=6, transient=True) as lv:
        while True:
            try:
                d = get(f"/api/bench_live?offset={offset}" + (f"&label={urllib.parse.quote(label)}" if label else ""))
            except OSError:
                time.sleep(2); continue
            if offset == 0 and d["events"]:
                lv.console.print(Rule(f"[bold]{LABELS.get(d['label'], d['label'])}[/]", style="magenta"))
            offset = d["offset"]
            evs = d["events"]
            # catch up quickly: everything but the question in progress becomes one line per finished problem
            last_q = max((i for i, e in enumerate(evs) if e["type"] == "question"), default=-1) if not caught_up else -1
            for i, e in enumerate(evs):
                t = e["type"]
                replay = not caught_up and i < last_q
                if t == "question":
                    it = Item(e)
                elif t == "thinking" and it:
                    it.thinking += e["text"]
                elif t == "token" and it:
                    it.answer += e["text"]
                elif t == "tool" and it:
                    it.tools.append(e); it.answer, it.thinking = "", ""
                elif t == "timings" and it:
                    it.tps = e.get("gen_tps")
                elif t == "result" and it:
                    verdict = "[bold green]✓ PASS[/]" if e["pass"] else "[bold red]✗ FAIL[/]"
                    if replay:
                        lv.console.print(f"{verdict} {it.q['n']}/{it.q['of']} {it.q['id']} [grey50]· {e['seconds']}s[/]")
                    else:
                        lv.console.print(question_panel(it.q))
                        for tl in it.tools:
                            lv.console.print(tool_panel(tl))
                        if it.answer.strip():
                            lv.console.print(Panel(Markdown(it.answer, code_theme="monokai"), title="[bold cyan]answer[/]", border_style="cyan", title_align="left"))
                        log = (e.get("log") or "").strip()
                        lv.console.print(Panel(Text(log[-600:] if log else ("all hidden tests passed" if e["pass"] else "(no output)"),
                                                    style="grey70"), border_style="green" if e["pass"] else "red", title_align="left",
                                               title=f"{verdict}  [grey62]{it.q['id']} · {e['seconds']}s"
                                                     + (f" · {e['gen_tps']} tok/s" if e.get("gen_tps") else "") + "[/]"))
                    it = None
                elif t == "end":
                    lv.console.print(Rule(f"[bold]finished[/] [grey62]{e['summary'].split(chr(9), 1)[-1].replace(chr(9), ' · ')}[/]", style="magenta"))
            caught_up = True
            lv.update(live_view(it, console.size.height) if it else Text("  waiting for the next question…" if d["running"] else
                                                                           "  no benchmark running right now", style="grey50"))
            time.sleep(0.4)


def main():
    a = sys.argv[1:]
    try:
        if not a:
            summary()
        elif a[0] == "--live":
            live()
        elif a[0] in ("-r", "--responses-detailed", "--responses", "--watch"):
            responses(a[1] if len(a) > 1 else "")
        elif len(a) == 1:
            detail(a[0])
        else:
            one(a[0], a[1])
    except KeyboardInterrupt:
        pass
    except OSError as e:
        console.print(f"[red]can't reach the AI backend at {AGENT}:[/] {e}")


if __name__ == "__main__":
    main()
