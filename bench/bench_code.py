#!/usr/bin/env python3
"""Coding benchmark for whatever model llama-server is serving on :8080.

Part A (write code): HumanEval problems (seeded random subset). The model writes the function; hidden tests grade it.
Part B (agent):      10 bug-fix tasks. The model gets a run_python tool, may test as much as it likes (up to 8 steps),
                     and must answer with the fixed code; hidden tests grade it.
All code runs in the same sandbox as the chat tool: user llmtools, no network, 512 MB, time limit.

usage: bench_code.py LABEL [--n 40] [--effort low|medium|high] [--temp 0.3] [--parts AB]
Results: results/LABEL.json (every answer kept) + one summary line in results/summary.tsv
"""
import argparse
import gzip
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fix_tasks import TASKS

LLM = "http://100.82.180.15:8080/v1/chat/completions"
HERE = Path(__file__).parent
LIVE = None   # results/LABEL.live.jsonl: question, thinking, answer (as it streams), tool runs, verdict -- for aibench -r / web


def live(ev):
    if LIVE:
        ev["t"] = round(time.time(), 2)
        LIVE.write(json.dumps(ev) + "\n")
        LIVE.flush()


def sandbox(code, timeout=15):
    try:
        p = subprocess.run(["sudo", "-n", "unshare", "--net", "--", "setpriv", "--reuid=llmtools", "--regid=llmtools", "--init-groups", "--",
                            "env", "-i", "-C", "/home/llmtools/work", "HOME=/home/llmtools", "PATH=/usr/bin:/bin", "OPENBLAS_NUM_THREADS=1",
                            "prlimit", "--as=536870912", "--nproc=64", "--fsize=52428800", "--",
                            "timeout", str(timeout), "python3", "-I", "-c", code],
                           capture_output=True, text=True, timeout=timeout + 10, stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout[-3000:] + ("\n[stderr]\n" + p.stderr[-3000:] if p.stderr.strip() else ""))
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def chat(messages, args, tools=None):
    body = {"messages": messages, "max_tokens": args.max_tokens, "temperature": args.temp, "top_p": 0.95, "stream": True}
    if args.effort:
        body["chat_template_kwargs"] = {"reasoning_effort": args.effort}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    content, reasoning, calls, timings = [], [], {}, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            timings = chunk.get("timings") or timings
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"]); live({"type": "token", "text": d["content"]})
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"]); live({"type": "thinking", "text": d["reasoning_content"]})
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
    live({"type": "timings", "gen_tps": round(timings.get("predicted_per_second", 0), 1), "gen_tokens": timings.get("predicted_n", 0),
          "prompt_tps": round(timings.get("prompt_per_second", 0), 1)})
    return {"choices": [{"message": msg}], "timings": timings, "usage": {"completion_tokens": timings.get("predicted_n", 0)}}


def last_code_block(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def part_a(args, out):
    probs = [json.loads(l) for l in gzip.open(HERE / "HumanEval.jsonl.gz", "rt")]
    random.Random(1234).shuffle(probs)
    probs = probs[:args.n]
    for i, p in enumerate(probs, 1):
        msg = [{"role": "system", "content": "You are an expert Python programmer."},
               {"role": "user", "content": "Complete this Python function. Reply with the complete function (including the "
                                           "signature and any imports it needs) in a single ```python code block.\n\n```python\n"
                                           + p["prompt"] + "```"}]
        t0 = time.time()
        live({"type": "question", "part": "A", "id": p["task_id"], "n": i, "of": len(probs),
              "text": "Complete this Python function:\n\n" + p["prompt"]})
        try:
            r = chat(msg, args)
            text = r["choices"][0]["message"].get("content") or ""
            t, usage = r.get("timings", {}), r.get("usage", {})
        except Exception as e:
            text, t, usage = f"(request failed: {e})", {}, {}
        code = last_code_block(text) or text
        if f"def {p['entry_point']}" not in code:
            code = p["prompt"] + code                      # model answered with the body only
        rc, log = sandbox(code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n")
        res = {"part": "A", "id": p["task_id"], "pass": rc == 0, "seconds": round(time.time() - t0, 1),
               "gen_tokens": usage.get("completion_tokens", 0), "gen_tps": round(t.get("predicted_per_second", 0), 1),
               "answer": text[-4000:], "log": log[-800:]}
        out.append(res)
        live({"type": "result", "id": p["task_id"], "pass": res["pass"], "seconds": res["seconds"], "log": res["log"][-600:],
              "gen_tokens": res["gen_tokens"], "gen_tps": res["gen_tps"]})
        print(f"A {i:2}/{len(probs)} {p['task_id']:14} {'PASS' if res['pass'] else 'fail'}  {res['seconds']:5.1f}s "
              f"{res['gen_tokens']:4} tok @ {res['gen_tps']} t/s", flush=True)


TOOL = [{"type": "function", "function": {
    "name": "run_python", "description": "Run a Python 3 program and see its stdout/stderr and exit code. No internet.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]


def part_b(args, out):
    for i, t in enumerate(TASKS, 1):
        msgs = [{"role": "system", "content": "You are an expert Python programmer and debugger. Use the run_python tool to "
                                             "reproduce bugs and test your fixes before answering."},
                {"role": "user", "content": f"This code has a bug.\n\nWhat it should do: {t['desc']}\n\n```python\n{t['code']}```\n\n"
                                            f"Observed problem: {t['example']}\n\nFind and fix the bug(s). Test your fix with "
                                            "run_python. When done, reply with the complete fixed code in one ```python block."}]
        t0, steps, tool_calls, gen_tok, answer = time.time(), 0, 0, 0, ""
        live({"type": "question", "part": "B", "id": t["name"], "n": i, "of": len(TASKS),
              "text": f"Fix the bug. {t['desc']}\n\n{t['code']}\nObserved problem: {t['example']}"})
        try:
            while steps < 8:
                steps += 1
                r = chat(msgs, args, tools=TOOL)
                m = r["choices"][0]["message"]
                gen_tok += r.get("usage", {}).get("completion_tokens", 0)
                calls = m.get("tool_calls") or []
                msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "reasoning_content")})
                if not calls:
                    answer = m.get("content") or ""
                    break
                for c in calls:
                    tool_calls += 1
                    try:
                        code = json.loads(c["function"].get("arguments") or "{}").get("code", "")
                    except json.JSONDecodeError:
                        code = ""
                    rc, log = sandbox(code, timeout=10)
                    live({"type": "tool", "code": code[-3000:], "output": log[-1500:], "exit": rc})
                    msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": f"{log}\n[exit code {rc}]"})
        except Exception as e:
            answer = f"(request failed: {e})"
        code = last_code_block(answer)
        rc, log = sandbox((code or "raise SystemExit('no code block in the answer')") + "\n\n" + t["tests"]) if code else (1, "no code block")
        res = {"part": "B", "id": t["name"], "pass": rc == 0, "seconds": round(time.time() - t0, 1), "gen_tokens": gen_tok,
               "tool_calls": tool_calls, "steps": steps, "answer": answer[-4000:], "log": log[-800:]}
        out.append(res)
        live({"type": "result", "id": t["name"], "pass": res["pass"], "seconds": res["seconds"], "log": res["log"][-600:],
              "gen_tokens": gen_tok, "tool_calls": tool_calls})
        print(f"B {i:2}/{len(TASKS)} {t['name']:16} {'PASS' if res['pass'] else 'fail'}  {res['seconds']:6.1f}s  {tool_calls} tool runs  "
              f"{gen_tok} tok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--effort", default=None, help="gpt-oss reasoning effort: low / medium / high")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--parts", default="AB")
    args = ap.parse_args()
    global LIVE
    (HERE / "results").mkdir(exist_ok=True)
    LIVE = open(HERE / "results" / f"{args.label}.live.jsonl", "w")
    live({"type": "start", "label": args.label, "total": (args.n if "A" in args.parts else 0) + (len(TASKS) if "B" in args.parts else 0)})
    out, t0 = [], time.time()
    if "A" in args.parts:
        part_a(args, out)
    if "B" in args.parts:
        part_b(args, out)
    (HERE / "results").mkdir(exist_ok=True)
    json.dump({"label": args.label, "args": vars(args), "results": out}, open(HERE / "results" / f"{args.label}.json", "w"), indent=1)
    a = [r for r in out if r["part"] == "A"]; b = [r for r in out if r["part"] == "B"]
    pct = lambda xs: f"{100 * sum(r['pass'] for r in xs) / len(xs):.0f}% ({sum(r['pass'] for r in xs)}/{len(xs)})" if xs else "-"  # noqa: E731
    avg = lambda xs, k: sum(r[k] for r in xs) / len(xs) if xs else 0  # noqa: E731
    line = (f"{args.label}\tA {pct(a)}\tB {pct(b)}\tA avg {avg(a, 'seconds'):.1f}s {avg(a, 'gen_tokens'):.0f} tok\t"
            f"B avg {avg(b, 'seconds'):.1f}s\tgen {avg(a, 'gen_tps'):.1f} t/s\ttotal {(time.time() - t0) / 60:.1f} min")
    with open(HERE / "results" / "summary.tsv", "a") as f:
        f.write(line + "\n")
    live({"type": "end", "summary": line})
    print("\n" + line)


if __name__ == "__main__":
    main()
