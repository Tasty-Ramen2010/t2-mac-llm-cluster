#!/usr/bin/env python3
"""local-coder: hand a coding job to the local model on the Mac minis and get back tested code.

The model writes code, runs it in the sandbox (no network, 512 MB, 30 s) as often as it wants, and answers with the
final code. If you give tests (Python asserts), they are run against the answer; failures go back to the model to fix,
up to --rounds times. Meant to be called by Claude Code (or you) to farm out well-defined work without burning tokens.

usage:
  local-coder "write a function slugify(s) that ..." [--tests tests.py] [--file context.py ...] [--out result.py]
  echo "task" | local-coder - --tests tests.py
exit code 0 = tests passed (or no tests given and an answer came back), 1 = failed
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLM = "http://100.82.180.15:8080/v1/chat/completions"
SYSTEM = ("You are a senior Python engineer working as a coding sub-agent. Be terse: no introductions, no explanations "
          "unless asked, no summaries. Write the code, test it with run_python (include quick asserts), fix what fails, "
          "and finish with ONLY the final complete code in a single ```python block.")
TOOL = [{"type": "function", "function": {
    "name": "run_python", "description": "Run a Python 3 program; returns stdout, stderr and exit code. No internet.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]


def sandbox(code, timeout=30):
    try:
        p = subprocess.run(["sudo", "-n", "unshare", "--net", "--", "setpriv", "--reuid=llmtools", "--regid=llmtools", "--init-groups", "--",
                            "env", "-i", "-C", "/home/llmtools/work", "HOME=/home/llmtools", "PATH=/usr/bin:/bin", "OPENBLAS_NUM_THREADS=1",
                            "prlimit", "--as=536870912", "--nproc=64", "--fsize=52428800", "--",
                            "timeout", str(timeout), "python3", "-I", "-c", code],
                           capture_output=True, text=True, timeout=timeout + 10, stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout[-4000:] + ("\n[stderr]\n" + p.stderr[-4000:] if p.stderr.strip() else ""))
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def chat(messages, effort):
    body = {"messages": messages, "tools": TOOL, "max_tokens": 6000, "temperature": 0.3, "top_p": 0.95}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def solve(messages, effort, log, max_steps=10):
    """run the tool loop until the model answers; returns the final text"""
    for _ in range(max_steps):
        r = chat(messages, effort)
        m = r["choices"][0]["message"]
        messages.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "reasoning_content")})
        calls = m.get("tool_calls") or []
        if not calls:
            return m.get("content") or ""
        for c in calls:
            try:
                code = json.loads(c["function"].get("arguments") or "{}").get("code", "")
            except json.JSONDecodeError:
                code = ""
            rc, out = sandbox(code)
            log(f"  ran code ({len(code.splitlines())} lines) -> exit {rc}")
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": f"{out}\n[exit code {rc}]"})
    return ""


def main():
    ap = argparse.ArgumentParser(description="Delegate a coding task to the local model and get tested code back.")
    ap.add_argument("task", help="task description, or - to read it from stdin")
    ap.add_argument("--tests", help="file with Python asserts the result must pass")
    ap.add_argument("--file", action="append", default=[], help="context file to show the model (repeatable)")
    ap.add_argument("--out", help="write the final code here")
    ap.add_argument("--rounds", type=int, default=3, help="fix attempts when the tests fail")
    ap.add_argument("--effort", default="medium", help="gpt-oss reasoning effort (ignored by other models)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    log = (lambda s: None) if a.quiet else (lambda s: print(s, file=sys.stderr, flush=True))
    task = sys.stdin.read() if a.task == "-" else a.task
    ctx = "".join(f"\n\nFile `{f}`:\n```\n{Path(f).read_text()}\n```" for f in a.file)
    tests = Path(a.tests).read_text() if a.tests else ""
    prompt = task + ctx + (f"\n\nYour code must pass these tests:\n```python\n{tests}\n```" if tests else "")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    t0, code, passed = time.time(), None, False
    for rnd in range(1, a.rounds + 1):
        log(f"round {rnd}: asking the local model...")
        text = solve(messages, a.effort, log)
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
        code = blocks[-1] if blocks else None
        if code is None:
            messages.append({"role": "user", "content": "Reply with the final complete code in one ```python block."})
            continue
        if not tests:
            passed = True
            break
        rc, out = sandbox(code + "\n\n" + tests)
        if rc == 0:
            passed = True
            log(f"round {rnd}: tests PASS")
            break
        log(f"round {rnd}: tests FAIL (exit {rc})")
        messages.append({"role": "user", "content": f"Your code fails the tests:\n```\n{out[-2500:]}\n```\nFix it. Answer with the complete fixed code only."})
    status = "PASS" if passed and tests else ("DONE (no tests given)" if passed else "FAIL")
    log(f"{status} after {time.time() - t0:.0f}s")
    if code and a.out:
        Path(a.out).write_text(code)
        log(f"wrote {a.out}")
    if code and not a.out:
        print(code)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
