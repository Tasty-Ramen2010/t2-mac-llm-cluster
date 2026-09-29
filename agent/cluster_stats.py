"""Live statistics for the two-Mac AI cluster: one source of truth for the web dashboard, `aitop` and `ai`.

Both Macs are sampled with the same code (SAMPLER): it runs in-process on node1 and as a small python loop over one ssh
connection on node2. Every second it reads raw counters (/proc, /sys); rates are computed here from consecutive samples
using the sampling machine's own clock. The AI server is polled 4x per second (/slots for live progress) and its
Prometheus /metrics counters give the exact per-request timings llama-server measured.
"""
import base64
import json
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

LLM = "http://100.82.180.15:8080"
NODE2 = "node2@10.10.10.2"
HIST = 120                                    # graph points (0.5 s apart -> last minute)

# --------------------------------------------------------------------------- sampler (runs on each Mac)

SAMPLER = r'''
import os, re, time, json, glob
AI_EXE = re.compile(r"^(llama-[\w-]+|ggml-rpc-server|ollama|vllm|koboldcpp|lms|whisper[\w-]*|text-generation-launcher|mlc_llm|llamafile)$")
AI_LIBS = (("libtorch", "PyTorch"), ("libopenvino", "OpenVINO"), ("onnxruntime", "ONNX Runtime"), ("libtensorflow", "TensorFlow"),
           ("libllama", "llama.cpp"), ("libggml", "llama.cpp"), ("jaxlib", "JAX"))
CLK = os.sysconf("SC_CLK_TCK")

def rd(p, d=""):
    try:
        with open(p) as f: return f.read()
    except OSError: return d

def hw(name):
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(h + "/name").strip() == name: return h
    return None

def ai_procs():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        raw = rd(f"/proc/{pid}/cmdline")
        if not raw: continue
        args = [a for a in raw.split("\0") if a]
        exe = os.path.basename(args[0])
        kind = None
        if AI_EXE.match(exe): kind = "llama.cpp" if exe.startswith(("llama", "ggml")) else exe
        elif exe.startswith("python"):
            maps = rd(f"/proc/{pid}/maps")
            kind = next((k for lib, k in AI_LIBS if lib in maps), None)
        if not kind: continue
        model = ""
        for flag in ("-m", "--model"):
            if flag in args and args.index(flag) + 1 < len(args): model = os.path.basename(args[args.index(flag) + 1])
        script = next((os.path.basename(a) for a in args[1:] if a.endswith(".py")), "")
        st = rd(f"/proc/{pid}/stat").rsplit(")", 1)[-1].split()
        rss = next((int(l.split()[1]) for l in rd(f"/proc/{pid}/status").splitlines() if l.startswith("VmRSS:")), 0)
        out.append({"pid": int(pid), "exe": exe, "kind": kind, "model": model, "script": script,
                    "rss_mb": rss // 1024, "cpu_s": (int(st[11]) + int(st[12])) / CLK if len(st) > 12 else 0.0,
                    "user": os.stat(f"/proc/{pid}").st_uid})
    return out

def sample():
    s = {"t": time.monotonic(), "host": os.uname().nodename}
    cores = {}
    for l in rd("/proc/stat").splitlines():
        if l.startswith("cpu") and l[3:4].isdigit():
            f = l.split(); v = list(map(int, f[1:9])); cores[int(f[0][3:])] = [sum(v), v[3] + v[4]]
    s["cpu"] = [cores[k] for k in sorted(cores)]
    mi = {}
    for l in rd("/proc/meminfo").splitlines():
        k, _, v = l.partition(":")
        if v.strip(): mi[k] = int(v.split()[0])
    s["mem"] = {k: mi.get(k, 0) for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "SwapCached", "Cached")}
    for l in rd("/proc/net/dev").splitlines():
        if l.strip().startswith("enp4s0:"):
            f = l.split(":")[1].split(); s["net"] = [int(f[0]), int(f[8])]
    h = hw("coretemp"); s["temp_cpu"] = int(rd(h + "/temp1_input", "0")) // 1000 if h else 0
    h = hw("nvme"); s["temp_ssd"] = int(rd(h + "/temp1_input", "0")) // 1000 if h else 0
    fan = glob.glob("/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/*/APP0001:00/fan1_input")
    s["fan"] = int(rd(fan[0], "0")) if fan else 0
    s["fan_max"] = int(rd(fan[0].replace("input", "max"), "0")) if fan else 0
    g = glob.glob("/sys/class/drm/card*/gt_act_freq_mhz"); s["gpu_mhz"] = int(rd(g[0], "0")) if g else 0
    g = glob.glob("/sys/class/drm/card*/gt_max_freq_mhz"); s["gpu_max"] = int(rd(g[0], "0")) if g else 0
    s["load"] = float(rd("/proc/loadavg", "0").split()[0]); s["uptime"] = float(rd("/proc/uptime", "0").split()[0])
    d = os.statvfs("/"); s["disk"] = [d.f_bavail * d.f_frsize, d.f_blocks * d.f_frsize]
    s["procs"] = ai_procs()
    return s
'''
_ns = {}
exec(SAMPLER, _ns)
sample_local = _ns["sample"]


# --------------------------------------------------------------------------- friendly model names

MODEL_NAMES = [  # file name fragment -> (friendly name, type)
    ("gptoss", "gpt-oss-20B", "MoE 21B, 3.6B active"), ("gpt-oss", "gpt-oss-20B", "MoE 21B, 3.6B active"),
    ("mellum-think", "Mellum 2 12B Thinking", "MoE 12B, 2.5B active"), ("mellum2-12b-a2.5b-thinking", "Mellum 2 12B Thinking", "MoE 12B, 2.5B active"),
    ("mellum", "Mellum 2 12B Instruct", "MoE 12B, 2.5B active"),
    ("qcoder14", "Qwen2.5-Coder 14B", "dense 14B"), ("qwen2.5-coder-14b", "Qwen2.5-Coder 14B", "dense 14B"),
    ("qcoder7", "Qwen2.5-Coder 7B", "dense 7B"), ("qwen2.5-coder-7b", "Qwen2.5-Coder 7B", "dense 7B"),
    ("gemma12", "Gemma 3 12B QAT", "dense 12B"), ("gemma-3-12b", "Gemma 3 12B", "dense 12B"),
    ("dscoder", "DeepSeek-Coder-V2-Lite", "MoE 16B, 2.4B active"), ("deepseek-coder-v2-lite", "DeepSeek-Coder-V2-Lite", "MoE 16B, 2.4B active"),
]


def friendly(fname):
    low = fname.lower()
    for frag, name, kind in MODEL_NAMES:
        if frag in low:
            return name, kind
    return fname.replace(".gguf", ""), ""


# --------------------------------------------------------------------------- per-node rates

class Node:
    def __init__(self, name, role):
        self.name, self.role, self.prev, self.cur, self.online, self.last_feed = name, role, None, None, False, 0.0
        self.hist = {"cpu": deque([0.0] * HIST, maxlen=HIST), "rx": deque([0.0] * HIST, maxlen=HIST), "tx": deque([0.0] * HIST, maxlen=HIST)}
        self.rates = {"cores": [], "cpu": 0.0, "rx": 0.0, "tx": 0.0, "proc_cpu": {}}
        self.lock = threading.Lock()

    def feed(self, s):
        with self.lock:
            prev, self.prev, self.cur, self.online, self.last_feed = self.cur, self.cur, s, True, time.monotonic()
            if not prev:
                return
            dt = max(s["t"] - prev["t"], 1e-3)
            cores = []
            for (tot, idle), (ptot, pidle) in zip(s["cpu"], prev["cpu"]):
                d = tot - ptot
                cores.append(0.0 if d <= 0 else max(0.0, min(100.0, 100.0 * (1 - (idle - pidle) / d))))
            rx = tx = 0.0
            if "net" in s and "net" in prev:
                rx, tx = (s["net"][0] - prev["net"][0]) / dt, (s["net"][1] - prev["net"][1]) / dt
            pc = {p["pid"]: p["cpu_s"] for p in prev["procs"]}
            proc_cpu = {p["pid"]: 100.0 * (p["cpu_s"] - pc[p["pid"]]) / dt for p in s["procs"] if p["pid"] in pc}
            self.rates = {"cores": cores, "cpu": sum(cores) / len(cores) if cores else 0.0, "rx": rx, "tx": tx, "proc_cpu": proc_cpu}

    def tick_hist(self):
        with self.lock:
            self.hist["cpu"].append(self.rates["cpu"]); self.hist["rx"].append(self.rates["rx"]); self.hist["tx"].append(self.rates["tx"])

    def snapshot(self):
        with self.lock:
            s, r = self.cur, self.rates
            if not s:
                return {"name": self.name, "role": self.role, "online": False}
            m = s["mem"]
            procs = []
            for p in sorted(s["procs"], key=lambda p: -p["rss_mb"]):
                name, kind = friendly(p["model"]) if p["model"] else ("", "")
                procs.append({"pid": p["pid"], "program": p["exe"] + (f" {p['script']}" if p["script"] else ""), "framework": p["kind"],
                              "model": name, "model_file": p["model"], "ram_mb": p["rss_mb"], "cpu_pct": round(r["proc_cpu"].get(p["pid"], 0.0))})
            return {
                "name": self.name, "host": s["host"], "role": self.role, "online": self.online and time.monotonic() - self.last_feed < 5,
                "cpu_pct": round(r["cpu"], 1), "cores": [round(c) for c in r["cores"]],
                "ram": {"used_mb": (m["MemTotal"] - m["MemAvailable"]) // 1024, "total_mb": m["MemTotal"] // 1024, "available_mb": m["MemAvailable"] // 1024},
                "swap": {"used_mb": (m["SwapTotal"] - m["SwapFree"]) // 1024, "total_mb": m["SwapTotal"] // 1024,
                         "also_in_ram_mb": m["SwapCached"] // 1024},
                "net": {"rx_bps": round(r["rx"]), "tx_bps": round(r["tx"])},
                "temp_cpu": s["temp_cpu"], "temp_ssd": s["temp_ssd"], "fan_rpm": s["fan"], "fan_max": s["fan_max"],
                "gpu_mhz": s["gpu_mhz"], "gpu_max_mhz": s["gpu_max"], "load": s["load"], "uptime_s": int(s["uptime"]),
                "disk": {"free_gb": round(s["disk"][0] / 1e9, 1), "total_gb": round(s["disk"][1] / 1e9, 1)},
                "ai_processes": procs,
                "hist": {k: [round(v, 1) for v in d] for k, d in self.hist.items()},
            }

def _local_loop(node):
    while True:
        try:
            node.feed(sample_local())
        except Exception as e:
            print("stats: local sample failed:", e, flush=True)
        time.sleep(1)


def _remote_loop(node):
    code = SAMPLER + "\nimport sys\nwhile True:\n    print(json.dumps(sample()), flush=True)\n    time.sleep(1)\n"
    while True:
        try:
            b64 = base64.b64encode(code.encode()).decode()          # ship the sampler without any shell quoting issues
            p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5", NODE2,
                                  f"python3 -u -c \"import base64; exec(base64.b64decode('{b64}'))\""],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                node.feed(json.loads(line))
        except Exception as e:
            print("stats: node2 stream failed:", e, flush=True)
        node.online = False
        time.sleep(3)


# --------------------------------------------------------------------------- the AI server

TIMING_RE = re.compile(r"task (\d+) \|\s+(prompt eval|eval) time =\s+([\d.]+) ms /\s+(\d+) tokens")


class ServerLog:
    """follows llama-server's log: it prints exact prompt and generation timings for every request (by task id)"""

    def __init__(self):
        self.timings, self.lock = {}, threading.Lock()
        self.totals = {"requests": 0, "prompt_tokens": 0, "prompt_ms": 0.0, "gen_tokens": 0, "gen_ms": 0.0}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                p = subprocess.Popen(["journalctl", "--user", "-u", "llm-ep", "-u", "llm-server", "-f", "-n", "0", "-o", "cat"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    m = TIMING_RE.search(line)
                    if m:
                        task, kind, ms, n = int(m[1]), m[2], float(m[3]), int(m[4])
                        with self.lock:
                            t = self.timings.setdefault(task, {})
                            t["prompt" if kind == "prompt eval" else "gen"] = (n, ms)
                            if "prompt" in t and "gen" in t and not t.get("counted"):   # every request, exactly once
                                t["counted"] = True
                                tot = self.totals
                                tot["requests"] += 1
                                tot["prompt_tokens"] += t["prompt"][0]; tot["prompt_ms"] += t["prompt"][1]
                                tot["gen_tokens"] += t["gen"][0]; tot["gen_ms"] += t["gen"][1]
                            if len(self.timings) > 500:
                                for k in sorted(self.timings)[:100]:
                                    del self.timings[k]
            except Exception as e:
                print("stats: server log follower failed:", e, flush=True)
            time.sleep(3)

    def get(self, task, wait=2.0):
        t0 = time.time()
        while True:
            with self.lock:
                t = self.timings.get(task)
                if t and "prompt" in t and "gen" in t:
                    return dict(t)
            if time.time() - t0 > wait:
                return None
            time.sleep(0.1)


class AI:
    def __init__(self):
        self.lock = threading.Lock()
        self.up, self.phase, self.task, self.model = False, "offline", None, {}
        self.t_start = self.t_gen0 = 0.0
        self.prompt_done = self.prompt_total = self.cached = self.decoded = 0
        self.gen_pts = deque(); self.pre_rate = self.gen_rate = 0.0
        self.hist_gen, self.hist_pre = deque([0.0] * HIST, maxlen=HIST), deque([0.0] * HIST, maxlen=HIST)
        self.last, self.requests, self.n_ctx = None, 0, 0
        self.log = ServerLog()                          # exact per-request timings and totals

    def refresh_model(self):
        try:
            with urllib.request.urlopen(f"{LLM}/props", timeout=3) as r:
                p = json.load(r)
            f = Path(p.get("model_path", "")).name
            name, kind = friendly(f)
            split = "-tp-" in f or "tpa" in f
            self.model = {"up": True, "name": name, "kind": kind + (" · split over node1 + node2" if split else ""), "file": f,
                          "n_ctx": p.get("default_generation_settings", {}).get("n_ctx", 0)}
        except Exception:
            self.model = {"up": False, "name": "offline", "kind": "", "file": "", "n_ctx": 0}

    def poll(self):
        try:
            with urllib.request.urlopen(f"{LLM}/slots", timeout=2) as r:
                slots = json.load(r)
            # servers started with several slots (e.g. without -np 1): follow the busy one, else the last one used
            s = next((x for x in slots if x.get("is_processing")), None) or max(slots, key=lambda x: x.get("id_task") or -1)
        except Exception:
            with self.lock:
                self.up, self.phase = False, "offline"
            return
        now = time.time()
        nt = s.get("next_token") or {}
        nt = nt[0] if isinstance(nt, list) else nt
        busy, task = s.get("is_processing", False), s.get("id_task")
        decoded, processed = nt.get("n_decoded", 0), s.get("n_prompt_tokens_processed", 0) or 0
        cached = s.get("n_prompt_tokens_cache", 0) or 0
        with self.lock:
            self.up, self.n_ctx = True, s.get("n_ctx", 0)
            if busy and task != self.task:                              # a new request started
                self._finish(now)
                self.task, self.t_start, self.t_gen0 = task, now, 0.0
                self.gen_pts.clear(); self.pre_rate = self.gen_rate = 0.0
                self.prompt_done = self.decoded = 0
                self.requests += 1
            if busy:
                self.prompt_done = max(self.prompt_done, processed)
                self.cached = cached
                self.prompt_total = max((s.get("n_prompt_tokens", 0) or 0) - cached, self.prompt_done)
                if decoded == 0:
                    self.phase = "reading"
                    if self.prompt_done:
                        self.pre_rate = self.prompt_done / max(now - self.t_start, 1e-3)
                else:
                    if self.phase != "writing":
                        self.t_gen0 = now
                        if self.prompt_total:
                            self.pre_rate = self.prompt_total / max(now - self.t_start, 1e-3)
                    self.phase = "writing"
                    self.gen_pts.append((now, decoded))
                    while self.gen_pts and now - self.gen_pts[0][0] > 2.0:
                        self.gen_pts.popleft()
                    if len(self.gen_pts) >= 2 and self.gen_pts[-1][0] > self.gen_pts[0][0]:
                        self.gen_rate = (self.gen_pts[-1][1] - self.gen_pts[0][1]) / (self.gen_pts[-1][0] - self.gen_pts[0][0])
                self.decoded = decoded
            else:
                if self.task is not None:
                    self._finish(now)
                self.phase = "idle"

    def _finish(self, now):
        """request ended: exact numbers from llama-server's log for this task id, else our live estimate"""
        if self.task is None:
            return
        gen_s = (now - self.t_gen0) if self.t_gen0 else 0.0
        last = {"prompt_tokens": self.prompt_total, "prompt_tps": round(self.pre_rate, 1), "gen_tokens": self.decoded,
                "gen_tps": round(self.decoded / gen_s if gen_s > 0.5 else self.gen_rate, 1), "seconds": round(now - self.t_start, 1),
                "cached_tokens": self.cached, "exact": False, "ended": time.strftime("%H:%M:%S"), "task": self.task}
        self.last, self.task = last, None
        threading.Thread(target=self._exact, args=(last,), daemon=True).start()

    def _exact(self, last):
        t = self.log.get(last["task"], wait=3.0)
        if not t:
            return                                      # e.g. a stopped request: keep the live estimate
        (pn, pms), (gn, gms) = t["prompt"], t["gen"]
        with self.lock:
            last.update(prompt_tokens=pn, prompt_tps=round(pn / pms * 1000, 1) if pms else 0.0,
                        gen_tokens=gn, gen_tps=round(gn / gms * 1000, 1) if gms else 0.0, exact=True)

    def _totals(self):
        with self.log.lock:
            t = dict(self.log.totals)
        t["avg_prompt_tps"] = round(t["prompt_tokens"] / t["prompt_ms"] * 1000, 1) if t["prompt_ms"] else 0.0
        t["avg_gen_tps"] = round(t["gen_tokens"] / t["gen_ms"] * 1000, 1) if t["gen_ms"] else 0.0
        return t

    def tick_hist(self):
        with self.lock:
            self.hist_gen.append(self.gen_rate if self.phase == "writing" else 0.0)
            self.hist_pre.append(self.pre_rate if self.phase == "reading" else 0.0)

    def snapshot(self):
        with self.lock:
            return {"up": self.up, "phase": self.phase, "model": self.model, "n_ctx": self.n_ctx,
                    "prompt_done": self.prompt_done, "prompt_total": self.prompt_total, "cached_tokens": self.cached,
                    "prefill_tps": round(self.pre_rate, 1), "gen_tps": round(self.gen_rate, 1), "gen_tokens": self.decoded,
                    "elapsed_s": round(time.time() - self.t_start, 1) if self.phase in ("reading", "writing") else 0,
                    "last": dict(self.last) if self.last else None, "requests_seen": self.requests,
                    "totals": self._totals(), "hist": {"gen": [round(v, 1) for v in self.hist_gen], "prefill": [round(v, 1) for v in self.hist_pre]}}



def _ai_loop(ai):
    i = 0
    while True:
        if i % 20 == 0:
            ai.refresh_model()
        ai.poll()
        i += 1
        time.sleep(0.25)


# --------------------------------------------------------------------------- collector

class Collector:
    def __init__(self):
        self.nodes = [Node("node1", "leader · serves the AI"), Node("node2", "follower · computes its half")]
        self.ai = AI()
        for target, arg in ((_local_loop, self.nodes[0]), (_remote_loop, self.nodes[1]), (_ai_loop, self.ai), (self._hist_loop, None)):
            threading.Thread(target=target, args=(arg,) if arg is not None else (), daemon=True).start()

    def _hist_loop(self):
        while True:
            time.sleep(0.5)
            for n in self.nodes:
                n.tick_hist()
            self.ai.tick_hist()

    def snapshot(self):
        guard = ""
        try:
            guard = Path.home().joinpath("cluster/cooldown.status").read_text().strip()
        except OSError:
            pass
        return {"time": time.strftime("%H:%M:%S"), "ai": self.ai.snapshot(), "nodes": [n.snapshot() for n in self.nodes], "cooldown": guard}
