import json, subprocess, sys, time, urllib.request
SRV = "/home/node1/llama.cpp/build/bin/llama-server"
M = "/home/node1/models/"
RPC = ["--rpc", "127.0.0.1:50051,100.109.16.15:50052", "-ngl", "999", "-t", "4"]
PROMPTS = ["Write a Python function that checks whether a number is prime, with a docstring and comments.",
           "Explain in a short paragraph how vaccines train the immune system."]
def ask(p, tokens=200):
    body = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": tokens, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
    return r["timings"]
def run(name, args):
    srv = subprocess.Popen([SRV, "--host", "127.0.0.1", "--port", "8091", "-c", "2048", *RPC, *args], stdout=subprocess.DEVNULL, stderr=open(f"/tmp/spec_{name[:12].replace(' ','_')}.err", "w"))
    try:
        for _ in range(600):
            try:
                if b"ok" in urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2).read(): break
            except Exception:
                if srv.poll() is not None: print(f"{name:48s} server exited ({srv.returncode})", flush=True); return
                time.sleep(1)
        ask("hi", 8)
        res = [ask(p) for p in PROMPTS]
        tps = sum(t["predicted_per_second"] for t in res) / len(res)
        dn = sum(t.get("draft_n", 0) for t in res); da = sum(t.get("draft_n_accepted", 0) for t in res)
        acc = f"{100*da/dn:.0f}% of {dn} drafted accepted" if dn else "-"
        print(f"{name:48s} {tps:6.2f} words/s   {acc}", flush=True)
    finally:
        srv.terminate(); srv.wait()
G = ["-m", M + "gemma-3-12b-it-Q4_K_M.gguf"]
GD = ["-md", M + "gemma-3-1b-it-Q4_K_M.gguf", "--spec-type", "draft-simple", "-ngld", "0", "-td", "4"]
O = ["-m", M + "gpt-oss-20b-MXFP4.gguf"]
OE = ["-md", M + "eagle3-gpt-oss-20b-Q8_0.gguf", "--spec-type", "draft-eagle3", "-ngld", "0", "-td", "4"]
tests = {
 "g0": ("Gemma 12B alone (2 Macs)", G),
 "g3": ("Gemma 12B + Gemma 1B draft, 3 guesses", G + GD + ["--spec-draft-n-max", "3"]),
 "g5": ("Gemma 12B + Gemma 1B draft, 5 guesses", G + GD + ["--spec-draft-n-max", "5"]),
 "o0": ("gpt-oss-20B alone (2 Macs)", O),
 "o3": ("gpt-oss-20B + EAGLE3, 3 guesses", O + OE + ["--spec-draft-n-max", "3"]),
}
for k in (sys.argv[1:] or tests): run(*tests[k])
print("DONE", flush=True)
