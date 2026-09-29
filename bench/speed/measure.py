"""measure.py MODEL STAGE [NOTE]: wait for llama-server, then time a fixed ~600-token prompt + 200 forced words
(greedy, no prompt cache), twice; append the result to results.jsonl."""
import json, os, sys, time, urllib.request
URL = "http://100.82.180.15:8080"
PROMPT = open("/home/node1/cluster/README.md").read()[3000:5600] + "\n\nSummarize the notes above in a few paragraphs:\n"


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    model, stage, note = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
    t0 = time.time()
    while True:
        try:
            if json.load(urllib.request.urlopen(URL + "/health", timeout=5)).get("status") == "ok":
                break
        except Exception:
            pass
        if time.time() - t0 > 900:
            print(f"{model} {stage}: server never came up"); return
        time.sleep(3)
    load_s = time.time() - t0
    post("/completion", {"prompt": "Hello", "n_predict": 8, "cache_prompt": False})          # warm-up
    runs = []
    for _ in range(int(os.environ.get("REPS", "2"))):
        t = post("/completion", {"prompt": PROMPT, "n_predict": 200, "ignore_eos": True, "temperature": 0,
                                 "cache_prompt": False})["timings"]
        runs.append((t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]))
    r = {"model": model, "stage": stage, "note": note, "time": time.strftime("%H:%M:%S"), "load_s": round(load_s),
         "prompt_tokens": runs[0][0], "prompt_tps": round(sum(x[1] for x in runs) / len(runs), 2),
         "gen_tps": round(sum(x[3] for x in runs) / len(runs), 2), "gen_runs": [round(x[3], 2) for x in runs]}
    open("/home/node1/cluster/bench/speed/results.jsonl", "a").write(json.dumps(r) + "\n")
    print(f"{model:10} {stage:28} prompt {r['prompt_tps']:6.1f} t/s | writing {r['gen_tps']:5.2f} t/s {r['gen_runs']}", flush=True)


main()
