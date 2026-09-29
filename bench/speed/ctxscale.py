"""ctxscale.py LABEL: writing speed with ~0.1k, 2k, 4k, 8k tokens of prior text (greedy, 100 forced tokens)"""
import json, sys, urllib.request
src = open("/home/node1/cluster/agent/agent_server.py").read() + open("/home/node1/cluster/agent/index.html").read()
res = []
for chars in (300, 7000, 14000, 28000):
    p = src[:chars] + "\n\n# Summary of the code above:\n"
    b = {"prompt": p, "n_predict": 100, "ignore_eos": True, "temperature": 0, "cache_prompt": False}
    t = json.load(urllib.request.urlopen(urllib.request.Request("http://100.82.180.15:8080/completion", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))["timings"]
    res.append(f"{t['prompt_n']:5}ctx {t['predicted_per_second']:5.1f}t/s")
print(f"{sys.argv[1]:32}", " | ".join(res), flush=True)
