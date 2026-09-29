"""gpucheck.py OUT: greedy continuations of a ~6k-token prompt + top-5 next-token probabilities (GPU on vs off)"""
import json, sys, urllib.request
src = open("/home/node1/cluster/agent/agent_server.py").read()[:22000]
res = []
for tail in ("\n\n# Summary of the code above:\n", "\n\n# The most important function above is"):
    b = {"prompt": src + tail, "n_predict": 60, "temperature": 0, "cache_prompt": False, "n_probs": 5}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://100.82.180.15:8080/completion", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
    res.append({"text": r["content"], "top": [(t["token"], round(t["logprob"], 3)) for t in r["completion_probabilities"][0]["top_logprobs"]]})
json.dump(res, open(sys.argv[1], "w")); print(json.dumps(res)[:600])
