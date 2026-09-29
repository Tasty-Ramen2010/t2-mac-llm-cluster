import json, sys, urllib.request
P = ["The three laws of thermodynamics are", "# Python: read a CSV file and print the average of column 'price'\n", "Q: What is the capital of Australia and why was it chosen?\nA:"]
out = []
for p in P:
    b = {"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 5, "cache_prompt": False}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://100.82.180.15:8080/completion", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=300))
    top = r["completion_probabilities"][0]["top_logprobs"]
    out.append([(t["token"], round(t["logprob"], 4)) for t in top])
json.dump(out, open(sys.argv[1], "w")); print(out)
