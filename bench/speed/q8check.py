"""q8check.py OUT.json : greedy completions for fixed prompts (to compare fp16 vs Q8 sync)"""
import json, sys, urllib.request
P = ["def quicksort(arr):", "The three laws of thermodynamics are", "# Python: read a CSV file and print the average of column 'price'\n",
     "Q: What is the capital of Australia and why was it chosen?\nA:", "class LinkedList:\n    def __init__(self):"]
out = []
for p in P:
    b = {"prompt": p, "n_predict": 150, "temperature": 0, "cache_prompt": False}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://100.82.180.15:8080/completion", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600))
    out.append(r["content"])
json.dump(out, open(sys.argv[1], "w"))
