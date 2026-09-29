"""spec.py LABEL: an edit task (output mostly repeats the input) and a fresh question; tokens/s as the user sees it"""
import json, sys, time, urllib.request
code = open("/home/node1/cluster/agent/netgate.py").read()[:3500]
T = [("edit", f"Here is a Python file:\n```python\n{code}\n```\nRename the function `site_of` to `site_for_host` everywhere and output the complete updated file, nothing else."),
     ("fresh", "Explain how a hash map handles collisions, with a short Python example.")]
for name, q in T:
    b = {"messages": [{"role": "user", "content": q}], "max_tokens": 700, "temperature": 0}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://100.82.180.15:8080/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    tm = r["timings"]
    extra = f" | drafted {tm.get('draft_n', 0)} accepted {tm.get('draft_n_accepted', 0)}" if "draft_n" in tm else ""
    print(f"{sys.argv[1]:22} {name:6} {tm['predicted_n']:4} tok at {tm['predicted_per_second']:5.1f} t/s (prompt {tm['prompt_n']} at {tm['prompt_per_second']:.0f} t/s){extra}", flush=True)
