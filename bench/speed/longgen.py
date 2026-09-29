"""longgen.py LABEL [CHARS]: read a long prompt once, then time 3 x 100 tokens reusing its cache"""
import json, sys, urllib.request
chars = int(sys.argv[2]) if len(sys.argv) > 2 else 28000
src = (open("/home/node1/cluster/agent/agent_server.py").read() + open("/home/node1/cluster/agent/index.html").read())[:chars]
def go(n):
    b = {"prompt": src + "\n\n# Summary of the code above:\n", "n_predict": n, "ignore_eos": True, "temperature": 0, "cache_prompt": True}
    return json.load(urllib.request.urlopen(urllib.request.Request("http://100.82.180.15:8080/completion", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))["timings"]
go(1)
r = [go(100) for _ in range(3)]
print(f"{sys.argv[1]:34} ctx {r[0]['prompt_n'] + r[0].get('cache_n', 0):5}  " + "  ".join(f"{x['predicted_per_second']:.2f}" for x in r) + f"  avg {sum(x['predicted_per_second'] for x in r) / 3:.2f} t/s", flush=True)
