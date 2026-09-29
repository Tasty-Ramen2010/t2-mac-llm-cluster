import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for x, y in zip(a, b):
    n = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
    print(f"identical for {n:4d} of {max(len(x), len(y)):4d} chars" + ("  (fully identical)" if x == y else f"  | fp16: {x[n:n+40]!r} | q8: {y[n:n+40]!r}"))
