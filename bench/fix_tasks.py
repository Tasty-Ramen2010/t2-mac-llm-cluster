"""Bug-fix tasks for the agentic part of the coding benchmark.

Each task: a description, a buggy program, and hidden tests (asserts) the fixed code must pass. The model sees the
description, the code and one failing example; it can run code with a Python tool, and must answer with the fixed code.
"""

TASKS = [
    dict(name="binary_search",
         desc="`find(xs, target)` should return the index of target in the sorted list xs, or -1 if it is absent.",
         code='''def find(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid
        else:
            hi = mid
    return -1
''',
         example="find([1, 3, 5, 7], 7) never returns (hangs).",
         tests='''assert find([1, 3, 5, 7], 7) == 3
assert find([1, 3, 5, 7], 1) == 0
assert find([1, 3, 5, 7], 4) == -1
assert find([], 3) == -1
assert find([2], 2) == 0
assert all(find(list(range(0, 100, 2)), v) == v // 2 for v in range(0, 100, 2))
assert all(find(list(range(0, 100, 2)), v) == -1 for v in range(1, 100, 2))
'''),
    dict(name="median",
         desc="`median(xs)` returns the median of a non-empty list of numbers (average of the two middle values for even length). The input list must not be modified.",
         code='''def median(xs):
    xs.sort()
    n = len(xs)
    return xs[n // 2]
''',
         example="median([4, 1, 3, 2]) returns 3, expected 2.5; it also reorders the caller's list.",
         tests='''assert median([4, 1, 3, 2]) == 2.5
assert median([5]) == 5
assert median([3, 1, 2]) == 2
a = [9, 1, 5]; median(a); assert a == [9, 1, 5]
assert median([1.5, 2.5]) == 2.0
'''),
    dict(name="word_count",
         desc="`word_count(text)` returns a dict of how often each word occurs, case-insensitive, ignoring punctuation (.,!?;:) around words.",
         code='''def word_count(text):
    counts = {}
    for w in text.split(" "):
        counts[w] = counts.get(w, 0) + 1
    return counts
''',
         example='word_count("Hi hi, HI!") returns {"Hi": 1, "hi,": 1, "HI!": 1}, expected {"hi": 3}.',
         tests='''assert word_count("Hi hi, HI!") == {"hi": 3}
assert word_count("a b  a") == {"a": 2, "b": 1}
assert word_count("") == {}
assert word_count("Well; well. WELL?  done:") == {"well": 3, "done": 1}
'''),
    dict(name="transpose",
         desc="`transpose(m)` returns the transpose of a rectangular matrix given as a list of rows (it may be non-square).",
         code='''def transpose(m):
    n = len(m)
    return [[m[j][i] for j in range(n)] for i in range(n)]
''',
         example="transpose([[1, 2, 3], [4, 5, 6]]) returns [[1, 4], [2, 5]], expected [[1, 4], [2, 5], [3, 6]].",
         tests='''assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert transpose([[1], [2], [3]]) == [[1, 2, 3]]
assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
assert transpose([]) == []
'''),
    dict(name="roman",
         desc="`to_int(s)` converts a Roman numeral (I, V, X, L, C, D, M, with subtractive forms like IV, IX, XL, XC, CD, CM) to an integer.",
         code='''def to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(vals[c] for c in s)
''',
         example='to_int("IV") returns 6, expected 4.',
         tests='''assert to_int("IV") == 4
assert to_int("IX") == 9
assert to_int("MCMXCIV") == 1994
assert to_int("LVIII") == 58
assert to_int("MMXXVI") == 2026
assert to_int("XL") == 40
'''),
    dict(name="merge_intervals",
         desc="`merge(intervals)` merges overlapping [start, end] intervals (touching ones like [1,2],[2,3] also merge) and returns them sorted by start.",
         code='''def merge(intervals):
    out = []
    for s, e in intervals:
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
         example="merge([[5, 6], [1, 3], [2, 4]]) returns [[5, 6], [1, 3], [2, 4]], expected [[1, 4], [5, 6]].",
         tests='''assert merge([[5, 6], [1, 3], [2, 4]]) == [[1, 4], [5, 6]]
assert merge([[1, 2], [2, 3]]) == [[1, 3]]
assert merge([]) == []
assert merge([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
assert merge([[3, 4], [1, 2]]) == [[1, 2], [3, 4]]
'''),
    dict(name="flatten",
         desc="`flatten(x)` flattens arbitrarily nested lists and tuples into one flat list. Strings are single items (not split into characters).",
         code='''def flatten(x):
    out = []
    for item in x:
        if hasattr(item, "__iter__"):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
''',
         example='flatten(["ab", [1]]) crashes with RecursionError, expected ["ab", 1].',
         tests='''assert flatten(["ab", [1]]) == ["ab", 1]
assert flatten([1, [2, (3, [4])], 5]) == [1, 2, 3, 4, 5]
assert flatten([]) == []
assert flatten([[[]], "x", ("y", ["zz"])]) == ["x", "y", "zz"]
'''),
    dict(name="rle",
         desc="`rle(s)` run-length encodes a string: 'aaabcc' -> 'a3b1c2'. Empty string gives ''.",
         code='''def rle(s):
    out = ""
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            out += s[i - 1] + str(count)
            count = 1
    return out
''',
         example="rle('aaabcc') returns 'a3b1', expected 'a3b1c2'.",
         tests='''assert rle("aaabcc") == "a3b1c2"
assert rle("") == ""
assert rle("a") == "a1"
assert rle("abc") == "a1b1c1"
assert rle("zzzzzzzzzzzz") == "z12"
'''),
    dict(name="moving_average",
         desc="`moving_average(xs, k)` returns the list of averages of each window of k consecutive values (len(xs)-k+1 values; [] if k > len(xs)).",
         code='''def moving_average(xs, k):
    out = []
    total = sum(xs[:k])
    for i in range(k, len(xs)):
        out.append(total / k)
        total += xs[i] - xs[i - k]
    return out
''',
         example="moving_average([1, 2, 3, 4], 2) returns [1.5, 2.5], expected [1.5, 2.5, 3.5].",
         tests='''assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
assert moving_average([5], 1) == [5.0]
assert moving_average([1, 2], 3) == []
assert moving_average([2, 4, 6], 3) == [4.0]
'''),
    dict(name="parse_duration",
         desc="`parse_duration(s)` converts strings like '1h30m', '45s', '2h5s', '10m' to a total number of seconds.",
         code='''import re

def parse_duration(s):
    total = 0
    for num, unit in re.findall(r"(\\d)([hms])", s):
        total += int(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total
''',
         example="parse_duration('10m') returns 0, expected 600.",
         tests='''assert parse_duration("10m") == 600
assert parse_duration("1h30m") == 5400
assert parse_duration("45s") == 45
assert parse_duration("2h5s") == 7205
assert parse_duration("100h") == 360000
'''),
]
