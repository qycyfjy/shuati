from typing import List, Optional, Self, Mapping, Tuple
from itertools import permutations, accumulate
from functools import reduce, cache
from math import inf
from collections import deque, defaultdict


def minimum_string(s: str):
    ans = list(s)
    ls = list(s)
    ls.sort()

    for i, c in enumerate(ls):
        if s[i] != c:
            idx = s.rfind(c)
            ans[idx] = s[i]
            ans[i] = c
            break
    return ''.join(ans)

print(minimum_string('abc'))
print(minimum_string('bca'))
print(minimum_string('abbaca'))