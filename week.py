from typing import List
import bisect

MAX = 99999999999


def geometric_mean(nums: List[int], L: int):
    n = len(nums)
    muls = [1] * (n + 1)
    muls[0] = 1
    ans = MAX
    ans_len = MAX
    for i in range(1, n + 1):
        muls[i] = nums[i] * muls[i - 1]
    for i in range(n - L + 1):
        for j in range(i + L - 1, i - 1, -1):
            mul = muls[j] / muls[i]
            l = j - l + 1
            gm = mul ** (1 / l)
            if gm > ans:
                ans = gm
                ans_len = l
            elif gm == ans and l < ans_len:
                ans_len = l
    return ans


def triangle(nums: List[int]):
    n = len(nums)
    nums.sort()
    ans = 0
    for i in range(n):
        si = i * i
        for x in range(i + 2, n):
            another = (x * x - si) ** (1 / 2)
            pos = bisect.bisect_left(nums, another)
            while pos >= 0 and pos < n and nums[pos] == another:
                ans += 1
                pos += 1
    return ans


print(triangle([7, 3, 4, 5, 6, 5, 12, 13]))
