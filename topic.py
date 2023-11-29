from typing import List, Dict, Optional
from collections import deque, defaultdict
import bisect


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    # 最短路floyd https://leetcode.cn/problems/network-delay-time/
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        MAX = 99999999
        G = [[MAX] * n for _ in range(n)]
        for u, v, w in times:
            G[u - 1][v - 1] = w
        for i in range(n):
            G[i][i] = 0
        for m in range(n):
            for i in range(n):
                for j in range(n):
                    G[i][j] = min(G[i][j], G[i][m] + G[m][j])
        return max(G[k - 1]) if all(d != MAX for d in G[k - 1]) else -1

    # dp https://leetcode.cn/problems/house-robber-ii/
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        def rob_partial(rooms: List[int]):
            a, b = 0, 0
            for room in rooms:
                a, b = b, max(a + room, b)
            return b

        return max(rob_partial(nums[:-1]), rob_partial(nums[1:]))

    # 滑动窗口 https://leetcode.cn/problems/longest-substring-without-repeating-characters/
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        longest = 0
        l, r = 0, 0
        chrs = set()
        while r < n:
            while s[r] in chrs:
                chrs.remove(s[l])
                l += 1
            chrs.add(s[r])
            longest = max(longest, len(chrs))
            r += 1

        return longest

    # 双指针 https://leetcode.cn/problems/volume-of-histogram-lcci/
    def trap(self, height: List[int]) -> int:
        n = len(height)
        l, r = 0, n - 1
        level = 1
        total = 0
        while l <= r:
            while l <= r and height[l] < level:
                l += 1
            while l <= r and height[r] < level:
                r -= 1
            total += r - l + 1
            level += 1
        return total - sum(height)

    # 二分 https://leetcode.cn/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + r >> 1
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target and target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target and target <= nums[-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return l if nums[l] == target else -1

    # 拓扑排序 https://leetcode.cn/problems/course-schedule/
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegrees = [0] * numCourses
        G: defaultdict[int, List[int]] = defaultdict(list)
        for a, b in prerequisites:
            G[b].append(a)
            indegrees[a] += 1

        q = deque()
        completed = 0
        for i in range(numCourses):
            if indegrees[i] == 0:
                q.append(i)
                completed += 1
        while len(q) != 0:
            f = q.popleft()
            for post in G[f]:
                indegrees[post] -= 1
                if indegrees[post] == 0:
                    completed += 1
                    q.append(post)

        return completed == numCourses

    # bfs https://leetcode.cn/problems/word-ladder/
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wl = set(wordList)
        words = deque()
        words.append(beginWord)
        level = 1
        while len(words) != 0:
            n = len(words)
            for _ in range(n):
                w = words.popleft()
                if w == endWord:
                    return level
                for i in range(len(w)):
                    for k in range(26):
                        nw = w[:i] + chr(ord("a") + k) + w[i + 1 :]
                        if nw not in wl:
                            continue
                        words.append(nw)
                        wl.remove(nw)
            level += 1
        return 0

    # dfs https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        total = sum(nums)
        target = total / k
        if target != int(target):
            return False

        nums.sort(reverse=True)
        sums = [0] * k

        def dfs(i):
            if i == n:
                return True
            for g in range(k):
                if g > 0 and sums[g] == sums[g - 1]:
                    continue
                if sums[g] + nums[i] > target:
                    continue
                sums[g] += nums[i]
                if dfs(i + 1):
                    return True
                sums[g] -= nums[i]
            return False

        return dfs(0)


class FrontMiddleBackQueue:
    def __init__(self):
        self.left: deque[int] = deque()
        self.right: deque[int] = deque()

    def pushFront(self, val: int) -> None:
        self.left.appendleft(val)
        self.balance()

    def pushMiddle(self, val: int) -> None:
        m = len(self.left)
        n = len(self.right)
        if m > n:
            self.right.appendleft(self.left.pop())
            self.left.append(val)
            return
        self.left.append(val)

    def pushBack(self, val: int) -> None:
        self.right.append(val)
        self.balance()

    def popFront(self) -> int:
        if len(self.left) == 0:
            return -1
        n = self.left.popleft()
        self.balance()
        return n

    def popMiddle(self) -> int:
        if len(self.left) == 0:
            return -1
        n = self.left.pop()
        self.balance()
        return n

    def popBack(self) -> int:
        if len(self.left) != 0:
            if len(self.right) != 0:
                n = self.right.pop()
                self.balance()
                return n
            return self.left.pop()
        return -1

    def balance(self):
        m = len(self.left)
        n = len(self.right)
        if m == n or m == n + 1:
            return
        if m < n:
            self.left.append(self.right.popleft())
        else:
            self.right.appendleft(self.left.pop())


class SmallestInfiniteSet:
    def __init__(self):
        self.not_exists = set()

    def popSmallest(self) -> int:
        i = 1
        while True:
            if i not in self.not_exists:
                self.not_exists.add(i)
                return i
            i += 1

    def addBack(self, num: int) -> None:
        if num in self.not_exists:
            self.not_exists.remove(num)


s = Solution()
s.networkDelayTime([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2)
