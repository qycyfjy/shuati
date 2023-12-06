from typing import List, Optional, Self, Mapping, Tuple
from itertools import permutations, accumulate
from functools import reduce, cache
from math import inf
from collections import deque, defaultdict
import heapq
import bisect


from sortedcontainers import SortedDict


class Node:
    def __init__(
        self,
        val: int = 0,
        left: Optional[Self] = None,
        right: Optional[Self] = None,
        next: Optional[Self] = None,
    ):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution:
    def test(self):
        assert self.minimumTotalPrice(
            4, [[0, 1], [1, 2], [1, 3]], [2, 2, 10, 6], [[0, 3], [2, 1], [2, 3]]
        )

    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        G = [[] for _ in range(n)]
        for a, b in connections:
            G[a].append((b, True))
            G[b].append((a, False))
        ans = 0
        def dfs(cur, prev):
            nonlocal ans
            for nxt, dirTo in G[cur]:
                if nxt != prev:
                    if dirTo:
                        ans += 1
                    dfs(nxt, cur)
        dfs(0, -1)

    def minimumTotalPrice(
        self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]
    ) -> int:
        G = [[] for _ in range(n)]
        for a, b in edges:
            G[a].append(b)
            G[b].append(a)

        visitCount = [0] * n

        def dfs(cur, prev, to):
            if cur == to:
                visitCount[cur] += 1
                return True
            for neighbor in G[cur]:
                if neighbor != prev:
                    if dfs(neighbor, cur, to):
                        visitCount[cur] += 1
                        return True
            return False

        for trip_start, trip_end in trips:
            dfs(trip_start, -1, trip_end)

        def dfs_rob(cur, prev):
            no = price[cur] * visitCount[cur]
            yes = no // 2
            for neighbor in G[cur]:
                if neighbor != prev:
                    cno, cyes = dfs_rob(neighbor, cur)
                    no += min(cno, cyes)
                    yes += cno
            return (no, yes)

        return min(dfs_rob(0, -1))

    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        n = len(roads) + 1
        G = [[] for _ in range(n)]
        for a, b in roads:
            G[a].append(b)
            G[b].append(a)
        ans = 0

        def dfs(init, prev):
            nonlocal ans
            p = 1
            for neighbor in G[init]:
                if neighbor != prev:
                    np = dfs(neighbor, init)
                    p += np
                    ans += (np + seats - 1) // seats
            return p

        dfs(0, -1)
        return ans

    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        presum = [0] * (n + 1)
        for i in range(n):
            presum[i + 1] = presum[i] + cardPoints[i]
        rest = presum[n] + 1
        for i in range(k + 1):
            rest = min(rest, presum[n - k + i] - presum[i])
        return presum[n] - rest

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        trips.sort(key=lambda t: t[1])
        pq = []
        passengers = 0
        for n, f, t in trips:
            while len(pq) != 0 and pq[0][0] <= f:
                passengers -= pq[0][1]
                heapq.heappop(pq)
            if passengers + n > capacity:
                return False
            passengers += n
            heapq.heappush(pq, (t, n))
        return True

    def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
        m = len(mat)
        n = len(mat[0])

        pos = {}
        for i in range(m):
            for j in range(n):
                pos[mat[i][j]] = (i, j)

        row_count = [0] * m
        col_count = [0] * n
        for i, num in enumerate(arr):
            x, y = pos[num]
            row_count[x] += 1
            col_count[y] += 1
            if row_count[x] == n or col_count[y] == m:
                return i

        return 0

    def minDeletion(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        normal = True
        for i in range(n - 1):
            if normal and i % 2 == 0:
                if nums[i] == nums[i + 1]:
                    ans += 1
                    normal = False
            elif not normal and i % 2 == 1:
                if nums[i] == nums[i + 1]:
                    ans += 1
                    normal = True
        if (n - ans) % 2 != 0:
            ans += 1
        return ans

    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)

    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        ans = []
        n = len(nums)
        if n < 3 * k:
            return ans
        sub1_sum, sub1_max, sub1_idx = 0, 0, 0
        sub2_sum, sub12_max, sub12_idx = 0, 0, ()
        sub3_sum, sub123_max = 0, 0
        for i in range(k * 2, n):
            sub1_sum += nums[i - k * 2]
            sub2_sum += nums[i - k]
            sub3_sum += nums[i]
            if i >= k * 3 - 1:
                if sub1_sum > sub1_max:
                    sub1_max = sub1_sum
                    sub1_idx = i - k * 3 + 1
                if sub1_max + sub2_sum > sub12_max:
                    sub12_max = sub1_max + sub2_sum
                    sub12_idx = (sub1_idx, i - k * 2 + 1)
                if sub12_max + sub3_sum > sub123_max:
                    sub123_max = sub12_max + sub3_sum
                    ans = [*sub12_idx, i - k + 1]
                sub1_sum -= nums[i - k * 3 + 1]
                sub2_sum -= nums[i - k * 2 + 1]
                sub3_sum -= nums[i - k + 1]

        return ans

    # https://leetcode.cn/problems/max-sum-of-a-pair-with-equal-sum-of-digits
    def maximumSum(self, nums: List[int]) -> int:
        cache = {}
        ans = -1
        for num in nums:
            digit_sum = 0
            num_c = num
            while num_c != 0:
                digit_sum += num_c % 10
                num_c //= 10
            if digit_sum not in cache:
                cache[digit_sum] = num
            else:
                ans = max(ans, num + cache[digit_sum])
                cache[digit_sum] = max(num, cache[digit_sum])
        return ans

    # https://leetcode.cn/problems/couples-holding-hands
    def minSwapsCouples(self, row: List[int]) -> int:
        n = len(row) // 2
        uf = list(range(n))
        rank = [1] * n
        gs = n

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa == pb:
                return
            if rank[pa] == rank[pb]:
                uf[pa] = pb
                rank[pb] += 1
            elif rank[pa] < rank[pb]:
                uf[pa] = pb
            else:
                uf[pb] = pa
            nonlocal gs
            gs -= 1

        def find(x):
            if uf[x] != x:
                uf[x] = find(uf[x])
            return uf[x]

        for i in range(0, 2 * n, 2):
            union(row[i] // 2, row[i + 1] // 2)

        return n - gs

    # https://leetcode.cn/problems/successful-pairs-of-spells-and-potions
    def successfulPairs(
        self, spells: List[int], potions: List[int], success: int
    ) -> List[int]:
        potions.sort()
        n = len(spells)
        m = len(potions)
        ans = [0] * n
        for i, spell in enumerate(spells):
            need = (success - 1) // spell
            bound = bisect.bisect_right(potions, need)
            ans[i] = m - bound
        return ans

    # https://leetcode.cn/problems/smallest-string-with-swaps
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        uf = list(range(n))
        rank = [1] * n

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa == pb:
                return
            if rank[pa] == rank[pb]:
                uf[pa] = pb
                rank[pb] += 1
            elif rank[pa] < rank[pb]:
                uf[pa] = pb
            else:
                uf[pb] = pa

        def find(x):
            if uf[x] != x:
                uf[x] = find(uf[x])
            return uf[x]

        for a, b in pairs:
            union(a, b)

        groups: Mapping[int, List[str]] = defaultdict(list)
        for i in range(n):
            heapq.heappush(groups[find(i)], s[i])

        ans = [""] * n
        for i in range(n):
            k = find(i)
            ans[i] = heapq.heappop(groups[k])

        return "".join(ans)

    # https://leetcode.cn/problems/escape-the-spreading-fire
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        UPPER = 100000000
        r = len(grid)
        c = len(grid[0])
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        will_fire = [[UPPER] * c for _ in range(r)]

        def bfs():
            fires = deque()
            for i in range(r):
                for j in range(c):
                    if grid[i][j] == 1:
                        fires.append((i, j))
                        will_fire[i][j] = 0
            minutes = 1
            while len(fires) != 0:
                n = len(fires)
                for i in range(n):
                    x, y = fires.popleft()
                    for dirx, diry in dirs:
                        nx, ny = x + dirx, y + diry
                        if nx >= 0 and nx < r and ny >= 0 and ny < c:
                            if grid[nx][ny] == 2 or will_fire[nx][ny] != UPPER:
                                continue
                            will_fire[nx][ny] = minutes
                            fires.append((nx, ny))
                minutes += 1

        bfs()

        def check(t):
            visited = [[False] * c for _ in range(r)]
            visited[0][0] = True
            q = deque()
            q.append((0, 0, t))
            while len(q) != 0:
                x, y, time = q.popleft()
                for dir in dirs:
                    nx, ny = x + dir[0], y + dir[1]
                    if nx >= 0 and nx < r and ny >= 0 and ny < c:
                        if visited[nx][ny] or grid[nx][ny] == 2:
                            continue
                        if nx == r - 1 and ny == c - 1:
                            return will_fire[nx][ny] >= time + 1
                        if will_fire[nx][ny] > time + 1:
                            visited[nx][ny] = True
                            q.append((nx, ny, time + 1))
            return False

        ans = -1
        l, h = 0, r * c
        while l <= h:
            m = l + ((h - l) >> 1)
            if check(m):
                ans = m
                l = m + 1
            else:
                h = m - 1
        return 1000000000 if ans >= r * c else ans

    # https://leetcode.cn/problems/find-the-longest-balanced-substring-of-a-binary-string
    def findTheLongestBalancedSubstring(self, s: str) -> int:
        ans = 0
        zs, os = 0, 0
        for c in s:
            if c == "1":
                os += 1
                ans = max(ans, min(zs, os) * 2)
            else:
                if os != 0:
                    os = 0
                    zs = 0
                zs += 1
        return ans

    # https://leetcode.cn/problems/maximum-product-of-word-lengths/description/
    def maxProduct(self, words: List[str]) -> int:
        mapping: dict[int, int] = {}
        for word in words:
            mask = 0
            for c in word:
                mask |= 1 << (ord(c) - ord("a"))
            if mask not in mapping or len(word) > mapping[mask]:
                mapping[mask] = len(word)
        ans = 0
        for k, v in mapping.items():
            for k1, v1 in mapping.items():
                if k & k1 == 0:
                    ans = max(ans, v * v1)
        return ans

    # https://leetcode.cn/problems/maximum-balanced-subsequence-sum/
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = nums.copy()
        for i in range(1, n):
            for j in range(0, i):
                if nums[j] - j <= nums[i] - i:
                    dp[i] = max(dp[i], max(0, dp[j]) + nums[i])
        return max(dp)

    # https://leetcode.cn/problems/maximum-score-after-applying-operations-on-a-tree/
    def maximumScoreAfterOperations(
        self, edges: List[List[int]], values: List[int]
    ) -> int:
        n = len(values)
        G = [[] for _ in range(n)]
        G[0].append(-1)
        for edge in edges:
            G[edge[0]].append(edge[1])
            G[edge[1]].append(edge[0])

        def dfs(root, parent):
            if len(G[root]) == 1:
                return values[root]
            loss = 0
            for c in G[root]:
                if c != parent:
                    loss += dfs(c, root)
            return min(values[root], loss)

        return sum(values) - dfs(0, -1)

    # https://leetcode.cn/problems/find-champion-ii
    def findChampion(self, n: int, edges: List[List[int]]) -> int:
        loser = set()
        for edge in edges:
            loser.add(edge[1])
        if len(loser) != n - 1:
            return -1
        for i in range(n):
            if i not in loser:
                return i
        return -1

    # https://leetcode.cn/problems/find-champion-i
    def findChampionEasy(self, grid: List[List[int]]) -> int:
        n = len(grid)
        loser = set()
        for i in range(n):
            for j in range(n):
                if i != j and grid[i][j] == 1:
                    loser.add(j)
        for i in range(n):
            if i not in loser:
                return i
        return -1

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        mapping = {"A": 0, "G": 1, "C": 2, "T": 3}
        counter = defaultdict(int)
        n = len(s)
        if n < 10:
            return []
        ans = []
        V = 0
        for c in s[:9]:
            V = (V << 2) | mapping[c]
        for r in range(9, n):
            V = ((V << 2) | mapping[s[r]]) & ((1 << 20) - 1)
            counter[V] += 1
            if counter[V] == 2:
                ans.append(s[r - 9 : r + 1])
        return ans

    def findMaximumXOR(self, nums: List[int]) -> int:
        MAX_BIT = 30
        ans = 0
        mask = 0
        for i in range(MAX_BIT, -1, -1):
            mask = mask | (1 << i)
            prefixs = set()
            for num in nums:
                prefixs.add(num & mask)
            try_ans = ans | (1 << i)
            for prefix in prefixs:
                if prefix ^ try_ans in prefixs:
                    ans = try_ans
                    break
        return ans

    def connect(self, root: Node) -> Node:
        if root is None:
            return root
        q: deque[Node] = deque()
        q.append(root)
        cur_level = 1
        next_level = 0
        while len(q) != 0:
            front = q.popleft()
            cur_level -= 1
            if front.left is not None:
                q.append(front.left)
                next_level += 1
            if front.right is not None:
                q.append(front.right)
                next_level += 1
            if cur_level != 0:
                right = q[0]
                front.next = right
            else:
                cur_level = next_level
                next_level = 0

        return root

    def countPoints(self, rings: str) -> int:
        cnt = [0] * 10
        pos = 0
        for c in rings:
            if c == "R":
                pos = 0
            elif c == "G":
                pos = 1
            elif c == "B":
                pos = 2
            else:
                cnt[ord(c) - ord("0")] |= 1 << pos
        ans = 0
        for c in cnt:
            if c == 7:
                ans += 1
        return ans

    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        indeg = [0] * n
        for fav in favorite:
            indeg[fav] += 1

        visited = [False] * n
        dp = [1] * n
        q: deque[int] = deque()
        for i in range(n):
            if indeg[i] == 0:
                q.append(i)
        while len(q) != 0:
            front = q.popleft()
            visited[front] = True
            fav = favorite[front]
            dp[fav] = max(dp[fav], dp[front] + 1)
            indeg[fav] -= 1
            if indeg[fav] == 0:
                q.append(fav)

        ring = 0
        chain = 0
        for i in range(n):
            if not visited[i]:
                fav = favorite[i]
                if favorite[fav] == i:
                    chain += dp[i] + dp[fav]
                    visited[i], visited[fav] = True, True
                else:
                    node, cnt = i, 0
                    while True:
                        cnt += 1
                        node = favorite[node]
                        visited[node] = True
                        if node == i:
                            break
                    ring = max(ring, cnt)

        return max(ring, chain)

    def smallestMissingValueSubtree(
        self, parents: List[int], nums: List[int]
    ) -> List[int]:
        n = len(parents)
        G = [[] for _ in range(n)]
        for i in range(1, n):
            G[parents[i]].append(i)

        ans = [1] * n
        genes = [set() for _ in range(n)]

        def dfs(node):
            genes[node].add(nums[node])
            for child in G[node]:
                ans[node] = max(ans[node], dfs(child))
                if len(genes[node]) < len(genes[child]):
                    genes[node], genes[child] = genes[child], genes[node]
                genes[node].update(genes[child])
            while ans[node] in genes[node]:
                ans[node] += 1
            return ans[node]

        dfs(0)
        return ans

    # https://leetcode.cn/problems/maximum-points-after-collecting-coins-from-all-nodes/
    def maximumPoints(self, edges: List[List[int]], coins: List[int], k: int) -> int:
        n = len(coins)
        G = [[] for _ in range(n)]
        for edge in edges:
            G[edge[0]].append(edge[1])
            G[edge[1]].append(edge[0])

        @cache
        def dfs(i, shift, parent):
            res1 = (coins[i] >> shift) - k
            res2 = coins[i] >> (shift + 1)
            for child in G[i]:
                if child != parent:
                    res1 += dfs(child, shift, i)
                    if shift < 13:
                        res2 += dfs(child, shift + 1, i)
            return max(res1, res2)

        return dfs(0, 0, -1)

    # https://leetcode.cn/problems/minimum-increment-operations-to-make-array-beautiful/
    def minIncrementOperations(self, nums: List[int], k: int) -> int:
        f0, f1, f2 = max(0, k - nums[0]), max(0, k - nums[1]), max(0, k - nums[2])
        for num in nums[3:]:
            inc = min(f0, f1, f2) + max(0, k - num)
            f0 = f1
            f1 = f2
            f2 = inc
        return min(f0, f1, f2)

    # https://leetcode.cn/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1, s2 = sum(max(num, 1) for num in nums1), sum(max(num, 1) for num in nums2)
        if (s1 < s2 and 0 not in nums1) or (s2 < s1 and 0 not in nums2):
            return -1
        return max(s1, s2)

    # https://leetcode.cn/problems/find-the-k-or-of-an-array/
    def findKOr(self, nums: List[int], k: int) -> int:
        cnt = [0] * 32
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    cnt[i] += 1
        ans = 0
        for i in range(32):
            if cnt[i] >= k:
                ans |= 1 << i
        return ans

    def minimumMoves(self, grid: List[List[int]]) -> int:
        stones = []
        slots = []
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    slots.append((i, j))
                if grid[i][j] > 1:
                    stones.extend([(i, j)] * (grid[i][j] - 1))
        l = len(stones)
        ans = 0x7FFFFFFF
        for p in permutations(stones):
            distance = 0
            for i in range(l):
                a1, b1 = slots[i]
                a2, b2 = p[i]
                distance += abs(a2 - a1) + abs(b2 - b1)
            ans = min(ans, distance)
        return ans

    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        citations.sort(reverse=True)
        h = 0
        for i in range(1, n + 1):
            if citations[i - 1] > h:
                h = i
        return h

    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        if sx == fx and sy == fy:
            return t != 1
        return max(abs(sx - fx), abs(sy - fy)) <= t

    def numberOfPoints(self, nums: List[List[int]]) -> int:
        nums.sort(key=lambda p: p[0])

        ans = 0
        last_r = 0
        for pair in nums:
            if pair[0] <= last_r:
                ans += max(0, pair[1] - last_r)
                last_r = max(last_r, pair[1])
            else:
                ans += pair[1] - pair[0] + 1
                last_r = pair[1]

        return ans


class RangeModule:
    def __init__(self):
        self.ivs: SortedDict = SortedDict()

    def addRange(self, left: int, right: int) -> None:
        idx = self.ivs.bisect_right(left)
        if idx != 0:
            prev = idx - 1
            prev_left = self.ivs.keys()[prev]
            prev_right = self.ivs.values()[prev]
            if prev_right >= right:  # 包含
                return
            if prev_right >= left:  # pl left pr right 需要看情况更新pl对应的r
                left = prev_left
                self.ivs.popitem(prev)
                idx -= 1
        while idx < len(self.ivs) and self.ivs.keys()[idx] <= right:  # 之后所有区间的l都在r内
            right = max(right, self.ivs.values()[idx])
            self.ivs.popitem(idx)

        self.ivs[left] = right

    def queryRange(self, left: int, right: int) -> bool:
        idx = self.ivs.bisect_right(left)
        if idx == 0:
            return False
        return right <= self.ivs.values()[idx - 1]

    def removeRange(self, left: int, right: int) -> None:
        idx = self.ivs.bisect_right(left)
        if idx != 0:
            prev = idx - 1
            prev_left = self.ivs.keys()[prev]
            prev_right = self.ivs.values()[prev]
            if prev_right >= right:
                if prev_left == left:  # 左边界重合
                    self.ivs.popitem(prev)
                else:
                    self.ivs[prev_left] = left
                if right != prev_right:
                    self.ivs[right] = prev_right
            elif prev_right > left:  # pl left pr right
                if prev_left == left:
                    self.ivs.popitem(prev)
                    idx -= 1
                else:
                    self.ivs[prev_left] = left
        while idx < len(self.ivs) and self.ivs.keys()[idx] < right:
            if self.ivs.values()[idx] <= right:
                self.ivs.popitem(idx)
            else:
                self.ivs[right] = self.ivs.values()[idx]
                self.ivs.popitem(idx)
                break


class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.n = len(nums)
        self.tree = [0] * (self.n + 1)
        for i in range(self.n):
            self.add(i + 1, nums[i])

    def update(self, index: int, val: int) -> None:
        self.add(index + 1, val - self.nums[index])
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        return self.query(right + 1) - self.query(left)

    def query(self, index):
        ans = 0
        while index > 0:
            ans += self.tree[index]
            index -= self.low_bit(index)
        return ans

    def add(self, index, value):
        while index <= self.n:
            self.tree[index] += value
            index += self.low_bit(index)

    def low_bit(self, x):
        return x & -x


s = Solution()
s.test()
