from typing import List
from functools import cache
from itertools import accumulate
from collections import Counter


class Solution:
    # https://leetcode.cn/problems/maximal-square
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) + 1
                    ans = max(ans, dp[i][j])
        return ans * ans

    # https://leetcode.cn/problems/minimum-falling-path-sum
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        dp = [[0] * m for _ in range(m)]
        dp[-1][:] = matrix[-1][:]
        for i in range(m - 2, -1, -1):
            for j in range(m):
                c = dp[i + 1][j]
                if j > 0:
                    c = min(c, dp[i + 1][j - 1])
                if j < m - 1:
                    c = min(c, dp[i + 1][j + 1])
                dp[i][j] = c + matrix[i][j]
        return min(dp[0])

    # https://leetcode.cn/problems/triangle
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        m = len(triangle)
        n = len(triangle[-1])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = triangle[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + triangle[i][0]
            for j in range(1, i):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
            dp[i][i] = dp[i - 1][i - 1] + triangle[i][i]
        return min(dp[-1])

    # https://leetcode.cn/problems/unique-paths-ii
    def uniquePathsWithObstacles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [1] * n
        ob = False
        for i in range(n):
            if ob:
                dp[i] = 0
            if grid[0][i] == 1:
                ob = True
                dp[i] = 0
        for i in range(1, m):
            for j in range(n):
                if grid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] += dp[j - 1]
        return dp[-1]

    # https://leetcode.cn/problems/minimum-path-sum
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = list(accumulate(grid[0]))
        for i in range(1, m):
            for j in range(n):
                t = dp[j] if j == 0 else dp[j - 1]
                dp[j] = min(t, dp[j]) + grid[i][j]
        return dp[-1]

    # https://leetcode.cn/problems/unique-paths
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[n - 1]

    # https://leetcode.cn/problems/delete-and-earn/
    def deleteAndEarn(self, nums: List[int]) -> int:
        maximum = max(nums)
        sums = [0] * (maximum + 1)
        for num in nums:
            sums[num] += num
        return self.rob(sums)

    # https://leetcode.cn/problems/house-robber
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(dp[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[n - 1]

    # https://leetcode.cn/problems/min-cost-climbing-stairs
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        a, b = 0, 0
        for i in range(2, n + 1):
            a, b = b, min(a + cost[i - 2], b + cost[i - 1])
        return b

    # https://leetcode.cn/problems/n-th-tribonacci-number
    def tribonacci(self, n: int) -> int:
        a, b, c, d = 0, 1, 1, 2
        for _ in range(n):
            a = b
            b = c
            c = d
            d = a + b + c
        return a

    # https://leetcode.cn/problems/fibonacci-number
    def fib(self, n: int) -> int:
        a, b, c = 0, 1, 1
        for _ in range(n):
            a = b
            b = c
            c = a + b
        return a

    # https://leetcode.cn/problems/climbing-stairs/
    @cache
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)


s = Solution()
s.uniquePathsWithObstacles([[1]])
