from typing import List
from collections import defaultdict


class Solution:
    # https://leetcode.cn/problems/two-sum/
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in dic:
                return [dic[diff], i]
            dic[num] = i
        return [-1, -1]

    # https://leetcode.cn/problems/group-anagrams/
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = defaultdict(list)
        for s in strs:
            ss = "".join(sorted(s))
            ans[ss].append(s)
        return list(ans.values())

    # https://leetcode.cn/problems/longest-consecutive-sequence/
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        ans = 0
        for num in nums:
            if num - 1 not in nums:
                cur = num
                cur_len = 1
                while cur + 1 in nums:
                    cur += 1
                    cur_len += 1
                ans = max(ans, cur_len)
        return ans

    # https://leetcode.cn/problems/move-zeroes/
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        zi = 0
        for i in range(n):
            if nums[i] != 0:
                nums[i], nums[zi] = nums[zi], nums[i]
                zi += 1

    # https://leetcode.cn/problems/container-with-most-water/
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            ans = max(ans, (r-l) * min(height[l], height[r]))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans