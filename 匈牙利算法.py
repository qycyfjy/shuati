from typing import List


class Buddy:
    """
    两数和为质数的最大分组数
    """

    def __init__(self):
        self.primes = set()
        self.init_primes()

    # 筛可能更好
    def init_primes(self):
        for i in range(5, 60000 + 1):
            k = 2
            prime = True
            while k * k <= i:
                if i % k == 0:
                    prime = False
                    break
                k += 1
            if prime:
                self.primes.add(i)

    def is_prime(self, x: int):
        return x in self.primes

    def buddy(self, nums: List[int]):
        evens = []
        odds = []
        for num in nums:
            if num % 2 == 0:
                evens.append(num)
            else:
                odds.append(num)
        n_evens = len(evens)
        n_odds = len(odds)
        matches = [-1] * n_odds
        checked = [False] * n_odds

        def match(xi: int):
            x = evens[xi]
            for oi, odd in enumerate(odds):
                if self.is_prime(x + odd) and not checked[oi]:  # 如果能成而且不是绕一圈又回来
                    checked[oi] = True  # 标记
                    if matches[oi] == -1 or match(
                        matches[oi]
                    ):  # 如果该odd没分配even或者前even能找到另一个合适的
                        matches[oi] = xi  # 配对
                        return True
            return False

        cnt = 0
        for i in range(n_evens):  # 为每个even找合适的odd
            for j in range(n_odds):  # 假装能随便自己选
                checked[j] = False
            if match(i):  # 如果能找到
                cnt += 1
        return cnt
