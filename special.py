from typing import List, Mapping, Optional, TypeVar
from collections import defaultdict

TNode = TypeVar("TNode", bound="ListNode")


class ListNode:
    def __init__(self, val=0, next=Optional[TNode]):
        self.val = val
        self.next = next


class Solution:
    # https://leetcode.cn/problems/merge-two-sorted-lists/
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy
        while list1 is not None and list2 is not None:
            if list1.val <= list2.val:
                current.next = list1
                current = list1
                list1 = list1.next
            else:
                current.next = list2
                current = list2
                list2 = list2.next
        while list1 is not None:
            current.next = list1
            current = list1
            list1 = list1.next
        while list2 is not None:
            current.next = list2
            current = list2
            list2 = list2.next
        return dummy.next

    # https://leetcode.cn/problems/remove-nth-node-from-end-of-list/
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        f_node = dummy
        b_node = dummy
        for _ in range(n + 1):
            if f_node is None:
                return head
            f_node = f_node.next
        while f_node != None:
            b_node = b_node.next
            f_node = f_node.next
        b_node.next = b_node.next.next
        return dummy.next

    # https://leetcode.cn/problems/group-anagrams/
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = defaultdict(list)
        for s in strs:
            sorted_s = "".join(sorted(s))
            dic[sorted_s].append(s)
        return list(dic.values())

    # https://leetcode.cn/problems/add-two-numbers/
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode()
        current = dummy
        carry = 0
        while l1 != None or l2 != None:
            n1 = 0 if l1 == None else l1.val
            l1 = l1.next if l1 != None else l1
            n2 = 0 if l2 == None else l2.val
            l2 = l2.next if l2 != None else l2
            s = n1 + n2 + carry
            carry = s // 10
            n = s % 10
            node = ListNode(n)
            current.next = node
            current = node
        if carry:
            current.next = ListNode(1)
        return dummy.next

    # https://leetcode.cn/problems/two-sum/
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in dic:
                return [i, dic[diff]]
            dic[num] = i


# https://leetcode.cn/problems/lru-cache/
TLruNode = TypeVar("TLruNode", bound="LRUNode")


class LRUNode:
    def __init__(
        self, key: int = 0, value: int = 0, prev: TLruNode = None, next: TLruNode = None
    ) -> None:
        self.key = key
        self.value = value
        self.next = next
        self.prev = prev


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.count = 0

        self.head = LRUNode()
        self.tail = LRUNode()
        self.head.next = self.tail
        self.tail.prev = self.head

        self.kv: Mapping[int, LRUNode] = {}

    def get(self, key: int) -> int:
        if key in self.kv:
            node = self.kv[key]
            self.move_to_head(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.kv:
            node = self.kv[key]
            node.value = value
            self.move_to_head(node)
            return
        if self.count == self.cap:
            self.remove_last()
        self.count += 1
        node = LRUNode(key, value, self.head, self.head.next)
        self.kv[key] = node
        self.head.next.prev = node
        self.head.next = node

    def move_to_head(self, node: LRUNode):
        if node.prev == self.head:
            return
        prev = node.prev
        next = node.next
        hnext = self.head.next

        node.next = hnext
        node.prev = self.head
        self.head.next = node
        hnext.prev = node
        next.prev = prev
        prev.next = next

    def remove_last(self):
        removed = self.tail.prev
        removed.prev.next = self.tail
        self.tail.prev = removed.prev

        del self.kv[removed.key]

        self.count -= 1
