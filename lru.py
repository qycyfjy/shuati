from typing import Mapping, TypeVar

TLRUNode = TypeVar("TLRUNode", bound="LRUNode")


class LRUNode:
    def __init__(
        self, key = None, value=None, prev: TLRUNode = None, next: TLRUNode = None
    ) -> None:
        self.key = key
        self.value = value
        self.prev = prev
        self.next = next


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cnt = 0
        self.head = LRUNode()
        self.tail = LRUNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.kv: Mapping[int, LRUNode] = {}

    def get(self, key: int) -> int:
        if key in self.kv:
            node = self.kv[key]
            self.refresh(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.kv:
            node = self.kv[key]
            node.value = value
            self.refresh(node)
            return
        if self.cnt == self.cap:
            self.evict()
            self.cnt -= 1
        node = LRUNode(key, value, self.head, self.head.next)
        self.head.next.prev = node
        self.head.next = node
        self.kv[key] = node
        self.cnt += 1

    def refresh(self, node: LRUNode) -> None:
        if node.prev == self.head:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        origin = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = origin
        origin.prev = node

    def evict(self) -> None:
        evicted = self.tail.prev
        evicted.prev.next = self.tail
        self.tail.prev = evicted.prev
        del self.kv[evicted.key]
