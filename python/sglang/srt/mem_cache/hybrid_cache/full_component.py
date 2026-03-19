from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.hybrid_cache.tree_component import (
    BASE_COMPONENT_NAME,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_radix_cache import HybridTreeNode


class FullComponent(TreeComponent):
    @property
    def name(self) -> str:
        return BASE_COMPONENT_NAME

    def create_match_validator(self) -> Callable[["HybridTreeNode"], bool]:
        return lambda node: True

    def redistribute_on_node_split(
        self, new_parent: "HybridTreeNode", child: "HybridTreeNode"
    ):
        new_parent.component(self.name).lock_ref = child.component(self.name).lock_ref

    def evict_component(self, node: "HybridTreeNode", is_leaf: bool) -> int:
        self.cache.token_to_kv_pool_allocator.free(node.full_value)
        freed = len(node.full_value)
        self.cache.component_evictable_size_[self.name] -= freed
        return freed

    def acquire_component_lock(
        self, node: "HybridTreeNode", result: IncLockRefResult
    ) -> IncLockRefResult:
        cur = node
        while cur != self.cache.root_node:
            if cur.component(self.name).lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(cur.full_value)
                self.cache.component_protected_size_[self.name] += len(cur.full_value)
            cur.component(self.name).lock_ref += 1
            cur = cur.parent
        return result

    def release_component_lock(
        self, node: "HybridTreeNode", params: Optional[DecLockRefParams]
    ) -> None:
        cur = node
        while cur != self.cache.root_node:
            assert cur.component(self.name).lock_ref > 0
            if cur.component(self.name).lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(cur.full_value)
                self.cache.component_protected_size_[self.name] -= len(cur.full_value)
            cur.component(self.name).lock_ref -= 1
            cur = cur.parent
