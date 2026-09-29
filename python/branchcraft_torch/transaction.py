"""Small, inspectable copy-on-write page transaction for tree verification.

This is a teaching-sized host allocator, not vLLM's scheduler. Snapshots own
page references, so committing to a shared/remembered partial page allocates a
copy. Rollback therefore restores both page mappings and the previous bytes.
"""

from dataclasses import dataclass

import torch

from . import commit_path_, tree_verify


@dataclass
class Snapshot:
    request: int
    pages: list[int]
    length: int
    active: bool = True


class PagedTreeCache:
    def __init__(self, physical_pages: int, kv_heads: int, head_dim: int,
                 device: str = "cuda"):
        if head_dim not in (64, 128) or physical_pages < 1:
            raise ValueError("head_dim must be 64/128; physical_pages must be positive")
        self.k = torch.zeros((physical_pages, 16, kv_heads, head_dim),
                             device=device, dtype=torch.float16)
        self.v = torch.zeros_like(self.k)
        self.refcount = [0] * physical_pages
        self.free = list(range(physical_pages - 1, -1, -1))
        self.pages: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        self._next_request = 0

    def _alloc(self) -> int:
        if not self.free:
            raise RuntimeError("KV page pool exhausted")
        page = self.free.pop()
        assert self.refcount[page] == 0
        self.refcount[page] = 1
        return page

    def _release(self, page: int):
        self.refcount[page] -= 1
        if self.refcount[page] == 0:
            self.free.append(page)

    def new(self, prefix_k: torch.Tensor, prefix_v: torch.Tensor) -> int:
        """Create one request from [tokens, KV heads, head dim] FP16 tensors."""
        if prefix_k.shape != prefix_v.shape or prefix_k.ndim != 3:
            raise ValueError("prefix K/V shapes must match [tokens, heads, dim]")
        if prefix_k.dtype != torch.float16 or prefix_k.device != self.k.device:
            raise ValueError("prefix must be FP16 on the cache device")
        if tuple(prefix_k.shape[1:]) != tuple(self.k.shape[2:]):
            raise ValueError("prefix head shape differs from cache")
        count = (prefix_k.shape[0] + 15) // 16
        if len(self.free) < count:
            raise RuntimeError("KV page pool exhausted")
        request = self._next_request
        self._next_request += 1
        self.pages[request] = []
        self.lengths[request] = prefix_k.shape[0]
        for start in range(0, prefix_k.shape[0], 16):
            page = self._alloc()
            self.pages[request].append(page)
            n = min(16, prefix_k.shape[0] - start)
            self.k[page, :n].copy_(prefix_k[start:start + n])
            self.v[page, :n].copy_(prefix_v[start:start + n])
        return request

    def fork(self, source: int) -> int:
        request = self._next_request
        self._next_request += 1
        self.pages[request] = self.pages[source].copy()
        self.lengths[request] = self.lengths[source]
        for page in self.pages[request]:
            self.refcount[page] += 1
        return request

    def checkpoint(self, request: int) -> Snapshot:
        pages = self.pages[request].copy()
        for page in pages:
            self.refcount[page] += 1
        return Snapshot(request, pages, self.lengths[request])

    def rollback(self, snapshot: Snapshot):
        if not snapshot.active:
            raise RuntimeError("snapshot already consumed")
        for page in self.pages[snapshot.request]:
            self._release(page)
        self.pages[snapshot.request] = snapshot.pages
        self.lengths[snapshot.request] = snapshot.length
        snapshot.active = False  # transfer snapshot's refs to the request

    def accept(self, snapshot: Snapshot):
        if not snapshot.active:
            raise RuntimeError("snapshot already consumed")
        for page in snapshot.pages:
            self._release(page)
        snapshot.active = False

    def verify(self, request: int, q: torch.Tensor, draft_k: torch.Tensor,
               draft_v: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        """Verify one request; draft inputs have leading batch dimension 1."""
        pages = self.pages[request] or [0]  # ignored when prefix length is zero
        page_table = torch.tensor([pages], device=self.k.device, dtype=torch.int32)
        length = torch.tensor([self.lengths[request]], device=self.k.device,
                              dtype=torch.int32)
        return tree_verify(q, self.k, self.v, page_table, length,
                           draft_k, draft_v, parent)

    def commit(self, request: int, path: list[int], draft_k: torch.Tensor,
               draft_v: torch.Tensor):
        """Append an accepted root-to-leaf path, preserving shared pages."""
        if not path or len(path) > draft_k.shape[1]:
            raise ValueError("path is empty or longer than draft")
        if draft_k.shape != draft_v.shape or draft_k.shape[0] != 1:
            raise ValueError("draft K/V must match and have batch size 1")
        if any(node < 0 or node >= draft_k.shape[1] for node in path):
            raise ValueError("path node out of range")
        old_length = self.lengths[request]
        old_pages = self.pages[request]
        new_page_count = (old_length + len(path) + 15) // 16
        copy_partial = old_length % 16 != 0 and self.refcount[old_pages[-1]] > 1
        pages_needed = new_page_count - len(old_pages) + int(copy_partial)
        if len(self.free) < pages_needed:
            raise RuntimeError("KV page pool exhausted")
        if copy_partial:
            old = old_pages[-1]
            fresh = self._alloc()
            self.k[fresh].copy_(self.k[old])
            self.v[fresh].copy_(self.v[old])
            old_pages[-1] = fresh
            self._release(old)
        while len(old_pages) < new_page_count:
            old_pages.append(self._alloc())
        destination_pages = [old_pages[pos // 16]
                             for pos in range(old_length, old_length + len(path))]
        destination_slots = [pos % 16
                             for pos in range(old_length, old_length + len(path))]
        device = self.k.device
        commit_path_(
            self.k, self.v, draft_k, draft_v,
            torch.tensor(path, device=device, dtype=torch.int32),
            torch.tensor(destination_pages, device=device, dtype=torch.int32),
            torch.tensor(destination_slots, device=device, dtype=torch.int32), 0,
        )
        self.lengths[request] += len(path)
