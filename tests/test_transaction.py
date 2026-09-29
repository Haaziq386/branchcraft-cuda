import pytest
import torch

from branchcraft_torch.transaction import PagedTreeCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_partial_page_commit_then_rollback():
    torch.manual_seed(17)
    store = PagedTreeCache(8, 2, 64)
    prefix_k = torch.randn(19, 2, 64, device="cuda", dtype=torch.float16)
    prefix_v = torch.randn_like(prefix_k)
    original = store.new(prefix_k, prefix_v)
    sibling = store.fork(original)
    snapshot = store.checkpoint(original)
    initial_pages = store.pages[original].copy()
    old_bytes = store.k[initial_pages[-1]].clone()
    draft_k = torch.randn(1, 7, 2, 64, device="cuda", dtype=torch.float16)
    draft_v = torch.randn_like(draft_k)
    store.commit(original, [0, 2, 6], draft_k, draft_v)
    assert store.pages[original][-1] != store.pages[sibling][-1]
    assert store.lengths[original] == 22
    torch.testing.assert_close(store.k[store.pages[original][-1], 3], draft_k[0, 0])
    torch.testing.assert_close(store.k[store.pages[original][-1], 5], draft_k[0, 6])
    torch.testing.assert_close(store.k[store.pages[sibling][-1]], old_bytes)
    store.rollback(snapshot)
    assert store.pages[original] == initial_pages
    assert store.lengths[original] == 19
    torch.testing.assert_close(store.k[store.pages[original][-1]], old_bytes)
