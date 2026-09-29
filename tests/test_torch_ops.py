import math

import pytest
import torch

import branchcraft_torch as bc


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _device():
    return "cuda"


@pytest.mark.parametrize("dim", [64, 128])
def test_tree_mask_and_compile(dim):
    torch.manual_seed(7)
    device = _device()
    batch, nodes, hq, hkv = 2, 7, 4, 2
    q = torch.randn(batch, nodes, hq, dim, device=device, dtype=torch.float16)
    prefix_k = torch.randn(7, 16, hkv, dim, device=device, dtype=torch.float16)
    prefix_v = torch.randn_like(prefix_k)
    pages = torch.tensor([[4, 1], [5, 2]], device=device, dtype=torch.int32)
    lengths = torch.tensor([23, 0], device=device, dtype=torch.int32)
    draft_k = torch.randn(batch, nodes, hkv, dim, device=device, dtype=torch.float16)
    draft_v = torch.randn_like(draft_k)
    parent = torch.tensor([[-1, 0, 0, 1, 1, 2, 2]] * batch,
                          device=device, dtype=torch.int32)
    actual = bc.tree_verify(q, prefix_k, prefix_v, pages, lengths, draft_k,
                            draft_v, parent)
    expected = torch.empty_like(actual)
    for b in range(batch):
        for node in range(nodes):
            path = []
            cur = node
            while cur >= 0:
                path.append(cur)
                cur = int(parent[b, cur])
            path.reverse()
            for h in range(hq):
                kvh = h // (hq // hkv)
                keys = []
                vals = []
                for t in range(int(lengths[b])):
                    page = int(pages[b, t // 16])
                    keys.append(prefix_k[page, t % 16, kvh].float())
                    vals.append(prefix_v[page, t % 16, kvh].float())
                keys += [draft_k[b, p, kvh].float() for p in path]
                vals += [draft_v[b, p, kvh].float() for p in path]
                scores = torch.stack(keys) @ q[b, node, h].float() / math.sqrt(dim)
                expected[b, node, h] = torch.softmax(scores, 0) @ torch.stack(vals)
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)
    compiled = torch.compile(bc.tree_verify, backend="inductor")
    torch.testing.assert_close(compiled(q, prefix_k, prefix_v, pages, lengths,
                                        draft_k, draft_v, parent), actual)
    torch.library.opcheck(torch.ops.branchcraft.tree_verify,
                          (q, prefix_k, prefix_v, pages, lengths, draft_k,
                           draft_v, parent), test_utils=("test_schema", "test_faketensor"))


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_packed_decode_and_graph(layout):
    torch.manual_seed(11)
    device = _device()
    batch, hq, hkv, dim = 3, 8, 2, 64
    if layout == "HND":
        cache = torch.randn(9, hkv, 16, 2 * dim, device=device, dtype=torch.float16)
    else:
        cache = torch.randn(9, 16, hkv, 2 * dim, device=device,
                            dtype=torch.float16).transpose(1, 2)
    q = torch.randn(batch, hq, dim, device=device, dtype=torch.float16)
    pages = torch.tensor([[4, 1], [2, 7], [8, 6]], device=device, dtype=torch.int32)
    lengths = torch.tensor([23, 9, 0], device=device, dtype=torch.int32)
    out = torch.empty_like(q)
    bc.packed_decode_out(q, cache, pages, lengths, out, 1 / math.sqrt(dim))
    expected = torch.zeros_like(q)
    for b in range(batch):
        for h in range(hq):
            kvh = h // (hq // hkv)
            keys, vals = [], []
            for t in range(int(lengths[b])):
                token = cache[int(pages[b, t // 16]), kvh, t % 16]
                keys.append(token[:dim].float())
                vals.append(token[dim:].float())
            if keys:
                scores = torch.stack(keys) @ q[b, h].float() / math.sqrt(dim)
                expected[b, h] = (torch.softmax(scores, 0) @ torch.stack(vals)).half()
    torch.testing.assert_close(out, expected, atol=6e-3, rtol=6e-3)
    # Capturing the same op on fixed allocations verifies stream correctness.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bc.packed_decode_out(q, cache, pages, lengths, out, 1 / math.sqrt(dim))
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out, expected, atol=6e-3, rtol=6e-3)


def test_commit_isolation():
    device = _device()
    cache_k = torch.zeros(4, 16, 2, 64, device=device, dtype=torch.float16)
    cache_v = torch.zeros_like(cache_k)
    draft_k = torch.arange(3, device=device, dtype=torch.float16).view(1, 3, 1, 1).expand(1, 3, 2, 64).contiguous()
    draft_v = draft_k + 10
    path = torch.tensor([0, 2], device=device, dtype=torch.int32)
    dest_pages = torch.tensor([3, 3], device=device, dtype=torch.int32)
    dest_slots = torch.tensor([6, 7], device=device, dtype=torch.int32)
    bc.commit_path_(cache_k, cache_v, draft_k, draft_v, path, dest_pages,
                    dest_slots, 0)
    torch.testing.assert_close(cache_k[3, 6], draft_k[0, 0])
    torch.testing.assert_close(cache_k[3, 7], draft_k[0, 2])
    torch.testing.assert_close(cache_v[3, 7], draft_v[0, 2])
    assert torch.count_nonzero(cache_k[:3]) == 0
    assert torch.count_nonzero(cache_k[3, 8:]) == 0
