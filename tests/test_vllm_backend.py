"""Contract test at the vLLM V1 attention implementation boundary."""

import math
from types import SimpleNamespace

import pytest
import torch


vllm = pytest.importorskip("vllm")
from branchcraft_vllm.backend import BranchcraftImpl  # noqa: E402
from vllm.v1.attention.backend import AttentionType  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_custom_backend_reads_vllm_packed_cache():
    torch.manual_seed(23)
    n, hq, hkv, d = 2, 4, 2, 64
    query = torch.randn(n, hq, d, device="cuda", dtype=torch.float16)
    cache = torch.randn(6, 16, hkv, 2 * d, device="cuda",
                        dtype=torch.float16).transpose(1, 2)  # logical HND, physical NHD
    pages = torch.tensor([[3, 1], [4, 2]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([19, 7], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=n, max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        causal=True, use_cascade=False, mm_prefix_range_tensor=None,
        rswa_window=None, block_table=pages, seq_lens=lengths,
    )
    impl = BranchcraftImpl.__new__(BranchcraftImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = hq, hkv, d
    impl.scale = 1 / math.sqrt(d)
    impl.attn_type = AttentionType.DECODER
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = impl.sinks = None
    impl.logits_soft_cap = 0
    impl.kv_cache_dtype = "auto"
    impl.chunk_lookback = -1
    output = torch.empty(n, hq * d, device="cuda", dtype=torch.float16)
    result = impl.forward(None, query, query, query, cache, metadata, output)
    assert result.data_ptr() == output.data_ptr()
    expected = torch.empty_like(output).view(n, hq, d)
    for b in range(n):
        for h in range(hq):
            kh = h // (hq // hkv)
            tokens = [cache[int(pages[b, t // 16]), kh, t % 16]
                      for t in range(int(lengths[b]))]
            k = torch.stack([item[:d].float() for item in tokens])
            v = torch.stack([item[d:].float() for item in tokens])
            expected[b, h] = (torch.softmax(k @ query[b, h].float() * impl.scale, 0) @ v).half()
    torch.testing.assert_close(output.view(n, hq, d), expected,
                               atol=6e-3, rtol=6e-3)
    impl.sliding_window = (127, 0)
    assert not impl._can_decode(query, cache, metadata, output, None, None)
