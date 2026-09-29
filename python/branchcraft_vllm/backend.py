"""vLLM V1 backend: direct packed-KV CUDA decode with Triton fallback.

Only the simple one-token causal decoder case uses Branchcraft. All other
calls keep vLLM's Triton semantics. This boundary is deliberately explicit:
prefill, sliding-window, quantized KV, and multimodal attention differ.
"""

import os

import torch

import branchcraft_torch  # noqa: F401 - registers torch.ops.branchcraft
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)


class BranchcraftBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        # vLLM resolves this name through AttentionBackendEnum. The concrete
        # implementation remains distinguishable by its module/class and trace.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["BranchcraftImpl"]:
        return BranchcraftImpl


class BranchcraftImpl(TritonAttentionImpl):
    _reported_fast_path = False
    _reported_prefill_fallback = False

    def _can_decode(self, query, kv_cache, metadata, output, output_scale,
                    output_block_scale):
        if metadata is None:
            return False
        n = metadata.num_actual_tokens
        return (
            n > 0
            and metadata.max_query_len == 1
            and metadata.query_start_loc.numel() == n + 1
            and metadata.causal
            and not metadata.use_cascade
            and self.attn_type == AttentionType.DECODER
            and self.sliding_window == (-1, -1)
            and self.alibi_slopes is None
            and self.sinks is None
            and not self.logits_soft_cap
            and self.kv_cache_dtype in ("auto", "float16")
            and self.chunk_lookback == -1
            and metadata.mm_prefix_range_tensor is None
            and metadata.rswa_window is None
            and query.dtype == torch.float16
            and kv_cache.dtype == torch.float16
            and output.dtype == torch.float16
            and query.is_contiguous()
            and output.is_contiguous()
            and self.head_size in (64, 128)
            and kv_cache.shape[2] == 16
            and kv_cache.shape[3] == 2 * self.head_size
            and kv_cache.stride(3) == 1
            and output_scale is None
            and output_block_scale is None
            and metadata.block_table.dtype == torch.int32
            and metadata.seq_lens.dtype == torch.int32
            and metadata.block_table.is_contiguous()
            and metadata.seq_lens.is_contiguous()
        )

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if self._can_decode(query, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale):
            n = attn_metadata.num_actual_tokens
            if os.getenv("BRANCHCRAFT_TRACE") and not type(self)._reported_fast_path:
                print("BRANCHCRAFT_FAST_PATH packed_decode_out", flush=True)
                type(self)._reported_fast_path = True
            torch.ops.branchcraft.packed_decode_out(
                query[:n],
                kv_cache,
                attn_metadata.block_table[:n],
                attn_metadata.seq_lens[:n],
                output[:n].view(n, self.num_heads, self.head_size),
                self.scale,
            )
            return output
        if (os.getenv("BRANCHCRAFT_TRACE") and attn_metadata is not None
                and attn_metadata.max_query_len > 1
                and not type(self)._reported_prefill_fallback):
            print("BRANCHCRAFT_PREFILL_FALLBACK TritonAttentionImpl", flush=True)
            type(self)._reported_prefill_fallback = True
        return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                               output, output_scale, output_block_scale)
