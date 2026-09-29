"""PyTorch dispatcher bindings for Branchcraft's CUDA kernels.

Importing this module loads the native operator library. The public API is
``torch.ops.branchcraft`` so dynamo/compile sees proper operator schemas.
"""

from pathlib import Path

import torch


_library = next(Path(__file__).parent.glob("_C*.so"), None)
if _library is None:
    raise ImportError("Branchcraft CUDA extension is missing; run `pip install -e . --no-build-isolation`")
torch.ops.load_library(str(_library))


@torch.library.register_fake("branchcraft::tree_verify")
def _fake_tree_verify(q, prefix_k, prefix_v, pages, lengths, draft_k, draft_v, parent):
    return q.new_empty(q.shape, dtype=torch.float32)


@torch.library.register_fake("branchcraft::packed_decode_out")
def _fake_packed_decode_out(q, packed_kv, pages, lengths, output, scale):
    return output


@torch.library.register_fake("branchcraft::commit_path_")
def _fake_commit_path(cache_k, cache_v, draft_k, draft_v, path_nodes, dest_pages,
                      dest_slots, request):
    return None


def tree_verify(q, prefix_k, prefix_v, pages, lengths, draft_k, draft_v, parent):
    """Return FP32 [batch, candidate, query head, dimension] outputs."""
    return torch.ops.branchcraft.tree_verify(
        q, prefix_k, prefix_v, pages, lengths, draft_k, draft_v, parent
    )


def packed_decode_out(q, packed_kv, pages, lengths, output, scale):
    """Decode into a preallocated FP16 output (safe for CUDA graph capture)."""
    return torch.ops.branchcraft.packed_decode_out(
        q, packed_kv, pages, lengths, output, scale
    )


def commit_path_(cache_k, cache_v, draft_k, draft_v, path_nodes, dest_pages,
                 dest_slots, request):
    """Write accepted draft K/V to preallocated physical destinations."""
    return torch.ops.branchcraft.commit_path_(
        cache_k, cache_v, draft_k, draft_v, path_nodes, dest_pages, dest_slots, request
    )
