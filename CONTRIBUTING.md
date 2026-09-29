# Contributing to Branchcraft

Branchcraft is intentionally small enough to inspect end to end. Changes are
welcome when they keep the attention semantics and KV ownership rules visible.

## A useful change includes

1. A specific contract: supported dtype, head dimension, page layout, mask,
   and output precision.
2. A correctness check against an independent implementation or a failing
   cache transaction case. Comparing two calls to the same kernel is not enough.
3. A reproducer for performance changes, with shape, batch, context length,
   GPU, timing method, and raw results. Show regressions as well as wins.
4. Documentation updates when a supported vLLM shape or PyTorch operator
   schema changes. The backend targets vLLM 0.27.0; newer versions require
   revalidation of metadata and the plugin interface.

## Good next problems

- **Split long packed KV decode across CTAs.** The vLLM fast path currently
  assigns one warp per query head. Reuse the standalone split-KV online-softmax
  merge idea while preserving vLLM's packed HND/NHD strides. A useful result
  includes the dispatch threshold and cases where splitting loses.
- **Vectorize GQA reuse.** Several query heads read the same K/V. Study a CTA
  mapping that shares K/V loads among those heads and quantify register and
  occupancy costs.
- **Extend the vLLM guard deliberately.** BF16, larger pages, or another
  decoder feature should get an independent reference test and model-level
  parity before joining the fast path. Do not broaden the guard by assumption.
- **Harden the page owner.** Add explicit metadata validation and a bounded
  free-list stress test for repeated fork, checkpoint, commit, accept, and
  rollback sequences.

## Development commands

```bash
make CUDA_ARCH=sm_120
./build/branchcraft test
pytest -q tests/test_torch_ops.py tests/test_transaction.py tests/test_vllm_backend.py
python examples/vllm_parity.py --output benchmarks/vllm_parity.json
```

Use the [integration notes](docs/framework-integration.md) for toolkit setup
and supported vLLM shapes. The [benchmark protocol](benchmarks/README.md)
defines the existing measurements.
