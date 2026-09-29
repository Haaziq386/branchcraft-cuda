# Benchmark protocol and interpretation

The checked-in CSV files were measured on an NVIDIA RTX PRO 5000 Blackwell (compute capability 12.0), driver 580.95.05, with code compiled by CUDA 12.8.1 in `nvidia/cuda:12.8.1-devel-ubuntu22.04`. The binary ran directly on the host. Measurements were collected on 2026-09-29. GPU clocks were not locked, so exact microsecond values may change.

## Tree verification (`tree_results.csv`)

For each shape, the fixture constructs a seven-node binary tree, with 32 query heads, 8 KV heads, head dimension 128, FP16 Q/K/V, FP32 output, and 16-token pages. All requests share identical physical prefix pages. Each candidate node stores one draft K/V pair; `parent` indices encode paths. The baseline physically expands each candidate into an independent paged sequence containing the prefix and its ancestors, then performs **one batched launch** of this repo's ordinary paged GQA decode kernel. Both implementations receive the same query values and must agree within `3e-4` maximum absolute error before either is timed.

`tree_us` and `expanded_us` include their respective single attention kernel launch and device work. They exclude host fixture construction, GPU allocation and copies, and output checking. `latency_ratio = expanded_us / tree_us`: above one favors the tree path. This baseline is deliberately materialized and should not be confused with FlashInfer, DeFT, cuDNN, or an optimized production tree verifier.

`shared_kv_mib` counts one page-rounded prefix plus compact draft K/V for every node. `expanded_kv_mib` counts the page-rounded independent sequence for every candidate path. Both count K and V as FP16 and exclude allocator reserve pages, page-table metadata, queries, outputs, and scratch. `memory_saving_pct = 100 × (1 - shared/expanded)`. It describes the **live KV representation**, not exact process VRAM use; the test fixture reserves extra pages for the commit experiment.

## Ordinary decode (`decode_results.csv`)

This secondary sweep uses ragged request lengths, shuffled physical pages, 32 query heads, 8 KV heads, dimension 128, and FP16 K/V. `single_us` is the one-partition kernel; `split_us` is the heuristic-selected split count and includes the stable merge kernel when split > 1. `speedup = single_us / split_us`. The heuristic targets roughly two CTAs per SM, caps at 16 splits, requires at least 128 tokens per partition, and keeps contexts shorter than 512 on one partition. The kernel paths are compared for numerical agreement before timing.

## Timing method

Every timed path gets five warmup launches. Seven trials then run twenty back-to-back launches each. CUDA events bracket each trial; the CSV reports the median per-launch time across trials. This reduces host launch jitter but does not lock clocks or eliminate contention. Data generation is deterministic, with the fixture seed recorded in `src/main.cu`. The Python chart generator reads the CSV without hand-edited numbers.

Run `./build/branchcraft tree-bench --quick` and `./build/branchcraft bench --quick` for a fast smoke test. Rebuilding the full checked-in results takes longer and may produce different values on another device or driver.
