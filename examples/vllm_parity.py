"""Run a real model through stock Triton and the Branchcraft CUSTOM backend.

Two subprocesses isolate vLLM's backend registry and CUDA context. The script
fails unless token IDs match *and* Branchcraft reports at least one decode.
"""

import argparse
from importlib.metadata import version
import json
import os
import subprocess
import sys


PROMPTS = [
    "Explain why paged KV caches need a block table in one sentence.",
    "Write three numbered steps to make CUDA softmax numerically stable.",
    "A GPU kernel sees a speculative tree. Which candidate tokens may a leaf attend to?",
]
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def run_backend(model, revision, backend):
    # Greedy parity does not need FlashInfer's JIT sampler; the Torch sampler
    # also makes this demo usable on hosts with an older CUDA driver.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    # Plugin registration happens through the installed entry point.
    llm = LLM(
        model=model,
        revision=revision,
        dtype="float16",
        attention_backend=backend,
        block_size=16,
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        trust_remote_code=False,
    )
    samples = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=24))
    return [sample.outputs[0].token_ids for sample in samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="model revision; default pins Qwen3-0.6B")
    parser.add_argument("--backend", choices=["TRITON_ATTN", "CUSTOM"])
    parser.add_argument("--output", help="write a machine-readable parity artifact")
    args = parser.parse_args()
    revision = args.revision or (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)
    if args.backend:
        tokens = run_backend(args.model, revision, args.backend)
        print("BRANCHCRAFT_TOKENS=" + json.dumps(tokens), flush=True)
        return
    results = {}
    for backend in ("TRITON_ATTN", "CUSTOM"):
        env = dict(os.environ, BRANCHCRAFT_TRACE="1")
        process = subprocess.run(
            [sys.executable, __file__, "--backend", backend, "--model", args.model,
             *(["--revision", revision] if revision else [])],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(f"{backend} failed:\n{process.stdout[-8000:]}")
        lines = process.stdout.splitlines()
        marker = next((line for line in lines if line.startswith("BRANCHCRAFT_TOKENS=")), None)
        if marker is None:
            raise RuntimeError(process.stdout[-4000:])
        results[backend] = json.loads(marker.split("=", 1)[1])
        if backend == "CUSTOM" and "BRANCHCRAFT_FAST_PATH" not in process.stdout:
            raise AssertionError("CUSTOM generated tokens without entering CUDA fast path")
        if backend == "CUSTOM" and "BRANCHCRAFT_PREFILL_FALLBACK" not in process.stdout:
            raise AssertionError("CUSTOM did not show the Triton prefill fallback")
        print(f"{backend}: {len(results[backend])} requests, fast path: "
              f"{'BRANCHCRAFT_FAST_PATH' in process.stdout}")
    if results["TRITON_ATTN"] != results["CUSTOM"]:
        raise AssertionError(f"Token mismatch: {results}")
    if args.output:
        artifact = {
            "model": args.model,
            "revision": revision,
            "vllm_version": version("vllm"),
            "torch_version": version("torch"),
            "prompts": PROMPTS,
            "sampling": {"temperature": 0, "max_tokens": 24},
            "backends": results,
            "exact_token_parity": True,
            "custom_fast_path_observed": True,
            "prefill_fallback_observed": True,
        }
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2)
            file.write("\n")
    print("PASS: exact greedy token parity and CUDA fast path observed")


if __name__ == "__main__":
    main()
