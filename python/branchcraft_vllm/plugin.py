"""vLLM discovers this entry point when the package is installed."""


def register():
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "branchcraft_vllm.backend.BranchcraftBackend",
    )
