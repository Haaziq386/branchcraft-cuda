from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent

setup(
    ext_modules=[
        CUDAExtension(
            "branchcraft_torch._C",
            [
                "bindings/torch_ops.cpp",
                "src/paged_gqa.cu",
                "src/tree_attention.cu",
                "src/packed_decode.cu",
            ],
            include_dirs=[str(ROOT / "include")],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
