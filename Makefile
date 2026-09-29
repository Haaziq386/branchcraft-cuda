NVCC ?= nvcc
CUDA_ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all test bench clean
all: build/branchcraft

build/branchcraft: src/main.cu src/paged_gqa.cu src/tree_attention.cu include/branchcraft.hpp
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -Iinclude src/main.cu src/paged_gqa.cu src/tree_attention.cu -o $@

test: build/branchcraft
	./build/branchcraft test

bench: build/branchcraft
	./build/branchcraft tree-bench --output benchmarks/tree_results.csv
	./build/branchcraft bench --output benchmarks/decode_results.csv
	python3 scripts/make_charts.py benchmarks/decode_results.csv

clean:
	rm -rf build
