#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace branchcraft {

constexpr int kPageSize = 16;
constexpr int kWarpsPerBlock = 4;

struct Shape {
  int batch;
  int query_heads;
  int kv_heads;
  int dim;
  int max_pages;
  int physical_pages;
};

// All pointers are device pointers. K/V: [physical_pages, 16, kv_heads, dim].
// Q: [batch, query_heads, dim]. Pages: [batch, max_pages].
// Output: FP32 [batch, query_heads, dim]. Empty sequences produce zeros.
struct Inputs {
  const half* q;
  const half* k;
  const half* v;
  const int* pages;
  const int* lengths;
  float* output;
};

// Caller allocates scratch for splits > 1: partial_a = splits*B*Hq*D floats;
// partial_ml = splits*B*Hq*2 floats. Both may be null for splits == 1.
struct Scratch {
  float* partial_a;
  float* partial_ml;
};

cudaError_t decode(Shape shape, Inputs in, Scratch scratch, int splits,
                   cudaStream_t stream = nullptr);
int auto_splits(Shape shape, int max_length, int sm_count);

// Draft nodes are stored in topological order, with parent[b,n] < n or -1.
// Every node attends to the request's paged prefix and its own ancestor chain
// (including itself). Draft Q: [B,N,Hq,D]; K/V: [B,N,Hkv,D].
struct TreeInputs {
  Inputs prefix;
  const half* draft_q;
  const half* draft_k;
  const half* draft_v;
  const int* parent;
  float* output;  // [B,N,Hq,D]
};

cudaError_t tree_verify(Shape shape, TreeInputs in, int nodes,
                        cudaStream_t stream = nullptr);

// Copy accepted draft nodes into physical KV destinations. `dest_pages` and
// `dest_slots` are device arrays of length accepted; `path_nodes` is the
// root-to-leaf path. The caller updates page-table metadata after completion.
cudaError_t commit_path(Shape shape, const half* draft_k, const half* draft_v,
                        half* cache_k, half* cache_v, const int* path_nodes,
                        const int* dest_pages, const int* dest_slots,
                        int request, int nodes, int accepted,
                        cudaStream_t stream = nullptr);

// vLLM-style packed cache. Logical shape [P,Hkv,16,2D], with runtime strides
// so either NHD or HND physical layout works without a KV copy. Q and output
// are contiguous [B,Hq,D]. Output is FP16, accumulation stays FP32.
cudaError_t packed_decode(Shape shape, const half* q, const half* packed_kv,
                          const int* pages, const int* lengths, half* output,
                          long long block_stride, long long head_stride,
                          long long slot_stride, float scale,
                          cudaStream_t stream = nullptr);

}  // namespace branchcraft
