#include "branchcraft.hpp"

#include <math_constants.h>

namespace branchcraft {
namespace {

__device__ __forceinline__ float warp_sum_tree(float x) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}

template <int D>
__global__ void tree_kernel(Shape s, TreeInputs in, int nodes) {
  constexpr int R = D / 32;
  const int lane = threadIdx.x & 31;
  const int h = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int b = blockIdx.y / nodes;
  const int node = blockIdx.y % nodes;
  if (h >= s.query_heads) return;
  const int kvh = h / (s.query_heads / s.kv_heads);
  const int qbase = ((b * nodes + node) * s.query_heads + h) * D;
  const float scale = rsqrtf(float(D));
  float q[R], acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    q[r] = __half2float(in.draft_q[qbase + r * 32 + lane]);
    acc[r] = 0.f;
  }
  float m = -CUDART_INF_F, l = 0.f;
  const int len = in.prefix.lengths[b];
  for (int t = 0; t < len; ++t) {
    const int physical = in.prefix.pages[b * s.max_pages + t / kPageSize];
    const int base = ((physical * kPageSize + t % kPageSize) * s.kv_heads + kvh) * D;
    float dot = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r)
      dot = fmaf(q[r], __half2float(in.prefix.k[base + r * 32 + lane]), dot);
    const float score = warp_sum_tree(dot) * scale;
    const float next_m = fmaxf(m, score);
    const float alpha = __expf(m - next_m);
    const float p = __expf(score - next_m);
    l = l * alpha + p;
#pragma unroll
    for (int r = 0; r < R; ++r)
      acc[r] = acc[r] * alpha + p * __half2float(in.prefix.v[base + r * 32 + lane]);
    m = next_m;
  }
  // Parent pointers are topological and bounded by `nodes`; host validation
  // ensures they form a tree. Walking leaf-to-root gives the same softmax.
  for (int cur = node; cur >= 0; cur = in.parent[b * nodes + cur]) {
    const int base = ((b * nodes + cur) * s.kv_heads + kvh) * D;
    float dot = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r)
      dot = fmaf(q[r], __half2float(in.draft_k[base + r * 32 + lane]), dot);
    const float score = warp_sum_tree(dot) * scale;
    const float next_m = fmaxf(m, score);
    const float alpha = __expf(m - next_m);
    const float p = __expf(score - next_m);
    l = l * alpha + p;
#pragma unroll
    for (int r = 0; r < R; ++r)
      acc[r] = acc[r] * alpha + p * __half2float(in.draft_v[base + r * 32 + lane]);
    m = next_m;
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
    in.output[qbase + r * 32 + lane] = acc[r] / l;
}

template <int D>
__global__ void commit_kernel(Shape s, const half* draft_k, const half* draft_v,
                              half* cache_k, half* cache_v,
                              const int* path_nodes, const int* dest_pages,
                              const int* dest_slots, int request, int nodes) {
  const int accepted = blockIdx.x;
  const int node = path_nodes[accepted];
  const int page = dest_pages[accepted];
  const int slot = dest_slots[accepted];
  const int elems = s.kv_heads * D;
  const int src = (request * nodes + node) * elems;
  const int dst = (page * kPageSize + slot) * elems;
  for (int x = threadIdx.x; x < elems; x += blockDim.x) {
    cache_k[dst + x] = draft_k[src + x];
    cache_v[dst + x] = draft_v[src + x];
  }
}

}  // namespace

cudaError_t tree_verify(Shape s, TreeInputs in, int nodes, cudaStream_t stream) {
  if (s.batch < 1 || s.query_heads < 1 || s.kv_heads < 1 ||
      s.query_heads % s.kv_heads || (s.dim != 64 && s.dim != 128) ||
      s.max_pages < 1 || nodes < 1 || nodes > 256 ||
      !in.prefix.k || !in.prefix.v || !in.prefix.pages || !in.prefix.lengths ||
      !in.draft_q || !in.draft_k || !in.draft_v || !in.parent || !in.output)
    return cudaErrorInvalidValue;
  const dim3 grid((s.query_heads + kWarpsPerBlock - 1) / kWarpsPerBlock,
                  s.batch * nodes);
  if (s.dim == 64) tree_kernel<64><<<grid, 128, 0, stream>>>(s, in, nodes);
  else tree_kernel<128><<<grid, 128, 0, stream>>>(s, in, nodes);
  return cudaGetLastError();
}

cudaError_t commit_path(Shape s, const half* draft_k, const half* draft_v,
                        half* cache_k, half* cache_v, const int* path_nodes,
                        const int* dest_pages, const int* dest_slots,
                        int request, int nodes, int accepted, cudaStream_t stream) {
  if (!draft_k || !draft_v || !cache_k || !cache_v || !path_nodes ||
      !dest_pages || !dest_slots || request < 0 || request >= s.batch ||
      nodes < 1 || accepted < 1 || accepted > nodes) return cudaErrorInvalidValue;
  if (s.dim != 64 && s.dim != 128) return cudaErrorInvalidValue;
  if (s.dim == 64)
    commit_kernel<64><<<accepted, 128, 0, stream>>>(s, draft_k, draft_v,
        cache_k, cache_v, path_nodes, dest_pages, dest_slots, request, nodes);
  else
    commit_kernel<128><<<accepted, 128, 0, stream>>>(s, draft_k, draft_v,
        cache_k, cache_v, path_nodes, dest_pages, dest_slots, request, nodes);
  return cudaGetLastError();
}

}  // namespace branchcraft
