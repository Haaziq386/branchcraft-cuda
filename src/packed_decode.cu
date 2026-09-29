#include "branchcraft.hpp"

#include <math_constants.h>

namespace branchcraft {
namespace {

__device__ __forceinline__ float warp_sum_packed(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    value += __shfl_down_sync(0xffffffff, value, delta);
  return __shfl_sync(0xffffffff, value, 0);
}

template <int D>
__global__ void packed_decode_kernel(Shape s, const half* q,
                                     const half* cache, const int* pages,
                                     const int* lengths, half* output,
                                     long long block_stride,
                                     long long head_stride,
                                     long long slot_stride, float scale) {
  constexpr int R = D / 32;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int batch = blockIdx.y;
  if (head >= s.query_heads) return;
  const int kv_head = head / (s.query_heads / s.kv_heads);
  const int qbase = (batch * s.query_heads + head) * D;
  float query[R], weighted[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    query[r] = __half2float(q[qbase + r * 32 + lane]);
    weighted[r] = 0.f;
  }
  float max_score = -CUDART_INF_F;
  float denom = 0.f;
  for (int token = 0; token < lengths[batch]; ++token) {
    const int physical = pages[batch * s.max_pages + token / kPageSize];
    const long long offset = physical * block_stride + kv_head * head_stride
                           + (token % kPageSize) * slot_stride;
    float dot = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r)
      dot = fmaf(query[r], __half2float(cache[offset + r * 32 + lane]), dot);
    const float score = warp_sum_packed(dot) * scale;
    const float new_max = fmaxf(max_score, score);
    const float alpha = __expf(max_score - new_max);
    const float probability = __expf(score - new_max);
    denom = denom * alpha + probability;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = __half2float(cache[offset + D + r * 32 + lane]);
      weighted[r] = weighted[r] * alpha + probability * v;
    }
    max_score = new_max;
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
    output[qbase + r * 32 + lane] = __float2half(denom > 0.f ? weighted[r] / denom : 0.f);
}

}  // namespace

cudaError_t packed_decode(Shape s, const half* q, const half* packed_kv,
                          const int* pages, const int* lengths, half* output,
                          long long block_stride, long long head_stride,
                          long long slot_stride, float scale,
                          cudaStream_t stream) {
  if (s.batch < 1 || s.query_heads < 1 || s.kv_heads < 1 ||
      s.query_heads % s.kv_heads || (s.dim != 64 && s.dim != 128) ||
      s.max_pages < 1 || s.physical_pages < 1 || !q || !packed_kv ||
      !pages || !lengths || !output || block_stride < 1 || head_stride < 1 ||
      slot_stride < 1 || !(scale > 0.f)) return cudaErrorInvalidValue;
  const dim3 grid((s.query_heads + kWarpsPerBlock - 1) / kWarpsPerBlock, s.batch);
  if (s.dim == 64)
    packed_decode_kernel<64><<<grid, 128, 0, stream>>>(s, q, packed_kv,
        pages, lengths, output, block_stride, head_stride, slot_stride, scale);
  else
    packed_decode_kernel<128><<<grid, 128, 0, stream>>>(s, q, packed_kv,
        pages, lengths, output, block_stride, head_stride, slot_stride, scale);
  return cudaGetLastError();
}

}  // namespace branchcraft
