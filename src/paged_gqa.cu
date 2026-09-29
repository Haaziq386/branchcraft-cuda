#include "branchcraft.hpp"

#include <cmath>
#include <math_constants.h>

namespace branchcraft {
namespace {

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    x += __shfl_down_sync(0xffffffff, x, delta);
  return __shfl_sync(0xffffffff, x, 0);
}

template <int D>
__global__ void decode_kernel(Shape s, Inputs in, Scratch scratch, int splits) {
  constexpr int R = D / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int h = blockIdx.x * kWarpsPerBlock + warp;
  const int b = blockIdx.y;
  const int part = blockIdx.z;
  if (h >= s.query_heads) return;

  const int kvh = h / (s.query_heads / s.kv_heads);
  const int len = in.lengths[b];
  const int chunk = (len + splits - 1) / splits;
  const int begin = part * chunk;
  const int end = min(begin + chunk, len);
  const int qbase = (b * s.query_heads + h) * D;
  const float scale = rsqrtf(float(D));
  float q[R], acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    q[r] = __half2float(in.q[qbase + r * 32 + lane]);
    acc[r] = 0.f;
  }

  float m = -CUDART_INF_F;
  float l = 0.f;
  for (int t = begin; t < end; ++t) {
    const int physical = in.pages[b * s.max_pages + t / kPageSize];
    const int base = ((physical * kPageSize + t % kPageSize) * s.kv_heads + kvh) * D;
    float dot = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r)
      dot = fmaf(q[r], __half2float(in.k[base + r * 32 + lane]), dot);
    const float score = warp_sum(dot) * scale;
    const float next_m = fmaxf(m, score);
    const float alpha = __expf(m - next_m);
    const float p = __expf(score - next_m);
    l = l * alpha + p;
#pragma unroll
    for (int r = 0; r < R; ++r)
      acc[r] = acc[r] * alpha + p * __half2float(in.v[base + r * 32 + lane]);
    m = next_m;
  }

  if (splits == 1) {
#pragma unroll
    for (int r = 0; r < R; ++r)
      in.output[qbase + r * 32 + lane] = l > 0.f ? acc[r] / l : 0.f;
  } else {
    const int row = (part * s.batch + b) * s.query_heads + h;
    if (lane == 0) {
      scratch.partial_ml[row * 2] = m;
      scratch.partial_ml[row * 2 + 1] = l;
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
      scratch.partial_a[row * D + r * 32 + lane] = acc[r];
  }
}

template <int D>
__global__ void merge_kernel(Shape s, Inputs in, Scratch scratch, int splits) {
  constexpr int R = D / 32;
  const int lane = threadIdx.x & 31;
  const int h = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int b = blockIdx.y;
  if (h >= s.query_heads) return;
  float max_m = -CUDART_INF_F;
  for (int part = 0; part < splits; ++part) {
    const int row = (part * s.batch + b) * s.query_heads + h;
    max_m = fmaxf(max_m, scratch.partial_ml[row * 2]);
  }
  const int out = (b * s.query_heads + h) * D;
  if (max_m == -CUDART_INF_F) {
#pragma unroll
    for (int r = 0; r < R; ++r) in.output[out + r * 32 + lane] = 0.f;
    return;
  }
  float denom = 0.f, acc[R] = {};
  for (int part = 0; part < splits; ++part) {
    const int row = (part * s.batch + b) * s.query_heads + h;
    const float weight = __expf(scratch.partial_ml[row * 2] - max_m);
    denom += weight * scratch.partial_ml[row * 2 + 1];
#pragma unroll
    for (int r = 0; r < R; ++r)
      acc[r] += weight * scratch.partial_a[row * D + r * 32 + lane];
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
    in.output[out + r * 32 + lane] = acc[r] / denom;
}

template <int D>
cudaError_t dispatch(Shape shape, Inputs in, Scratch scratch, int splits,
                     cudaStream_t stream) {
  const dim3 grid((shape.query_heads + kWarpsPerBlock - 1) / kWarpsPerBlock,
                  shape.batch, splits);
  decode_kernel<D><<<grid, 128, 0, stream>>>(shape, in, scratch, splits);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess || splits == 1) return err;
  merge_kernel<D><<<dim3(grid.x, grid.y), 128, 0, stream>>>(shape, in, scratch, splits);
  return cudaGetLastError();
}

}  // namespace

cudaError_t decode(Shape s, Inputs in, Scratch scratch, int splits,
                   cudaStream_t stream) {
  if (s.batch < 1 || s.query_heads < 1 || s.kv_heads < 1 ||
      s.query_heads % s.kv_heads || (s.dim != 64 && s.dim != 128) ||
      s.max_pages < 1 || s.physical_pages < 1 || splits < 1 || splits > 16 ||
      !in.q || !in.k || !in.v || !in.pages || !in.lengths || !in.output ||
      (splits > 1 && (!scratch.partial_a || !scratch.partial_ml)))
    return cudaErrorInvalidValue;
  return s.dim == 64 ? dispatch<64>(s, in, scratch, splits, stream)
                     : dispatch<128>(s, in, scratch, splits, stream);
}

int auto_splits(Shape s, int max_length, int sm_count) {
  if (max_length < 512) return 1;
  const int base_ctas = s.batch * ((s.query_heads + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const int wanted = (2 * sm_count + base_ctas - 1) / base_ctas;
  int splits = 1;
  while (splits < wanted && splits < 16) splits *= 2;
  while (splits > 1 && max_length / splits < 128) splits /= 2;
  return splits;
}

}  // namespace branchcraft
