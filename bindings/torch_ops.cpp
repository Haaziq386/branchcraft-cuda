#include "branchcraft.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <string>

namespace {

void check_cuda_contiguous(const at::Tensor& tensor, const char* name,
                           at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
}

void check_launch(cudaError_t err, const char* op) {
  TORCH_CHECK(err == cudaSuccess, op, ": ", cudaGetErrorString(err));
}

at::Tensor tree_verify_cuda(const at::Tensor& q, const at::Tensor& prefix_k,
                            const at::Tensor& prefix_v, const at::Tensor& pages,
                            const at::Tensor& lengths, const at::Tensor& draft_k,
                            const at::Tensor& draft_v, const at::Tensor& parent) {
  check_cuda_contiguous(q, "q", at::kHalf);
  check_cuda_contiguous(prefix_k, "prefix_k", at::kHalf);
  check_cuda_contiguous(prefix_v, "prefix_v", at::kHalf);
  check_cuda_contiguous(draft_k, "draft_k", at::kHalf);
  check_cuda_contiguous(draft_v, "draft_v", at::kHalf);
  check_cuda_contiguous(pages, "pages", at::kInt);
  check_cuda_contiguous(lengths, "lengths", at::kInt);
  check_cuda_contiguous(parent, "parent", at::kInt);
  TORCH_CHECK(q.dim() == 4 && prefix_k.dim() == 4 && prefix_v.dim() == 4 &&
              draft_k.dim() == 4 && draft_v.dim() == 4 && pages.dim() == 2 &&
              lengths.dim() == 1 && parent.dim() == 2, "invalid tree tensor rank");
  const int b = q.size(0), nodes = q.size(1), hq = q.size(2), d = q.size(3);
  const int hkv = prefix_k.size(2);
  TORCH_CHECK(b > 0 && nodes > 0 && nodes <= 256 && hq > 0 && hkv > 0 &&
              hq % hkv == 0 && (d == 64 || d == 128), "unsupported tree shape");
  TORCH_CHECK(prefix_k.size(1) == branchcraft::kPageSize &&
              prefix_k.size(3) == d && prefix_v.sizes() == prefix_k.sizes(),
              "prefix K/V shape mismatch");
  TORCH_CHECK(draft_k.sizes() == at::IntArrayRef({b, nodes, hkv, d}) &&
              draft_v.sizes() == draft_k.sizes(), "draft K/V shape mismatch");
  TORCH_CHECK(pages.size(0) == b && lengths.size(0) == b &&
              parent.sizes() == at::IntArrayRef({b, nodes}),
              "tree metadata shape mismatch");
  TORCH_CHECK(q.device() == prefix_k.device() && q.device() == prefix_v.device() &&
              q.device() == draft_k.device() && q.device() == draft_v.device() &&
              q.device() == pages.device() && q.device() == lengths.device() &&
              q.device() == parent.device(), "all tensors must share a device");
  auto output = at::empty(q.sizes(), q.options().dtype(at::kFloat));
  branchcraft::Shape shape{b, hq, hkv, d, int(pages.size(1)), int(prefix_k.size(0))};
  branchcraft::Inputs prefix{nullptr,
      reinterpret_cast<const half*>(prefix_k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(prefix_v.data_ptr<at::Half>()),
      pages.data_ptr<int>(), lengths.data_ptr<int>(), nullptr};
  branchcraft::TreeInputs input{prefix,
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(draft_k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(draft_v.data_ptr<at::Half>()),
      parent.data_ptr<int>(), output.data_ptr<float>()};
  check_launch(branchcraft::tree_verify(shape, input, nodes,
      at::cuda::getCurrentCUDAStream().stream()), "tree_verify");
  return output;
}

at::Tensor packed_decode_out_cuda(const at::Tensor& q,
                                  const at::Tensor& packed_kv,
                                  const at::Tensor& pages,
                                  const at::Tensor& lengths,
                                  at::Tensor& output, double scale) {
  check_cuda_contiguous(q, "q", at::kHalf);
  check_cuda_contiguous(pages, "pages", at::kInt);
  check_cuda_contiguous(lengths, "lengths", at::kInt);
  check_cuda_contiguous(output, "output", at::kHalf);
  TORCH_CHECK(packed_kv.is_cuda() && packed_kv.scalar_type() == at::kHalf &&
              packed_kv.dim() == 4 && packed_kv.stride(3) == 1,
              "packed_kv must be a CUDA FP16 tensor with contiguous content dim");
  TORCH_CHECK(q.dim() == 3 && pages.dim() == 2 && lengths.dim() == 1,
              "invalid packed decode tensor rank");
  const int b = q.size(0), hq = q.size(1), d = q.size(2);
  const int hkv = packed_kv.size(1);
  TORCH_CHECK(b > 0 && hq > 0 && hkv > 0 && hq % hkv == 0 &&
              (d == 64 || d == 128), "unsupported packed decode shape");
  TORCH_CHECK(packed_kv.size(2) == branchcraft::kPageSize &&
              packed_kv.size(3) == 2 * d, "packed KV shape mismatch");
  TORCH_CHECK(pages.size(0) == b && lengths.size(0) == b &&
              output.numel() == q.numel(), "packed decode metadata/output mismatch");
  TORCH_CHECK(q.device() == packed_kv.device() && q.device() == pages.device() &&
              q.device() == lengths.device() && q.device() == output.device(),
              "all tensors must share a device");
  branchcraft::Shape shape{b, hq, hkv, d, int(pages.size(1)), int(packed_kv.size(0))};
  check_launch(branchcraft::packed_decode(shape,
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(packed_kv.data_ptr<at::Half>()),
      pages.data_ptr<int>(), lengths.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      packed_kv.stride(0), packed_kv.stride(1), packed_kv.stride(2),
      float(scale), at::cuda::getCurrentCUDAStream().stream()),
      "packed_decode_out");
  return output;
}

void commit_path_cuda(at::Tensor& cache_k, at::Tensor& cache_v,
                      const at::Tensor& draft_k, const at::Tensor& draft_v,
                      const at::Tensor& path_nodes, const at::Tensor& dest_pages,
                      const at::Tensor& dest_slots, int64_t request) {
  check_cuda_contiguous(cache_k, "cache_k", at::kHalf);
  check_cuda_contiguous(cache_v, "cache_v", at::kHalf);
  check_cuda_contiguous(draft_k, "draft_k", at::kHalf);
  check_cuda_contiguous(draft_v, "draft_v", at::kHalf);
  check_cuda_contiguous(path_nodes, "path_nodes", at::kInt);
  check_cuda_contiguous(dest_pages, "dest_pages", at::kInt);
  check_cuda_contiguous(dest_slots, "dest_slots", at::kInt);
  TORCH_CHECK(cache_k.dim() == 4 && cache_v.sizes() == cache_k.sizes() &&
              draft_k.dim() == 4 && draft_v.sizes() == draft_k.sizes() &&
              path_nodes.dim() == 1 && dest_pages.sizes() == path_nodes.sizes() &&
              dest_slots.sizes() == path_nodes.sizes(), "commit tensor shape mismatch");
  const int b = draft_k.size(0), nodes = draft_k.size(1), hkv = draft_k.size(2);
  const int d = draft_k.size(3);
  TORCH_CHECK(request >= 0 && request < b && path_nodes.numel() > 0 &&
              path_nodes.numel() <= nodes && cache_k.size(1) == branchcraft::kPageSize &&
              cache_k.size(2) == hkv && cache_k.size(3) == d,
              "invalid commit destination");
  TORCH_CHECK(cache_k.device() == draft_k.device() && cache_k.device() == cache_v.device() &&
              cache_k.device() == draft_v.device() && cache_k.device() == path_nodes.device() &&
              cache_k.device() == dest_pages.device() && cache_k.device() == dest_slots.device(),
              "all tensors must share a device");
  branchcraft::Shape shape{b, hkv, hkv, d, 1, int(cache_k.size(0))};
  check_launch(branchcraft::commit_path(shape,
      reinterpret_cast<const half*>(draft_k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(draft_v.data_ptr<at::Half>()),
      reinterpret_cast<half*>(cache_k.data_ptr<at::Half>()),
      reinterpret_cast<half*>(cache_v.data_ptr<at::Half>()),
      path_nodes.data_ptr<int>(), dest_pages.data_ptr<int>(),
      dest_slots.data_ptr<int>(), int(request), nodes, int(path_nodes.numel()),
      at::cuda::getCurrentCUDAStream().stream()), "commit_path_");
}

}  // namespace

TORCH_LIBRARY(branchcraft, m) {
  m.def("tree_verify(Tensor q, Tensor prefix_k, Tensor prefix_v, Tensor pages, Tensor lengths, Tensor draft_k, Tensor draft_v, Tensor parent) -> Tensor");
  m.def("packed_decode_out(Tensor q, Tensor packed_kv, Tensor pages, Tensor lengths, Tensor(a!) output, float scale) -> Tensor(a!)");
  m.def("commit_path_(Tensor(a!) cache_k, Tensor(b!) cache_v, Tensor draft_k, Tensor draft_v, Tensor path_nodes, Tensor dest_pages, Tensor dest_slots, int request) -> ()");
}

TORCH_LIBRARY_IMPL(branchcraft, CUDA, m) {
  m.impl("tree_verify", tree_verify_cuda);
  m.impl("packed_decode_out", packed_decode_out_cuda);
  m.impl("commit_path_", commit_path_cuda);
}
