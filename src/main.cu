#include "branchcraft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using branchcraft::Inputs;
using branchcraft::Scratch;
using branchcraft::Shape;
using branchcraft::kPageSize;

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

template <typename T> struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) : count(n) { CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  void upload(const std::vector<T>& v) {
    if (v.size() != count) throw std::runtime_error("upload size mismatch");
    CHECK(cudaMemcpy(ptr, v.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> v(count);
    CHECK(cudaMemcpy(v.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

struct Fixture {
  Shape shape;
  std::vector<half> q, k, v;
  std::vector<int> pages, lengths;
};

Fixture make_fixture(int batch, int hq, int hkv, int dim,
                     std::vector<int> lengths, uint32_t seed = 17) {
  if (int(lengths.size()) != batch) throw std::runtime_error("length count mismatch");
  const int max_len = *std::max_element(lengths.begin(), lengths.end());
  const int max_pages = std::max(1, (max_len + kPageSize - 1) / kPageSize);
  int used_pages = 0;
  for (int n : lengths) {
    if (n < 0) throw std::runtime_error("negative length");
    used_pages += (n + kPageSize - 1) / kPageSize;
  }
  const int physical_pages = used_pages + 7;
  Fixture f{{batch, hq, hkv, dim, max_pages, physical_pages}};
  f.lengths = std::move(lengths);
  f.pages.assign(batch * max_pages, -1);
  std::mt19937 gen(seed);
  std::vector<int> ids(physical_pages);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), gen);
  int cursor = 0;
  for (int b = 0; b < batch; ++b)
    for (int p = 0; p < (f.lengths[b] + kPageSize - 1) / kPageSize; ++p)
      f.pages[b * max_pages + p] = ids[cursor++];
  std::normal_distribution<float> dist(0.f, 0.5f);
  auto fill = [&](std::vector<half>& out) {
    for (half& x : out) x = __float2half(dist(gen));
  };
  f.q.resize(size_t(batch) * hq * dim);
  f.k.resize(size_t(physical_pages) * kPageSize * hkv * dim);
  f.v.resize(f.k.size());
  fill(f.q); fill(f.k); fill(f.v);
  return f;
}

std::vector<float> cpu_reference(const Fixture& f) {
  const Shape s = f.shape;
  std::vector<float> out(size_t(s.batch) * s.query_heads * s.dim, 0.f);
  std::vector<float> scores;
  for (int b = 0; b < s.batch; ++b) {
    for (int h = 0; h < s.query_heads; ++h) {
      const int kvh = h / (s.query_heads / s.kv_heads);
      const int qbase = (b * s.query_heads + h) * s.dim;
      scores.resize(f.lengths[b]);
      float max_score = -INFINITY;
      for (int t = 0; t < f.lengths[b]; ++t) {
        const int physical = f.pages[b * s.max_pages + t / kPageSize];
        const int base = ((physical * kPageSize + t % kPageSize) * s.kv_heads + kvh) * s.dim;
        double dot = 0;
        for (int d = 0; d < s.dim; ++d)
          dot += double(__half2float(f.q[qbase + d])) * __half2float(f.k[base + d]);
        scores[t] = float(dot / std::sqrt(double(s.dim)));
        max_score = std::max(max_score, scores[t]);
      }
      double denom = 0;
      for (int t = 0; t < f.lengths[b]; ++t) {
        const double p = std::exp(double(scores[t] - max_score));
        denom += p;
        const int physical = f.pages[b * s.max_pages + t / kPageSize];
        const int base = ((physical * kPageSize + t % kPageSize) * s.kv_heads + kvh) * s.dim;
        for (int d = 0; d < s.dim; ++d)
          out[qbase + d] += float(p * __half2float(f.v[base + d]));
      }
      if (denom) for (int d = 0; d < s.dim; ++d) out[qbase + d] /= float(denom);
    }
  }
  return out;
}

struct GpuFixture {
  DeviceBuffer<half> q, k, v;
  DeviceBuffer<int> pages, lengths;
  DeviceBuffer<float> output, partial_a, partial_ml;
  Shape shape;
  explicit GpuFixture(const Fixture& f)
      : q(f.q.size()), k(f.k.size()), v(f.v.size()), pages(f.pages.size()),
        lengths(f.lengths.size()),
        output(size_t(f.shape.batch) * f.shape.query_heads * f.shape.dim),
        partial_a(16 * output.count),
        partial_ml(16 * size_t(f.shape.batch) * f.shape.query_heads * 2),
        shape(f.shape) {
    q.upload(f.q); k.upload(f.k); v.upload(f.v);
    pages.upload(f.pages); lengths.upload(f.lengths);
  }
  Inputs inputs() { return {q.ptr, k.ptr, v.ptr, pages.ptr, lengths.ptr, output.ptr}; }
  Scratch scratch() { return {partial_a.ptr, partial_ml.ptr}; }
  void run(int splits) {
    CHECK(branchcraft::decode(shape, inputs(), scratch(), splits));
    CHECK(cudaDeviceSynchronize());
  }
};

struct TreeFixture {
  Fixture prefix;
  int nodes;
  std::vector<half> draft_q, draft_k, draft_v;
  std::vector<int> parent;
};

TreeFixture make_tree_fixture(int batch, int prefix_len, int nodes, int dim,
                              uint32_t seed = 31) {
  if (nodes < 1 || nodes > 15) throw std::runtime_error("fixture nodes out of range");
  Fixture base = make_fixture(1, 32, 8, dim, {prefix_len}, seed);
  TreeFixture tree{};
  tree.nodes = nodes;
  Fixture& f = tree.prefix;
  f.shape = {batch, 32, 8, dim, base.shape.max_pages + 2,
             base.shape.physical_pages + batch * 2};
  f.lengths.assign(batch, prefix_len);
  f.pages.assign(size_t(batch) * f.shape.max_pages, -1);
  for (int b = 0; b < batch; ++b)
    for (int p = 0; p < (prefix_len + kPageSize - 1) / kPageSize; ++p)
      f.pages[b * f.shape.max_pages + p] = base.pages[p];
  f.q.resize(size_t(batch) * 32 * dim);
  for (int b = 0; b < batch; ++b)
    std::copy(base.q.begin(), base.q.end(), f.q.begin() + size_t(b) * 32 * dim);
  f.k = std::move(base.k); f.v = std::move(base.v);
  f.k.resize(size_t(f.shape.physical_pages) * kPageSize * 8 * dim);
  f.v.resize(f.k.size());
  std::mt19937 gen(seed + 99);
  std::normal_distribution<float> dist(0.f, .5f);
  tree.draft_q.resize(size_t(batch) * nodes * 32 * dim);
  tree.draft_k.resize(size_t(batch) * nodes * 8 * dim);
  tree.draft_v.resize(tree.draft_k.size());
  auto fill = [&](std::vector<half>& v) {
    for (half& x : v) x = __float2half(dist(gen));
  };
  fill(tree.draft_q); fill(tree.draft_k); fill(tree.draft_v);
  tree.parent.resize(batch * nodes);
  for (int b = 0; b < batch; ++b)
    for (int n = 0; n < nodes; ++n)
      tree.parent[b * nodes + n] = n == 0 ? -1 : (n - 1) / 2;
  return tree;
}

struct TreeGpu {
  GpuFixture prefix;
  DeviceBuffer<half> draft_q, draft_k, draft_v;
  DeviceBuffer<int> parent;
  DeviceBuffer<float> output;
  int nodes;
  explicit TreeGpu(const TreeFixture& f)
      : prefix(f.prefix), draft_q(f.draft_q.size()), draft_k(f.draft_k.size()),
        draft_v(f.draft_v.size()), parent(f.parent.size()),
        output(f.draft_q.size()), nodes(f.nodes) {
    draft_q.upload(f.draft_q); draft_k.upload(f.draft_k);
    draft_v.upload(f.draft_v); parent.upload(f.parent);
  }
  branchcraft::TreeInputs inputs() {
    return {prefix.inputs(), draft_q.ptr, draft_k.ptr, draft_v.ptr,
            parent.ptr, output.ptr};
  }
  void run() {
    CHECK(branchcraft::tree_verify(prefix.shape, inputs(), nodes));
    CHECK(cudaDeviceSynchronize());
  }
};

std::vector<float> cpu_tree_reference(const TreeFixture& f) {
  const Shape s = f.prefix.shape;
  std::vector<float> out(f.draft_q.size(), 0.f);
  for (int b = 0; b < s.batch; ++b) for (int node = 0; node < f.nodes; ++node)
    for (int h = 0; h < s.query_heads; ++h) {
      const int kvh = h / (s.query_heads / s.kv_heads);
      const int qbase = ((b * f.nodes + node) * s.query_heads + h) * s.dim;
      struct Token { int base; bool draft; };
      std::vector<Token> tokens;
      for (int t = 0; t < f.prefix.lengths[b]; ++t) {
        const int phys = f.prefix.pages[b * s.max_pages + t / kPageSize];
        tokens.push_back({((phys * kPageSize + t % kPageSize) * s.kv_heads + kvh) * s.dim, false});
      }
      for (int cur = node; cur >= 0; cur = f.parent[b * f.nodes + cur])
        tokens.push_back({((b * f.nodes + cur) * s.kv_heads + kvh) * s.dim, true});
      std::vector<double> scores(tokens.size());
      double maximum = -INFINITY;
      for (size_t t = 0; t < tokens.size(); ++t) {
        double dot = 0;
        const auto& k = tokens[t].draft ? f.draft_k : f.prefix.k;
        for (int d = 0; d < s.dim; ++d)
          dot += double(__half2float(f.draft_q[qbase + d])) * __half2float(k[tokens[t].base + d]);
        scores[t] = dot / std::sqrt(double(s.dim));
        maximum = std::max(maximum, scores[t]);
      }
      double denom = 0;
      for (size_t t = 0; t < tokens.size(); ++t) {
        const double p = std::exp(scores[t] - maximum);
        denom += p;
        const auto& v = tokens[t].draft ? f.draft_v : f.prefix.v;
        for (int d = 0; d < s.dim; ++d)
          out[qbase + d] += float(p * __half2float(v[tokens[t].base + d]));
      }
      for (int d = 0; d < s.dim; ++d) out[qbase + d] /= float(denom);
    }
  return out;
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
  float worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
    worst = std::max(worst, std::fabs(a[i] - b[i]));
  }
  return worst;
}

void run_tree_tests() {
  int checks = 0;
  for (const auto& spec : std::vector<std::pair<int, int>>{{0, 64}, {17, 64}, {257, 128}}) {
    TreeFixture f = make_tree_fixture(2, spec.first, 7, spec.second);
    TreeGpu gpu(f);
    gpu.run();
    const float error = max_abs_error(cpu_tree_reference(f), gpu.output.download());
    if (error > 3e-4f)
      throw std::runtime_error("tree CPU-oracle mismatch, max error=" + std::to_string(error));
    ++checks;
  }

  // A transaction starts with two requests sharing a partial final page.
  // Clone that page, write only the accepted root->leaf path, verify against
  // tree attention, then roll the page-table metadata back.
  TreeFixture f = make_tree_fixture(2, 17, 7, 128, 49);
  TreeGpu gpu(f);
  gpu.run();
  const auto tree_out = gpu.output.download();
  gpu.prefix.run(1);
  const auto before = gpu.prefix.output.download();
  const int new_page = f.prefix.shape.physical_pages - 1;
  const int old_page = f.prefix.pages[1];
  const size_t page_elems = size_t(kPageSize) * f.prefix.shape.kv_heads * f.prefix.shape.dim;
  CHECK(cudaMemcpy(gpu.prefix.k.ptr + size_t(new_page) * page_elems,
                   gpu.prefix.k.ptr + size_t(old_page) * page_elems,
                   page_elems * sizeof(half), cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(gpu.prefix.v.ptr + size_t(new_page) * page_elems,
                   gpu.prefix.v.ptr + size_t(old_page) * page_elems,
                   page_elems * sizeof(half), cudaMemcpyDeviceToDevice));
  DeviceBuffer<int> path(3), dest_pages(3), dest_slots(3);
  path.upload({0, 1, 3});
  dest_pages.upload({new_page, new_page, new_page});
  dest_slots.upload({1, 2, 3});
  CHECK(branchcraft::commit_path(f.prefix.shape, gpu.draft_k.ptr, gpu.draft_v.ptr,
      gpu.prefix.k.ptr, gpu.prefix.v.ptr, path.ptr, dest_pages.ptr, dest_slots.ptr,
      0, f.nodes, 3));
  CHECK(cudaDeviceSynchronize());
  auto committed_pages = f.prefix.pages;
  auto committed_lengths = f.prefix.lengths;
  committed_pages[1] = new_page;
  committed_lengths[0] += 3;
  gpu.prefix.pages.upload(committed_pages);
  gpu.prefix.lengths.upload(committed_lengths);
  const size_t query_elems = size_t(f.prefix.shape.query_heads) * f.prefix.shape.dim;
  CHECK(cudaMemcpy(gpu.prefix.q.ptr, gpu.draft_q.ptr + 3 * query_elems,
                   query_elems * sizeof(half), cudaMemcpyDeviceToDevice));
  gpu.prefix.run(1);
  const auto after = gpu.prefix.output.download();
  float accepted_error = 0, sibling_error = 0;
  for (size_t i = 0; i < query_elems; ++i) {
    accepted_error = std::max(accepted_error, std::fabs(after[i] - tree_out[3 * query_elems + i]));
    sibling_error = std::max(sibling_error, std::fabs(after[query_elems + i] - before[query_elems + i]));
  }
  if (accepted_error > 3e-4f || sibling_error > 1e-7f)
    throw std::runtime_error("commit changed accepted path or shared sibling");
  gpu.prefix.pages.upload(f.prefix.pages);
  gpu.prefix.lengths.upload(f.prefix.lengths);
  gpu.prefix.q.upload(f.prefix.q);
  gpu.prefix.run(1);
  const float rollback_error = max_abs_error(before, gpu.prefix.output.download());
  if (rollback_error > 1e-7f) throw std::runtime_error("rollback failed");
  std::cout << "PASS " << checks << " tree CPU-oracle cases plus shared-page"
            << " commit, sibling isolation, and rollback.\n";
}

void run_tests() {
  struct Case { int b, hq, hkv, d; std::vector<int> lens; };
  const std::vector<Case> cases = {
    {1, 8, 2, 64, {0}}, {1, 8, 2, 64, {1}},
    {1, 8, 2, 64, {16}}, {1, 8, 2, 64, {17}},
    {3, 10, 2, 64, {0, 37, 91}},
    {3, 12, 3, 128, {7, 128, 257}},
    {2, 32, 8, 128, {513, 3}},
  };
  int checks = 0;
  for (const auto& c : cases) {
    Fixture f = make_fixture(c.b, c.hq, c.hkv, c.d, c.lens);
    const auto expected = cpu_reference(f);
    GpuFixture gpu(f);
    for (int splits : {1, 2, 4, 8, 16}) {
      gpu.run(splits);
      const float error = max_abs_error(expected, gpu.output.download());
      if (error > 3e-4f) {
        throw std::runtime_error("FAIL B=" + std::to_string(c.b) + " D=" +
            std::to_string(c.d) + " splits=" + std::to_string(splits) +
            " max_abs_error=" + std::to_string(error));
      }
      ++checks;
    }
  }
  // Public API must reject unsupported dimensions and missing split scratch.
  Fixture f = make_fixture(1, 8, 2, 64, {17});
  GpuFixture gpu(f);
  Shape bad = f.shape; bad.dim = 96;
  if (branchcraft::decode(bad, gpu.inputs(), gpu.scratch(), 1) != cudaErrorInvalidValue ||
      branchcraft::decode(f.shape, gpu.inputs(), {nullptr, nullptr}, 2) != cudaErrorInvalidValue)
    throw std::runtime_error("invalid API arguments were accepted");
  std::cout << "PASS " << checks << " CPU-oracle checks across empty, page-boundary,"
            << " ragged, GQA, D64/D128, and split-KV cases.\n";
}

double median_latency_us(GpuFixture& gpu, int splits, int repeats = 20) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i) CHECK(branchcraft::decode(gpu.shape, gpu.inputs(), gpu.scratch(), splits));
  CHECK(cudaDeviceSynchronize());
  std::vector<double> times;
  for (int trial = 0; trial < 7; ++trial) {
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i)
      CHECK(branchcraft::decode(gpu.shape, gpu.inputs(), gpu.scratch(), splits));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(double(ms) * 1000.0 / repeats);
  }
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

Fixture make_expanded_fixture(const TreeFixture& tree) {
  const Shape ps = tree.prefix.shape;
  const int sequences = ps.batch * tree.nodes;
  int max_len = 0, total_pages = 0;
  std::vector<int> lengths(sequences);
  std::vector<std::vector<int>> paths(sequences);
  for (int b = 0; b < ps.batch; ++b) for (int node = 0; node < tree.nodes; ++node) {
    const int seq = b * tree.nodes + node;
    for (int cur = node; cur >= 0; cur = tree.parent[b * tree.nodes + cur])
      paths[seq].push_back(cur);
    std::reverse(paths[seq].begin(), paths[seq].end());
    lengths[seq] = tree.prefix.lengths[b] + int(paths[seq].size());
    max_len = std::max(max_len, lengths[seq]);
    total_pages += (lengths[seq] + kPageSize - 1) / kPageSize;
  }
  const int max_pages = (max_len + kPageSize - 1) / kPageSize;
  Fixture flat{{sequences, ps.query_heads, ps.kv_heads, ps.dim,
                max_pages, total_pages + 7}};
  flat.lengths = lengths;
  flat.pages.assign(size_t(sequences) * max_pages, -1);
  flat.q = tree.draft_q;
  const size_t token_elems = size_t(ps.kv_heads) * ps.dim;
  flat.k.resize(size_t(flat.shape.physical_pages) * kPageSize * token_elems);
  flat.v.resize(flat.k.size());
  int next_page = 0;
  for (int seq = 0; seq < sequences; ++seq) {
    const int b = seq / tree.nodes;
    for (int p = 0; p < (lengths[seq] + kPageSize - 1) / kPageSize; ++p)
      flat.pages[seq * max_pages + p] = next_page++;
    for (int t = 0; t < lengths[seq]; ++t) {
      const int dst_page = flat.pages[seq * max_pages + t / kPageSize];
      const size_t dst = (size_t(dst_page) * kPageSize + t % kPageSize) * token_elems;
      const bool is_prefix = t < tree.prefix.lengths[b];
      size_t src;
      if (is_prefix) {
        const int page = tree.prefix.pages[b * ps.max_pages + t / kPageSize];
        src = (size_t(page) * kPageSize + t % kPageSize) * token_elems;
      } else {
        const int ancestor = paths[seq][t - tree.prefix.lengths[b]];
        src = (size_t(b) * tree.nodes + ancestor) * token_elems;
      }
      const auto& k = is_prefix ? tree.prefix.k : tree.draft_k;
      const auto& v = is_prefix ? tree.prefix.v : tree.draft_v;
      std::copy_n(k.begin() + src, token_elems, flat.k.begin() + dst);
      std::copy_n(v.begin() + src, token_elems, flat.v.begin() + dst);
    }
  }
  return flat;
}

double median_tree_latency_us(TreeGpu& gpu, int repeats = 20) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 5; ++i)
    CHECK(branchcraft::tree_verify(gpu.prefix.shape, gpu.inputs(), gpu.nodes));
  CHECK(cudaDeviceSynchronize());
  std::vector<double> times;
  for (int trial = 0; trial < 7; ++trial) {
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i)
      CHECK(branchcraft::tree_verify(gpu.prefix.shape, gpu.inputs(), gpu.nodes));
    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(double(ms) * 1000. / repeats);
  }
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

void run_tree_bench(const std::string& output_path, bool quick) {
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("cannot open output: " + output_path);
  out << "gpu,sm,cuda_runtime,requests,tree_nodes,prefix_tokens,query_heads,"
         "kv_heads,head_dim,page_size,tree_us,expanded_us,latency_ratio,"
         "max_abs_diff,shared_kv_mib,expanded_kv_mib,memory_saving_pct\n";
  const auto batches = quick ? std::vector<int>{1, 4} : std::vector<int>{1, 4, 8};
  const auto contexts = quick ? std::vector<int>{256, 1024} : std::vector<int>{256, 1024, 2048};
  for (int b : batches) for (int context : contexts) {
    TreeFixture f = make_tree_fixture(b, context, 7, 128, 91 + b + context);
    Fixture expanded = make_expanded_fixture(f);
    TreeGpu tree(f);
    GpuFixture flat(expanded);
    tree.run(); flat.run(1);
    const float diff = max_abs_error(tree.output.download(), flat.output.download());
    if (diff > 3e-4f) throw std::runtime_error("expanded-cache output mismatch");
    const double tree_us = median_tree_latency_us(tree);
    const double expanded_us = median_latency_us(flat, 1);
    const double bytes_per_token = double(f.prefix.shape.kv_heads) *
        f.prefix.shape.dim * sizeof(half) * 2;
    const double shared_tokens = double((context + kPageSize - 1) / kPageSize * kPageSize)
        + b * f.nodes;
    int expanded_slots = 0;
    for (int len : expanded.lengths)
      expanded_slots += (len + kPageSize - 1) / kPageSize * kPageSize;
    const double shared_mib = shared_tokens * bytes_per_token / (1024. * 1024.);
    const double expanded_mib = expanded_slots * bytes_per_token / (1024. * 1024.);
    out << '"' << prop.name << '\"' << ',' << prop.major << '.' << prop.minor
        << ',' << CUDART_VERSION << ',' << b << ",7," << context
        << ",32,8,128,16," << std::fixed << std::setprecision(3)
        << tree_us << ',' << expanded_us << ',' << expanded_us / tree_us
        << ',' << std::scientific << std::setprecision(8) << diff
        << std::fixed << std::setprecision(3) << ',' << shared_mib << ',' << expanded_mib
        << ',' << 100. * (1. - shared_mib / expanded_mib) << '\n';
    out.flush();
    std::cout << "tree B=" << b << " S=" << context << " tree=" << tree_us
              << "us expanded=" << expanded_us << "us ratio=" << expanded_us / tree_us
              << "x KV saved=" << 100. * (1. - shared_mib / expanded_mib) << "%\n";
  }
  std::cout << "Wrote " << output_path << '\n';
}

void run_bench(const std::string& output_path, bool quick) {
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("cannot open output: " + output_path);
  out << "gpu,sm,cuda_runtime,batch,query_heads,kv_heads,head_dim,context,"
         "page_size,splits,single_us,split_us,speedup,max_abs_diff,kv_mib\n";
  const std::vector<int> batches = quick ? std::vector<int>{1, 8} : std::vector<int>{1, 4, 16, 32};
  const std::vector<int> contexts = quick ? std::vector<int>{256, 2048} : std::vector<int>{256, 1024, 4096, 8192};
  for (int b : batches) for (int context : contexts) {
    std::vector<int> lengths(b);
    for (int i = 0; i < b; ++i) lengths[i] = context - (i % 4) * context / 8;
    Fixture f = make_fixture(b, 32, 8, 128, lengths, 17 + b + context);
    GpuFixture gpu(f);
    const int splits = branchcraft::auto_splits(f.shape, context, prop.multiProcessorCount);
    gpu.run(1);
    const auto baseline_out = gpu.output.download();
    gpu.run(splits);
    const float diff = max_abs_error(baseline_out, gpu.output.download());
    if (diff > 3e-4f) throw std::runtime_error("benchmark split output mismatch");
    const double single = median_latency_us(gpu, 1);
    const double split = splits == 1 ? single : median_latency_us(gpu, splits);
    const double kv_mib = double(f.k.size() + f.v.size()) * sizeof(half) / (1024. * 1024.);
    out << '"' << prop.name << '\"' << ',' << prop.major << '.' << prop.minor
        << ',' << CUDART_VERSION << ',' << b << ",32,8,128," << context << ','
        << kPageSize << ',' << splits << ',' << std::fixed << std::setprecision(3)
        << single << ',' << split << ',' << single / split << ','
        << std::scientific << std::setprecision(8) << diff
        << std::fixed << std::setprecision(3) << ',' << kv_mib << '\n';
    out.flush();
    std::cout << "B=" << b << " S=" << context << " split=" << splits
              << " single=" << single << "us chosen=" << split << "us"
              << " speedup=" << single / split << "x diff=" << diff << '\n';
  }
  std::cout << "Wrote " << output_path << '\n';
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: branchcraft test | bench | tree-bench [--quick] [--output FILE]\n";
      return 2;
    }
    const std::string mode = argv[1];
    if (mode == "test") { run_tests(); run_tree_tests(); }
    else if (mode == "bench" || mode == "tree-bench") {
      bool quick = false;
      std::string output = mode == "bench" ? "benchmarks/decode_results.csv"
                                             : "benchmarks/tree_results.csv";
      for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--output" && i + 1 < argc) output = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
      }
      if (mode == "bench") run_bench(output, quick);
      else run_tree_bench(output, quick);
    } else throw std::runtime_error("unknown mode: " + mode);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
