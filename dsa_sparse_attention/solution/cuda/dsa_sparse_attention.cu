/*
 * DSA sparse attention — CUDA flash-decoding with bf16 tensor cores (WMMA),
 * cp.async K gather, 8-warp layout.
 *
 * History:
 *   v1 scalar FMA, split-KV: 156 us
 *   v2 WMMA, 128 threads, 2 CTAs/SM: 28 us
 *   v3 WMMA, 256 threads (PV split 8 ways): 25.5 us
 *   v4 + cp.async for K gather: 25.2 us  ← this
 *   v5 ping-pong 2 tiles/CTA: 29 us (regressed — 1 CTA/SM cost > pipeline gain)
 *
 * Layout: one tile per CTA (K_SPLITS=32). Each CTA reads 64 KV tokens,
 * computes [H=16, D_TOT=576] × [D_TOT, 64]^T via WMMA, softmaxes one row
 * at a time, then [H, 64] × [64, DC=512] for PV. First 4 warps do QK;
 * all 8 warps split PV's DC across them (64 cols each).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

using namespace nvcuda;

namespace {

constexpr int H        = 16;
constexpr int DC       = 512;
constexpr int DP       = 64;
constexpr int D_TOT    = DC + DP;                 // 576
constexpr int BLOCK_N  = 64;
constexpr int K_MAX    = 2048;
constexpr int K_SPLITS = 32;                      // BLOCK_N * K_SPLITS = K_MAX
constexpr int THREADS  = 256;                     // 8 warps
constexpr int N_WARPS  = THREADS / 32;
constexpr int QK_WARPS = 4;                        // 4 warps × 1 N-tile each (others idle during QK)
constexpr int SOFT_THREADS = 128;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define NEG_INF (-__builtin_huge_valf())
constexpr float LOG2E = 1.4426950408889634f;

// Shared-memory budget (per CTA):
//   sQ: 18 KB, sK: 74 KB, sL: 4 KB, sP: 2 KB, sIdx+sM+sLn+sMbar: ~0.5 KB
//   total                                    ≈ 98 KB (2 CTAs/SM)
constexpr int SMEM_BYTES =
    H * D_TOT * 2
  + BLOCK_N * D_TOT * 2
  + H * BLOCK_N * 4
  + H * BLOCK_N * 2
  + BLOCK_N * 4
  + H * 4
  + H * 4
  + 32;   // sMbar + padding

__device__ __forceinline__ void cp_async_16(
    uint32_t smem_int_ptr, const void* gmem_ptr)
{
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_int_ptr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n");
}

// ---- TMA (cp.async.bulk) primitives (sm_90+) ----
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_shm, int count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar_shm);
  asm volatile(
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(mbar_ptr), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar_shm, uint32_t tx_count)
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar_shm);
  asm volatile(
    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(mbar_ptr), "r"(tx_count));
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_shm, int parity) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar_shm);
  asm volatile(
    "{ .reg .pred P;\n"
    "LAB_WAIT%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
    "@P bra DONE%=;\n"
    "bra LAB_WAIT%=;\n"
    "DONE%=: }\n"
    :: "r"(mbar_ptr), "r"(parity));
}
__device__ __forceinline__ void cp_async_bulk_shared_global(
    uint32_t smem_int_ptr, const void* gmem_ptr,
    uint32_t bytes, uint64_t* mbar_shm)
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar_shm);
  asm volatile(
    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
    "[%0], [%1], %2, [%3];\n"
    :: "r"(smem_int_ptr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_ptr)
    : "memory");
}

__device__ __forceinline__ float warp_reduce_max_8(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
  return v;
}
__device__ __forceinline__ float warp_reduce_sum_8(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

__global__ __launch_bounds__(THREADS, 2)
void attn_split_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t*       __restrict__ sparse_idx,
    float*               __restrict__ partial_m,
    float*               __restrict__ partial_l,
    float*               __restrict__ partial_o,
    float sm_scale)
{
  const int t     = blockIdx.x;
  const int split = blockIdx.y;
  const int tid   = threadIdx.x;
  const int warp  = tid / 32;

  extern __shared__ char smem[];
  __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sK   = sQ  + H * D_TOT;
  float*         sL   = reinterpret_cast<float*>(sK + BLOCK_N * D_TOT);
  __nv_bfloat16* sP   = reinterpret_cast<__nv_bfloat16*>(sL + H * BLOCK_N);
  int32_t*       sIdx = reinterpret_cast<int32_t*>(sP + H * BLOCK_N);
  float*         sM   = reinterpret_cast<float*>(sIdx + BLOCK_N);
  float*         sLn  = sM + H;
  // 8-byte-aligned mbarrier slot (must fit within SMEM_BYTES budget below)
  uint64_t*      sMbar = reinterpret_cast<uint64_t*>(sLn + H + 2);  // +2 pads to 8B

  const int k_base = split * BLOCK_N;

  // ---- Load Q (nope ‖ pe) to sQ [H, D_TOT] ----
  {
    const float4* src_n = reinterpret_cast<const float4*>(q_nope + t * H * DC);
    #pragma unroll 4
    for (int i = tid; i < (H * DC) / 8; i += THREADS) {
      int h   = i / (DC / 8);
      int d_f = i % (DC / 8);
      reinterpret_cast<float4*>(sQ + h * D_TOT)[d_f] = src_n[i];
    }
    const float4* src_p = reinterpret_cast<const float4*>(q_pe + t * H * DP);
    #pragma unroll
    for (int i = tid; i < (H * DP) / 8; i += THREADS) {
      int h   = i / (DP / 8);
      int d_f = i % (DP / 8);
      reinterpret_cast<float4*>(sQ + h * D_TOT + DC)[d_f] = src_p[i];
    }
  }

  if (tid < BLOCK_N) {
    sIdx[tid] = sparse_idx[t * K_MAX + k_base + tid];
  }
  __syncthreads();

  // ---- Gather K via cp.async (16B lane-wise). TMA / cp.async.bulk was tried
  //      but regressed (+7 us): 1024B rows are below the amortization point
  //      where TMA + mbarrier overhead pays off for this workload size. ----
  {
    const uint32_t sK_int = __cvta_generic_to_shared(sK);
    constexpr int KC_F4_PER_ROW = DC / 8;
    constexpr int KC_F4_TOTAL   = BLOCK_N * KC_F4_PER_ROW;
    #pragma unroll 4
    for (int i = tid; i < KC_F4_TOTAL; i += THREADS) {
      int n    = i / KC_F4_PER_ROW;
      int d_f  = i % KC_F4_PER_ROW;
      int idx  = sIdx[n];
      int safe = (idx == -1) ? 0 : idx;
      const void* src = ckv_cache + (size_t)safe * DC + d_f * 8;
      uint32_t dst = sK_int + (n * D_TOT + d_f * 8) * sizeof(__nv_bfloat16);
      cp_async_16(dst, src);
    }
    constexpr int KP_F4_PER_ROW = DP / 8;
    constexpr int KP_F4_TOTAL   = BLOCK_N * KP_F4_PER_ROW;
    #pragma unroll 2
    for (int i = tid; i < KP_F4_TOTAL; i += THREADS) {
      int n    = i / KP_F4_PER_ROW;
      int d_f  = i % KP_F4_PER_ROW;
      int idx  = sIdx[n];
      int safe = (idx == -1) ? 0 : idx;
      const void* src = kpe_cache + (size_t)safe * DP + d_f * 8;
      uint32_t dst = sK_int + (n * D_TOT + DC + d_f * 8) * sizeof(__nv_bfloat16);
      cp_async_16(dst, src);
    }
    cp_async_commit();
    cp_async_wait_all();
  }
  __syncthreads();

  // ---- QK matmul (first 4 warps; others idle here) ----
  // Software-pipelined: prefetch next K-tile's fragments while current tile's
  // mma_sync is executing. The compiler schedules ldmatrix well ahead of mma
  // when given two independent fragment variables.
  if (warp < QK_WARPS) {
    using QFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                 __nv_bfloat16, wmma::row_major>;
    using KFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                 __nv_bfloat16, wmma::col_major>;
    using CFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    const int warp_n_start = warp * WMMA_N;

    CFrag c;
    wmma::fill_fragment(c, 0.0f);

    // Double-buffered fragments
    QFrag q0, q1;
    KFrag k0, k1;

    // Prefetch iter 0
    wmma::load_matrix_sync(q0, sQ + 0, D_TOT);
    wmma::load_matrix_sync(k0, sK + warp_n_start * D_TOT + 0, D_TOT);

    // D_TOT = 576 = 36 K-tiles. Pipeline iter k (computes) with iter k+1
    // (loads). We unroll 2 to process pairs with swapped frag usage.
    #pragma unroll
    for (int k = WMMA_K; k < D_TOT; k += 2 * WMMA_K) {
      // Load next (k) into q1/k1
      wmma::load_matrix_sync(q1, sQ + k, D_TOT);
      wmma::load_matrix_sync(k1, sK + warp_n_start * D_TOT + k, D_TOT);
      // MMA on previous (k - WMMA_K) that lives in q0/k0
      wmma::mma_sync(c, q0, k0, c);

      if (k + WMMA_K < D_TOT) {
        // Load k + WMMA_K into q0/k0
        wmma::load_matrix_sync(q0, sQ + k + WMMA_K, D_TOT);
        wmma::load_matrix_sync(k0, sK + warp_n_start * D_TOT + k + WMMA_K, D_TOT);
      }
      // MMA on k (in q1/k1)
      wmma::mma_sync(c, q1, k1, c);
    }
    // Final MMA if D_TOT / WMMA_K is odd (36/16 → 36 iters, even, so no tail)
    if ((D_TOT / WMMA_K) & 1) {
      wmma::mma_sync(c, q0, k0, c);
    }

    #pragma unroll
    for (int i = 0; i < c.num_elements; i++) c.x[i] *= sm_scale;
    wmma::store_matrix_sync(sL + warp_n_start, c, BLOCK_N, wmma::mem_row_major);
  }
  __syncthreads();

  // ---- Softmax (first 128 threads) ----
  if (tid < SOFT_THREADS) {
    const int h = tid / 8;
    const int s = tid & 7;
    const int n_base = s * 8;

    float logits[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int n = n_base + i;
      float v = sL[h * BLOCK_N + n];
      if (sIdx[n] == -1) v = NEG_INF;
      logits[i] = v;
    }

    float my_max = logits[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) my_max = fmaxf(my_max, logits[i]);
    float row_max = warp_reduce_max_8(my_max);

    float p_vals[8];
    float my_sum = 0.0f;
    if (row_max > NEG_INF) {
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        p_vals[i] = __expf(logits[i] - row_max);
        my_sum += p_vals[i];
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 8; i++) p_vals[i] = 0.0f;
    }
    float row_sum = warp_reduce_sum_8(my_sum);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      sP[h * BLOCK_N + n_base + i] = __float2bfloat16(p_vals[i]);
    }
    if (s == 0) {
      sM[h]  = row_max;
      sLn[h] = row_sum;
    }
  }
  __syncthreads();

  // ---- PV matmul (8 warps split DC=512 → 64 cols per warp) ----
  {
    using PFrag  = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                  __nv_bfloat16, wmma::row_major>;
    using KcFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                  __nv_bfloat16, wmma::row_major>;
    using AccFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    constexpr int COLS_PER_WARP    = DC / N_WARPS;           // 64
    constexpr int N_TILES_PER_WARP = COLS_PER_WARP / WMMA_N; // 4

    float* po = partial_o + ((size_t)t * K_SPLITS + split) * (H * DC);

    #pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
      int n_start = warp * COLS_PER_WARP + nt * WMMA_N;

      AccFrag acc;
      wmma::fill_fragment(acc, 0.0f);

      #pragma unroll
      for (int k = 0; k < BLOCK_N; k += WMMA_K) {
        PFrag  pfrag;
        KcFrag kfrag;
        wmma::load_matrix_sync(pfrag, sP + k, BLOCK_N);
        wmma::load_matrix_sync(kfrag, sK + k * D_TOT + n_start, D_TOT);
        wmma::mma_sync(acc, pfrag, kfrag, acc);
      }
      wmma::store_matrix_sync(po + n_start, acc, DC, wmma::mem_row_major);
    }
  }

  if (tid < H) {
    size_t off = ((size_t)t * K_SPLITS + split) * H + tid;
    partial_m[off] = sM[tid];
    partial_l[off] = sLn[tid];
  }
}

constexpr int MERGE_THREADS = 64;
static_assert(DC % MERGE_THREADS == 0, "merge dc coverage");
constexpr int DC_PER_MERGE_THREAD = DC / MERGE_THREADS;

__global__ __launch_bounds__(MERGE_THREADS, 4)
void attn_merge_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_o,
    __nv_bfloat16* __restrict__ out,
    float*          __restrict__ lse)
{
  const int t   = blockIdx.x;
  const int h   = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float sM[K_SPLITS];
  __shared__ float sL[K_SPLITS];
  __shared__ float sW[K_SPLITS];
  __shared__ float sMg;
  __shared__ float sLg;

  if (tid < K_SPLITS) {
    size_t off = ((size_t)t * K_SPLITS + tid) * H + h;
    sM[tid] = partial_m[off];
    sL[tid] = partial_l[off];
  }
  __syncthreads();

  if (tid == 0) {
    float mg = NEG_INF;
    #pragma unroll
    for (int s = 0; s < K_SPLITS; s++) mg = fmaxf(mg, sM[s]);
    float lg = 0.0f;
    #pragma unroll
    for (int s = 0; s < K_SPLITS; s++) {
      float w = (sM[s] == NEG_INF) ? 0.0f : __expf(sM[s] - mg);
      sW[s] = w;
      lg += w * sL[s];
    }
    sMg = mg;
    sLg = lg;
  }
  __syncthreads();

  const float mg = sMg;
  const float lg = sLg;
  const float inv_l = (lg > 0.0f) ? (1.0f / lg) : 0.0f;

  const int d_base = tid * DC_PER_MERGE_THREAD;
  float o_acc[DC_PER_MERGE_THREAD];
  #pragma unroll
  for (int i = 0; i < DC_PER_MERGE_THREAD; i++) o_acc[i] = 0.0f;

  #pragma unroll 8
  for (int s = 0; s < K_SPLITS; s++) {
    float w = sW[s];
    if (w == 0.0f) continue;
    const float* po =
        partial_o + ((size_t)t * K_SPLITS + s) * (H * DC) + h * DC + d_base;
    #pragma unroll
    for (int i = 0; i < DC_PER_MERGE_THREAD; i++) {
      o_acc[i] += w * po[i];
    }
  }

  __nv_bfloat16* out_row = out + (size_t)t * H * DC + h * DC + d_base;
  if (mg == NEG_INF || lg == 0.0f) {
    #pragma unroll
    for (int i = 0; i < DC_PER_MERGE_THREAD; i++) {
      out_row[i] = __float2bfloat16(0.0f);
    }
  } else {
    #pragma unroll
    for (int i = 0; i < DC_PER_MERGE_THREAD; i++) {
      out_row[i] = __float2bfloat16(o_acc[i] * inv_l);
    }
  }

  if (tid == 0) {
    float lse_v = (mg == NEG_INF || lg == 0.0f) ? NEG_INF
                                                : (mg + __logf(lg)) * LOG2E;
    lse[(size_t)t * H + h] = lse_v;
  }
}

struct Workspace {
  float* d_m = nullptr;
  float* d_l = nullptr;
  float* d_o = nullptr;
  int    Nt  = 0;
};
static Workspace g_ws;

static void ensure_workspace(int Nt) {
  if (Nt <= g_ws.Nt) return;
  if (g_ws.d_m) cudaFree(g_ws.d_m);
  if (g_ws.d_l) cudaFree(g_ws.d_l);
  if (g_ws.d_o) cudaFree(g_ws.d_o);
  size_t ml = (size_t)Nt * K_SPLITS * H * sizeof(float);
  size_t od = (size_t)Nt * K_SPLITS * H * DC * sizeof(float);
  cudaMalloc(&g_ws.d_m, ml);
  cudaMalloc(&g_ws.d_l, ml);
  cudaMalloc(&g_ws.d_o, od);
  g_ws.Nt = Nt;
}

static bool g_shmem_attr_set = false;

}  // namespace

using tvm::ffi::TensorView;

static void run_impl(
    TensorView q_nope,
    TensorView q_pe,
    TensorView ckv_cache,
    TensorView kpe_cache,
    TensorView sparse_indices,
    double sm_scale_d,
    TensorView output,
    TensorView lse)
{
  const int Nt = (int)q_nope.shape()[0];
  if (Nt == 0) return;
  const float sm_scale = (float)sm_scale_d;

  ensure_workspace(Nt);

  if (!g_shmem_attr_set) {
    cudaFuncSetAttribute(
        attn_split_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES);
    g_shmem_attr_set = true;
  }

  dim3 grid_split(Nt, K_SPLITS);
  attn_split_kernel<<<grid_split, THREADS, SMEM_BYTES, 0>>>(
      static_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      static_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
      static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
      static_cast<const int32_t*>(sparse_indices.data_ptr()),
      g_ws.d_m, g_ws.d_l, g_ws.d_o,
      sm_scale);

  dim3 grid_merge(Nt, H);
  attn_merge_kernel<<<grid_merge, MERGE_THREADS, 0, 0>>>(
      g_ws.d_m, g_ws.d_l, g_ws.d_o,
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<float*>(lse.data_ptr()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_dsa_sparse_attn, run_impl);
