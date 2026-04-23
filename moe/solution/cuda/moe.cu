/*
 * Fused FP8 block-scale MoE — full CUDA implementation from scratch.
 *
 * Pipeline:
 *   1. route_kernel           — DeepSeek-V3 no-aux routing
 *   2. perm_count_kernel      — counts[le] per local expert
 *   3. scan_offsets_kernel    — offs[E_LOCAL+1] and write_ptr
 *   4. perm_fill_kernel       — perm_token[N_sel], perm_weight[N_sel]
 *   5. dequant_hs_kernel      — hidden_states fp8 -> bf16 [T, H]
 *   6. gather_hs_kernel       — A_perm[N_sel, H] = hs_bf16[perm_token]
 *   7. dequant_w_kernel       — W13, W2 fp8 -> bf16
 *   8. gemm_bf16_kernel       — A @ B^T, bf16 tensor cores (WMMA)
 *   9. swiglu_kernel          — silu(X2)*X1 -> C[Tk, I]
 *  10. fin_acc_kernel         — fp32 atomic-add weighted scatter
 *  11. fin_cast_kernel        — fp32 -> bf16 output
 *
 * Constants (DeepSeek-V3/R1, moe_fp8_block_scale track):
 *   H=7168, I=2048, BS=128, E_LOCAL=32, E_GLOBAL=256,
 *   TOP_K=8, N_GROUP=8, TOPK_GROUP=4
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

using namespace nvcuda;

namespace {

constexpr int H          = 7168;
constexpr int I          = 2048;
constexpr int II         = 2 * I;               // 4096
constexpr int E_LOCAL    = 32;
constexpr int E_GLOBAL   = 256;
constexpr int TOP_K      = 8;
constexpr int N_GROUP    = 8;
constexpr int TOPK_GROUP = 4;
constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP;  // 32
constexpr int BS         = 128;                 // quant block size
constexpr int NH_BLKS    = H / BS;              // 56
constexpr int NI_BLKS    = I / BS;              // 16
constexpr int NII_BLKS   = II / BS;             // 32

#define NEG_INF_F (-CUDART_INF_F)

// ======================================================================
// Kernel 1: routing.
// One CTA per token, 256 threads. One thread per expert.
// Uses argmax-with-lane-tiebreak so top-2 group scores are correct with ties.
// ======================================================================
__global__ void route_kernel(
    const float* __restrict__ logits,          // [T, E_GLOBAL]
    const __nv_bfloat16* __restrict__ bias,    // [E_GLOBAL]
    float scaling_factor,
    int T,
    int32_t* __restrict__ topk_idx_out,        // [T, TOP_K]
    float*   __restrict__ topk_w_out)          // [T, TOP_K]
{
  const int t = blockIdx.x;
  if (t >= T) return;
  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ float s_sig[E_GLOBAL];
  __shared__ float s_wb [E_GLOBAL];
  __shared__ float gs[N_GROUP];
  __shared__ int   group_sel[N_GROUP];
  __shared__ int   tk_idx[TOP_K];
  __shared__ float tk_sig[TOP_K];
  __shared__ int   s_best_idx;
  __shared__ float warp_v[8];
  __shared__ int   warp_i[8];

  // sigmoid + add bias
  float lg = logits[(size_t)t * E_GLOBAL + tid];
  float si = 1.0f / (1.0f + __expf(-lg));
  s_sig[tid] = si;
  float b = __bfloat162float(bias[tid]);
  s_wb[tid] = si + b;
  if (tid < N_GROUP) group_sel[tid] = 0;
  __syncthreads();

  // per-warp top-2 sum: argmax tiebreak by (value, -lane)
  {
    float v  = s_wb[warp * GROUP_SIZE + lane];
    float bv = v;
    int   bl = lane;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, m);
      int   ol = __shfl_xor_sync(0xffffffff, bl, m);
      if (ov > bv || (ov == bv && ol < bl)) { bv = ov; bl = ol; }
    }
    // Exclude exactly the argmax lane, find second max.
    float v2 = (lane == bl) ? NEG_INF_F : v;
    float m2 = v2;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
      m2 = fmaxf(m2, __shfl_xor_sync(0xffffffff, m2, m));
    if (lane == 0) gs[warp] = bv + m2;
  }
  __syncthreads();

  // top-4 of 8 groups (warp 0)
  if (warp == 0) {
    float gv = (lane < N_GROUP) ? gs[lane] : NEG_INF_F;
    int   gi = lane;
    #pragma unroll
    for (int r = 0; r < TOPK_GROUP; r++) {
      float bv = gv;
      int   bi = gi;
      #pragma unroll
      for (int m = 16; m >= 1; m >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, m);
        int   oi = __shfl_xor_sync(0xffffffff, bi, m);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) group_sel[bi] = 1;
      if (gi == bi) gv = NEG_INF_F;
    }
  }
  __syncthreads();

  // top-8 of E_GLOBAL among kept groups.
  float sp = (group_sel[tid / GROUP_SIZE] == 1) ? s_wb[tid] : NEG_INF_F;

  for (int r = 0; r < TOP_K; r++) {
    float bv = sp;
    int   bi = tid;
    // warp reduce
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, m);
      int   oi = __shfl_xor_sync(0xffffffff, bi, m);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { warp_v[warp] = bv; warp_i[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
      float bv2 = (lane < 8) ? warp_v[lane] : NEG_INF_F;
      int   bi2 = (lane < 8) ? warp_i[lane] : 0;
      #pragma unroll
      for (int m = 4; m >= 1; m >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv2, m);
        int   oi = __shfl_xor_sync(0xffffffff, bi2, m);
        if (ov > bv2 || (ov == bv2 && oi < bi2)) { bv2 = ov; bi2 = oi; }
      }
      if (lane == 0) {
        s_best_idx = bi2;
        tk_idx[r]  = bi2;
        tk_sig[r]  = s_sig[bi2];
      }
    }
    __syncthreads();
    if (tid == s_best_idx) sp = NEG_INF_F;
  }

  // normalize + scale
  if (tid == 0) {
    float ssum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) ssum += tk_sig[k];
    float inv = 1.0f / (ssum + 1e-20f);
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) {
      topk_idx_out[(size_t)t * TOP_K + k] = tk_idx[k];
      topk_w_out  [(size_t)t * TOP_K + k] = tk_sig[k] * inv * scaling_factor;
    }
  }
}

// ======================================================================
// Kernel 2: count tokens per local expert.
// ======================================================================
__global__ void perm_count_kernel(
    const int32_t* __restrict__ topk_idx,
    int T,
    int local_expert_offset,
    int32_t* __restrict__ counts)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  #pragma unroll
  for (int k = 0; k < TOP_K; k++) {
    int ge = topk_idx[(size_t)t * TOP_K + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)E_LOCAL) {
      atomicAdd(&counts[le], 1);
    }
  }
}

// ======================================================================
// Kernel 3: exclusive scan counts -> offs[E_LOCAL+1], write_ptr[E_LOCAL].
// ======================================================================
__global__ void scan_offsets_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offs,
    int32_t* __restrict__ write_ptr,
    int32_t* __restrict__ n_sel_out)
{
  if (threadIdx.x != 0) return;
  int acc = 0;
  offs[0] = 0;
  #pragma unroll
  for (int i = 0; i < E_LOCAL; i++) {
    write_ptr[i] = acc;
    acc += counts[i];
    offs[i + 1] = acc;
  }
  *n_sel_out = acc;
}

// ======================================================================
// Kernel 4: fill perm arrays grouped by local expert.
// ======================================================================
__global__ void perm_fill_kernel(
    const int32_t* __restrict__ topk_idx,
    const float*   __restrict__ topk_w,
    int T,
    int local_expert_offset,
    int32_t* __restrict__ write_ptr,
    int32_t* __restrict__ perm_token,
    float*   __restrict__ perm_weight)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  #pragma unroll
  for (int k = 0; k < TOP_K; k++) {
    int ge = topk_idx[(size_t)t * TOP_K + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)E_LOCAL) {
      int slot = atomicAdd(&write_ptr[le], 1);
      perm_token [slot] = t;
      perm_weight[slot] = topk_w[(size_t)t * TOP_K + k];
    }
  }
}

// ======================================================================
// Kernel 5: hidden_states fp8 -> bf16.
// Scale[H/BS, T] -> row (hidden-block), col (token).
// Grid=(T, NH_BLKS), threads=BS. Block handles one (token, hb).
// ======================================================================
__global__ void dequant_hs_kernel(
    const __nv_fp8_e4m3* __restrict__ hs_fp8,
    const float*         __restrict__ hs_scale,
    int T,
    __nv_bfloat16* __restrict__ hs_bf16)
{
  const int t  = blockIdx.x;
  const int hb = blockIdx.y;
  const int d  = threadIdx.x;
  if (t >= T) return;
  float sc = hs_scale[(size_t)hb * T + t];
  int   h  = hb * BS + d;
  float v  = (float)hs_fp8[(size_t)t * H + h] * sc;
  hs_bf16[(size_t)t * H + h] = __float2bfloat16(v);
}

// ======================================================================
// Kernel 6: gather hs_bf16 rows into A_perm by perm_token.
// ======================================================================
__global__ void gather_hs_kernel(
    const __nv_bfloat16* __restrict__ hs_bf16,
    const int32_t*       __restrict__ perm_token,
    int N_sel,
    __nv_bfloat16* __restrict__ A_e)
{
  const int r  = blockIdx.x;
  const int hb = blockIdx.y;
  const int d  = threadIdx.x;
  if (r >= N_sel) return;
  int t = perm_token[r];
  int h = hb * BS + d;
  A_e[(size_t)r * H + h] = hs_bf16[(size_t)t * H + h];
}

// ======================================================================
// Kernel 7: weights fp8 -> bf16.
// Grid=(E_LOCAL, N/BS, K/BS), threads=256, 64 elems per thread.
// ======================================================================
__global__ void dequant_w_kernel(
    const __nv_fp8_e4m3* __restrict__ W_fp8,
    const float*         __restrict__ W_scale,
    const int32_t*       __restrict__ active_le,  // [A] local-expert ids to dequant
    int N, int K, int NB, int KB,
    __nv_bfloat16* __restrict__ W_bf16)
{
  const int le = active_le[blockIdx.x];
  const int nb = blockIdx.y;
  const int kb = blockIdx.z;
  const int tid = threadIdx.x;
  const int n0 = nb * BS;
  const int k0 = kb * BS;

  float sc = W_scale[((size_t)le * NB + nb) * KB + kb];

  constexpr int TILE  = BS * BS;       // 16384
  constexpr int ITERS = TILE / 256;    // 64
  #pragma unroll
  for (int it = 0; it < ITERS; it++) {
    int id = tid + it * 256;
    int r  = id / BS;
    int c  = id % BS;
    int n  = n0 + r;
    int k  = k0 + c;
    if (n < N && k < K) {
      size_t idx = ((size_t)le * N + n) * K + k;
      W_bf16[idx] = __float2bfloat16((float)W_fp8[idx] * sc);
    }
  }
}

// ======================================================================
// Kernel 8: bf16 GEMM, C[M,N] = A[M,K] @ B[N,K]^T, row-major.
// BM=128, BN=128, BK=32. 8 warps laid out 4 (M) x 2 (N).
// Each warp: 32 x 64 output = 2 x 4 WMMA 16x16x16.
// 2-stage cp.async pipeline.
// ======================================================================
constexpr int BM           = 128;
constexpr int BN           = 128;
constexpr int BK           = 32;
constexpr int NWARPS_M     = 4;
constexpr int NWARPS_N     = 2;
constexpr int NWARPS       = NWARPS_M * NWARPS_N;   // 8
constexpr int THREADS_GEMM = NWARPS * 32;           // 256
constexpr int WM           = BM / NWARPS_M;         // 32
constexpr int WN           = BN / NWARPS_N;         // 64
constexpr int W_TILES_M    = WM / 16;               // 2
constexpr int W_TILES_N    = WN / 16;               // 4
constexpr int LDA_S        = BK + 8;                // 40 (bf16 pad)
constexpr int LDB_S        = BK + 8;                // 40
constexpr int STAGES       = 2;

static_assert(BM * BK % (THREADS_GEMM * 8) == 0, "A load not evenly tileable by 16B");
static_assert(BN * BK % (THREADS_GEMM * 8) == 0, "B load not evenly tileable by 16B");

__device__ __forceinline__ void cp_async_16(void* sdst, const void* gsrc, bool pred) {
  unsigned sdstaddr = __cvta_generic_to_shared(sdst);
  if (pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(sdstaddr), "l"(gsrc));
  } else {
    *reinterpret_cast<uint4*>(sdst) = make_uint4(0, 0, 0, 0);
  }
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void stage_load_A(
    __nv_bfloat16 (*sA)[LDA_S],
    const __nv_bfloat16* A, int m0,
    int k0, int M, int K, int tid)
{
  constexpr int A_ITERS = (BM * BK) / (THREADS_GEMM * 8);
  #pragma unroll
  for (int it = 0; it < A_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 8);
    int c  = (id % (BK / 8)) * 8;
    bool ok = (m0 + r) < M;
    const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
    cp_async_16(&sA[r][c], gptr, ok);
  }
}

__device__ __forceinline__ void stage_load_B_bf16(
    __nv_bfloat16 (*sB)[LDB_S],
    const __nv_bfloat16* B, int n0,
    int k0, int N, int K, int tid)
{
  constexpr int B_ITERS = (BN * BK) / (THREADS_GEMM * 8);
  #pragma unroll
  for (int it = 0; it < B_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 8);
    int c  = (id % (BK / 8)) * 8;
    bool ok = (n0 + r) < N;
    const void* gptr = &B[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB[r][c], gptr, ok);
  }
}

// LDB_S padded stride for the bf16 sB buffer (kept in bytes the same as LDB_S).
constexpr int LDB_FP8 = BK + 16;       // fp8 pad (16 elems = 16 bytes)

// Async FP8 load: cp.async 16B of fp8 per thread into a scratch buffer.
// No conversion happens here — fp8 stays as raw bytes in shmem.
__device__ __forceinline__ void stage_load_B_fp8_async(
    __nv_fp8_e4m3 (*sB_fp8)[LDB_FP8],
    const __nv_fp8_e4m3* B_fp8,
    int n0, int k0, int N, int K, int tid)
{
  constexpr int B_ITERS = (BN * BK) / (THREADS_GEMM * 16);
  #pragma unroll
  for (int it = 0; it < B_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 16);          // 0..127 (since BK/16 = 2)
    int c  = (id % (BK / 16)) * 16;    // 0, 16
    bool ok = (n0 + r) < N;
    const void* gptr = &B_fp8[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB_fp8[r][c], gptr, ok);
  }
}

// Convert sB_fp8 scratch -> sB bf16 with fused per-tile scale multiply.
// Called once per main-loop iter after async loads complete.
__device__ __forceinline__ void convert_sB_fp8_to_bf16(
    const __nv_fp8_e4m3 (*sB_fp8)[LDB_FP8],
    __nv_bfloat16 (*sB)[LDB_S],
    float sc, int tid)
{
  constexpr int ITERS = (BN * BK) / (THREADS_GEMM * 16);
  #pragma unroll
  for (int it = 0; it < ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 16);
    int c  = (id % (BK / 16)) * 16;
    uint4 u = *reinterpret_cast<const uint4*>(&sB_fp8[r][c]);
    __nv_fp8_e4m3 b[16];
    *reinterpret_cast<uint4*>(b) = u;
    __nv_bfloat16 dst[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) dst[i] = __float2bfloat16((float)b[i] * sc);
    reinterpret_cast<uint4*>(&sB[r][c])[0] = reinterpret_cast<const uint4*>(dst)[0];
    reinterpret_cast<uint4*>(&sB[r][c])[1] = reinterpret_cast<const uint4*>(dst)[1];
  }
}

// Dynamic shared memory layout (in bytes, offset from smem_raw):
//   [0            , SA_BYTES)          : sA[STAGES][BM][LDA_S] bf16
//   [SA_BYTES     , SA_BYTES+SB_BYTES) : sB[STAGES][BN][LDB_S] bf16
// Epilogue reuses same buffer (main loop done):
//   [0, SC_BYTES)                      : sW[NWARPS][16][16] fp32 (per-warp 1KB)
constexpr size_t SA_BYTES = (size_t)STAGES * BM * LDA_S * sizeof(__nv_bfloat16);
constexpr size_t SB_BYTES = (size_t)STAGES * BN * LDB_S * sizeof(__nv_bfloat16);
constexpr size_t SC_BYTES = (size_t)NWARPS * 16 * 16 * sizeof(float);
constexpr size_t GEMM_SMEM_BYTES =
    (SA_BYTES + SB_BYTES) > SC_BYTES ? (SA_BYTES + SB_BYTES) : SC_BYTES;

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    int M, int N, int K,
    __nv_bfloat16* __restrict__ C,
    int ldC)
{
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_bfloat16 (*sB)[BN][LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BN][LDB_S]>(smem_raw + SA_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_bf16(sB[0], B, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_bf16(sB[next_stage], B, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[stage][warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  // Main loop complete — reuse smem as fp32 accumulator buffer.
  // Per-warp epilogue: one 16x16 shmem scratch per warp keeps total epilogue
  // smem at 8*1KB = 8KB (vs 64KB for a full BM×BN fp32 buffer).
  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4;
        int c = e & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          C[(size_t)gm * ldC + gn] = __float2bfloat16(sW[warp][r][c]);
        }
      }
      __syncwarp();
    }
}

// Fused FP8-weight GEMM: A bf16, B fp8 + scale.
// B is async-loaded as fp8 via cp.async to scratch, then converted in-shmem to bf16.
constexpr size_t SA_BYTES_V2  = (size_t)STAGES * BM * LDA_S    * sizeof(__nv_bfloat16);
constexpr size_t SBFP8_BYTES  = (size_t)STAGES * BN * LDB_FP8  * sizeof(__nv_fp8_e4m3);
constexpr size_t SBBF16_BYTES = (size_t)       BN * LDB_S      * sizeof(__nv_bfloat16);
constexpr size_t MAIN_BYTES_V2 = SA_BYTES_V2 + SBFP8_BYTES + SBBF16_BYTES;
constexpr size_t SC_BYTES_V2   = (size_t)NWARPS * 16 * 16 * sizeof(float);
constexpr size_t GEMM_SMEM_BYTES_V2 =
    MAIN_BYTES_V2 > SC_BYTES_V2 ? MAIN_BYTES_V2 : SC_BYTES_V2;

__global__ void gemm_bf16_fp8w_kernel(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const __nv_fp8_e4m3* __restrict__ B_fp8,  // [N, K]
    const float*         __restrict__ B_scale,// [NB, KB]
    int M, int N, int K, int NB, int KB,
    __nv_bfloat16* __restrict__ C,
    int ldC)
{
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int nb = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  // Preload stage 0: A bf16 + B fp8 (both via cp.async).
  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Convert current stage's fp8 to bf16 with per-tile scale.
    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4;
        int c = e & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          C[(size_t)gm * ldC + gn] = __float2bfloat16(sW[warp][r][c]);
        }
      }
      __syncwarp();
    }
}

// ======================================================================
// Grouped GEMM: one persistent kernel covers all active experts.
// Each block reads its (expert, tile_m) from a schedule array; tile_n = blockIdx.x.
// 2-stage cp.async pipeline (same as per-expert kernel).
// ======================================================================
constexpr size_t GEMM_SMEM_BYTES_GRP = GEMM_SMEM_BYTES_V2;

__global__ void gemm_bf16_fp8w_grouped_kernel(
    const __nv_bfloat16* __restrict__ A_base,     // [N_sel, K]
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E_LOCAL, N, K]
    const float*         __restrict__ S_base,     // [E_LOCAL, NB, KB]
    const int32_t*       __restrict__ offs,       // [E_LOCAL+1]
    const int32_t*       __restrict__ sched_e,    // [total_m_tiles]
    const int32_t*       __restrict__ sched_tm,   // [total_m_tiles]
    int N, int K, int NB, int KB,
    __nv_bfloat16* __restrict__ C_base,           // [N_sel, ldC]
    int ldC)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_bfloat16* A       = A_base + (size_t)rs * K;
  const __nv_fp8_e4m3* B_fp8   = W_base + (size_t)e * (size_t)N * (size_t)K;
  const float*         B_scale = S_base + (size_t)e * (size_t)NB * (size_t)KB;
  __nv_bfloat16*       C       = C_base + (size_t)rs * (size_t)ldC;

  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;
  const int nb     = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int ee = lane; ee < 256; ee += 32) {
        int r = ee >> 4;
        int c = ee & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          C[(size_t)gm * ldC + gn] = __float2bfloat16(sW[warp][r][c]);
        }
      }
      __syncwarp();
    }
}

// ======================================================================
// Grouped BF16 GEMM with FP8 weights + fused fin_acc:
//   scatters weighted fp32 output directly to per-token accumulator,
//   skipping the bf16 O intermediate that caused ~1 bf16-LSB precision loss
//   per element (~2048 at output magnitudes of 5e5 seen in random workloads).
// ======================================================================
__global__ void gemm_bf16_fp8w_fused_acc_grouped_kernel(
    const __nv_bfloat16* __restrict__ A_base,
    const __nv_fp8_e4m3* __restrict__ W_base,
    const float*         __restrict__ S_base,
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    const int32_t*       __restrict__ perm_token,
    const float*         __restrict__ perm_weight,
    int N, int K, int NB, int KB,
    float* __restrict__ acc_out,
    int ldAcc)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_bfloat16* A       = A_base + (size_t)rs * K;
  const __nv_fp8_e4m3* B_fp8   = W_base + (size_t)e * (size_t)N * (size_t)K;
  const float*         B_scale = S_base + (size_t)e * (size_t)NB * (size_t)KB;
  const int32_t*       Ptok    = perm_token + rs;
  const float*         Pwgt    = perm_weight + rs;

  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;
  const int nb     = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  // Fused epilogue: write fp32 × per-row weight atomically to acc_out[token, col].
  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int ee = lane; ee < 256; ee += 32) {
        int r = ee >> 4;
        int c = ee & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          int t = Ptok[gm];
          float w = Pwgt[gm];
          float v = sW[warp][r][c] * w;
          atomicAdd(&acc_out[(size_t)t * ldAcc + gn], v);
        }
      }
      __syncwarp();
    }
}

// ======================================================================
// FP8 MMA grouped GEMM1: A fp8 [N_sel, K] × W fp8 [E, N, K]^T → bf16 [N_sel, N]
// A scale is per-row per-kblock: A_scale[N_sel, KB]
// W scale is per-nblock per-kblock: W_scale[E, NB, KB]
// BK = 128 (one kblock per BK iter, so block-scale apply happens at BK boundary).
// ======================================================================
constexpr int BK_FP8          = 128;
constexpr int LDA_FP8_PAD     = BK_FP8 + 16;      // row stride in fp8 bytes
constexpr int LDB_FP8_PAD     = BK_FP8 + 16;
constexpr int STAGES_FP8      = 2;
constexpr int WN_TILES_FP8    = WN / 8;            // 8 n=8 sub-tiles per warp
constexpr size_t SA_FP8_BYTES = (size_t)STAGES_FP8 * BM * LDA_FP8_PAD;  // fp8 bytes
constexpr size_t SB_FP8_BYTES = (size_t)STAGES_FP8 * BN * LDB_FP8_PAD;
constexpr size_t SC_FP8_BYTES = (size_t)NWARPS * 16 * 16 * sizeof(float);
constexpr size_t GEMM_FP8_SMEM_BYTES =
    (SA_FP8_BYTES + SB_FP8_BYTES) > SC_FP8_BYTES
        ? (SA_FP8_BYTES + SB_FP8_BYTES) : SC_FP8_BYTES;

__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1));
}

__device__ __forceinline__ void stage_load_A_fp8_BK128(
    __nv_fp8_e4m3 (*sA)[LDA_FP8_PAD],
    const __nv_fp8_e4m3* A, int m0, int k0, int M, int K, int tid)
{
  constexpr int A_ITERS = (BM * BK_FP8) / (THREADS_GEMM * 16);  // 4
  #pragma unroll
  for (int it = 0; it < A_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK_FP8 / 16);          // 0..127
    int c  = (id % (BK_FP8 / 16)) * 16;    // 0,16,...,112
    bool ok = (m0 + r) < M;
    const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
    cp_async_16(&sA[r][c], gptr, ok);
  }
}

__device__ __forceinline__ void stage_load_B_fp8_BK128(
    __nv_fp8_e4m3 (*sB)[LDB_FP8_PAD],
    const __nv_fp8_e4m3* B, int n0, int k0, int N, int K, int tid)
{
  constexpr int B_ITERS = (BN * BK_FP8) / (THREADS_GEMM * 16);  // 4
  #pragma unroll
  for (int it = 0; it < B_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK_FP8 / 16);
    int c  = (id % (BK_FP8 / 16)) * 16;
    bool ok = (n0 + r) < N;
    const void* gptr = &B[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB[r][c], gptr, ok);
  }
}

__global__ void gemm_fp8_fp8_grouped_kernel(
    const __nv_fp8_e4m3* __restrict__ A_base,     // [N_sel, K] fp8
    const float*         __restrict__ A_scale,    // [N_sel, KB]
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E_LOCAL, N, K] fp8
    const float*         __restrict__ S_base,     // [E_LOCAL, NB, KB]
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    int N, int K, int NB, int KB,
    float*        __restrict__ C_base,            // [N_sel, ldC] fp32
    int ldC)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_fp8_e4m3* A       = A_base   + (size_t)rs * (size_t)K;
  const float*         Asc     = A_scale  + (size_t)rs * (size_t)KB;
  const __nv_fp8_e4m3* B_fp8   = W_base   + (size_t)e  * (size_t)N * (size_t)K;
  const float*         B_scale = S_base   + (size_t)e  * (size_t)NB * (size_t)KB;
  float*               C       = C_base   + (size_t)rs * (size_t)ldC;

  const int tid          = threadIdx.x;
  const int warp         = tid >> 5;
  const int lane         = tid & 31;
  const int warp_m       = warp / NWARPS_N;
  const int warp_n       = warp % NWARPS_N;
  const int nb           = n0 / BS;
  const int group_id     = lane >> 2;     // 0..7 (row base in a 16-row tile)
  const int tid_in_group = lane & 3;      // 0..3 (col/row step of 4 fp8)

  extern __shared__ __align__(16) char smem_raw[];
  __nv_fp8_e4m3 (*sA)[BM][LDA_FP8_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BM][LDA_FP8_PAD]>(smem_raw);
  __nv_fp8_e4m3 (*sB)[BN][LDB_FP8_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8_PAD]>(smem_raw + SA_FP8_BYTES);

  // Outer fp32 accumulators: 2 M-tiles × 8 N-tiles × 4 fp32 per thread.
  float outer[W_TILES_M][WN_TILES_FP8][4];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < WN_TILES_FP8; j++)
      #pragma unroll
      for (int q = 0; q < 4; q++) outer[i][j][q] = 0.0f;

  // Preload stage 0.
  stage_load_A_fp8_BK128(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_BK128(sB[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int kb = 0; kb < KB; kb++) {
    int k      = kb * BK_FP8;
    int next_k = k + BK_FP8;
    if (next_k < K) {
      int ns = 1 - stage;
      stage_load_A_fp8_BK128(sA[ns], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_BK128(sB[ns], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Per-kblock fp32 accumulator (accumulates 4 m16n8k32 mmas).
    float inner[W_TILES_M][WN_TILES_FP8][4];
    #pragma unroll
    for (int i = 0; i < W_TILES_M; i++)
      #pragma unroll
      for (int j = 0; j < WN_TILES_FP8; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++) inner[i][j][q] = 0.0f;

    #pragma unroll
    for (int kk = 0; kk < BK_FP8; kk += 32) {
      // Load A fragments (2 M-tiles × 4 u32).
      uint32_t Aregs[W_TILES_M][4];
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++) {
        int rbase = warp_m * WM + i * 16;
        int row0  = rbase + group_id;
        int row1  = rbase + group_id + 8;
        int col0  = kk + tid_in_group * 4;
        int col1  = col0 + 16;
        Aregs[i][0] = *reinterpret_cast<const uint32_t*>(&sA[stage][row0][col0]);
        Aregs[i][1] = *reinterpret_cast<const uint32_t*>(&sA[stage][row1][col0]);
        Aregs[i][2] = *reinterpret_cast<const uint32_t*>(&sA[stage][row0][col1]);
        Aregs[i][3] = *reinterpret_cast<const uint32_t*>(&sA[stage][row1][col1]);
      }
      // Load B fragments (8 N-tiles × 2 u32).
      uint32_t Bregs[WN_TILES_FP8][2];
      #pragma unroll
      for (int j = 0; j < WN_TILES_FP8; j++) {
        int cbase = warp_n * WN + j * 8;
        int col_n = cbase + group_id;
        int rowk0 = kk + tid_in_group * 4;
        int rowk1 = rowk0 + 16;
        Bregs[j][0] = *reinterpret_cast<const uint32_t*>(&sB[stage][col_n][rowk0]);
        Bregs[j][1] = *reinterpret_cast<const uint32_t*>(&sB[stage][col_n][rowk1]);
      }
      // MMA: inner += A × B^T
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WN_TILES_FP8; j++) {
          mma_m16n8k32_e4m3(
              inner[i][j][0], inner[i][j][1], inner[i][j][2], inner[i][j][3],
              Aregs[i][0], Aregs[i][1], Aregs[i][2], Aregs[i][3],
              Bregs[j][0], Bregs[j][1]);
        }
      }
    }

    // Apply per-kblock scales and accumulate into outer.
    // Each thread covers 2 rows per M-tile (group_id and group_id+8).
    float w_sc = B_scale[(size_t)nb * KB + kb];
    #pragma unroll
    for (int i = 0; i < W_TILES_M; i++) {
      int rbase    = warp_m * WM + i * 16;
      int rlo_loc  = rbase + group_id;
      int rhi_loc  = rlo_loc + 8;
      int rlo_glob = m0 + rlo_loc;
      int rhi_glob = m0 + rhi_loc;
      float a_lo = (rlo_glob < M) ? Asc[(size_t)rlo_glob * KB + kb] : 0.0f;
      float a_hi = (rhi_glob < M) ? Asc[(size_t)rhi_glob * KB + kb] : 0.0f;
      float s_lo = a_lo * w_sc;
      float s_hi = a_hi * w_sc;
      #pragma unroll
      for (int j = 0; j < WN_TILES_FP8; j++) {
        outer[i][j][0] += inner[i][j][0] * s_lo;
        outer[i][j][1] += inner[i][j][1] * s_lo;
        outer[i][j][2] += inner[i][j][2] * s_hi;
        outer[i][j][3] += inner[i][j][3] * s_hi;
      }
    }

    stage = 1 - stage;
  }

  // Epilogue: write outer fp32 → bf16.  Fragment layout for m16n8 fp32:
  //   d0: (row = group_id    , col = tid_in_group*2    )
  //   d1: (row = group_id    , col = tid_in_group*2 + 1)
  //   d2: (row = group_id + 8, col = tid_in_group*2    )
  //   d3: (row = group_id + 8, col = tid_in_group*2 + 1)
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++) {
    int rbase = m0 + warp_m * WM + i * 16;
    int rlo   = rbase + group_id;
    int rhi   = rlo + 8;
    #pragma unroll
    for (int j = 0; j < WN_TILES_FP8; j++) {
      int cbase = n0 + warp_n * WN + j * 8;
      int clo   = cbase + tid_in_group * 2;
      int chi   = clo + 1;
      if (rlo < M && clo < N) C[(size_t)rlo * ldC + clo] = outer[i][j][0];
      if (rlo < M && chi < N) C[(size_t)rlo * ldC + chi] = outer[i][j][1];
      if (rhi < M && clo < N) C[(size_t)rhi * ldC + clo] = outer[i][j][2];
      if (rhi < M && chi < N) C[(size_t)rhi * ldC + chi] = outer[i][j][3];
    }
  }
}

// ======================================================================
// FP8 MMA GEMM with fused per-token weighted scatter to fp32 accumulator.
// Used for GEMM2: A fp8 [N_sel, K] × W fp8 [E, N, K]^T → atomicAdd to acc[T, N]
// ======================================================================
__global__ void gemm_fp8_fp8_fused_acc_grouped_kernel(
    const __nv_fp8_e4m3* __restrict__ A_base,     // [N_sel, K] fp8
    const float*         __restrict__ A_scale,    // [N_sel, KB]
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E_LOCAL, N, K] fp8
    const float*         __restrict__ S_base,     // [E_LOCAL, NB, KB]
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    const int32_t*       __restrict__ perm_token,
    const float*         __restrict__ perm_weight,
    int N, int K, int NB, int KB,
    float* __restrict__ acc_out,
    int ldAcc)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_fp8_e4m3* A       = A_base   + (size_t)rs * (size_t)K;
  const float*         Asc     = A_scale  + (size_t)rs * (size_t)KB;
  const __nv_fp8_e4m3* B_fp8   = W_base   + (size_t)e  * (size_t)N * (size_t)K;
  const float*         B_scale = S_base   + (size_t)e  * (size_t)NB * (size_t)KB;
  const int32_t*       Ptok    = perm_token + rs;
  const float*         Pwgt    = perm_weight + rs;

  const int tid          = threadIdx.x;
  const int warp         = tid >> 5;
  const int lane         = tid & 31;
  const int warp_m       = warp / NWARPS_N;
  const int warp_n       = warp % NWARPS_N;
  const int nb           = n0 / BS;
  const int group_id     = lane >> 2;
  const int tid_in_group = lane & 3;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_fp8_e4m3 (*sA)[BM][LDA_FP8_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BM][LDA_FP8_PAD]>(smem_raw);
  __nv_fp8_e4m3 (*sB)[BN][LDB_FP8_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8_PAD]>(smem_raw + SA_FP8_BYTES);

  float outer[W_TILES_M][WN_TILES_FP8][4];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < WN_TILES_FP8; j++)
      #pragma unroll
      for (int q = 0; q < 4; q++) outer[i][j][q] = 0.0f;

  stage_load_A_fp8_BK128(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_BK128(sB[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int kb = 0; kb < KB; kb++) {
    int k      = kb * BK_FP8;
    int next_k = k + BK_FP8;
    if (next_k < K) {
      int ns = 1 - stage;
      stage_load_A_fp8_BK128(sA[ns], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_BK128(sB[ns], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    float inner[W_TILES_M][WN_TILES_FP8][4];
    #pragma unroll
    for (int i = 0; i < W_TILES_M; i++)
      #pragma unroll
      for (int j = 0; j < WN_TILES_FP8; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++) inner[i][j][q] = 0.0f;

    #pragma unroll
    for (int kk = 0; kk < BK_FP8; kk += 32) {
      uint32_t Aregs[W_TILES_M][4];
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++) {
        int rbase = warp_m * WM + i * 16;
        int row0  = rbase + group_id;
        int row1  = rbase + group_id + 8;
        int col0  = kk + tid_in_group * 4;
        int col1  = col0 + 16;
        Aregs[i][0] = *reinterpret_cast<const uint32_t*>(&sA[stage][row0][col0]);
        Aregs[i][1] = *reinterpret_cast<const uint32_t*>(&sA[stage][row1][col0]);
        Aregs[i][2] = *reinterpret_cast<const uint32_t*>(&sA[stage][row0][col1]);
        Aregs[i][3] = *reinterpret_cast<const uint32_t*>(&sA[stage][row1][col1]);
      }
      uint32_t Bregs[WN_TILES_FP8][2];
      #pragma unroll
      for (int j = 0; j < WN_TILES_FP8; j++) {
        int cbase = warp_n * WN + j * 8;
        int col_n = cbase + group_id;
        int rowk0 = kk + tid_in_group * 4;
        int rowk1 = rowk0 + 16;
        Bregs[j][0] = *reinterpret_cast<const uint32_t*>(&sB[stage][col_n][rowk0]);
        Bregs[j][1] = *reinterpret_cast<const uint32_t*>(&sB[stage][col_n][rowk1]);
      }
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WN_TILES_FP8; j++) {
          mma_m16n8k32_e4m3(
              inner[i][j][0], inner[i][j][1], inner[i][j][2], inner[i][j][3],
              Aregs[i][0], Aregs[i][1], Aregs[i][2], Aregs[i][3],
              Bregs[j][0], Bregs[j][1]);
        }
      }
    }

    float w_sc = B_scale[(size_t)nb * KB + kb];
    #pragma unroll
    for (int i = 0; i < W_TILES_M; i++) {
      int rbase    = warp_m * WM + i * 16;
      int rlo_loc  = rbase + group_id;
      int rhi_loc  = rlo_loc + 8;
      int rlo_glob = m0 + rlo_loc;
      int rhi_glob = m0 + rhi_loc;
      float a_lo = (rlo_glob < M) ? Asc[(size_t)rlo_glob * KB + kb] : 0.0f;
      float a_hi = (rhi_glob < M) ? Asc[(size_t)rhi_glob * KB + kb] : 0.0f;
      float s_lo = a_lo * w_sc;
      float s_hi = a_hi * w_sc;
      #pragma unroll
      for (int j = 0; j < WN_TILES_FP8; j++) {
        outer[i][j][0] += inner[i][j][0] * s_lo;
        outer[i][j][1] += inner[i][j][1] * s_lo;
        outer[i][j][2] += inner[i][j][2] * s_hi;
        outer[i][j][3] += inner[i][j][3] * s_hi;
      }
    }

    stage = 1 - stage;
  }

  // Fused epilogue: scale each output fragment by the row's routing weight
  // and atomically add into acc_out[perm_token[row], col].
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++) {
    int rbase    = m0 + warp_m * WM + i * 16;
    int rlo      = rbase + group_id;
    int rhi      = rlo + 8;
    int tlo = (rlo < M) ? Ptok[rlo] : 0;
    int thi = (rhi < M) ? Ptok[rhi] : 0;
    float wlo = (rlo < M) ? Pwgt[rlo] : 0.0f;
    float whi = (rhi < M) ? Pwgt[rhi] : 0.0f;
    #pragma unroll
    for (int j = 0; j < WN_TILES_FP8; j++) {
      int cbase = n0 + warp_n * WN + j * 8;
      int clo   = cbase + tid_in_group * 2;
      int chi   = clo + 1;
      if (rlo < M) {
        if (clo < N) atomicAdd(&acc_out[(size_t)tlo * ldAcc + clo], outer[i][j][0] * wlo);
        if (chi < N) atomicAdd(&acc_out[(size_t)tlo * ldAcc + chi], outer[i][j][1] * wlo);
      }
      if (rhi < M) {
        if (clo < N) atomicAdd(&acc_out[(size_t)thi * ldAcc + clo], outer[i][j][2] * whi);
        if (chi < N) atomicAdd(&acc_out[(size_t)thi * ldAcc + chi], outer[i][j][3] * whi);
      }
    }
  }
}

// Fused SwiGLU + per-(row, kblock) fp8 quantization.
// Reads fp32 G1 [N_sel, II], writes fp8 C [N_sel, I] + fp32 scale [N_sel, NI_BLKS].
__global__ void swiglu_fused_quant_kernel(
    const float* __restrict__ G1,      // [N_sel, II] fp32
    int M,
    __nv_fp8_e4m3* __restrict__ C_fp8, // [N_sel, I]
    float* __restrict__ C_scale)       // [N_sel, NI_BLKS]
{
  const int kb = blockIdx.x;
  const int r  = blockIdx.y;
  const int d  = threadIdx.x;
  if (r >= M) return;

  const int d_global = kb * BS + d;
  float x1   = G1[(size_t)r * II + d_global];
  float x2   = G1[(size_t)r * II + I + d_global];
  float silu = x2 / (1.0f + __expf(-x2));
  float val  = silu * x1;

  // Per-(row, kb) max_abs across BS=128 threads (= 4 warps of 32).
  float a = fabsf(val);
  #pragma unroll
  for (int m = 16; m >= 1; m >>= 1)
    a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
  __shared__ float warp_max[4];
  if ((d & 31) == 0) warp_max[d >> 5] = a;
  __syncthreads();
  __shared__ float blk_sc;
  if (d < 32) {
    float m4 = (d < 4) ? warp_max[d] : 0.0f;
    #pragma unroll
    for (int m = 2; m >= 1; m >>= 1)
      m4 = fmaxf(m4, __shfl_xor_sync(0xffffffff, m4, m));
    if (d == 0) {
      const float FP8_MAX = 448.0f;
      blk_sc = (m4 > 1e-30f) ? (m4 / FP8_MAX) : 1.0f;
    }
  }
  __syncthreads();

  float sc = blk_sc;
  float q  = val / sc;
  if (q >  448.0f) q =  448.0f;
  if (q < -448.0f) q = -448.0f;
  C_fp8[(size_t)r * I + d_global] = (__nv_fp8_e4m3)q;

  if (d == 0) C_scale[(size_t)r * NI_BLKS + kb] = sc;
}

// Gather permuted fp8 hidden states: A_fp8[r, h] = hs_fp8[perm_token[r], h]
__global__ void gather_hs_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ hs_fp8,
    const int32_t*       __restrict__ perm_token,
    int N_sel,
    __nv_fp8_e4m3* __restrict__ A_fp8)
{
  const int r  = blockIdx.x;
  const int hb = blockIdx.y;
  const int d  = threadIdx.x;
  if (r >= N_sel) return;
  int t = perm_token[r];
  int h = hb * BS + d;
  A_fp8[(size_t)r * H + h] = hs_fp8[(size_t)t * H + h];
}

// Gather per-row per-kblock scales: A_scale[r, kb] = hs_scale[kb, perm_token[r]]
__global__ void gather_hs_scale_kernel(
    const float*   __restrict__ hs_scale,    // [NH_BLKS, T]
    const int32_t* __restrict__ perm_token,  // [N_sel]
    int T, int N_sel,
    float* __restrict__ A_scale)             // [N_sel, NH_BLKS]
{
  int r  = blockIdx.x;
  int kb = threadIdx.x;
  if (r >= N_sel || kb >= NH_BLKS) return;
  int t = perm_token[r];
  A_scale[(size_t)r * NH_BLKS + kb] = hs_scale[(size_t)kb * T + t];
}

// ======================================================================
// Kernel 9: SwiGLU.  G1 [M, 2I] -> out [M, I]; out = silu(G1[:, I:]) * G1[:, :I]
// ======================================================================
__global__ void swiglu_kernel(
    const float* __restrict__ G1,
    int M,
    __nv_bfloat16* __restrict__ out)
{
  const int r = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M || d >= I) return;
  float x1 = G1[(size_t)r * II + d];
  float x2 = G1[(size_t)r * II + I + d];
  float silu = x2 / (1.0f + __expf(-x2));
  out[(size_t)r * I + d] = __float2bfloat16(silu * x1);
}

// ======================================================================
// Kernel 10+11: fp32-accumulate scatter, then cast to bf16.
// ======================================================================
__global__ void fin_acc_kernel(
    const __nv_bfloat16* __restrict__ O,        // [N_sel, H]
    const int32_t* __restrict__ perm_token,     // [N_sel]
    const float*   __restrict__ perm_weight,    // [N_sel]
    int N_sel,
    float* __restrict__ acc)                     // [T, H] fp32
{
  const int r = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N_sel || d >= H) return;
  int   t = perm_token[r];
  float w = perm_weight[r];
  float v = __bfloat162float(O[(size_t)r * H + d]) * w;
  atomicAdd(&acc[(size_t)t * H + d], v);
}

__global__ void fin_cast_kernel(
    const float* __restrict__ acc,
    int T,
    __nv_bfloat16* __restrict__ output)
{
  const int t = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T || d >= H) return;
  output[(size_t)t * H + d] = __float2bfloat16(acc[(size_t)t * H + d]);
}

__global__ void zero_f32_kernel(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.0f;
}

__global__ void zero_i32_kernel(int32_t* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0;
}

// ======================================================================
// Persistent workspace.
// ======================================================================
struct Workspace {
  int32_t* d_topk_idx    = nullptr;
  float*   d_topk_w      = nullptr;
  int32_t* d_counts      = nullptr;
  int32_t* d_offs        = nullptr;
  int32_t* d_write_ptr   = nullptr;
  int32_t* d_n_sel       = nullptr;
  int32_t* d_perm_token  = nullptr;
  float*   d_perm_weight = nullptr;
  int32_t* d_active_le   = nullptr;  // [E_LOCAL] active-expert local ids (pinned host + mirrored device)
  int32_t* d_sched_e     = nullptr;  // grouped-GEMM schedule: expert per M-tile
  int32_t* d_sched_tm    = nullptr;  // grouped-GEMM schedule: tile_m per M-tile
  int32_t* h_sched_e     = nullptr;  // pinned host staging
  int32_t* h_sched_tm    = nullptr;  // pinned host staging
  int      sched_cap     = 0;

  __nv_bfloat16* d_hs_bf16    = nullptr;
  __nv_bfloat16* d_W13_bf16   = nullptr;
  __nv_bfloat16* d_W2_bf16    = nullptr;
  __nv_bfloat16* d_A_e        = nullptr;  // bf16 permuted (unused with FP8 GEMM1)
  __nv_fp8_e4m3* d_A_fp8      = nullptr;  // fp8 permuted (GEMM1 input path)
  float*         d_A_scale    = nullptr;  // [N_sel, NH_BLKS] per-row-per-kblock
  float*         d_G1         = nullptr;  // fp32 [N_sel, II] (GEMM1 → SwiGLU)
  __nv_bfloat16* d_C_bf       = nullptr;  // bf16 C (SwiGLU, unused with fp8 path)
  __nv_fp8_e4m3* d_C_fp8      = nullptr;  // [N_sel, I] fp8 (SwiGLU fused-quant output)
  float*         d_C_scale    = nullptr;  // [N_sel, NI_BLKS] per-row per-kblock scale
  __nv_bfloat16* d_O          = nullptr;  // unused with fused GEMM2-acc
  float*         d_out_f32    = nullptr;

  int  cap_T          = 0;
  bool singletons_ok  = false;
  bool weights_alloc  = false;
};
static Workspace g_ws;

static void ensure_singletons() {
  if (g_ws.singletons_ok) return;
  cudaMalloc(&g_ws.d_counts,    E_LOCAL       * sizeof(int32_t));
  cudaMalloc(&g_ws.d_offs,      (E_LOCAL + 1) * sizeof(int32_t));
  cudaMalloc(&g_ws.d_write_ptr, E_LOCAL       * sizeof(int32_t));
  cudaMalloc(&g_ws.d_n_sel,     1             * sizeof(int32_t));
  cudaMalloc(&g_ws.d_active_le, E_LOCAL       * sizeof(int32_t));
  g_ws.singletons_ok = true;
}

static void ensure_schedule(int T) {
  int max_tiles = E_LOCAL + (T * TOP_K + BM - 1) / BM;
  if (g_ws.d_sched_e != nullptr && max_tiles <= g_ws.sched_cap) return;
  if (g_ws.d_sched_e)  cudaFree(g_ws.d_sched_e);
  if (g_ws.d_sched_tm) cudaFree(g_ws.d_sched_tm);
  if (g_ws.h_sched_e)  cudaFreeHost(g_ws.h_sched_e);
  if (g_ws.h_sched_tm) cudaFreeHost(g_ws.h_sched_tm);
  cudaMalloc(&g_ws.d_sched_e,  max_tiles * sizeof(int32_t));
  cudaMalloc(&g_ws.d_sched_tm, max_tiles * sizeof(int32_t));
  cudaMallocHost(&g_ws.h_sched_e,  max_tiles * sizeof(int32_t));
  cudaMallocHost(&g_ws.h_sched_tm, max_tiles * sizeof(int32_t));
  g_ws.sched_cap = max_tiles;
}

static void ensure_weights() {
  if (g_ws.weights_alloc) return;
  cudaMalloc(&g_ws.d_W13_bf16, (size_t)E_LOCAL * II * H * sizeof(__nv_bfloat16));
  cudaMalloc(&g_ws.d_W2_bf16,  (size_t)E_LOCAL * H  * I * sizeof(__nv_bfloat16));
  g_ws.weights_alloc = true;
}

static void ensure_per_token(int T) {
  if (T <= g_ws.cap_T) return;
  if (g_ws.d_topk_idx)    cudaFree(g_ws.d_topk_idx);
  if (g_ws.d_topk_w)      cudaFree(g_ws.d_topk_w);
  if (g_ws.d_hs_bf16)     cudaFree(g_ws.d_hs_bf16);
  if (g_ws.d_perm_token)  cudaFree(g_ws.d_perm_token);
  if (g_ws.d_perm_weight) cudaFree(g_ws.d_perm_weight);
  if (g_ws.d_A_e)         cudaFree(g_ws.d_A_e);
  if (g_ws.d_A_fp8)       cudaFree(g_ws.d_A_fp8);
  if (g_ws.d_A_scale)     cudaFree(g_ws.d_A_scale);
  if (g_ws.d_G1)          cudaFree(g_ws.d_G1);
  if (g_ws.d_C_bf)        cudaFree(g_ws.d_C_bf);
  if (g_ws.d_C_fp8)       cudaFree(g_ws.d_C_fp8);
  if (g_ws.d_C_scale)     cudaFree(g_ws.d_C_scale);
  if (g_ws.d_O)           cudaFree(g_ws.d_O);
  if (g_ws.d_out_f32)     cudaFree(g_ws.d_out_f32);

  // Per-expert Tk ≤ T (each token picks each expert at most once).
  // Total N_sel ≤ T * TOP_K, but we stack buffers by expert so max total rows is T*TOP_K.
  size_t nsmax = (size_t)T * TOP_K;

  cudaMalloc(&g_ws.d_topk_idx,    (size_t)T * TOP_K * sizeof(int32_t));
  cudaMalloc(&g_ws.d_topk_w,      (size_t)T * TOP_K * sizeof(float));
  cudaMalloc(&g_ws.d_hs_bf16,     (size_t)T * H     * sizeof(__nv_bfloat16));
  cudaMalloc(&g_ws.d_perm_token,  nsmax * sizeof(int32_t));
  cudaMalloc(&g_ws.d_perm_weight, nsmax * sizeof(float));
  cudaMalloc(&g_ws.d_A_e,         nsmax * H  * sizeof(__nv_bfloat16));
  cudaMalloc(&g_ws.d_A_fp8,       nsmax * H  * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&g_ws.d_A_scale,     nsmax * NH_BLKS * sizeof(float));
  cudaMalloc(&g_ws.d_G1,          nsmax * II * sizeof(float));
  cudaMalloc(&g_ws.d_C_bf,        nsmax * I  * sizeof(__nv_bfloat16));
  cudaMalloc(&g_ws.d_C_fp8,       nsmax * I  * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&g_ws.d_C_scale,     nsmax * NI_BLKS * sizeof(float));
  cudaMalloc(&g_ws.d_O,           nsmax * H  * sizeof(__nv_bfloat16));
  cudaMalloc(&g_ws.d_out_f32,     (size_t)T * H * sizeof(float));
  g_ws.cap_T = T;
}

}  // namespace

// ======================================================================
// Host entry.
// ======================================================================
using tvm::ffi::TensorView;

static void run_impl(
    TensorView routing_logits,       // [T, 256] fp32
    TensorView routing_bias,         // [256] bf16
    TensorView hidden_states,        // [T, 7168] fp8_e4m3
    TensorView hidden_states_scale,  // [56, T] fp32
    TensorView gemm1_weights,        // [32, 4096, 7168] fp8_e4m3
    TensorView gemm1_weights_scale,  // [32, 32, 56] fp32
    TensorView gemm2_weights,        // [32, 7168, 2048] fp8_e4m3
    TensorView gemm2_weights_scale,  // [32, 56, 16] fp32
    int64_t    local_expert_offset,
    double     routed_scaling_factor,
    TensorView output)               // [T, 7168] bf16
{
  const int T = (int)routing_logits.shape()[0];
  if (T == 0) return;
  const int leo   = (int)local_expert_offset;
  const float sf  = (float)routed_scaling_factor;

  cudaStream_t stream = 0;

  ensure_singletons();
  ensure_weights();
  ensure_per_token(T);
  ensure_schedule(T);

  static bool gemm_shmem_attr_set = false;
  if (!gemm_shmem_attr_set) {
    cudaFuncSetAttribute(gemm_bf16_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES);
    cudaFuncSetAttribute(gemm_bf16_fp8w_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES_V2);
    cudaFuncSetAttribute(gemm_bf16_fp8w_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES_GRP);
    cudaFuncSetAttribute(gemm_bf16_fp8w_fused_acc_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES_GRP);
    cudaFuncSetAttribute(gemm_fp8_fp8_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_FP8_SMEM_BYTES);
    cudaFuncSetAttribute(gemm_fp8_fp8_fused_acc_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_FP8_SMEM_BYTES);
    gemm_shmem_attr_set = true;
  }

  const float*         p_logits  = static_cast<const float*>(routing_logits.data_ptr());
  const __nv_bfloat16* p_bias    = static_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const __nv_fp8_e4m3* p_hs_fp8  = static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr());
  const float*         p_hs_sc   = static_cast<const float*>(hidden_states_scale.data_ptr());
  const __nv_fp8_e4m3* p_w13_fp8 = static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr());
  const float*         p_w13_sc  = static_cast<const float*>(gemm1_weights_scale.data_ptr());
  const __nv_fp8_e4m3* p_w2_fp8  = static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr());
  const float*         p_w2_sc   = static_cast<const float*>(gemm2_weights_scale.data_ptr());
  __nv_bfloat16*       p_out     = static_cast<__nv_bfloat16*>(output.data_ptr());

  // 1. routing
  route_kernel<<<T, E_GLOBAL, 0, stream>>>(
      p_logits, p_bias, sf, T, g_ws.d_topk_idx, g_ws.d_topk_w);

  // 2. counts / scan / fill
  zero_i32_kernel<<<1, E_LOCAL, 0, stream>>>(g_ws.d_counts, E_LOCAL);
  {
    int th = 256;
    perm_count_kernel<<<(T + th - 1) / th, th, 0, stream>>>(
        g_ws.d_topk_idx, T, leo, g_ws.d_counts);
  }
  scan_offsets_kernel<<<1, 1, 0, stream>>>(
      g_ws.d_counts, g_ws.d_offs, g_ws.d_write_ptr, g_ws.d_n_sel);
  {
    int th = 256;
    perm_fill_kernel<<<(T + th - 1) / th, th, 0, stream>>>(
        g_ws.d_topk_idx, g_ws.d_topk_w, T, leo,
        g_ws.d_write_ptr, g_ws.d_perm_token, g_ws.d_perm_weight);
  }

  // 3. Read offs to host (blocks until perm done).
  int32_t offs_host[E_LOCAL + 1];
  cudaMemcpyAsync(offs_host, g_ws.d_offs, (E_LOCAL + 1) * sizeof(int32_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  const int N_sel = offs_host[E_LOCAL];

  // 5. Build active-expert list on host, copy to device.
  int32_t active_host[E_LOCAL];
  int num_active = 0;
  for (int e = 0; e < E_LOCAL; e++) {
    if (offs_host[e + 1] > offs_host[e]) active_host[num_active++] = e;
  }
  if (num_active > 0) {
    cudaMemcpyAsync(g_ws.d_active_le, active_host,
                    num_active * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
  }

  // 6. Gather permuted fp8 hidden states + per-row per-kblock scale (for FP8 MMA GEMM1).
  if (N_sel > 0) {
    {
      dim3 grid(N_sel, NH_BLKS);
      gather_hs_fp8_kernel<<<grid, BS, 0, stream>>>(
          p_hs_fp8, g_ws.d_perm_token, N_sel, g_ws.d_A_fp8);
    }
    gather_hs_scale_kernel<<<N_sel, NH_BLKS, 0, stream>>>(
        p_hs_sc, g_ws.d_perm_token, T, N_sel, g_ws.d_A_scale);

    // BF16 fallback path also needs d_hs_bf16 + d_A_e.
    {
      dim3 grid(T, NH_BLKS);
      dequant_hs_kernel<<<grid, BS, 0, stream>>>(
          p_hs_fp8, p_hs_sc, T, g_ws.d_hs_bf16);
    }
    {
      dim3 grid(N_sel, NH_BLKS);
      gather_hs_kernel<<<grid, BS, 0, stream>>>(
          g_ws.d_hs_bf16, g_ws.d_perm_token, N_sel, g_ws.d_A_e);
    }
  }

  // 7. zero fp32 accumulator
  zero_f32_kernel<<<(((size_t)T * H) + 255) / 256, 256, 0, stream>>>(
      g_ws.d_out_f32, (size_t)T * H);

  // Build grouped-GEMM schedule on host: one entry per (expert, M-tile).
  int total_m_tiles = 0;
  for (int e = 0; e < E_LOCAL; e++) {
    int Tk = offs_host[e + 1] - offs_host[e];
    int nmt = (Tk + BM - 1) / BM;
    for (int m = 0; m < nmt; m++) {
      g_ws.h_sched_e[total_m_tiles]  = e;
      g_ws.h_sched_tm[total_m_tiles] = m;
      total_m_tiles++;
    }
  }
  if (total_m_tiles > 0) {
    cudaMemcpyAsync(g_ws.d_sched_e,  g_ws.h_sched_e,
                    total_m_tiles * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_ws.d_sched_tm, g_ws.h_sched_tm,
                    total_m_tiles * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // 8a. Grouped GEMM1 — FP8×FP8 tensor cores, per-kblock block-scale accumulation.
    {
      dim3 grid((II + BN - 1) / BN, total_m_tiles);
      gemm_fp8_fp8_grouped_kernel<<<grid, THREADS_GEMM, GEMM_FP8_SMEM_BYTES, stream>>>(
          g_ws.d_A_fp8, g_ws.d_A_scale,
          p_w13_fp8, p_w13_sc,
          g_ws.d_offs, g_ws.d_sched_e, g_ws.d_sched_tm,
          II, H, NII_BLKS, NH_BLKS,
          g_ws.d_G1, II);
    }
    // 8b. SwiGLU over all N_sel rows (fp32 G1 → bf16 C).
    {
      int th = 128;
      dim3 grid((I + th - 1) / th, N_sel);
      swiglu_kernel<<<grid, th, 0, stream>>>(g_ws.d_G1, N_sel, g_ws.d_C_bf);
    }
    // 8c. Grouped GEMM2 + fused per-token weighted scatter (skips bf16 O intermediate).
    {
      dim3 grid((H + BN - 1) / BN, total_m_tiles);
      gemm_bf16_fp8w_fused_acc_grouped_kernel<<<grid, THREADS_GEMM, GEMM_SMEM_BYTES_GRP, stream>>>(
          g_ws.d_C_bf, p_w2_fp8, p_w2_sc,
          g_ws.d_offs, g_ws.d_sched_e, g_ws.d_sched_tm,
          g_ws.d_perm_token, g_ws.d_perm_weight,
          H, I, NH_BLKS, NI_BLKS,
          g_ws.d_out_f32, H);
    }
  }

  // 9. cast fp32 -> bf16 output
  {
    int th = 128;
    dim3 grid((H + th - 1) / th, T);
    fin_cast_kernel<<<grid, th, 0, stream>>>(g_ws.d_out_f32, T, p_out);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_moe, run_impl);
