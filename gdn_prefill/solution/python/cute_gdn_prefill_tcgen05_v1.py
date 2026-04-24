"""
GDN prefill — WY-chunked, tcgen05 via dense_gemm for the 4 per-chunk matmuls.

Session 4 of the tcgen05 staging plan.  Structurally identical to
cute_gdn_prefill_wy_chunked.py, but each of `KS/QS/KK/QK` is computed
by the tcgen05-backed `dense_gemm(A, B) = A @ B^T` Blackwell kernel.
`Attn = QKmb @ u` and the state update `K.T @ u` stay in torch for now
(u is fp32; dense_gemm wants bf16 K-contiguous operands, so those need
rearrangement that would cost more than they save at C=64).

Known limitations that motivate the next sessions:
  * `dense_gemm` has `mma_tiler_mn = (128, 128)` hardcoded.  At `CHUNK=64`
    each call only half-fills the MMA in the M dim, so ~50 % of the mma
    bandwidth is wasted.  The wall-clock win over torch.matmul is
    therefore modest.  Real wins need either:
      - a dense_gemm_m64 variant (smaller mma_tiler),
      - CHUNK=128 (needs log-space γ to avoid the fp32 underflow we
        document in the plan), OR
      - a fused kernel where the 4 matmuls share TMA loads of Q/K/S.
  * dense_gemm returns bf16.  Downstream (tri-solve / QKmb) wants fp32.
    We cast .float() at the matmul boundary, costing a small amount of
    precision vs the current CUDA kernel's in-shmem fp32 accumulator.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .cute_dense_gemm import dense_gemm


CHUNK = 64


def _dense_gemm_ab_t(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrap dense_gemm — which wants bf16 inputs and returns bf16 — with
    fp32 casting at the boundaries for the GDN scalar paths."""
    A_bf = A.to(torch.bfloat16).contiguous()
    B_bf = B.to(torch.bfloat16).contiguous()
    return dense_gemm(A_bf, B_bf).float()


def _wy_chunk_tcgen05(
    Q: torch.Tensor,    # [C, D] fp32
    K: torch.Tensor,    # [C, D] fp32
    V: torch.Tensor,    # [C, D] fp32
    S0: torch.Tensor,   # [D, D] fp32  (internal [K_dim, V_dim]; but k-last gmem
                        #              storage already has K contiguous, so
                        #              dense_gemm(X, S_gmem) directly gives
                        #              X @ S_internal — see below)
    S0_gmem_klast: torch.Tensor,  # [D, D] fp32 with K contiguous (≡ S0.T)
    g: torch.Tensor,    # [C]    fp32
    beta: torch.Tensor, # [C]    fp32
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk of WY-form GDN with tcgen05 matmuls."""
    C = Q.shape[0]
    device = Q.device

    gamma = torch.cumprod(g, dim=0)
    gamma_last = gamma[-1]

    # --- 4 per-chunk matmuls via tcgen05 ---
    # KS[c, v] = Σ_k K[c, k] · S_internal[k, v]
    #         = Σ_k K[c, k] · S_gmem_klast[v, k]      (since S_gmem_klast = S_internal.T)
    #         = (K @ S_gmem_klast.T)[c, v]
    #         = dense_gemm(K, S_gmem_klast)[c, v]    (dense_gemm computes A @ B.T)
    KS = _dense_gemm_ab_t(K, S0_gmem_klast)           # [C, D]
    QS = _dense_gemm_ab_t(Q, S0_gmem_klast)           # [C, D]
    KK = _dense_gemm_ab_t(K, K)                        # [C, C]
    QK = _dense_gemm_ab_t(Q, K)                        # [C, C]

    # --- WY tri-solve ---
    v_hat = V / gamma.unsqueeze(-1)
    rhs = beta.unsqueeze(-1) * (v_hat - KS)
    tri = torch.ones(C, C, device=device, dtype=torch.bool).tril(-1)
    L = (beta.unsqueeze(-1) * KK) * tri
    u = torch.empty_like(rhs)
    u[0] = rhs[0]
    for t in range(1, C):
        u[t] = rhs[t] - L[t, :t] @ u[:t]

    # --- output (Attn matmul stays in torch: u is fp32 and small) ---
    causal = torch.ones(C, C, device=device, dtype=torch.bool).tril(0)
    QKmb = QK * causal
    o = scale * gamma.unsqueeze(-1) * (QS + QKmb @ u)

    # --- state update (also in torch: K.T layout fights dense_gemm) ---
    #   S_C_internal = γ_last · (S0_internal + K.T @ u)
    #   store in k-last gmem:  S_gmem[v, k] = S_C_internal[k, v]
    S_new_internal = S0 + K.T @ u                      # [D, D] (k, v)
    S_new_internal = gamma_last * S_new_internal
    S_new_gmem = S_new_internal.transpose(-1, -2).contiguous()

    return o, S_new_gmem


def _wy_chunk_tcgen05_partial(
    Q, K, V, S0, S0_gmem_klast, g, beta, scale, C_actual,
):
    """Handle tail chunks where C_actual < CHUNK (no dense_gemm here —
    small matmuls don't justify tcgen05's tile overhead and we'd need to
    pad to 128 anyway)."""
    gamma = torch.cumprod(g[:C_actual], dim=0)
    gamma_last = gamma[-1]
    Q_r = Q[:C_actual]; K_r = K[:C_actual]; V_r = V[:C_actual]
    b_r = beta[:C_actual]

    KS = K_r @ S0
    QS = Q_r @ S0
    KK = K_r @ K_r.T
    QK = Q_r @ K_r.T

    v_hat = V_r / gamma.unsqueeze(-1)
    rhs = b_r.unsqueeze(-1) * (v_hat - KS)
    tri = torch.ones(C_actual, C_actual, device=Q.device, dtype=torch.bool).tril(-1)
    L = (b_r.unsqueeze(-1) * KK) * tri
    u = torch.empty_like(rhs)
    u[0] = rhs[0]
    for t in range(1, C_actual):
        u[t] = rhs[t] - L[t, :t] @ u[:t]

    causal = torch.ones(C_actual, C_actual, device=Q.device, dtype=torch.bool).tril(0)
    QKmb = QK * causal
    o = scale * gamma.unsqueeze(-1) * (QS + QKmb @ u)

    S_new_internal = gamma_last * (S0 + K_r.T @ u)
    S_new_gmem = S_new_internal.transpose(-1, -2).contiguous()
    return o, S_new_gmem


@torch.no_grad()
def gdn_prefill_tcgen05_v1(
    q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None,
    chunk_size: int = CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4 and num_v_heads == 8 and head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_q_heads, dim=1)

    output = torch.zeros(
        total_seq_len, num_sab_heads, head_size,
        dtype=torch.bfloat16, device=device,
    )
    new_state = torch.zeros(
        num_seqs, num_sab_heads, head_size, head_size,
        dtype=torch.float32, device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end   = int(cu_seqlens[seq_idx + 1].item())
        seq_len   = seq_end - seq_start
        if seq_len <= 0:
            continue

        for h in range(num_sab_heads):
            if state is not None:
                # Contest state is [N, H_v, V_DIM, K_DIM] k-last; that IS the
                # S_gmem_klast layout dense_gemm wants for its B operand.
                # Internal [K_dim, V_dim] layout (for torch tri-solve /
                # state update) is S_gmem.T.
                S_gmem_klast = state[seq_idx, h].float().contiguous()
                S_internal   = S_gmem_klast.T.contiguous()
            else:
                S_gmem_klast = torch.zeros(head_size, head_size, dtype=torch.float32, device=device)
                S_internal   = torch.zeros(head_size, head_size, dtype=torch.float32, device=device)

            for chunk_start in range(0, seq_len, chunk_size):
                C_actual = min(chunk_size, seq_len - chunk_start)
                ts = seq_start + chunk_start
                te = seq_start + chunk_start + C_actual

                Q_c = q_exp[ts:te, h].float()
                K_c = k_exp[ts:te, h].float()
                V_c = v[ts:te, h].float()
                g_c = g[ts:te, h]
                b_c = beta[ts:te, h]

                if C_actual == chunk_size:
                    o_c, S_gmem_klast = _wy_chunk_tcgen05(
                        Q_c, K_c, V_c, S_internal, S_gmem_klast, g_c, b_c, scale,
                    )
                else:
                    o_c, S_gmem_klast = _wy_chunk_tcgen05_partial(
                        Q_c, K_c, V_c, S_internal, S_gmem_klast, g_c, b_c, scale, C_actual,
                    )
                S_internal = S_gmem_klast.T.contiguous()
                output[ts:te, h] = o_c.to(torch.bfloat16)

            new_state[seq_idx, h] = S_gmem_klast

    return output, new_state


# =============================================================================
# Self-test against the scalar reference
# =============================================================================

def _self_test():
    import os, sys
    sys.path.insert(0, os.getcwd())
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    ts = TraceSet.from_path(os.environ.get("FIB_DATASET_PATH", "mlsys26-contest"))
    defn = ts.definitions["gdn_prefill_qk4_v8_d128_k_last"]
    wls = ts.workloads["gdn_prefill_qk4_v8_d128_k_last"]

    test_indices = [0, 1, 22, 38, 90]
    for i in test_indices:
        wt = wls[i]
        w = wt.workload
        tensors = load_safetensors(defn, w, trace_set_root=ts.root)
        inputs = gen_inputs(defn, w, "cuda:0", safe_tensors=tensors)

        ref_runnable = BuilderRegistry.get_instance().build_reference(defn)
        exp_out, exp_state = ref_runnable(*inputs)
        ours_out, ours_state = gdn_prefill_tcgen05_v1(*inputs)

        abs_err_out = (ours_out.float() - exp_out.float()).abs().max().item()
        abs_err_st  = (ours_state - exp_state).abs().max().item()
        rel_err_out = (
            (ours_out.float() - exp_out.float()).abs()
            / exp_out.float().abs().clamp_min(1e-6)
        ).max().item()
        nan = torch.isnan(ours_out.float()).any().item() or torch.isnan(ours_state).any().item()
        print(
            f"idx={i:3d}  axes={w.axes}  out_abs={abs_err_out:.2e}  "
            f"state_abs={abs_err_st:.2e}  out_rel={rel_err_out:.2e}"
        )
        if nan:
            print(f"  ⚠ NaN produced")
        elif abs_err_out > 0.1 or abs_err_st > 0.1:
            print(f"  ⚠ mismatch exceeds 0.1")
        else:
            print(f"  OK")


if __name__ == "__main__":
    _self_test()
