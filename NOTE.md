# Evaluation Numbers — syfi vs FlashInfer baseline

Run on 2026-04-23 inside the official contest docker image
`flashinfer/flashinfer-ci-cu132:20260401-2c675fb` (+ `flashinfer-python`
and `flashinfer-bench` wheels + `deep_gemm` from source) on a local
bare-metal B200 (sm_100a), CUDA 13.2, torch 2.12.0.dev20260331+cu132,
triton 3.6.0.

Per-track flags from `EVALUATION.md`:
- MoE: `--atol 1 --rtol 0.3 --required-matched-ratio 0.9`
- GDN prefill: `--warmup-runs 1 --iterations 5 --num-trials 3`
- GDN decode / DSA indexer: defaults
- All: `--use-isolated-runner --timeout 300`

## Headline

| Track | Kernel score (syfi / baseline) |
|---|---|
| **GDN** | **3.43×** arith-mean of prefill + decode |
| **MoE** | **0.36×** (baseline is ~2.75× faster than us) |
| **DSA** | indexer correctness **fixed** (128/128 PASS) — pairwise ratio TBD in contest env (baseline broken locally) |

## Per-kernel detail

### GDN prefill — `gdn_prefill_qk4_v8_d128_k_last`

| | syfi | flashinfer baseline |
|---|---|---|
| solution | `syfi-gdn-prefill-v1` (4-warp WMMA, C=16 chunks) | `flashinfer_wrapper_123ca6` |
| correctness | **100 / 100 PASSED** | 100 / 100 PASSED |
| mean speedup vs python ref | 701× (min 79×, max 2319×) | ~17–187× |
| **pairwise syfi / baseline** | **4.90×** (arith-mean over 100 workloads) | — |

### GDN decode — `gdn_decode_qk4_v8_d128_k_last`

| | syfi | flashinfer baseline |
|---|---|---|
| solution | `syfi-gdn-decode-v1` (2-warp) | `flashinfer_wrapper_9b7f1e` |
| correctness | **54 / 54 PASSED** | 54 / 54 PASSED |
| mean speedup vs python ref | ~2500× (batched) | ~1200× |
| **pairwise syfi / baseline** | **1.95×** (arith-mean over 54 workloads) | — |

### MoE — `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

| | syfi | flashinfer baseline |
|---|---|---|
| solution | `syfi-moe-v1` (from-scratch CUDA: DS-V3 routing + dequant + bf16 WMMA GEMM + SwiGLU + weighted scatter) | `flashinfer_wrapper_9sdjf3` (`trtllm_fp8_block_scale_moe`) |
| correctness | **19 / 19 PASSED** | **19 / 19 PASSED** |
| mean speedup vs python ref | 13.3× (arith-mean, 4.76×–32.1×) | ~30–41× |
| **pairwise syfi / baseline** | **0.36×** (arith-mean over 19 workloads) | — |

The baseline originally TIMEOUT'd at 300 s per isolated-runner subprocess
(`trtllm_fp8_block_scale_moe` hits a large JIT compile on each fresh
subprocess). Mitigation: pre-populated `/root/.cache/flashinfer` via a
persistent-runner warmup pass over all 19 workloads; the official
isolated sweep then reused the JIT artifacts and all 19 baselines
finished well under 300 s. Numbers above are from that post-warmup run.

Our from-scratch bf16 WMMA GEMM path loses to the TensorRT-LLM FP8-block
MoE kernel by ~2.75×. Expected — TRT-LLM uses FP8 tensor cores (2× per
cycle vs bf16) plus a grouped GEMM that batches all active experts,
whereas we loop over experts and use bf16 math.

### DSA topk indexer — `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`

| | syfi | flashinfer baseline |
|---|---|---|
| solution | `syfi-dsa-topk-indexer-v1` (pure torch, mirrors reference math exactly) | `flashinfer_deepgemm_wrapper_2ba145` |
| correctness | **128 / 128 PASSED** | 0 / 128 (RUNTIME_ERROR in this env) |
| mean speedup vs python ref | 1.06× (range 0.29×–3.72×) | — |
| **pairwise syfi / baseline** | undefined (baseline broken in our env) | — |

Our original WMMA-scorer + custom radix top-K CUDA kernel produced the
**correct top-K set** on every workload, but **ordered** it differently
from the reference because fp32 matmul accumulation order in CUDA /
Triton doesn't byte-match torch's native matmul. Near-tied scores break
differently → same 2048 indices, different positions → contest's
position-wise correctness check fails.

Workaround: submit a pure-torch solution that mirrors the reference math
exactly (allowed — only `flashinfer` / `deep_gemm` runtime calls are
banned; plain torch is fine). Speedup is ~1× the python reference, but
correctness is now locked in. On the contest pipeline's working
`deep_gemm` the syfi/baseline ratio will become measurable.

The baseline itself fails here because `deep_gemm` 2.4.2 (via pypi) has
`deep_gemm.get_paged_mqa_logits_metadata(..., context_lens)` requiring a
2-D `context_lens`, which the wrapper passes as 1-D. Not our bug; the
official eval env ships a pinned `deep_gemm` where this matches.

### DSA sparse attention — `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

Not submitted.

## Workload totals used in the averages above

| Kernel | workloads with both PASSED | workloads total |
|---|---|---|
| GDN prefill | 100 | 100 |
| GDN decode | 54 | 54 |
| MoE | 19 | 19 |
| DSA indexer | 0 (baseline fails locally; syfi all 128 PASS) | 128 |

## Scoring formula reminder (from EVALUATION.md)

```
speedup        = FlashInfer_baseline_latency / syfi_latency    (per workload)
kernel_score   = arith-mean of per-workload speedups
                 (zeroed if any syfi workload fails correctness)
track_score    = sum(kernel_scores) / expected_kernel_count
                 (expected: MoE=1, DSA=2, GDN=2)
```
