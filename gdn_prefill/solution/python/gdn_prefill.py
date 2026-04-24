from __future__ import annotations

import torch

from .cute_gdn_prefill_tcgen05_v1 import gdn_prefill_tcgen05_v1


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    out, st = gdn_prefill_tcgen05_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale
    )
    output.copy_(out)
    new_state.copy_(st)
