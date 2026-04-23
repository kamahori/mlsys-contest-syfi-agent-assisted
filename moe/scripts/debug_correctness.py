"""Compare kernel vs python reference on one workload with strict tolerances."""
import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from scripts.pack_solution import pack_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    trace_set = TraceSet.from_path(os.environ["FIB_DATASET_PATH"])
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads[solution.definition]
    wl_trace = workloads[args.workload_index]
    wl = wl_trace.workload
    print(f"workload={wl.uuid} axes={wl.axes}")

    safe = load_safetensors(definition, wl, trace_set_root=trace_set.root)
    inputs = gen_inputs(definition, wl, args.device, safe_tensors=safe)
    input_names = list(definition.inputs.keys())

    # Reference
    ns = {"__name__": "ref_mod"}
    exec(compile(definition.reference, "<ref>", "exec"), ns)
    ref_fn = ns["run"]
    ref_kwargs = dict(zip(input_names, inputs))
    with torch.no_grad():
        ref_out = ref_fn(**ref_kwargs)
    ref_out = ref_out.to(torch.bfloat16).to(args.device)

    # Kernel
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)
    outputs = allocate_outputs(definition, inputs, args.device)
    # warmup
    with torch.no_grad():
        runnable(*inputs, *outputs)
    torch.cuda.synchronize()

    my_out = outputs[0]
    r = ref_out.to(torch.float32)
    m = my_out.to(torch.float32)
    diff = (m - r).abs()
    rel  = diff / (r.abs() + 1e-8)

    print(f"\n--- Output magnitudes ---")
    print(f"ref  range=[{r.min().item():.3f},{r.max().item():.3f}] |mean|={r.abs().mean().item():.4f}")
    print(f"mine range=[{m.min().item():.3f},{m.max().item():.3f}] |mean|={m.abs().mean().item():.4f}")
    print(f"\n--- Error ---")
    print(f"max abs_err={diff.max().item():.4f}  max rel_err={rel.max().item():.4f}")
    print(f"mean abs_err={diff.mean().item():.4f}  mean rel_err={rel.mean().item():.4f}")
    # Contest MoE tolerances: atol=1, rtol=0.3, required_matched_ratio=0.9
    print("\n--- Contest MoE tolerances (atol=1, rtol=0.3, ratio=0.9) ---")
    fail_c = (diff > 1.0) & (rel > 0.3)
    match_c = 1 - fail_c.float().mean().item()
    print(f"  match_ratio={match_c:.6f} ({'PASS' if match_c >= 0.9 else 'FAIL'}, need >=0.9)")
    for rtol in (1e-2, 5e-3, 1e-3):
        for atol in (1e-2, 1e-1, 1.0):
            fail = (diff > atol) & (rel > rtol)
            ratio = 1 - fail.float().mean().item()
            print(f"  rtol={rtol} atol={atol} match={ratio:.6f} fail={int(fail.sum())}/{fail.numel()}")

    # Per-row diagnostics at rtol=atol=1e-2
    fail = (diff > 1e-2) & (rel > 1e-2)
    per_row_fail = fail.float().mean(dim=1)
    per_row_max_abs = diff.max(dim=1).values
    per_row_ref_abs = r.abs().max(dim=1).values
    per_row_my_abs  = m.abs().max(dim=1).values
    print("\nPer-row:")
    for t in range(r.shape[0]):
        print(f"  t={t}: fail_frac={per_row_fail[t].item():.4f}"
              f"  max_abs_err={per_row_max_abs[t].item():.2f}"
              f"  ref_max_abs={per_row_ref_abs[t].item():.2f}"
              f"  my_max_abs={per_row_my_abs[t].item():.2f}")


if __name__ == "__main__":
    main()
