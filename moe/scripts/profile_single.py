"""Run one workload through the compiled kernel for ncu profiling.

Usage:
  CUDA_VISIBLE_DEVICES=3 FIB_DATASET_PATH=... \\
    ncu --set full --kernel-name 'regex:.*gemm.*grouped_kernel' \\
        --launch-count 2 --export /tmp/moe_prof \\
        python scripts/profile_single.py --workload-index 8
"""
import argparse, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from scripts.pack_solution import pack_solution


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload-index", type=int, default=8)  # big seq_len by default
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=1)
    args = p.parse_args()

    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    ts = TraceSet.from_path(os.environ["FIB_DATASET_PATH"])
    definition = ts.definitions[solution.definition]
    wls = ts.workloads[solution.definition]
    wl = wls[args.workload_index].workload
    print(f"workload={wl.uuid} seq_len={wl.axes.get('seq_len')}", flush=True)

    safe = load_safetensors(definition, wl, trace_set_root=ts.root)
    inputs = gen_inputs(definition, wl, args.device, safe_tensors=safe)

    runnable = BuilderRegistry.get_instance().build(definition, solution)
    outputs = allocate_outputs(definition, inputs, args.device)

    # Warmup (outside the ncu profile window by default — ncu attaches later).
    with torch.no_grad():
        runnable(*inputs, *outputs)
    torch.cuda.synchronize()

    # Profile region: run `iters` times.
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        for _ in range(args.iters):
            runnable(*inputs, *outputs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("done", flush=True)


if __name__ == "__main__":
    main()
