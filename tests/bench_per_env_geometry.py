"""Performance-regression bench for the per-environment geometry features.

Its purpose is to prove the zero-cost-default invariant: with the per-env geometry/scale options left
off, homogeneous scenes must compile identical kernels and keep the same steady-state FPS and peak GPU
memory. It reuses the scene factories and ``run_benchmark`` harness from ``test_rigid_benchmarks`` and
records ``runtime_fps`` plus peak GPU memory per case.

Each case runs in a fresh subprocess so Genesis initialization is isolated between scenes.

Usage:
    python -m tests.bench_per_env_geometry --baseline out.json [--backend gpu] [--n-envs 8192]
    python -m tests.bench_per_env_geometry --compare out.json  [--backend gpu] [--n-envs 8192]
    python -m tests.bench_per_env_geometry --run-case franka --backend gpu --n-envs 8192   # internal
"""

import argparse
import json
import subprocess
import sys

import genesis as gs
from tests.test_rigid_benchmarks import make_franka, make_go2, make_anymal, run_benchmark

CASES = {
    "franka": make_franka,
    "go2": make_go2,
    "anymal": make_anymal,
}

# A homogeneous-scene FPS regression beyond this fraction fails the compare (noise threshold).
FPS_REGRESSION_TOL = 0.03


def _run_case(case, backend, n_envs):
    gs.init(backend=getattr(gs, backend), precision="32" if backend == "gpu" else "64")
    scene, step, meta = CASES[case](n_envs=n_envs)
    peak_mem_mb = 0.0
    if backend == "gpu":
        import torch

        torch.cuda.reset_peak_memory_stats()
    result = run_benchmark(step, n_envs=n_envs, meta=meta)
    if backend == "gpu":
        import torch

        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {"runtime_fps": result["runtime_fps"], "peak_mem_mb": round(peak_mem_mb, 1)}


def _collect(backend, n_envs):
    results = {}
    for case in CASES:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.bench_per_env_geometry",
                "--run-case",
                case,
                "--backend",
                backend,
                "--n-envs",
                str(n_envs),
            ],
            capture_output=True,
            text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError(f"case {case} failed:\n{proc.stdout}\n{proc.stderr}")
        results[case] = json.loads(line[len("RESULT ") :])
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", metavar="OUT.json")
    group.add_argument("--compare", metavar="BASE.json")
    group.add_argument("--run-case", choices=list(CASES))
    parser.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--n-envs", type=int, default=8192)
    args = parser.parse_args()

    if args.run_case is not None:
        print("RESULT " + json.dumps(_run_case(args.run_case, args.backend, args.n_envs)))
        return

    results = _collect(args.backend, args.n_envs)

    if args.baseline is not None:
        with open(args.baseline, "w") as f:
            json.dump({"backend": args.backend, "n_envs": args.n_envs, "cases": results}, f, indent=2)
        print(f"wrote baseline to {args.baseline}: {results}")
        return

    with open(args.compare) as f:
        base = json.load(f)["cases"]
    regressed = False
    for case, cur in results.items():
        b = base.get(case)
        if b is None:
            print(f"{case}: NEW {cur}")
            continue
        fps_delta = (cur["runtime_fps"] - b["runtime_fps"]) / b["runtime_fps"]
        mem_delta = cur["peak_mem_mb"] - b["peak_mem_mb"]
        flag = " REGRESSION" if fps_delta < -FPS_REGRESSION_TOL else ""
        regressed |= fps_delta < -FPS_REGRESSION_TOL
        print(
            f"{case}: fps {b['runtime_fps']} -> {cur['runtime_fps']} ({fps_delta:+.1%}), "
            f"mem {b['peak_mem_mb']} -> {cur['peak_mem_mb']} MB ({mem_delta:+.1f}){flag}"
        )
    sys.exit(1 if regressed else 0)


if __name__ == "__main__":
    main()
