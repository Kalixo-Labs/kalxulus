# benchmark_kalxulus.py
from __future__ import annotations

import gc
import math
import time
from typing import Callable, Iterable, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from src.kalxulus import Kalxulus


# ---- X generators ------------------------------------------------------------

def x_uniform(n: int, L: float) -> np.ndarray:
    return np.linspace(0.0, L, n)

def x_power(n: int, L: float, p: float) -> np.ndarray:
    # Quadratic-ish spacing when p=2, cubic when p=3, etc. Nonuniform.
    return np.linspace(0.0, L, n) ** p

def x_geometric(n: int, L: float, r: float) -> np.ndarray:
    # Geometric progression (monotone increasing). r>1 increases clustering near 0.
    g = np.geomspace(1.0, r, n)
    g = (g - g.min()) / (g.max() - g.min())
    return g * L

def x_cluster_tanh(n: int, L: float, k: float) -> np.ndarray:
    # Cluster near the center with a tanh warp; k controls clustering strength.
    t = np.linspace(-1.0, 1.0, n)
    y = np.tanh(k * t)
    y = (y - y.min()) / (y.max() - y.min())
    return y * L


# ---- Benchmarking ------------------------------------------------------------

def time_call(fn: Callable[[], Any], repeats: int = 1) -> Tuple[float, Any]:
    """
    Run fn() 'repeats' times, return (median_duration, last_result).
    """
    times = []
    result = None
    for _ in range(repeats):
        gc.disable()
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        gc.enable()
        times.append(dt)
    times.sort()
    median = times[len(times) // 2]
    return median, result


def benchmark_error_estimator(
    solvers: Iterable[str],
    eos: Iterable[float],
    num_points_list: Iterable[int],
    x_specs: Iterable[Dict[str, Any]],
    *,
    max_points: int = 21,
    only_nonuniform: bool = False,
    uniform_tol: float = 1e-12,
    max_iters_factor: int = 3,
    repeats: int = 1,
    # If your Kalxulus uses "tolerance", set KALX_EXTRA_KWARGS={"tolerance": 1e-8}
    # If it uses "coeff_tolerance", set that instead. You can pass both; unused is ignored if you drop it.
    KALX_EXTRA_KWARGS: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Grid-search benchmark. x_specs is a list of dicts describing x generators.
    Each x_spec must define:
        - "name": str
        - "gen": callable -> np.ndarray
        - Any additional kwargs needed by the generator
        - "n": int       (number of points)
        - "L": float     (domain length)

    Example x_specs element:
        {"name": "power_p2_n200", "gen": x_power, "n": 200, "L": 3.0, "p": 2.0}
    """
    if KALX_EXTRA_KWARGS is None:
        KALX_EXTRA_KWARGS = {}

    records: List[Dict[str, Any]] = []

    for solver in solvers:
        for eo in eos:
            for np_init in num_points_list:
                for xs in x_specs:
                    # Build x_values
                    gen = xs["gen"]
                    n = int(xs["n"])
                    L = float(xs["L"])
                    # remaining kwargs (like p, r, k)
                    extra = {k: v for k, v in xs.items() if k not in {"name", "gen", "n", "L"}}
                    x = gen(n, L, **extra) if extra else gen(n, L)

                    # Ensure strictly increasing (avoid duplicates causing dt==0)
                    # If needed, jitter repeated points by a tiny epsilon:
                    if not np.all(np.diff(x) > 0):
                        # make it strictly increasing but keep shape
                        x = np.maximum.accumulate(x)
                        eps = np.finfo(float).eps
                        # bump equal neighbors minimally
                        d = np.diff(x)
                        ties = np.where(d == 0)[0]
                        for idx in ties:
                            x[idx + 1] += eps * (1 + idx)

                    # Build a Kalxulus with this combo
                    kalx_kwargs = dict(
                        x_values=x,
                        derivative_order=1,
                        num_points=int(np_init),
                        solver=solver,
                        eo=float(eo),
                        num_points_guess=int(np_init),
                        max_points=int(max_points),
                        only_nonuniform=bool(only_nonuniform),
                        uniform_tol=float(uniform_tol),
                        max_iters_factor=int(max_iters_factor),
                        **KALX_EXTRA_KWARGS,
                    )

                    # Time construction
                    build_time, kalx = time_call(lambda: Kalxulus(**kalx_kwargs), repeats=repeats)

                    # Time compute_stencil_requirements
                    def run_compute():
                        return kalx.compute_stencil_requirements()

                    compute_time, result = time_call(run_compute, repeats=repeats)

                    # Unpack results
                    num_points_res, error_res = result
                    # Basic summary stats
                    nan_mask = ~np.isfinite(error_res)
                    pct_nan = 100.0 * nan_mask.mean()
                    mean_err = float(np.nanmean(error_res))
                    max_err = float(np.nanmax(error_res))
                    mean_np = float(np.mean(num_points_res))
                    median_np = float(np.median(num_points_res))
                    max_np = int(np.max(num_points_res))

                    records.append(
                        dict(
                            solver=solver,
                            eo=float(eo),
                            num_points_init=int(np_init),
                            max_points=int(max_points),
                            only_nonuniform=bool(only_nonuniform),
                            uniform_tol=float(uniform_tol),
                            max_iters_factor=int(max_iters_factor),
                            x_name=xs["name"],
                            n_points=int(n),
                            L=float(L),

                            build_time_s=build_time,
                            compute_time_s=compute_time,
                            total_time_s=build_time + compute_time,

                            mean_error=mean_err,
                            max_error=max_err,
                            pct_nan=pct_nan,

                            mean_num_points=mean_np,
                            median_num_points=median_np,
                            max_num_points=max_np,
                        )
                    )

    df = pd.DataFrame.from_records(records)
    # Useful default ordering
    df = df.sort_values(["x_name", "solver", "eo", "num_points_init"]).reset_index(drop=True)
    return df


# ---- Example usage -----------------------------------------------------------

if __name__ == "__main__":
    # Define your grid
    SOLVERS = ["numpy", "scipy"]
    EOS = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    NUM_POINTS_INIT = [3, 5, 7, 9]

    X_SPECS = [
        {"name": "uniform_n200", "gen": x_uniform, "n": 200, "L": 10.0},
        {"name": "uniform_n25", "gen": x_uniform, "n": 25, "L": 10.0},
        {"name": "uniform_n50", "gen": x_uniform, "n": 50, "L": 10.0},
        {"name": "power_p2_n200", "gen": x_power, "n": 200, "L": 3.0, "p": 2.0},
        {"name": "power_p2_n50", "gen": x_power, "n": 50, "L": 3.0, "p": 2.0},
        {"name": "power_p2_n25", "gen": x_power, "n": 25, "L": 3.0, "p": 2.0},
        {"name": "power_p3_n200", "gen": x_power, "n": 200, "L": 3.0, "p": 3.0},
        {"name": "geom_r5_n200", "gen": x_geometric, "n": 200, "L": 3.0, "r": 5.0},
        {"name": "cluster_tanh_k3_n200", "gen": x_cluster_tanh, "n": 200, "L": 3.0, "k": 3.0},
        # Try a different size to see scaling
        {"name": "power_p2_n400", "gen": x_power, "n": 400, "L": 3.0, "p": 2.0},
    ]

    # If your Kalxulus uses "tolerance" instead of "coeff_tolerance", set it here:
    EXTRA = {
        # "tolerance": 1e-8,        # <- uncomment if your class wants this
        # "coeff_tolerance": 1e-8,  # <- or this, depending on your impl
    }

    df = benchmark_error_estimator(
        solvers=SOLVERS,
        eos=EOS,
        num_points_list=NUM_POINTS_INIT,
        x_specs=X_SPECS,
        max_points=21,
        only_nonuniform=False,
        uniform_tol=1e-12,
        max_iters_factor=3,
        repeats=1,  # bump to 3 or 5 for more stable timing
        KALX_EXTRA_KWARGS=EXTRA,
    )

    # Print a small summary view
    with pd.option_context("display.max_rows", 50, "display.max_columns", None):
        print(df.head(20))
        print("\nBEST combos by compute time:\n", df.nsmallest(10, "compute_time_s")[[
            "x_name", "solver", "eo", "num_points_init", "compute_time_s", "mean_error", "max_num_points"
        ]])

    # Save results to CSV if you want
    df.to_csv("kalxulus_error_estimator_bench.csv", index=False)
