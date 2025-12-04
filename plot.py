#!/usr/bin/env python3
"""
MMCN Results Analyzer
---------------------
Drop-in replacement for the ad-hoc script, with:
  • Automatic approach discovery (no hardcoded names)
  • Flexible include/exclude filters (regex or list)
  • Union/intersection control for instance sets
  • Robust loading of pickles
  • Vectorized computation of VBS and average gap curves
  • Easy plotting with sensible defaults (gap or primal bound)
  • Convenient API that returns DataFrames you can slice however you like

Assumptions about each pickle file (same as your script):
  Each file is a list[dict] with keys: "solving_time" (float) and "primal_bound" (float).

Quick start (in a notebook):

from mmcn_analyzer import analyze_and_plot
out = analyze_and_plot(
    result_folder="results/MMCN_hard_BI",
    include=[r"gurobi", r"25_5"],   # optional (regex or list of strings). None => all.
    exclude=None,                   # optional (regex)
    time_limit=1000,
    instance_mode="intersection",   # or "union"
    select=["gurobi", "gatIM_25_5"],# optional subset to plot (regex/list). None => all.
    logy=True,
    objective="min",                # or "max"
    metric="gap",                   # or "primal" to plot primal bounds
)

# Data objects you can use:
out["df_final"].head()      # per-instance final values for each approach + VBS
out["best_counts"]          # who wins per instance (final value)
out["df_area"].head()       # per-instance area-under-gap curve (lower is better)
out["area_best_counts"]     # who wins by area
out["curves"]               # average gap curves per approach
out["value_curves"]         # average primal-bound curves per approach
"""
from __future__ import annotations

import os
import re
import glob
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

@dataclass
class Run:
    times: np.ndarray  # ascending, includes 0 and final >= time_limit (after prep)
    values: np.ndarray # same length as times


def _is_dir(p: str) -> bool:
    try:
        return os.path.isdir(p)
    except Exception:
        return False


def _list_subdirs(path: str) -> List[str]:
    """Return immediate subdirectory names (not full paths), sorted."""
    subs = [d for d in glob.glob(os.path.join(path, "*")) if _is_dir(d)]
    return sorted(os.path.basename(d) for d in subs)


def _compile_patterns(pats: Optional[Sequence[str]]) -> Optional[List[re.Pattern]]:
    """Accept regex strings OR compiled patterns; case-insensitive by default.
    If a compiled pattern is provided, we keep its original flags.
    """
    if pats is None:
        return None
    # Normalize to list
    if isinstance(pats, (str, bytes, re.Pattern)):
        pats = [pats]  # type: ignore[list-item]
    out: List[re.Pattern] = []
    for p in pats:
        if isinstance(p, re.Pattern):
            out.append(p)
        else:
            out.append(re.compile(p, flags=re.IGNORECASE))
    return out


def _match_any(name: str, patterns: Optional[List[re.Pattern]]) -> bool:
    if patterns is None:
        return True
    return any(r.search(name) for r in patterns)


def discover_approaches(
    result_folder: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Discover approach subdirs under result_folder using optional include/exclude regexes.
    `include`/`exclude` may be regex strings or compiled patterns.
    """
    inc = _compile_patterns(include)
    exc = _compile_patterns(exclude)
    names = []
    for name in _list_subdirs(result_folder):
        if not _match_any(name, inc):
            continue
        if exc is not None and any(r.search(name) for r in exc):
            continue
        # Keep approach dirs that contain at least one plausible result file
        p = os.path.join(result_folder, name)
        any_pkl = glob.glob(os.path.join(p, "*.pkl"))
        any_file = [f for f in glob.glob(os.path.join(p, "*")) if os.path.isfile(f)]
        if any_pkl or any_file:
            names.append(name)
    if not names:
        raise ValueError("No approaches found. Check result_folder or your include/exclude filters.")
    return names


def discover_instances(result_folder: str, approaches: Sequence[str], mode: str = "intersection") -> List[str]:
    """Discover instance filenames present in approach directories.

    mode: "intersection" (only instances present in all selected approaches) or "union".
    Returns sorted list of filenames (e.g., foo.pkl)."""
    sets: List[set] = []
    for a in approaches:
        files = set(os.path.basename(p) for p in glob.glob(os.path.join(result_folder, a, "*.lp")))
        sets.append(files)
    if mode not in {"intersection", "union"}:
        raise ValueError("mode must be 'intersection' or 'union'")
    s = set.intersection(*sets) if mode == "intersection" else set.union(*sets)
    if not s:
        raise ValueError("No instances found with the chosen instance_mode. Try 'union' or relax filters.")
    return sorted(s)


def _safe_last(xs: Sequence[float]) -> float:
    return float(xs[-1]) if len(xs) else np.nan


def load_run(path: str) -> Run:
    """Load a single pickle and return stepwise (times, values) arrays.
    Ensures strictly increasing times and aligned values.
    If no valid entries exist, returns a NaN-only sentinel run.
    """
    with open(path, "rb") as fp:
        data = pickle.load(fp)

    # Extract and validate rows
    t_list, v_list = [], []
    for e in (data or []):
        try:
            tt = float(e["solving_time"])
            vv = float(e["primal_bound"])
            if np.isfinite(tt) and np.isfinite(vv):
                t_list.append(tt)
                v_list.append(vv)
        except Exception:
            continue  # skip malformed rows

    if not t_list:  # sentinel: "no data"
        # A single-step NaN run; downstream will detect and skip for averages.
        return Run(times=np.array([0.0], dtype=float),
                   values=np.array([np.nan], dtype=float))

    # Sort and deduplicate by time (keep last value at a given timestamp)
    order = np.argsort(t_list)
    t = np.asarray(t_list, dtype=float)[order]
    v = np.asarray(v_list, dtype=float)[order]
    uniq_t, idx = np.unique(t, return_index=True)
    t = uniq_t
    v = v[idx]

    # Ensure we start at t=0 with an initial value
    if t[0] > 0.0:
        t = np.concatenate([[0.0], t])
        v = np.concatenate([[v[0]], v])
    return Run(times=t, values=v)


def extend_to_limit(run: Run, limit: int) -> Run:
    """Ensure the run extends to >= limit seconds with stepwise hold of last value."""
    t, v = run.times, run.values
    if t[-1] < limit:
        t = np.concatenate([t, [float(limit)]])
        v = np.concatenate([v, [v[-1]]])
    return Run(times=t, values=v)


def step_values_on_grid(run: Run, limit: int) -> np.ndarray:
    """Return piecewise-constant values sampled at integer seconds [0, limit).
    If the run is NaN-only, return an all-NaN grid so callers can skip it.
    """
    # NaN-only sentinel?
    if np.all(~np.isfinite(run.values)):
        return np.full(limit, np.nan, dtype=float)

    r = extend_to_limit(run, limit)
    t = r.times
    vals = r.values

    grid = np.arange(limit, dtype=int)
    idx = np.searchsorted(t, grid, side="right") - 1
    idx[idx < 0] = 0
    y = vals[idx]
    return y


def _compute_gap(vals: np.ndarray, best: float, objective: str, eps: float) -> np.ndarray:
    """Compute nonnegative relative gap array given current values and best value.

    For minimization:
        gap = max(0, (vals - best) / max(|vals|, eps))

    For maximization:
        gap = max(0, (best - vals) / max(|best|, eps))
    """
    if not np.isfinite(best):
        return np.full_like(vals, np.nan, dtype=float)

    if objective == "min":
        denom = np.where(np.abs(vals) < eps, eps, np.abs(vals))
        gap = (vals - best) / denom
    elif objective == "max":
        denom_scalar = eps if abs(best) < eps else abs(best)
        denom = np.full_like(vals, denom_scalar, dtype=float)
        gap = (best - vals) / denom
    else:
        raise ValueError("objective must be 'min' or 'max'")

    gap = np.clip(gap, a_min=0.0, a_max=None)
    return gap


# -----------------------------
# Core analysis
# -----------------------------

def build_final_df(
    result_folder: str,
    approaches: Sequence[str],
    instances: Sequence[str],
    objective: str = "min",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """DataFrame of final *primal gaps* per (instance, approach), plus VBS column (primal bound).

    For each instance:
      - 'vbs' is the best final primal bound across approaches (min or max depending on `objective`)
      - each approach column is the final relative gap to that VBS, computed with `_compute_gap`.
    """
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    # First collect final primal bounds
    records = {}
    for a in approaches:
        col = []
        for ins in instances:
            path = os.path.join(result_folder, a, ins)
            if not os.path.exists(path):
                col.append(np.nan)
                continue
            run = load_run(path)
            col.append(_safe_last(run.values))  # final primal bound
        records[a] = col

    df = pd.DataFrame(records, index=list(instances))
    df.index.name = "instance"

    # Best final bound per instance
    if objective == "min":
        df["vbs"] = df.min(axis=1, numeric_only=True)
    else:  # max
        df["vbs"] = df.max(axis=1, numeric_only=True)

    # Convert approach columns from primal bounds -> primal gaps w.r.t. vbs
    for ins in instances:
        best = df.at[ins, "vbs"]
        if not np.isfinite(best):
            # No valid best => all gaps NaN for this instance
            for a in approaches:
                df.at[ins, a] = np.nan
            continue

        vals = df.loc[ins, list(approaches)].to_numpy(dtype=float)
        gaps = _compute_gap(vals, float(best), objective, eps)

        # Write back gaps
        df.loc[ins, list(approaches)] = gaps

    return df



def build_gap_curves(
    result_folder: str,
    approaches: Sequence[str],
    instances: Sequence[str],
    df_final: pd.DataFrame,
    time_limit: int = 1000,
    eps: float = 1e-12,
    objective: str = "min",
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Return (avg_gap_curves, df_area)

    avg_gap_curves: dict[approach -> np.ndarray] average over instances of per-second gap
    df_area: per-instance sum of gaps (area under the curve) per approach
    """
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    n = len(instances)
    curves: Dict[str, np.ndarray] = {}
    area_records: Dict[str, List[float]] = {a: [] for a in approaches}

    for a in approaches:
        agg = np.zeros(time_limit, dtype=float)
        areas: List[float] = []
        for ins in instances:
            path = os.path.join(result_folder, a, ins)
            if not os.path.exists(path):
                # If missing, skip this instance for curve average & area
                continue
            run = load_run(path)
            vals = step_values_on_grid(run, time_limit)
            best_raw = df_final.loc[ins, "vbs"]
            if not np.isfinite(best_raw):
                continue  # skip: no viable best for this instance
            best = float(best_raw)

            # If this approach has no data for the instance, vals will be NaN-only
            if not np.any(np.isfinite(vals)):
                continue

            gap = _compute_gap(vals, best, objective, eps)
            agg += np.nan_to_num(gap, nan=0.0)
            areas.append(float(np.nansum(gap)))

        # Average only over instances that actually existed for this approach
        m = max(1, len(areas))
        curves[a] = agg / m

        # Pad missing instances with NaN so df_area aligns to full instance index
        area_series: List[float] = []
        ins_set = set(i for i in instances if os.path.exists(os.path.join(result_folder, a, i)))
        for ins in instances:
            if ins in ins_set:
                run = load_run(os.path.join(result_folder, a, ins))
                vals = step_values_on_grid(run, time_limit)
                best = float(df_final.loc[ins, "vbs"]) if np.isfinite(df_final.loc[ins, "vbs"]) else np.nan
                gap = _compute_gap(vals, best, objective, eps)
                area_series.append(float(np.nansum(gap)))
            else:
                area_series.append(np.nan)
        area_records[a] = area_series

    df_area = pd.DataFrame(area_records, index=list(instances))
    df_area.index.name = "instance"
    return curves, df_area


def build_primal_curves(
    result_folder: str,
    approaches: Sequence[str],
    instances: Sequence[str],
    time_limit: int = 1000,
) -> Dict[str, np.ndarray]:
    """Return avg_value_curves: dict[approach -> np.ndarray] of average primal-bound vs time.

    For each approach, we:
      - sample each instance's run at integer seconds [0, time_limit)
      - average the primal bounds over all instances with data for that approach
    """
    curves: Dict[str, np.ndarray] = {}
    for a in approaches:
        agg = np.zeros(time_limit, dtype=float)
        m = 0
        for ins in instances:
            path = os.path.join(result_folder, a, ins)
            if not os.path.exists(path):
                continue
            run = load_run(path)
            vals = step_values_on_grid(run, time_limit)
            if not np.any(np.isfinite(vals)):
                continue
            agg += np.nan_to_num(vals, nan=0.0)
            m += 1
        if m == 0:
            curves[a] = np.full(time_limit, np.nan, dtype=float)
        else:
            curves[a] = agg / m
    return curves


# -----------------------------
# Plotting
# -----------------------------

def plot_curves(
    curves: Dict[str, np.ndarray],
    select: Optional[Sequence[str]] = None,
    time_limit: int = 1000,
    title: Optional[str] = None,
    logy: bool = True,
    figsize: Tuple[int, int] = (9, 6),
    legend_loc: str = "best",
) -> plt.Axes:
    """Plot average gap curves. `select` may be a list of approach names or regex strings."""
    sel_patterns = _compile_patterns(select) if select is not None else None
    to_plot: List[Tuple[str, np.ndarray]] = []
    for name, curve in curves.items():
        if _match_any(name, sel_patterns):
            to_plot.append((name, curve))
    if not to_plot:
        raise ValueError("No approaches matched `select`. Try a different filter or None.")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(time_limit)
    for name, y in to_plot:
        if "id" in name:
            ax.plot(x, y, linewidth=2, label="ID-PaS")
        elif "gurobi" in name:
            ax.plot(x, y, linewidth=2, label="Gurobi")
        else:
            ax.plot(x, y, linewidth=2, label="PaS")
    # for name, y in to_plot:
    #     ax.plot(x, y, linewidth=2, label=name)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Runtime (s)", fontsize=20)
    ax.set_ylabel("Primal Gap", fontsize=20)

    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    # if title:
    #     ax.set_title(title, fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()

    # New order (indices of original items)
    order = [2, 0, 1]
    
    plt.legend([handles[i] for i in order],
               [labels[i] for i in order], loc=legend_loc, fontsize=20)
    # plt.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return ax


def plot_primal_curves(
    curves: Dict[str, np.ndarray],
    select: Optional[Sequence[str]] = None,
    time_limit: int = 1000,
    title: Optional[str] = None,
    logy: bool = False,
    figsize: Tuple[int, int] = (9, 6),
    legend_loc: str = "best",
) -> plt.Axes:
    """Plot average primal-bound curves. `select` may be a list of approach names or regex strings."""
    sel_patterns = _compile_patterns(select) if select is not None else None
    to_plot: List[Tuple[str, np.ndarray]] = []
    for name, curve in curves.items():
        if _match_any(name, sel_patterns):
            to_plot.append((name, curve))
    if not to_plot:
        raise ValueError("No approaches matched `select`. Try a different filter or None.")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(time_limit)
    for name, y in to_plot:
        ax.plot(x, y, linewidth=2, label=name)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Runtime (s)", fontsize=14)
    ax.set_ylabel("Primal Bound", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    ax.legend(loc=legend_loc)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return ax


def compute_primal_integrals(
    curves: Dict[str, np.ndarray],
    cutoffs: Sequence[int] = (100, 200, 500, 1000),
) -> pd.DataFrame:
    """Return a DataFrame of primal integrals by approach (rows) and cutoff (cols).
    Uses the already-averaged gap curves: integral(T) = sum_{t=0}^{T-1} gap[t].
    """
    data = {}
    for approach, y in curves.items():
        vals = {}
        for T in cutoffs:
            T = int(T)
            if T <= 0:
                vals[T] = np.nan
            else:
                T = min(T, len(y))
                vals[T] = float(np.nansum(y[:T]))
        data[approach] = vals
    df_pi = pd.DataFrame.from_dict(data, orient="index")
    df_pi.index.name = "approach"
    df_pi.columns = [int(c) for c in df_pi.columns]  # ensure nice int column labels
    return df_pi.sort_index(axis=1)


def plot_primal_integral_bars_by_cutoff(
    curves: Dict[str, np.ndarray],
    cutoffs: Sequence[int] = (100, 200, 500, 1000),
    title: Optional[str] = "Primal Integral by Cutoff (lower is better)",
    figsize: Tuple[int, int] = (9, 5),
    legend_loc: str = "best",
    logy: bool = False,
) -> Tuple[pd.DataFrame, plt.Axes]:
    """Grouped by cutoff on x-axis; bars = approaches within each cutoff."""
    # Build the same table as before (rows=approaches, cols=cutoffs)
    df_pi = compute_primal_integrals(curves, cutoffs=cutoffs)

    # We'll plot by cutoff groups -> iterate approaches as bar offsets
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df_pi.columns))  # one slot per cutoff
    n_methods = max(1, len(df_pi.index))
    width = 0.8 / n_methods            # keep bars within 80% of the slot

    for j, approach in enumerate(df_pi.index):
        y = df_pi.loc[approach, df_pi.columns].values
        ax.bar(x + j * width, y, width=width, label=str(approach))

    # Center ticks on each cutoff group
    ax.set_xticks(x + (n_methods - 1) * width / 2)
    ax.set_xticklabels([f"{int(c)}s" for c in df_pi.columns])
    ax.set_ylabel("Primal Integral (area under avg gap)")
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(loc=legend_loc)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return df_pi, ax


def compute_primal_integral_stats(
    result_folder: str,
    approaches: Sequence[str],
    instances: Sequence[str],
    df_final: pd.DataFrame,                 # from analyze()
    cutoffs: Sequence[int] = (100, 200, 500, 1000),
    eps: float = 1e-12,
    objective: str = "min",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean_df, std_df) of primal integral across instances for each approach & cutoff."""
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    cutoffs = [int(c) for c in cutoffs]
    H = max(cutoffs)

    mean_rec, std_rec = {}, {}
    for a in approaches:
        rows = []
        for ins in instances:
            path = os.path.join(result_folder, a, ins)
            if not os.path.exists(path):
                continue
            run = load_run(path)
            vals = step_values_on_grid(run, H)        # shape (H,)
            best = float(df_final.loc[ins, "vbs"])
            gap = _compute_gap(vals, best, objective, eps)
            cs = np.cumsum(np.nan_to_num(gap, nan=0.0))  # integral up to t
            rows.append([float(cs[c-1]) for c in cutoffs])
        if rows:
            M = np.asarray(rows, dtype=float)         # (n_inst, n_cutoffs)
            mean_rec[a] = list(np.nanmean(M, axis=0))
            std_rec[a]  = list(np.nanstd(M, axis=0, ddof=0))  # use ddof=1 for sample std
        else:
            mean_rec[a] = [np.nan]*len(cutoffs)
            std_rec[a]  = [np.nan]*len(cutoffs)

    mean_df = pd.DataFrame(mean_rec, index=cutoffs).T
    std_df  = pd.DataFrame(std_rec,  index=cutoffs).T
    mean_df.index.name = std_df.index.name = "approach"
    mean_df.columns = std_df.columns = cutoffs
    return mean_df, std_df


def plot_primal_integral_bars_by_cutoff_with_err(
    result_folder: str,
    approaches: Sequence[str],
    instances: Sequence[str],
    df_final: pd.DataFrame,
    cutoffs: Sequence[int] = (100, 200, 500, 1000),
    title: Optional[str] = "Primal Integral by Cutoff (mean ± std; lower is better)",
    figsize: Tuple[int, int] = (9, 5),
    legend_loc: str = "best",
    logy: bool = False,
    capsize: float = 3.0,
    objective: str = "min",
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Axes]:
    """Grouped by cutoff on x-axis; bars = approaches within each cutoff, with std-dev error bars."""
    mean_df, std_df = compute_primal_integral_stats(
        result_folder, approaches, instances, df_final, cutoffs=cutoffs, objective=objective
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(mean_df.columns))  # one slot per cutoff
    n_methods = max(1, len(mean_df.index))
    width = 0.8 / n_methods              # keep bars within 80% of the slot

    for j, a in enumerate(mean_df.index):
        y = mean_df.loc[a].values
        e = std_df.loc[a].values
        ax.bar(x + j * width, y, width=width, label=str(a), yerr=e, capsize=capsize)

    # Center ticks on each cutoff group
    ax.set_xticks(x + (n_methods - 1) * width / 2)
    ax.set_xticklabels([f"{int(c)}s" for c in mean_df.columns])
    ax.set_ylabel("Primal Integral (area under per-instance gap)")
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(loc=legend_loc)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return mean_df, std_df, ax


# -----------------------------
# High-level API
# -----------------------------

def analyze(
    result_folder: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    time_limit: int = 1000,
    instance_mode: str = "intersection",
    objective: str = "min",
) -> Dict[str, object]:
    """Run the full analysis pipeline and return data objects.

    objective: "min" (default) or "max"

    Returns a dict with:
      - approaches: List[str]
      - instances: List[str]
      - df_final: DataFrame of final primal bounds per approach + VBS
      - best_counts: Series (who wins on final)
      - curves: Dict[str, np.ndarray] average gap curves per approach
      - value_curves: Dict[str, np.ndarray] average primal-bound curves per approach
      - df_area: DataFrame (per-instance area under gap curve)
      - area_best_counts: Series (who wins by area)
    """
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    approaches = discover_approaches(result_folder, include=include, exclude=exclude)
    print(approaches)
    instances = discover_instances(result_folder, approaches, mode=instance_mode)

    df_final = build_final_df(result_folder, approaches, instances, objective=objective)

    # Winner by final value (min or max)
    # if objective == "min":
    #     best_counts = df_final.drop(columns=["vbs"]).idxmin(axis=1).value_counts().sort_index()
    # else:
    #     best_counts = df_final.drop(columns=["vbs"]).idxmax(axis=1).value_counts().sort_index()

    best_counts = df_final.drop(columns=["vbs"]).idxmin(axis=1).value_counts().sort_index()

    curves, df_area = build_gap_curves(
        result_folder, approaches, instances, df_final,
        time_limit=time_limit, objective=objective
    )

    value_curves = build_primal_curves(
        result_folder, approaches, instances, time_limit=time_limit
    )

    # For area, lower is always better (gap >= 0), regardless of objective
    area_best_counts = df_area.idxmin(axis=1).value_counts().sort_index()

    return {
        "approaches": approaches,
        "instances": instances,
        "df_final": df_final,
        "best_counts": best_counts,
        "curves": curves,
        "value_curves": value_curves,
        "df_area": df_area,
        "area_best_counts": area_best_counts,
    }


def analyze_and_plot(
    result_folder: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    time_limit: int = 1000,
    instance_mode: str = "intersection",
    select: Optional[Sequence[str]] = None,
    logy: bool = True,
    title: Optional[str] = None,
    objective: str = "min",
    metric: str = "gap",  # "gap" or "primal"
) -> Dict[str, object]:
    """Convenience wrapper: run analysis and immediately plot curves.

    `select` can be approach names or regexes, e.g. ["gurobi", r"^gat"]
    `objective`: "min" (default) or "max"
    `metric`:
        - "gap"    => plot average primal gaps (default; old behavior)
        - "primal" => plot average primal bounds
    """
    if metric not in {"gap", "primal"}:
        raise ValueError("metric must be 'gap' or 'primal'")

    out = analyze(
        result_folder=result_folder,
        include=include,
        exclude=exclude,
        time_limit=time_limit,
        instance_mode=instance_mode,
        objective=objective,
    )

    if metric == "gap":
        ax = plot_curves(
            out["curves"],
            select=select,
            time_limit=time_limit,
            title=title,
            logy=logy,
        )
    else:  # "primal"
        ax = plot_primal_curves(
            out["value_curves"],
            select=select,
            time_limit=time_limit,
            title=title,
            logy=logy,
        )

    out["ax"] = ax
    return out


# -----------------------------
# If run as a script (optional CLI)
# -----------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Analyze MMCN result pickles and plot average curves.")
    p.add_argument("result_folder", type=str, help="Path to results root (e.g., results/MMCN_hard_BI)")
    p.add_argument("--include", nargs="*", default=None, help="Regex patterns to include approaches (default: all)")
    p.add_argument("--exclude", nargs="*", default=None, help="Regex patterns to exclude approaches")
    p.add_argument("--time_limit", type=int, default=1000, help="Seconds for curve horizon")
    p.add_argument("--instance_mode", choices=["intersection", "union"], default="intersection")
    p.add_argument("--select", nargs="*", default=None, help="Approach names or regexes to plot (default: all)")
    p.add_argument("--linear", action="store_true", help="Use linear y-scale instead of log")
    p.add_argument(
        "--sense",
        choices=["min", "max"],
        default="min",
        help="Objective sense: 'min' for minimization (default), 'max' for maximization",
    )
    p.add_argument(
        "--plot_metric",
        choices=["gap", "primal"],
        default="gap",
        help="What to plot on the y-axis: 'gap' (relative gap, default) or 'primal' (raw primal bounds).",
    )
    args = p.parse_args()

    out = analyze_and_plot(
        result_folder=args.result_folder,
        include=args.include,
        exclude=args.exclude,
        time_limit=args.time_limit,
        instance_mode=args.instance_mode,
        select=args.select,
        logy=not args.linear,
        objective=args.sense,
        metric=args.plot_metric,
    )
    plt.show()
