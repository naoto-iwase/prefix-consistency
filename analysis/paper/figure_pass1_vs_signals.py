#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy", "statsmodels"]
# ///
"""pass1_vs_signals figure: confidence baselines split by correctness.

Mirrors figure_pass1_vs_rates.py but for the two confidence baselines
used in the paper (DeepConf tail, P(True)). For each baseline the
figure shows the per-trial signal as a function of Pass@1, with two
curves per domain:

  * "Correct" curve: signal on samples whose init answer matches gold.
  * "Wrong"   curve: signal on samples whose init answer does not.

Three estimators share one inferential contract:

  * The point curve is the full-data fit.
  * CIs at ``+/- ci_sigma * SE`` come from a joint cluster bootstrap
    that resamples whole problems. Within each resample both curves are
    refit, so the gap between them inherits the correct joint sampling
    distribution.

Methods (selected via ``--method``):

  * ``glm``    : Gaussian OLS ``y = a + b * Pass@1`` per class and
                 domain. The continuous-response analogue of the
                 logistic GLM in figure_pass1_vs_rates.py.
  * ``lowess`` : trial-level LOWESS (``it=0``).
  * ``binned`` : trial-pooled mean over equal-count Pass@1 quantile
                 bins per domain. Bin edges are fixed from the full-data
                 quantiles.

One signal per call (``--signal``); compose the grid in LaTeX.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from _utils import (
    infer_benchmark_label, infer_model_label, find_init_file,
    METHOD_DISPLAY, setup_tex_rendering,
    signal_keys_in_order,
    is_usable_answer,
)
from _defs import (
    BENCHMARK_DOMAIN as DOMAIN_OF,
    DOMAIN_COLOR,
    SIGNAL_KEY_TO_METHOD,
)

setup_tex_rendering()


# ── Domain grouping ─────────────────────────────────────────────────────

DOMAIN_ORDER = ["Science", "Math"]


# ── Baseline signals ────────────────────────────────────────────────────
# (jsonl record key, METHOD_DISPLAY key, optional y-axis range).
# DeepConf metrics live inside ``confidences[i]`` (per-trial dict);
# verbal / P(True) metrics live in their own top-level list field.
# Order is derived from ALL_BASELINE_GROUPS (paper-natural) via
# signal_keys_in_order so this figure stays in sync with the rest of
# the paper as we tweak baseline ordering.

_BASELINE_YLIMS: dict[str, tuple[float, float]] = {
    "p_true": (0, 1),
    "verbal_0_100": (0, 100),
}
_BASELINE_SUBSET = ("tail_conf", "p_true")
BASELINE_SIGNALS: dict[str, tuple[str, tuple[float, float] | None]] = {
    k: (SIGNAL_KEY_TO_METHOD[k], _BASELINE_YLIMS.get(k))
    for k in signal_keys_in_order(_BASELINE_SUBSET)
}


# ── Data loading ────────────────────────────────────────────────────────

def compute_per_problem_signals(init_path: Path,
                                signal_key: str) -> list[dict]:
    """Per-problem Pass@1 and per-class signal values.

    Each problem contributes ``pass1 = n_correct / n_init`` plus two
    lists of signal values: ``c_vals`` for trials whose init answer
    matches gold, ``w_vals`` for the rest. Trials with a missing or
    non-finite signal value are dropped from the lists but still count
    toward Pass@1.
    """
    is_top_level = signal_key in ("p_true", "verbal_0_100")
    out: list[dict] = []
    with open(init_path) as f:
        for line in f:
            rec = json.loads(line)
            gold = str(rec["gold_answer"])
            if is_top_level:
                vals_field = (rec.get(f"{signal_key}_confidences")
                              or rec.get(signal_key) or [])
            else:
                vals_field = rec.get("confidences") or []
            n_init = n_correct = 0
            c_vals: list[float] = []
            w_vals: list[float] = []
            for i, entry in enumerate(rec["all_answers"]):
                if not is_usable_answer(entry):
                    continue
                init_correct = (str(entry[0]) == gold)
                n_init += 1
                n_correct += init_correct
                if i >= len(vals_field) or vals_field[i] is None:
                    continue
                if is_top_level:
                    raw = vals_field[i]
                else:
                    raw = vals_field[i].get(signal_key)
                if raw is None:
                    continue
                v = float(raw)
                if not np.isfinite(v):
                    continue
                (c_vals if init_correct else w_vals).append(v)
            if n_init == 0:
                continue
            out.append({
                "pass1": n_correct / n_init,
                "c_vals": c_vals,
                "w_vals": w_vals,
            })
    return out


def expand_trials(problems: list[dict],
                  kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-problem signal lists into per-trial (Pass@1, value)."""
    key = "c_vals" if kind == "C" else "w_vals"
    xs: list[float] = []
    ys: list[float] = []
    for p in problems:
        xs.extend([p["pass1"]] * len(p[key]))
        ys.extend(p[key])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# ── Stat helpers ────────────────────────────────────────────────────────

def sigma_to_alpha(sigma: float) -> float:
    return 2.0 * (1.0 - norm.cdf(sigma))


def _percentiles(samples: np.ndarray, sigma: float,
                 axis: int | None = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    alpha = sigma_to_alpha(sigma)
    return (np.nanpercentile(samples, 100 * alpha / 2, axis=axis),
            np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=axis))


def _quantile_edges(pass1: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(pass1, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = min(edges[0], 0.0) - 1e-9
    edges[-1] = max(edges[-1], 1.0) + 1e-9
    return np.unique(edges)


# ── Fit kernels ─────────────────────────────────────────────────────────

def _fit_glm(problems: list[dict], kind: str, xgrid: np.ndarray
             ) -> tuple[object | None, np.ndarray | None]:
    """Gaussian OLS on the expanded trial data."""
    xs, ys = expand_trials(problems, kind)
    if len(xs) < 3 or len(np.unique(xs)) < 2:
        return None, None
    try:
        result = sm.OLS(ys, sm.add_constant(xs)).fit()
    except Exception:
        return None, None
    return result, result.predict(sm.add_constant(xgrid))


def _fit_lowess(problems: list[dict], kind: str, xgrid: np.ndarray,
                frac: float, it: int) -> np.ndarray | None:
    xs, ys = expand_trials(problems, kind)
    if len(xs) < 3 or len(np.unique(xs)) < 2:
        return None
    smoothed = sm_lowess(ys, xs, frac=frac, it=it, return_sorted=True)
    return np.interp(xgrid, smoothed[:, 0], smoothed[:, 1],
                     left=np.nan, right=np.nan)


def _bin_pool_means(problems: list[dict], edges: np.ndarray,
                    kind: str) -> np.ndarray:
    """Per-bin trial-pooled mean: ``mean of all trials in the bin``."""
    key = "c_vals" if kind == "C" else "w_vals"
    pass1 = np.asarray([p["pass1"] for p in problems], dtype=float)
    nb = len(edges) - 1
    idx = np.clip(np.digitize(pass1, edges, right=False) - 1, 0, nb - 1)
    out = np.full(nb, np.nan)
    for b in range(nb):
        bin_vals: list[float] = []
        for p, m in zip(problems, idx == b):
            if m:
                bin_vals.extend(p[key])
        if bin_vals:
            out[b] = float(np.mean(bin_vals))
    return out


# ── Joint cluster bootstrap ─────────────────────────────────────────────

FitFn = Callable[[list[dict]], dict]


def cluster_bootstrap(problems: list[dict], fit_fn: FitFn,
                      n_boot: int, rng: np.random.Generator
                      ) -> tuple[dict, dict]:
    point = fit_fn(problems)
    acc: dict[str, list] = {k: [] for k in point}
    n_prob = len(problems)
    for _ in range(n_boot):
        sel = rng.integers(0, n_prob, size=n_prob)
        resampled = [problems[i] for i in sel]
        fit = fit_fn(resampled)
        for k in acc:
            acc[k].append(fit[k])
    samples = {k: np.asarray(v) for k, v in acc.items()}
    return point, samples


def _glm_fit_fn(xgrid: np.ndarray) -> FitFn:
    nan_grid = np.full_like(xgrid, np.nan, dtype=float)

    def fit(problems):
        res_c, pc = _fit_glm(problems, "C", xgrid)
        res_w, pw = _fit_glm(problems, "W", xgrid)
        return {
            "rc":     np.asarray(pc, dtype=float) if pc is not None
                      else nan_grid.copy(),
            "rw":     np.asarray(pw, dtype=float) if pw is not None
                      else nan_grid.copy(),
            "beta_c": float(res_c.params[1]) if res_c is not None
                      else float("nan"),
            "beta_w": float(res_w.params[1]) if res_w is not None
                      else float("nan"),
        }
    return fit


def _lowess_fit_fn(xgrid: np.ndarray, frac: float, it: int) -> FitFn:
    nan_grid = np.full_like(xgrid, np.nan, dtype=float)

    def fit(problems):
        pc = _fit_lowess(problems, "C", xgrid, frac, it)
        pw = _fit_lowess(problems, "W", xgrid, frac, it)
        return {
            "rc": pc if pc is not None else nan_grid.copy(),
            "rw": pw if pw is not None else nan_grid.copy(),
        }
    return fit


def _binned_fit_fn(edges: np.ndarray) -> FitFn:
    def fit(problems):
        return {
            "rc": _bin_pool_means(problems, edges, "C"),
            "rw": _bin_pool_means(problems, edges, "W"),
        }
    return fit


def _mask_continuous_out_of_support(xgrid: np.ndarray,
                                    problems: list[dict],
                                    point: dict, samples: dict) -> None:
    xs_all = np.asarray([p["pass1"] for p in problems], dtype=float)
    mask = (xgrid < xs_all.min()) | (xgrid > xs_all.max())
    for k in ("rc", "rw"):
        point[k][mask] = np.nan
        samples[k][:, mask] = np.nan


# ── Plotting ────────────────────────────────────────────────────────────

def _scatter(ax, problems, kind, color, marker) -> None:
    """Scatter per-problem mean signal (density overlay).

    One dot per problem, mirroring the rate ``k/n`` overlay in
    figure_pass1_vs_rates.py (since ``k/n`` is the per-problem
    mean of {0,1} trial outcomes, the per-problem mean of a continuous
    signal is the natural analogue)."""
    key = "c_vals" if kind == "C" else "w_vals"
    xs = [p["pass1"] * 100 for p in problems if p[key]]
    ys = [float(np.mean(p[key])) for p in problems if p[key]]
    ax.scatter(xs, ys, s=10, alpha=0.18, color=color,
               marker=marker, linewidths=0 if marker == "o" else 0.7)


def _draw_smooth(ax, xgrid, point, samples, sigma, color, domain) -> None:
    x = xgrid * 100
    lo_c, hi_c = _percentiles(samples["rc"], sigma)
    lo_w, hi_w = _percentiles(samples["rw"], sigma)
    ax.plot(x, point["rc"], "-", color=color, linewidth=2.0, label=domain)
    ax.fill_between(x, lo_c, hi_c, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, point["rw"], "--", color=color, linewidth=2.0)
    ax.fill_between(x, lo_w, hi_w, color=color, alpha=0.10, linewidth=0,
                    hatch="//", edgecolor=color)


def _draw_binned(ax, centers, point, samples, sigma, color, domain) -> None:
    xc = centers * 100
    lo_c, hi_c = _percentiles(samples["rc"], sigma)
    lo_w, hi_w = _percentiles(samples["rw"], sigma)
    ax.errorbar(xc, point["rc"],
                yerr=[point["rc"] - lo_c, hi_c - point["rc"]],
                fmt="-o", color=color, linewidth=2.0,
                markersize=5, capsize=3, label=domain)
    ax.errorbar(xc, point["rw"],
                yerr=[point["rw"] - lo_w, hi_w - point["rw"]],
                fmt="--x", color=color, linewidth=2.0,
                markersize=6, capsize=3)


def _print_beta_summary(glm_rows, sigma) -> None:
    if not glm_rows:
        return
    print("  Linear fit (y = a + b*Pass@1):")
    for domain, kind, beta_point, betas in glm_rows:
        valid = betas[~np.isnan(betas)]
        if len(valid) == 0:
            continue
        lo, hi = _percentiles(valid, sigma, axis=None)
        tail = min(float(np.mean(valid <= 0)),
                   float(np.mean(valid >= 0)))
        p = min(1.0, 2.0 * tail)
        print(f"    {domain:7s} {kind}: "
              f"beta={beta_point:+.3g} "
              f"[{float(lo):+.3g}, {float(hi):+.3g}]  p={p:.3g}")


# ── Main plot ───────────────────────────────────────────────────────────

def plot(domain_problems, model_label, signal_label, ylim, method,
         n_bins, out_path, *, n_boot, sigma, show_title,
         lowess_frac, lowess_it, seed) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    xgrid = np.linspace(0.0, 1.0, 201)
    glm_rows: list[tuple[str, str, float, np.ndarray]] = []

    for domain in DOMAIN_ORDER:
        problems = domain_problems.get(domain, [])
        if not problems:
            continue
        color = DOMAIN_COLOR[domain]
        _scatter(ax, problems, "C", color, "o")
        _scatter(ax, problems, "W", color, "x")
        rng = np.random.default_rng(seed)

        if method == "glm":
            point, samples = cluster_bootstrap(
                problems, _glm_fit_fn(xgrid), n_boot, rng)
            _mask_continuous_out_of_support(
                xgrid, problems, point, samples)
            _draw_smooth(ax, xgrid, point, samples, sigma, color, domain)
            glm_rows.append((domain, "Correct", point["beta_c"],
                             samples["beta_c"]))
            glm_rows.append((domain, "Wrong", point["beta_w"],
                             samples["beta_w"]))

        elif method == "lowess":
            point, samples = cluster_bootstrap(
                problems,
                _lowess_fit_fn(xgrid, lowess_frac, lowess_it),
                n_boot, rng)
            _mask_continuous_out_of_support(
                xgrid, problems, point, samples)
            _draw_smooth(ax, xgrid, point, samples, sigma, color, domain)

        elif method == "binned":
            pass1_all = np.asarray([p["pass1"] for p in problems])
            edges = _quantile_edges(pass1_all, n_bins)
            point, samples = cluster_bootstrap(
                problems, _binned_fit_fn(edges), n_boot, rng)
            centers = 0.5 * (np.clip(edges[:-1], 0, 1)
                             + np.clip(edges[1:], 0, 1))
            _draw_binned(ax, centers, point, samples, sigma, color, domain)

        else:
            raise ValueError(f"unknown method: {method}")

    ax.plot([], [], color="black", linestyle="-",
            label=r"Correct (solid, $\bullet$)")
    ax.plot([], [], color="black", linestyle="--",
            label=r"Wrong (dashed, $\times$)")
    ax.set_xlabel(r"Pass@1 (\%)")
    ax.set_ylabel(signal_label)
    ax.set_xlim(0, 100)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    if show_title:
        ax.set_title(model_label)

    _print_beta_summary(glm_rows, sigma)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="pass1_vs_signals figure: confidence baselines "
                    "split by correctness, with joint cluster-bootstrap "
                    "CIs shared by all three estimators.")
    ap.add_argument("jsonl_dirs", nargs="+", type=Path)
    ap.add_argument("--signal", required=True,
                    choices=list(BASELINE_SIGNALS.keys()),
                    help="Confidence baseline to plot.")
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--method", choices=["glm", "lowess", "binned"],
                    default="glm")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--ci-sigma", type=float, default=2.0,
                    help="CI half-width in sigma units "
                         "(2.0 ~= 95.45%%).")
    ap.add_argument("--lowess-frac", type=float, default=0.5)
    ap.add_argument("--lowess-it", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-title", action="store_true",
                    help="Suppress model-label title (for grid composition).")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    run_one(
        args.jsonl_dirs, args.output, signal=args.signal,
        model_label=args.model_label, method=args.method,
        n_bins=args.n_bins, n_boot=args.n_boot, ci_sigma=args.ci_sigma,
        lowess_frac=args.lowess_frac, lowess_it=args.lowess_it,
        seed=args.seed, no_title=args.no_title,
    )


def run_one(
    jsonl_dirs,
    output: Path,
    *,
    signal: str,
    model_label: str | None = None,
    method: str = "glm",
    n_bins: int = 5,
    n_boot: int = 1000,
    ci_sigma: float = 2.0,
    lowess_frac: float = 0.5,
    lowess_it: int = 0,
    seed: int = 0,
    no_title: bool = False,
) -> Path:
    """Compute and emit one pass1_vs_signals panel. CLI-equivalent entry."""
    display_key, ylim = BASELINE_SIGNALS[signal]
    signal_label = METHOD_DISPLAY.get(display_key, signal)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    inferred_model = None
    for d in jsonl_dirs:
        bl = infer_benchmark_label(d.name)
        domain = DOMAIN_OF.get(bl)
        if domain is None:
            print(f"  SKIP: {d.name} -> {bl!r}")
            continue
        init_path = find_init_file(d)
        if init_path is None:
            print(f"  WARNING: no init JSONL in {d.name}")
            continue
        if inferred_model is None:
            inferred_model = infer_model_label(d.name) or d.name
        stats = compute_per_problem_signals(init_path, signal)
        if not stats:
            print(f"  WARNING: no usable signal in {d.name}")
            continue
        n_c = sum(len(p["c_vals"]) for p in stats)
        n_w = sum(len(p["w_vals"]) for p in stats)
        if n_c == 0 and n_w == 0:
            print(f"  WARNING: no {signal} values in {d.name}")
            continue
        by_domain[domain].extend(stats)
        print(f"  {bl:20s} -> {domain:7s}  "
              f"n_problems={len(stats):3d}  "
              f"n_correct={n_c:5d}  n_wrong={n_w:5d}")

    if not by_domain:
        raise SystemExit("No data collected.")

    label = model_label or inferred_model or "Model"
    plot(by_domain, label, signal_label, ylim, method, n_bins, output,
         n_boot=n_boot, sigma=ci_sigma,
         show_title=not no_title,
         lowess_frac=lowess_frac, lowess_it=lowess_it, seed=seed)
    return output


if __name__ == "__main__":
    main()
