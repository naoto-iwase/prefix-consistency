#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy", "statsmodels"]
# ///
"""Generate reproduction rates vs. Pass@1 figure.

Three estimators share one inferential contract:

  * The point curve is the full-data fit.
  * CIs at ``+/- ci_sigma * SE`` come from a joint cluster bootstrap
    that resamples whole problems (not trials). Within each resample
    both ``r_C`` and ``r_W`` are refit, so the discrimination gap
    ``D = r_C - r_W`` inherits the correct joint sampling distribution.

Methods (selected via ``--method``):

  * ``glm``: logistic regression ``logit(p) = a + b * Pass@1`` per
    domain, returning the beta sample alongside the prediction band.
  * ``lowess``: trial-level LOWESS. Robustness reweightings are off
    (``it=0``): the bisquare weights are a continuous-residual
    construct and collapse on Bernoulli 0/1 outcomes.
  * ``binned``: trial-pooled reproduction rate over equal-count Pass@1
    quantile bins per domain. Bin edges are fixed from the full-data
    quantiles.

Each invocation emits a single panel (``--panel rates`` or
``--panel d``); the final 2x3 layout is composed in LaTeX.
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
    bench_sort_key,
    infer_benchmark_label, infer_model_label,
    find_regen_for_condition,
    setup_tex_rendering,
    is_usable_answer,
)
from _defs import (
    BENCHMARK_DOMAIN as DOMAIN_OF,
    DOMAIN_COLOR, BENCHMARK_COLOR,
)

setup_tex_rendering()


DOMAIN_ORDER = ["Math", "Science"]


# ── Data loading ────────────────────────────────────────────────────────

def compute_per_problem_stats(regen_path: Path) -> list[dict]:
    """Per-problem Pass@1 and (init, regen) trial counts.

    Each problem contributes ``c_pairs`` correct-init trials
    (``c_to_c`` of which reproduced the gold answer) and ``w_pairs``
    wrong-init trials (``w_match`` of which reproduced the init's
    wrong answer).
    """
    out = []
    with open(regen_path) as f:
        for line in f:
            rec = json.loads(line)
            gold = str(rec["gold_answer"])
            regen_groups = rec.get("regen_answers", [])
            n_init = n_correct = 0
            c_to_c = c_pairs = w_match = w_pairs = 0
            for i, pair in enumerate(rec["all_answers"]):
                if not is_usable_answer(pair):
                    continue
                init_ans = str(pair[0])
                init_correct = (init_ans == gold)
                n_init += 1
                n_correct += init_correct
                if i >= len(regen_groups):
                    continue
                for r in regen_groups[i]:
                    if not is_usable_answer(r):
                        continue
                    regen = str(r[0])
                    if init_correct:
                        c_pairs += 1
                        c_to_c += (regen == gold)
                    else:
                        w_pairs += 1
                        w_match += (regen == init_ans)
            if n_init == 0:
                continue
            out.append({
                "pass1": n_correct / n_init,
                "c_to_c": c_to_c, "c_pairs": c_pairs,
                "w_match": w_match, "w_pairs": w_pairs,
            })
    return out


def expand_trials(problems: list[dict],
                  kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-problem counts into per-trial (Pass@1, 0/1) arrays."""
    if kind == "C":
        succ_k, tot_k = "c_to_c", "c_pairs"
    else:
        succ_k, tot_k = "w_match", "w_pairs"
    xs: list[float] = []
    ys: list[int] = []
    for p in problems:
        n = p[tot_k]
        if n == 0:
            continue
        k = p[succ_k]
        xs.extend([p["pass1"]] * n)
        ys.extend([1] * k + [0] * (n - k))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


# ── Stat helpers ────────────────────────────────────────────────────────

def sigma_to_alpha(sigma: float) -> float:
    """CI at +/- sigma * SE corresponds to alpha = 2 * (1 - Phi(sigma))."""
    return 2.0 * (1.0 - norm.cdf(sigma))


def _percentiles(samples: np.ndarray, sigma: float,
                 axis: int | None = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    alpha = sigma_to_alpha(sigma)
    return (np.nanpercentile(samples, 100 * alpha / 2, axis=axis),
            np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=axis))


def samples_to_fit(beta_point: float, samples: np.ndarray,
                   ci_sigma: float = 2.0) -> dict:
    """Summarize a 1-D bootstrap distribution for table cells.

    Returns ``{"beta", "lo", "hi", "p"}``: point estimate, ``ci_sigma``
    cluster-bootstrap CI, and the two-sided bootstrap p-value for
    ``H_0: beta = 0``. ``p == 0`` indicates no replicate crossed zero
    (``"<.001"`` in tables).
    """
    valid = samples[~np.isnan(samples)]
    if not np.isfinite(beta_point) or len(valid) == 0:
        return {"beta": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    lo, hi = _percentiles(valid, ci_sigma, axis=None)
    tail = min(float(np.mean(valid <= 0)),
               float(np.mean(valid >= 0)))
    return {"beta": float(beta_point),
            "lo": float(lo), "hi": float(hi),
            "p": min(1.0, 2.0 * tail)}


def _quantile_edges(pass1: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count quantile edges, deduplicated. Endpoints are nudged
    outside ``[0, 1]`` so Pass@1 values at the extremes fall into the
    extreme bins; ``np.unique`` collapses duplicated quantiles (e.g.,
    from a mass at Pass@1 = 0), which can leave fewer than ``n_bins``
    effective bins."""
    edges = np.quantile(pass1, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = min(edges[0], 0.0) - 1e-9
    edges[-1] = max(edges[-1], 1.0) + 1e-9
    return np.unique(edges)


# ── Fit kernels ─────────────────────────────────────────────────────────

def _fit_glm(problems: list[dict], kind: str, xgrid: np.ndarray
             ) -> tuple[object | None, np.ndarray | None]:
    """Logistic GLM on the expanded trial data."""
    xs, ys = expand_trials(problems, kind)
    if len(xs) < 3 or len(np.unique(ys)) < 2:
        return None, None
    try:
        result = sm.GLM(ys, sm.add_constant(xs),
                        family=sm.families.Binomial()).fit()
    except Exception:
        return None, None
    return result, result.predict(sm.add_constant(xgrid))


def _fit_lowess(problems: list[dict], kind: str, xgrid: np.ndarray,
                frac: float, it: int) -> np.ndarray | None:
    """LOWESS smoother on expanded trial data; NaN outside support."""
    xs, ys = expand_trials(problems, kind)
    if len(xs) < 3 or len(np.unique(xs)) < 2:
        return None
    smoothed = sm_lowess(ys, xs, frac=frac, it=it, return_sorted=True)
    return np.interp(xgrid, smoothed[:, 0], smoothed[:, 1],
                     left=np.nan, right=np.nan)


def _bin_pool_rates(problems: list[dict], edges: np.ndarray,
                    kind: str) -> np.ndarray:
    """Per-bin trial-pooled rate: ``sum successes / sum trials``."""
    if kind == "C":
        succ_k, tot_k = "c_to_c", "c_pairs"
    else:
        succ_k, tot_k = "w_match", "w_pairs"
    pass1 = np.asarray([p["pass1"] for p in problems], dtype=float)
    nb = len(edges) - 1
    idx = np.clip(np.digitize(pass1, edges, right=False) - 1, 0, nb - 1)
    out = np.full(nb, np.nan)
    for b in range(nb):
        mask = idx == b
        if not mask.any():
            continue
        k_sum = sum(p[succ_k] for p, m in zip(problems, mask) if m)
        n_sum = sum(p[tot_k] for p, m in zip(problems, mask) if m)
        if n_sum > 0:
            out[b] = k_sum / n_sum
    return out


# ── Joint cluster bootstrap ─────────────────────────────────────────────

FitFn = Callable[[list[dict]], dict]


def cluster_bootstrap(problems: list[dict], fit_fn: FitFn,
                      n_boot: int,
                      rng: np.random.Generator
                      ) -> tuple[dict, dict]:
    """Joint cluster bootstrap over whole problems.

    ``fit_fn(problems)`` returns a dict mapping each quantity to a
    numpy array (or scalar). The same ``fit_fn`` is called on every
    resample, so quantities derived from multiple fits on the same
    resample -- e.g. ``D = r_C - r_W`` -- inherit the correct joint
    sampling distribution.

    Returns ``(point, samples)`` where ``point`` is ``fit_fn`` on the
    full data and ``samples[key]`` stacks the bootstrap replicates
    with leading axis ``n_boot``.
    """
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
            "rc": _bin_pool_rates(problems, edges, "C"),
            "rw": _bin_pool_rates(problems, edges, "W"),
        }
    return fit


def _mask_continuous_out_of_support(xgrid: np.ndarray,
                                    problems: list[dict],
                                    point: dict, samples: dict) -> None:
    """Set ``NaN`` where ``xgrid`` is outside the observed Pass@1
    support. Only relevant for the continuous estimators (GLM, LOWESS);
    for ``binned`` the bin centers are inside the support by
    construction."""
    xs_all = np.asarray([p["pass1"] for p in problems], dtype=float)
    mask = (xgrid < xs_all.min()) | (xgrid > xs_all.max())
    for k in ("rc", "rw"):
        point[k][mask] = np.nan
        samples[k][:, mask] = np.nan


# ── Plotting ────────────────────────────────────────────────────────────

def _scatter(ax, problems, kind, color, marker) -> None:
    """Scatter per-problem raw rates (density overlay)."""
    if kind == "C":
        succ_k, tot_k = "c_to_c", "c_pairs"
    else:
        succ_k, tot_k = "w_match", "w_pairs"
    xs = [p["pass1"] * 100 for p in problems if p[tot_k] > 0]
    ys = [p[succ_k] / p[tot_k] * 100
          for p in problems if p[tot_k] > 0]
    ax.scatter(xs, ys, s=10, alpha=0.18, color=color,
               marker=marker, linewidths=0 if marker == "o" else 0.7)


def _scatter_d(ax, problems, color) -> None:
    """Scatter per-problem ``D_i = r_C_i - r_W_i`` (density overlay on
    the D panel). Only problems with both classes of init are plotted;
    ``r_C`` is undefined at Pass@1 = 0 and ``r_W`` at Pass@1 = 1."""
    xs = []
    ys = []
    for p in problems:
        if p["c_pairs"] == 0 or p["w_pairs"] == 0:
            continue
        rc = p["c_to_c"] / p["c_pairs"]
        rw = p["w_match"] / p["w_pairs"]
        xs.append(p["pass1"] * 100)
        ys.append((rc - rw) * 100)
    ax.scatter(xs, ys, s=10, alpha=0.18, color=color,
               marker="o", linewidths=0)


def _draw_smooth(ax_rates, ax_d, xgrid, point, samples, sigma,
                 color, label) -> None:
    """Continuous r_C / r_W / D curves with pointwise cluster-bootstrap
    CI bands."""
    x = xgrid * 100
    lo_c, hi_c = _percentiles(samples["rc"], sigma)
    lo_w, hi_w = _percentiles(samples["rw"], sigma)
    lo_d, hi_d = _percentiles(samples["rc"] - samples["rw"], sigma)
    if ax_rates is not None:
        ax_rates.plot(x, point["rc"] * 100, "-", color=color,
                      linewidth=2.0, label=label)
        ax_rates.fill_between(x, lo_c * 100, hi_c * 100,
                              color=color, alpha=0.15, linewidth=0)
        ax_rates.plot(x, point["rw"] * 100, "--", color=color,
                      linewidth=2.0)
        ax_rates.fill_between(x, lo_w * 100, hi_w * 100,
                              color=color, alpha=0.10, linewidth=0,
                              hatch="//", edgecolor=color)
    if ax_d is not None:
        ax_d.plot(x, (point["rc"] - point["rw"]) * 100, "-",
                  color=color, linewidth=2.0, label=label)
        ax_d.fill_between(x, lo_d * 100, hi_d * 100,
                          color=color, alpha=0.15, linewidth=0)


def _draw_binned(ax_rates, ax_d, centers, point, samples, sigma,
                 color, label) -> None:
    """Discrete per-bin r_C / r_W / D with cluster-bootstrap error bars."""
    xc = centers * 100
    lo_c, hi_c = _percentiles(samples["rc"], sigma)
    lo_w, hi_w = _percentiles(samples["rw"], sigma)
    lo_d, hi_d = _percentiles(samples["rc"] - samples["rw"], sigma)
    if ax_rates is not None:
        ax_rates.errorbar(xc, point["rc"] * 100,
                          yerr=[(point["rc"] - lo_c) * 100,
                                (hi_c - point["rc"]) * 100],
                          fmt="-o", color=color, linewidth=2.0,
                          markersize=5, capsize=3, label=label)
        ax_rates.errorbar(xc, point["rw"] * 100,
                          yerr=[(point["rw"] - lo_w) * 100,
                                (hi_w - point["rw"]) * 100],
                          fmt="--x", color=color, linewidth=2.0,
                          markersize=6, capsize=3)
    if ax_d is not None:
        diff = point["rc"] - point["rw"]
        ax_d.errorbar(xc, diff * 100,
                      yerr=[(diff - lo_d) * 100, (hi_d - diff) * 100],
                      fmt="-o", color=color, linewidth=2.0,
                      markersize=5, capsize=3, label=label)


def _annotate_slope(ax, x_pct: np.ndarray, y_pct: np.ndarray,
                    beta_prefix: str, beta_point: float,
                    color, va: str = "center") -> None:
    """Place `β(r_X)=+X.XX` near the right end of a curve, in the
    curve's color, with a faint white bbox for legibility."""
    valid = np.where(~np.isnan(y_pct))[0]
    if len(valid) == 0 or not np.isfinite(beta_point):
        return
    idx = int(valid[-1])
    text = rf"${beta_prefix}\!=\!{beta_point:+.2f}$"
    ax.annotate(text,
                xy=(float(x_pct[idx]), float(y_pct[idx])),
                xytext=(-4, 0), textcoords="offset points",
                color=color, fontsize=10, va=va, ha="right",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=0.5))


def _print_beta_summary(glm_rows, sigma) -> None:
    if not glm_rows:
        return
    print("  GLM fit (logit(rate) = a + b*Pass@1):")
    for domain, kind, beta_point, betas in glm_rows:
        valid = betas[~np.isnan(betas)]
        if len(valid) == 0:
            continue
        lo, hi = _percentiles(valid, sigma, axis=None)
        tail = min(float(np.mean(valid <= 0)),
                   float(np.mean(valid >= 0)))
        p = min(1.0, 2.0 * tail)
        print(f"    {domain:7s} {kind}: "
              f"beta={beta_point:+.2f} "
              f"[{float(lo):+.2f}, {float(hi):+.2f}]  p={p:.3g}")


# ── Main plot ───────────────────────────────────────────────────────────

def plot(group_problems: dict[str, list[dict]], model_label: str,
         method: str, n_bins: int, out_path: Path,
         *, n_boot: int, sigma: float, panel: str, show_title: bool,
         lowess_frac: float, lowess_it: int, seed: int,
         group_order: list[str] | None = None,
         group_color: dict[str, str] | None = None,
         ) -> dict[str, dict[str, dict]]:
    """Render one panel and return GLM fit summaries.

    For ``method == "glm"`` returns ``{group_label: {"C": fit, "W": fit}}``
    where each ``fit`` is the dict from :func:`samples_to_fit`. For other
    methods returns an empty dict (no per-cell GLM betas to surface).
    """
    if panel == "both":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        ax_rates, ax_d = axes
    elif panel == "rates":
        fig, ax_rates = plt.subplots(figsize=(4.8, 3.8))
        ax_d = None
    elif panel == "d":
        fig, ax_d = plt.subplots(figsize=(4.8, 3.8))
        ax_rates = None
    else:
        raise ValueError(f"unknown panel: {panel}")

    xgrid = np.linspace(0.0, 1.0, 201)
    glm_rows: list[tuple[str, str, float, np.ndarray]] = []
    glm_curves: dict[str, dict] = {}
    glm_fits: dict[str, dict[str, dict]] = {}

    if group_order is None:
        group_order = DOMAIN_ORDER
    if group_color is None:
        group_color = DOMAIN_COLOR

    for group_label in group_order:
        problems = group_problems.get(group_label, [])
        if not problems:
            continue
        color = group_color[group_label]
        if ax_rates is not None:
            _scatter(ax_rates, problems, "C", color, "o")
            _scatter(ax_rates, problems, "W", color, "x")
        if ax_d is not None:
            _scatter_d(ax_d, problems, color)
        rng = np.random.default_rng(seed)

        if method == "glm":
            point, samples = cluster_bootstrap(
                problems, _glm_fit_fn(xgrid), n_boot, rng)
            _mask_continuous_out_of_support(
                xgrid, problems, point, samples)
            _draw_smooth(ax_rates, ax_d, xgrid, point, samples,
                         sigma, color, group_label)
            glm_curves[group_label] = {"point": point, "color": color}
            glm_rows.append((group_label, "r_C", point["beta_c"],
                             samples["beta_c"]))
            glm_rows.append((group_label, "r_W", point["beta_w"],
                             samples["beta_w"]))
            glm_fits[group_label] = {
                "C": samples_to_fit(point["beta_c"], samples["beta_c"], sigma),
                "W": samples_to_fit(point["beta_w"], samples["beta_w"], sigma),
            }

        elif method == "lowess":
            point, samples = cluster_bootstrap(
                problems,
                _lowess_fit_fn(xgrid, lowess_frac, lowess_it),
                n_boot, rng)
            _mask_continuous_out_of_support(
                xgrid, problems, point, samples)
            _draw_smooth(ax_rates, ax_d, xgrid, point, samples,
                         sigma, color, group_label)

        elif method == "binned":
            pass1_all = np.asarray([p["pass1"] for p in problems])
            edges = _quantile_edges(pass1_all, n_bins)
            point, samples = cluster_bootstrap(
                problems, _binned_fit_fn(edges), n_boot, rng)
            centers = 0.5 * (np.clip(edges[:-1], 0, 1)
                             + np.clip(edges[1:], 0, 1))
            _draw_binned(ax_rates, ax_d, centers, point, samples,
                         sigma, color, group_label)

        else:
            raise ValueError(f"unknown method: {method}")

    # Per-rate annotation pass: place each group's slope label so the
    # higher curve at the right end gets the label above (va="bottom"),
    # the lower one below (va="top"). This avoids collisions on models
    # where Math and Science cross between panels. With more than two
    # groups (per-benchmark mode), labels would overlap regardless of
    # placement, so we skip them and let the slope table carry the
    # numeric values.
    if (method == "glm" and ax_rates is not None and glm_curves
            and len(glm_curves) <= 2):
        x_pct = xgrid * 100
        for rate_key, beta_key, prefix in [
            ("rc", "beta_c", r"\beta(r_C)"),
            ("rw", "beta_w", r"\beta(r_W)"),
        ]:
            rightmost = {}
            for d, info in glm_curves.items():
                curve = info["point"][rate_key] * 100
                idxs = np.where(~np.isnan(curve))[0]
                if len(idxs) > 0:
                    rightmost[d] = float(curve[idxs[-1]])
            ranked = sorted(rightmost, key=lambda d: -rightmost[d])
            for rank, d in enumerate(ranked):
                info = glm_curves[d]
                va = "bottom" if rank == 0 else "top"
                _annotate_slope(ax_rates, x_pct,
                                info["point"][rate_key] * 100,
                                prefix, info["point"][beta_key],
                                info["color"], va=va)

    if ax_rates is not None:
        ax_rates.plot([], [], color="black", linestyle="-",
                      label=r"$r_C$ (solid, $\bullet$)")
        ax_rates.plot([], [], color="black", linestyle="--",
                      label=r"$r_W$ (dashed, $\times$)")
        ax_rates.set_xlabel(r"Pass@1 (\%)")
        ax_rates.set_ylabel(r"Reproduction rate (\%)")
        ax_rates.set_xlim(0, 100)
        ax_rates.set_ylim(0, 100)
        ax_rates.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax_rates.grid(True, alpha=0.3)
        if panel == "rates" and show_title:
            ax_rates.set_title(model_label)

    if ax_d is not None:
        ax_d.axhline(0, color="black", linestyle=":", linewidth=0.7)
        ax_d.set_xlabel(r"Pass@1 (\%)")
        ax_d.set_ylabel(r"$D = r_C - r_W$ (\%)")
        ax_d.set_xlim(0, 100)
        ax_d.set_ylim(-100, 100)
        ax_d.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax_d.grid(True, alpha=0.3)
        if panel == "d" and show_title:
            ax_d.set_title(model_label)

    if panel == "both" and show_title:
        fig.suptitle(model_label)

    _print_beta_summary(glm_rows, sigma)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return glm_fits


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="pass1_vs_rates figure: one panel per call, with "
                    "joint cluster-bootstrap CIs shared by all three "
                    "estimators.")
    ap.add_argument("jsonl_dirs", nargs="+", type=Path)
    ap.add_argument("--condition", default="rm25pct_full_x1")
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--method", choices=["glm", "lowess", "binned"],
                    default="glm")
    ap.add_argument("--n-bins", type=int, default=5,
                    help="Pass@1 quantile bins per domain (binned "
                         "method). Effective count may be smaller "
                         "when quantile edges collapse.")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Cluster-bootstrap replicates.")
    ap.add_argument("--ci-sigma", type=float, default=2.0,
                    help="CI half-width in sigma units "
                         "(2.0 ~= 95.45%%; matches Tables 3 and 4).")
    ap.add_argument("--lowess-frac", type=float, default=0.5)
    ap.add_argument("--lowess-it", type=int, default=0,
                    help="LOWESS robustness iterations. Default 0 "
                         "because the statsmodels default of 3 uses "
                         "bisquare residual weights that misbehave on "
                         "Bernoulli 0/1 outcomes.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel", choices=["both", "rates", "d"],
                    default="both")
    ap.add_argument("--group", choices=["domain", "benchmark"],
                    default="domain",
                    help="Aggregation axis. ``domain`` (default) "
                         "pools benchmarks by Math/Science. "
                         "``benchmark`` keeps each benchmark as a "
                         "separate curve, used for the per-benchmark "
                         "robustness check in Appendix D.3.")
    ap.add_argument("--no-title", action="store_true",
                    help="Suppress model-label title (use when "
                         "titles would repeat in a composed grid).")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    run_one(
        args.jsonl_dirs, args.output,
        condition=args.condition, model_label=args.model_label,
        method=args.method, group=args.group, panel=args.panel,
        n_bins=args.n_bins, n_boot=args.n_boot, ci_sigma=args.ci_sigma,
        lowess_frac=args.lowess_frac, lowess_it=args.lowess_it,
        seed=args.seed, no_title=args.no_title,
    )


def run_one(
    jsonl_dirs,
    output: Path,
    *,
    condition: str = "rm25pct_full_x1",
    model_label: str | None = None,
    method: str = "glm",
    group: str = "domain",
    panel: str = "both",
    n_bins: int = 5,
    n_boot: int = 1000,
    ci_sigma: float = 2.0,
    lowess_frac: float = 0.5,
    lowess_it: int = 0,
    seed: int = 0,
    no_title: bool = False,
) -> dict[str, dict[str, dict]]:
    """Compute and emit one pass1_vs_rates panel.

    Returns the per-group GLM fit summaries when ``method == "glm"`` so
    callers can build slope tables from the same bootstrap as the figure
    (eliminates CI drift from re-bootstrapping). Empty for other methods.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    inferred_model = None
    seen_benchmarks: list[str] = []
    for d in jsonl_dirs:
        bl = infer_benchmark_label(d.name)
        domain = DOMAIN_OF.get(bl)
        if domain is None:
            print(f"  SKIP: {d.name} -> {bl!r}")
            continue
        regen_path = find_regen_for_condition(d, condition)
        if regen_path is None:
            print(f"  WARNING: no regen for {condition} in {d.name}")
            continue
        if inferred_model is None:
            inferred_model = infer_model_label(d.name) or d.name
        stats = compute_per_problem_stats(regen_path)
        if group == "domain":
            by_group[domain].extend(stats)
            print(f"  {bl:20s} -> {domain:7s}  n={len(stats):3d}")
        else:
            by_group[bl].extend(stats)
            if bl not in seen_benchmarks:
                seen_benchmarks.append(bl)
            print(f"  {bl:20s} -> {bl:20s}  n={len(stats):3d}")

    if not by_group:
        raise SystemExit("No data collected.")

    if group == "domain":
        group_order = DOMAIN_ORDER
        group_color = DOMAIN_COLOR
    else:
        group_order = sorted(seen_benchmarks, key=bench_sort_key)
        group_color = {bl: BENCHMARK_COLOR.get(bl, "tab:gray")
                       for bl in group_order}

    label = model_label or inferred_model or "Model"
    return plot(by_group, label, method, n_bins, output,
                n_boot=n_boot, sigma=ci_sigma, panel=panel,
                show_title=not no_title,
                lowess_frac=lowess_frac, lowess_it=lowess_it, seed=seed,
                group_order=group_order, group_color=group_color)


if __name__ == "__main__":
    main()
