#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scikit-learn"]
# ///
"""Generate signal comparison panels for the paper.

By default emits the three panels used in main.tex:

  <stem>_pc.{ext}    -- Prefix consistency violin
  <stem>_ptrue.{ext} -- P(True) violin
  <stem>_tail.{ext}  -- DeepConf (tail) violin

Pass ``--panels`` to widen the selection. The full set of panels is

  pc, ptrue, tail, bottom10, verbal, robustness

i.e. include the three diagnostic panels (DeepConf bottom-10%, Verbal
0-100, and the consistency gap across tau).

Pass ``--stem signal_violin_120b_fsci`` (matching the paper's
includegraphics paths) to override the filename prefix.

All violin panels share the same figure size and fixed margins
so the plot area is identical across panels.

Usage:
    uv run python paper/figure_signal_violin.py \\
        gpt-oss-20b_frontierscience_olympiad_jsonl

    uv run python paper/figure_signal_violin.py \\
        gpt-oss-20b_frontierscience_olympiad_jsonl \\
        --panels pc ptrue tail bottom10 verbal robustness \\
        --formats png pdf --regen-k 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from _utils import (tau_from_rm_pct, find_regen_files,
                     load_init_records, load_regen_records,
                     setup_tex_rendering, METHOD_DISPLAY,
                     is_usable_answer)

setup_tex_rendering()

# ── Constants ──

COLOR_CORRECT = "#2ca02c"
COLOR_INCORRECT = "#d62728"

PANEL_FIGSIZE = (4.0, 2.5)
PANEL_MARGINS = dict(left=0.17, right=0.97, top=0.90, bottom=0.18)

# Baseline signals: (jsonl_key, method_display_key, panel_short).
# Display titles come from METHOD_DISPLAY in _defs.py to stay in sync
# with the paper tables. jsonl_key is the raw-record field name; the
# method_display_key is the wmv_result.json method name. panel_short
# is appended to ``--stem`` to form the output filename.
BASELINE_PANELS = [
    ("bottom10_conf", "deepconf_bottom10", "bottom10"),
    ("tail_conf",     "deepconf_tail",     "tail"),
    ("verbal_0_100",  "verbal_0_100_raw",  "verbal"),
    ("p_true",        "p_true_raw",        "ptrue"),
]

ALL_PANELS = ["pc", "ptrue", "tail", "bottom10", "verbal", "robustness"]
DEFAULT_PANELS = ["pc", "ptrue", "tail"]


# ── Signal extraction ──

def _extract_confidence(records, key):
    """Split confidence values by correctness.  Returns (correct, incorrect)."""
    correct, incorrect = [], []
    for rec in records:
        gold = rec["gold_answer"]
        if key in ("p_true", "verbal_0_100"):
            vals = rec.get(f"{key}_confidences") or rec.get(key) or []
            for i, entry in enumerate(rec["all_answers"]):
                if not is_usable_answer(entry) or i >= len(vals) or vals[i] is None:
                    continue
                v = float(vals[i])
                if np.isfinite(v):
                    (correct if str(entry[0]) == gold else incorrect).append(v)
        else:
            confs = rec.get("confidences") or []
            for i, entry in enumerate(rec["all_answers"]):
                if not is_usable_answer(entry) or i >= len(confs) or confs[i] is None:
                    continue
                v = confs[i].get(key)
                if v is not None and np.isfinite(v):
                    (correct if str(entry[0]) == gold else incorrect).append(v)
    return correct, incorrect


def _extract_confidence_per_problem(records, key):
    """Per-problem mean confidence by class, restricted to Q'
    (problems with at least one correct AND one wrong sample with valid signal).
    Returns (correct_means, incorrect_means), one entry per problem in Q'.
    """
    correct_means, incorrect_means = [], []
    for rec in records:
        gold = rec["gold_answer"]
        c_vals, i_vals = [], []
        if key in ("p_true", "verbal_0_100"):
            vals = rec.get(f"{key}_confidences") or rec.get(key) or []
            for i, entry in enumerate(rec["all_answers"]):
                if not is_usable_answer(entry) or i >= len(vals) or vals[i] is None:
                    continue
                v = float(vals[i])
                if not np.isfinite(v):
                    continue
                (c_vals if str(entry[0]) == gold else i_vals).append(v)
        else:
            confs = rec.get("confidences") or []
            for i, entry in enumerate(rec["all_answers"]):
                if not is_usable_answer(entry) or i >= len(confs) or confs[i] is None:
                    continue
                v = confs[i].get(key)
                if v is None or not np.isfinite(v):
                    continue
                (c_vals if str(entry[0]) == gold else i_vals).append(v)
        if c_vals and i_vals:
            correct_means.append(float(np.mean(c_vals)))
            incorrect_means.append(float(np.mean(i_vals)))
    return correct_means, incorrect_means


def _extract_pc_consistency(init_records, regen_data, k, mode="init"):
    """Prefix consistency scores split by correctness.

    mode="init"     : one sample per group, value = c_i(a_i) (init answer's
                      PC score in A_i), labeled by whether the init answer
                      is correct.
    mode="distinct" : one sample per distinct answer in A_i, value = c_i(a)
                      (that answer's PC score in A_i), labeled by whether
                      the candidate answer equals the gold answer.
    """
    if mode not in ("init", "distinct"):
        raise ValueError(f"mode must be 'init' or 'distinct', got {mode!r}")
    correct, incorrect = [], []
    for rec in init_records:
        pnum = rec["problem_num"]
        gold = rec["gold_answer"]
        regen_rec = regen_data.get(pnum)
        if not regen_rec:
            continue
        regens = regen_rec.get("regen_answers") or []
        for i, entry in enumerate(rec["all_answers"]):
            if not is_usable_answer(entry):
                continue
            if i >= len(regens) or not regens[i]:
                continue
            init_ans = str(entry[0])
            regen_answers = [str(r[0]) for r in regens[i]
                             if is_usable_answer(r)][:k]
            if not regen_answers:
                continue
            group = [init_ans] + regen_answers
            n = len(group)
            if mode == "init":
                freq = group.count(init_ans) / n
                (correct if init_ans == gold else incorrect).append(freq)
            else:  # distinct
                for a in set(group):
                    freq = group.count(a) / n
                    (correct if a == gold else incorrect).append(freq)
    return correct, incorrect


def _extract_pc_per_problem_rates(init_records, regen_data, k):
    """Per-problem reproduction rates r_{C,q} and r_{W,q}, restricted to Q'
    (problems with at least one correct AND one wrong initial sample).
    Macro-averaging the returned lists yields r_C(tau), r_W(tau) as in the paper.
    """
    rcq, rwq = [], []
    for rec in init_records:
        pnum = rec["problem_num"]
        gold = rec["gold_answer"]
        regen_rec = regen_data.get(pnum)
        if not regen_rec:
            continue
        regens = regen_rec.get("regen_answers") or []
        c_matches, i_matches = [], []
        for i, entry in enumerate(rec["all_answers"]):
            if not is_usable_answer(entry):
                continue
            if i >= len(regens) or not regens[i]:
                continue
            init_ans = str(entry[0])
            regen_answers = [str(r[0]) for r in regens[i]
                             if is_usable_answer(r)][:k]
            if not regen_answers:
                continue
            match_frac = sum(1 for ra in regen_answers if ra == init_ans) / len(regen_answers)
            (c_matches if init_ans == gold else i_matches).append(match_frac)
        if c_matches and i_matches:
            rcq.append(float(np.mean(c_matches)))
            rwq.append(float(np.mean(i_matches)))
    return rcq, rwq


def _compute_auroc(correct_vals, incorrect_vals):
    y = [1] * len(correct_vals) + [0] * len(incorrect_vals)
    s = list(correct_vals) + list(incorrect_vals)
    if len(y) < 10 or len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def _consistency_stats(init_records, regen_data, k, mode="init"):
    """Mean and SE of the PC score over the (init + regen) group."""
    c, ic = _extract_pc_consistency(init_records, regen_data, k, mode=mode)
    mean_c = np.mean(c) if c else 0.0
    mean_ic = np.mean(ic) if ic else 0.0
    se_c = np.std(c) / np.sqrt(len(c)) if len(c) > 1 else 0.0
    se_ic = np.std(ic) / np.sqrt(len(ic)) if len(ic) > 1 else 0.0
    return mean_c, mean_ic, se_c, se_ic


# ── Plotting ──

def _save(fig, path, formats):
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=300)
    print(f"  Saved: {path.with_suffix('')} ({', '.join(formats)})")


def _plot_violin(correct_vals, incorrect_vals, title, ylabel, auroc,
                 ylim=None, yticks=None, mean_labels=None,
                 xticklabels=("Correct", "Wrong")):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    if not correct_vals or not incorrect_vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    parts = ax.violinplot([correct_vals, incorrect_vals], positions=[0, 1],
                          showmeans=True, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([COLOR_CORRECT, COLOR_INCORRECT][i])
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(xticklabels), fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    means = [float(np.mean(correct_vals)), float(np.mean(incorrect_vals))]
    if mean_labels is None:
        # Label each mean line with its numeric value; prefix with "mean" so
        # the statistic is unambiguous (violin plots conventionally show the
        # median, so a bare number could be misread).
        fmt = (lambda v: f"{v:.2f}") if max(means) <= 1.5 else (lambda v: f"{v:.1f}")
        labels = [f"mean {fmt(m)}" for m in means]
    else:
        labels = list(mean_labels)
    # Place left-violin label to the right of the violin and right-violin
    # label to the left, so neither extends past the panel edge.
    for x, m, label in zip([0, 1], means, labels):
        if x == 0:
            xytext, ha = (6, 0), "left"
        else:
            xytext, ha = (-6, 0), "right"
        ax.annotate(
            label,
            xy=(x, m),
            xytext=xytext,
            textcoords="offset points",
            va="center",
            ha=ha,
            ma="center",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85),
        )

    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(**PANEL_MARGINS)
    return fig


def _plot_robustness(taus, mean_c, mean_ic, se_c, se_ic, k, ylim=None,
                     yticks=None):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    taus = np.array(taus)
    yc, yic = np.array(mean_c) * 100, np.array(mean_ic) * 100
    ec, eic = np.array(se_c) * 100, np.array(se_ic) * 100

    ax.plot(taus, yc, "o-", color=COLOR_CORRECT, lw=2, ms=7,
            label="Correct", zorder=3)
    ax.fill_between(taus, yc - 2 * ec, yc + 2 * ec,
                    color=COLOR_CORRECT, alpha=0.15)
    ax.plot(taus, yic, "o-", color=COLOR_INCORRECT, lw=2, ms=7,
            label="Wrong", zorder=3)
    ax.fill_between(taus, yic - 2 * eic, yic + 2 * eic,
                    color=COLOR_INCORRECT, alpha=0.15)

    ax.set_xlabel(r"Truncation fraction $\tau$", fontsize=11)
    ax.set_ylabel(r"Consistency (\%)", fontsize=11)
    ax.set_title(
        r"Robustness of consistency gap across $\tau$",
        fontsize=11)
    ax.set_xticks(taus)
    ax.set_xlim(taus[0] - 0.05, taus[-1] + 0.05)
    ax.set_ylim(ylim if ylim else (0, 105))
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(**PANEL_MARGINS)
    return fig


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate signal comparison panels for the paper")
    parser.add_argument("jsonl_dir", type=Path)
    parser.add_argument("--regen-k", type=int, default=1,
                        help="Regeneration count K (default: 1)")
    parser.add_argument("--mode", choices=["init", "distinct"], default="init",
                        help="Prefix consistency signal variant: 'init' scores "
                             "each group by c_i(a_i) (1 sample/group); 'distinct' "
                             "scores every distinct answer in A_i by c_i(a) "
                             "(|unique(A_i)| samples/group). Default: init.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--stem", default="signal_violin",
                        help="Filename prefix; outputs are <stem>_<panel>."
                             "{fmt} (default: 'signal_violin' giving e.g. "
                             "signal_violin_pc.pdf). Set to "
                             "'signal_violin_<model>_<bench>' for paper paths.")
    parser.add_argument("--panels", nargs="+", choices=ALL_PANELS,
                        default=DEFAULT_PANELS,
                        help=f"Which panels to emit (default: "
                             f"{' '.join(DEFAULT_PANELS)}; the panels used "
                             f"in main.tex). Choose any subset of "
                             f"{', '.join(ALL_PANELS)}.")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: <jsonl_dir>/plots_signal_comparison)")
    args = parser.parse_args()
    panels = set(args.panels)

    jsonl_dir = args.jsonl_dir
    k = args.regen_k
    mode = args.mode
    stem = args.stem
    out_dir = args.out_dir or (jsonl_dir / "plots_signal_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Mode suffix for the prefix-consistency panels (baselines don't depend on mode).
    pc_suffix = "" if mode == "init" else f"_{mode}"

    print(f"Loading data from {jsonl_dir}")
    init_records = load_init_records(jsonl_dir)
    print(f"  {len(init_records)} problems")

    # (a)-(d): baseline signals -- per-problem mean confidence by class,
    # restricted to Q' (problems with both correct and wrong samples).
    for key, display_key, panel in BASELINE_PANELS:
        if panel not in panels:
            continue
        label = METHOD_DISPLAY[display_key]
        c, ic = _extract_confidence_per_problem(init_records, key)
        auroc = _compute_auroc(c, ic)
        print(f"  {label}: per-problem AUROC={auroc:.3f} "
              f"(|Q'|={len(c)})")
        fig = _plot_violin(c, ic, label, "Mean confidence", auroc)
        _save(fig, out_dir / f"{stem}_{panel}.png", args.formats)
        plt.close(fig)

    pc_ylim = (-5, 105)
    pc_yticks = [0, 25, 50, 75, 100]

    if "pc" in panels or "robustness" in panels:
        regen_files = find_regen_files(jsonl_dir)
    else:
        regen_files = []

    # (e): prefix consistency violin
    if "pc" in panels:
        regen_path = None
        for rm_pct, scope, regen_k, path in regen_files:
            if rm_pct == 25 and regen_k >= k and scope == "full":
                regen_path = path
                break
        if not regen_path:
            print("ERROR: no regen file for rm25pct_full", file=sys.stderr)
            sys.exit(1)

        regen_data = load_regen_records(regen_path)
        pc_c, pc_ic = _extract_pc_per_problem_rates(init_records, regen_data, k)
        auroc_pc = _compute_auroc(pc_c, pc_ic)
        pc_c_pct = [f * 100 for f in pc_c]
        pc_ic_pct = [f * 100 for f in pc_ic]
        r_c = float(np.mean(pc_c)) if pc_c else float("nan")
        r_w = float(np.mean(pc_ic)) if pc_ic else float("nan")
        print(f"  Prefix Consistency: per-problem AUROC={auroc_pc:.3f}, "
              f"macro r_C={r_c:.3f}, r_W={r_w:.3f} (|Q'|={len(pc_c)})")
        mean_labels = [
            rf"$r_C(0.75) = {r_c * 100:.0f}\%$",
            rf"$r_W(0.75) = {r_w * 100:.0f}\%$",
        ]
        fig = _plot_violin(
            pc_c_pct, pc_ic_pct,
            "Prefix Consistency",
            r"Mean reproduction rate (\%)", auroc_pc,
            ylim=pc_ylim, yticks=pc_yticks,
            mean_labels=mean_labels)
        _save(fig, out_dir / f"{stem}_pc{pc_suffix}.png", args.formats)
        plt.close(fig)

    # (f): robustness across tau
    if "robustness" in panels:
        taus, mc_list, mic_list, sec_list, seic_list = [], [], [], [], []
        for rm_pct in [75, 50, 25]:
            tau = tau_from_rm_pct(rm_pct)
            rp = None
            for rp_pct, scope, regen_k, path in regen_files:
                if rp_pct == rm_pct and regen_k >= k and scope == "full":
                    rp = path
                    break
            if not rp:
                continue
            rd = load_regen_records(rp)
            mc, mic, sec, seic = _consistency_stats(init_records, rd, k,
                                                    mode=mode)
            taus.append(tau)
            mc_list.append(mc)
            mic_list.append(mic)
            sec_list.append(sec)
            seic_list.append(seic)
            print(f"  tau={tau:.2f}: correct={mc:.3f}, incorrect={mic:.3f}")

        fig = _plot_robustness(taus, mc_list, mic_list, sec_list, seic_list,
                               k, ylim=pc_ylim, yticks=pc_yticks)
        _save(fig, out_dir / f"{stem}_robustness{pc_suffix}.png", args.formats)
        plt.close(fig)

    print(f"\nDone. {len(panels)} panels saved to {out_dir}")


if __name__ == "__main__":
    main()
