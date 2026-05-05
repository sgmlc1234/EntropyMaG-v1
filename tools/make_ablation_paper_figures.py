#!/usr/bin/env python3
"""Build paper figures for the expanded validation-gate ablation study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
FIG_DIR = ROOT / "paper_submission_v2" / "figures"
ABLATION_CSV = REPO / "data/analysis/ablation_microstudy/ablation_candidates.csv"

BENCH_ORDER = ["math500", "aime2025", "gsm8k"]
BENCH_LABELS = {
    "math500": "MATH-500\nL4-5",
    "aime2025": "AIME\n2025",
    "gsm8k": "GSM8K",
}

COLORS = {
    "near_copy": "#AFCFF5",
    "same_answer": "#C8E8C0",
    "critical": "#FFD8A8",
    "shadow_fail": "#F5B7B1",
    "answer_mismatch": "#D9C6F3",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den.astype(float) > 0, num.astype(float) / den.astype(float) * 100.0, 0.0)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, condition), sub in df.groupby(["benchmark", "condition"], sort=False):
        n = len(sub)
        shadow_den = int(sub["shadow_solvability_checked"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "benchmark": benchmark,
                "condition": condition,
                "candidate_count": n,
                "near_copy_rate": 100.0 * sub["near_copy_candidate"].fillna(False).astype(bool).sum() / n,
                "parent_answer_same_rate": 100.0 * sub["parent_answer_same"].fillna(False).astype(bool).sum() / n,
                "answer_mismatch_rate": 100.0 * sub["answer_mismatch"].fillna(False).astype(bool).sum() / n,
                "critical_error_proxy_rate": 100.0 * sub["critical_error_proxy"].fillna(False).astype(bool).sum() / n,
                "shadow_solvability_fail_rate": (
                    100.0 * sub["shadow_solvability_fail"].fillna(False).astype(bool).sum() / shadow_den
                    if shadow_den
                    else 0.0
                ),
                "shadow_solvability_fail_candidate_rate": 100.0
                * sub["shadow_solvability_fail"].fillna(False).astype(bool).sum()
                / n,
            }
        )
    return pd.DataFrame(rows)


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.35,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#111827",
        )


def make_ablation_gate_stress_test() -> None:
    df = pd.read_csv(ABLATION_CSV)
    summary = _aggregate(df)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.36},
    )
    x = np.arange(len(BENCH_ORDER))

    panel_a = summary[summary["condition"].eq("no_near_copy")].set_index("benchmark").reindex(BENCH_ORDER)
    width = 0.28
    ax = axes[0]
    bars = ax.bar(
        x - width / 2,
        panel_a["near_copy_rate"].to_numpy(dtype=float),
        width=width,
        color=COLORS["near_copy"],
        edgecolor="#111827",
        linewidth=0.55,
        label="Near-copy candidates",
        zorder=3,
    )
    _bar_labels(ax, bars)
    bars = ax.bar(
        x + width / 2,
        panel_a["parent_answer_same_rate"].to_numpy(dtype=float),
        width=width,
        color=COLORS["same_answer"],
        edgecolor="#111827",
        linewidth=0.55,
        label="Same-answer descendants",
        zorder=3,
    )
    _bar_labels(ax, bars)
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Diversity-pressure\nrate (%)")
    ax.set_ylim(0, 18)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.text(-0.08, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8.0)

    panel_b = summary[summary["condition"].eq("no_solvability")].set_index("benchmark").reindex(BENCH_ORDER)
    width = 0.22
    ax = axes[1]
    metrics = [
        ("critical_error_proxy_rate", "Critical-error proxy", COLORS["critical"]),
        ("shadow_solvability_fail_candidate_rate", "Shadow solvability fail", COLORS["shadow_fail"]),
        ("answer_mismatch_rate", "Answer mismatch", COLORS["answer_mismatch"]),
    ]
    offsets = np.linspace(-width, width, len(metrics))
    for offset, (column, label, color) in zip(offsets, metrics):
        bars = ax.bar(
            x + offset,
            panel_b[column].to_numpy(dtype=float),
            width=width,
            color=color,
            edgecolor="#111827",
            linewidth=0.55,
            label=label,
            zorder=3,
        )
        _bar_labels(ax, bars)
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Consistency-pressure\nrate (%)")
    ax.set_ylim(0, 16)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.text(-0.08, 1.03, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8.0)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.08)
    _save(fig, "ablation_gate_stress_test")


def main() -> None:
    make_ablation_gate_stress_test()
    print(FIG_DIR / "ablation_gate_stress_test.pdf")


if __name__ == "__main__":
    main()
