#!/usr/bin/env python3
"""Build paper figures for expanded external evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
FIG_DIR = ROOT / "paper_submission_v2" / "figures"

SIG_CSV = REPO / "data/analysis/external_significance/external_significance_comparisons.csv"
SIG_RUNS_CSV = REPO / "data/analysis/external_significance/external_significance_runs.csv"
GEN_CELLS_CSV = REPO / "data/analysis/external_generation_eval/external_generation_cells.csv"
GEN_TRENDS_CSV = REPO / "data/analysis/external_generation_eval/external_generation_trends.csv"
EXPANDED_SUMMARY_JSON = REPO / "data/eval/external_ablation/expanded_full_summary.json"

BENCH_ORDER = ["math500", "aime2025", "gsm8k"]
BENCH_LABELS = {
    "math500": "MATH-500\nL4-5",
    "aime2025": "AIME\n2025",
    "gsm8k": "GSM8K",
}
MODEL_ORDER = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
]
MODEL_LABELS = {
    "openai/gpt-5.4-mini": "GPT-5.4-mini",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "google/gemini-3.1-flash-lite-preview": "Gemini Flash Lite",
}
MODEL_COLORS = {
    "openai/gpt-5.4-mini": "#AFCFF5",
    "anthropic/claude-haiku-4.5": "#C8E8C0",
    "google/gemini-3.1-flash-lite-preview": "#FFD8A8",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["benchmark"] = pd.Categorical(df["benchmark"], BENCH_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["benchmark", "model"])


def _classify(row: pd.Series) -> str:
    drop = float(row["problem_accuracy_drop_pp"])
    low = float(row["problem_accuracy_boot95_low_pp"])
    p = float(row["problem_accuracy_boot_p_one_sided"])
    if drop > 0 and low > 0 and p < 0.05:
        return "bootstrap CI > 0"
    if drop > 0 and p < 0.10:
        return "directional"
    return "mixed/inconclusive"


def _bootstrap_ci(values: list[float], seed: int, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def make_significance_hardness_bars() -> None:
    df = _ordered(pd.read_csv(SIG_CSV))
    df["class"] = df.apply(_classify, axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    x = np.arange(len(BENCH_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(MODEL_ORDER))

    hatches = {
        "bootstrap CI > 0": "",
        "directional": "///",
        "mixed/inconclusive": "xx",
    }
    alphas = {
        "bootstrap CI > 0": 0.96,
        "directional": 0.78,
        "mixed/inconclusive": 0.38,
    }

    for offset, model in zip(offsets, MODEL_ORDER):
        sub = df[df["model"].astype(str).eq(model)].set_index("benchmark").reindex(BENCH_ORDER)
        vals = sub["problem_accuracy_drop_pp"].astype(float).values
        low = sub["problem_accuracy_boot95_low_pp"].astype(float).values
        high = sub["problem_accuracy_boot95_high_pp"].astype(float).values
        classes = sub["class"].tolist()
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="#111827",
            linewidth=0.55,
            label=MODEL_LABELS[model],
            zorder=3,
        )
        for bar, cls in zip(bars, classes):
            bar.set_hatch(hatches[cls])
            bar.set_alpha(alphas[cls])
        ax.errorbar(
            x + offset,
            vals,
            yerr=np.vstack([vals - low, high - vals]),
            fmt="none",
            ecolor="#111827",
            elinewidth=0.75,
            capsize=2.0,
            zorder=4,
        )

    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Control - treatment accuracy (pp)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_ylim(-8, 62)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=3, frameon=False, fontsize=8.2)

    from matplotlib.patches import Patch

    class_handles = [
        Patch(facecolor="#9CA3AF", edgecolor="#111827", hatch="", label="CI above 0"),
        Patch(facecolor="#9CA3AF", edgecolor="#111827", hatch="///", label="directional"),
        Patch(facecolor="#9CA3AF", edgecolor="#111827", hatch="xx", label="mixed"),
    ]
    leg2 = ax.legend(handles=class_handles, loc="upper right", frameon=False, fontsize=7.8)
    ax.add_artist(leg2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=3, frameon=False, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "external_significance_hardness_bars")


def make_generation_delta_bars() -> None:
    df = _ordered(pd.read_csv(GEN_TRENDS_CSV))
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    x = np.arange(len(BENCH_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(MODEL_ORDER))

    for offset, model in zip(offsets, MODEL_ORDER):
        sub = df[df["model"].astype(str).eq(model)].set_index("benchmark").reindex(BENCH_ORDER)
        vals = sub["last_minus_first_pp"].astype(float).values
        slopes = sub["linear_slope_pp_per_generation"].astype(float).values
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="#111827",
            linewidth=0.55,
            label=MODEL_LABELS[model],
            alpha=0.92,
            zorder=3,
        )
        for xpos, val, slope in zip(x + offset, vals, slopes):
            ax.scatter(
                [xpos],
                [val + (2.2 if val >= 0 else -2.2)],
                marker="v" if slope < 0 else "^",
                s=16,
                color="#111827",
                zorder=5,
            )

    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Gen10 - Gen1 accuracy (pp)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_ylim(-48, 12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncols=3, frameon=False, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "external_generation_delta_bars")


def make_external_core_evidence_bars() -> None:
    sig = _ordered(pd.read_csv(SIG_CSV))
    sig["class"] = sig.apply(_classify, axis=1)
    gen = _ordered(pd.read_csv(GEN_TRENDS_CSV))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.55),
        gridspec_kw={"height_ratios": [1.12, 1.0], "hspace": 0.38},
    )
    x = np.arange(len(BENCH_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(MODEL_ORDER))
    hatches = {
        "bootstrap CI > 0": "",
        "directional": "///",
        "mixed/inconclusive": "xx",
    }
    alphas = {
        "bootstrap CI > 0": 0.96,
        "directional": 0.78,
        "mixed/inconclusive": 0.38,
    }

    ax = axes[0]
    for offset, model in zip(offsets, MODEL_ORDER):
        sub = sig[sig["model"].astype(str).eq(model)].set_index("benchmark").reindex(BENCH_ORDER)
        vals = sub["problem_accuracy_drop_pp"].astype(float).values
        low = sub["problem_accuracy_boot95_low_pp"].astype(float).values
        high = sub["problem_accuracy_boot95_high_pp"].astype(float).values
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="#111827",
            linewidth=0.55,
            label=MODEL_LABELS[model],
            zorder=3,
        )
        for bar, cls in zip(bars, sub["class"].tolist()):
            bar.set_hatch(hatches[cls])
            bar.set_alpha(alphas[cls])
        ax.errorbar(
            x + offset,
            vals,
            yerr=np.vstack([vals - low, high - vals]),
            fmt="none",
            ecolor="#111827",
            elinewidth=0.72,
            capsize=2.0,
            zorder=4,
        )
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Control - treatment\naccuracy (pp)")
    ax.set_ylim(-8, 62)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.text(-0.08, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    for offset, model in zip(offsets, MODEL_ORDER):
        sub = gen[gen["model"].astype(str).eq(model)].set_index("benchmark").reindex(BENCH_ORDER)
        vals = sub["last_minus_first_pp"].astype(float).values
        slopes = sub["linear_slope_pp_per_generation"].astype(float).values
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="#111827",
            linewidth=0.55,
            label=MODEL_LABELS[model],
            alpha=0.92,
            zorder=3,
        )
        for xpos, val, slope in zip(x + offset, vals, slopes):
            ax.scatter(
                [xpos],
                [val + (2.2 if val >= 0 else -2.2)],
                marker="v" if slope < 0 else "^",
                s=16,
                color="#111827",
                zorder=5,
            )
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Gen10 - Gen1\naccuracy (pp)")
    ax.set_ylim(-48, 12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.text(-0.08, 1.03, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch

    model_handles, model_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        model_handles,
        model_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=3,
        frameon=False,
        fontsize=8.3,
    )
    class_handles = [
        Patch(facecolor="#9CA3AF", edgecolor="#111827", hatch="", label="CI above 0"),
        Patch(facecolor="#9CA3AF", edgecolor="#111827", hatch="///", label="directional"),
        Patch(facecolor="#9CA3AF", edgecolor="#111827", hatch="xx", label="mixed"),
    ]
    axes[0].legend(handles=class_handles, loc="upper right", frameon=False, fontsize=7.8)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.13, hspace=0.40)
    _save(fig, "external_core_evidence_bars")


def make_generation_accuracy_by_generation() -> None:
    df = pd.read_csv(GEN_CELLS_CSV)
    agg = (
        df.groupby(["generation", "model"], as_index=False)
        .agg(problem_count=("problem_count", "sum"), run_count=("run_count", "sum"), correct_runs=("correct_runs", "sum"))
        .sort_values(["generation", "model"])
    )
    agg["run_accuracy_pct"] = agg["correct_runs"] / agg["run_count"] * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    gens = np.arange(1, 11)
    for model in MODEL_ORDER:
        series = agg[agg["model"].eq(model)].set_index("generation").reindex(gens)
        vals = series["run_accuracy_pct"].astype(float).values
        ax.plot(
            gens,
            vals,
            marker="o",
            markersize=4.2,
            linewidth=1.9,
            color=MODEL_COLORS[model],
            markeredgecolor="#111827",
            markeredgewidth=0.45,
            label=MODEL_LABELS[model],
            zorder=3,
        )
        slope, intercept = np.polyfit(gens, vals, deg=1)
        ax.plot(gens, slope * gens + intercept, linestyle="--", linewidth=1.0, color=MODEL_COLORS[model], alpha=0.68, zorder=2)

    counts = agg.groupby("generation")["problem_count"].max().reindex(gens).fillna(0).astype(int)
    for gen, count in zip(gens, counts):
        ax.text(gen, 67.2, f"n={count}", ha="center", va="bottom", fontsize=6.8, color="#374151")
    ax.set_xticks(gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Pooled 3-run accuracy (%)")
    ax.set_ylim(0, 72)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncols=3, frameon=False, fontsize=8.4)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, "external_generation_accuracy_by_generation")


def make_expanded_pool_yield_profile() -> None:
    summary = json.loads(EXPANDED_SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = []
    for bench in BENCH_ORDER:
        expanded = summary["benchmarks"][bench]["expanded_full"]
        source = sum(int(v) for v in expanded["source_counts_before_gate"].values())
        kept = source - int(expanded["quality_excluded_count"])
        usable = int(expanded["pool_count"])
        cap = int(expanded["exact_count"])
        rows.append({"benchmark": bench, "source": source, "quality kept": kept, "usable pool": usable, "250 cap": cap})

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(BENCH_ORDER))
    width = 0.19
    metrics = ["source", "quality kept", "usable pool", "250 cap"]
    colors = ["#D1D5DB", "#C8E8C0", "#AFCFF5", "#FFD8A8"]
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics))
    for offset, metric, color in zip(offsets, metrics, colors):
        vals = [row[metric] for row in rows]
        bars = ax.bar(x + offset, vals, width=width, color=color, edgecolor="#111827", linewidth=0.55, label=metric)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val}", ha="center", va="bottom", fontsize=7.3, color="#111827")
    ax.set_xticks(x, [BENCH_LABELS[b] for b in BENCH_ORDER])
    ax.set_ylabel("Rows")
    ax.set_ylim(0, 325)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=4, frameon=False, fontsize=8.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "external_trace_yield_profile")


def make_expanded_cap_provenance_profile() -> None:
    summary = json.loads(EXPANDED_SUMMARY_JSON.read_text(encoding="utf-8"))
    provenance_order = ["published_existing", "generated_full", "partial_recovered"]
    provenance_labels = {
        "published_existing": "Published-existing",
        "generated_full": "New full generation",
        "partial_recovered": "Partial recovered",
    }
    colors = {
        "published_existing": "#AFCFF5",
        "generated_full": "#C8E8C0",
        "partial_recovered": "#F5B7B1",
    }
    rows = []
    for bench in BENCH_ORDER:
        path = REPO / summary["benchmarks"][bench]["expanded_full"]["exact_path"]
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = pd.Series([row.get("_provenance_status", "") for row in data]).value_counts().to_dict()
        rows.append({"benchmark": bench, **{key: int(counts.get(key, 0)) for key in provenance_order}})

    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    y = np.arange(len(BENCH_ORDER))
    left = np.zeros(len(BENCH_ORDER))
    for key in provenance_order:
        vals = np.array([row[key] for row in rows], dtype=float)
        bars = ax.barh(
            y,
            vals,
            left=left,
            color=colors[key],
            edgecolor="white",
            linewidth=0.8,
            label=provenance_labels[key],
        )
        for bar, val, start in zip(bars, vals, left):
            if val >= 12:
                ax.text(start + val / 2, bar.get_y() + bar.get_height() / 2, f"{int(val)}", ha="center", va="center", fontsize=8.2, color="#111827")
        left += vals
    ax.set_yticks(y, [BENCH_LABELS[b].replace("\n", " ") for b in BENCH_ORDER])
    ax.set_xlabel("Rows")
    ax.set_xlim(0, 250)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=3, frameon=False, fontsize=8.1)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "external_operational_intervention_profile")


def write_provenance_summary() -> None:
    summary = json.loads(EXPANDED_SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = []
    totals = {"published_existing": 0, "generated_full": 0, "partial_recovered": 0}
    for bench in BENCH_ORDER:
        path = REPO / summary["benchmarks"][bench]["expanded_full"]["exact_path"]
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = pd.Series([row.get("_provenance_status", "") for row in data]).value_counts().to_dict()
        for key in totals:
            totals[key] += int(counts.get(key, 0))
        rows.append(
            {
                "benchmark": bench,
                "treatment_rows": len(data),
                "published_existing": int(counts.get("published_existing", 0)),
                "generated_full": int(counts.get("generated_full", 0)),
                "partial_recovered": int(counts.get("partial_recovered", 0)),
            }
        )
    rows.append({"benchmark": "total", "treatment_rows": sum(r["treatment_rows"] for r in rows), **totals})
    out = REPO / "data/analysis/external_significance/external_expanded_treatment_provenance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def write_quality_gate_yield_summary() -> None:
    summary = json.loads(EXPANDED_SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = []
    totals = {
        "source_before_gate": 0,
        "quality_excluded": 0,
        "quality_kept": 0,
        "dedup_excluded": 0,
        "usable_pool": 0,
        "treatment_cap": 0,
    }
    for bench in BENCH_ORDER:
        expanded = summary["benchmarks"][bench]["expanded_full"]
        source_before_gate = sum(int(v) for v in expanded["source_counts_before_gate"].values())
        quality_excluded = int(expanded["quality_excluded_count"])
        quality_kept = source_before_gate - quality_excluded
        dedup_excluded = int(expanded["dedup_excluded_count"])
        usable_pool = int(expanded["pool_count"])
        treatment_cap = int(expanded["exact_count"])
        row = {
            "benchmark": bench,
            "source_before_gate": source_before_gate,
            "quality_excluded": quality_excluded,
            "quality_kept": quality_kept,
            "quality_gate_yield": quality_kept / source_before_gate,
            "dedup_excluded": dedup_excluded,
            "usable_pool": usable_pool,
            "usable_pool_yield": usable_pool / source_before_gate,
            "treatment_cap": treatment_cap,
            "cap_fraction_of_source": treatment_cap / source_before_gate,
        }
        rows.append(row)
        for key in totals:
            totals[key] += int(row[key])
    totals_row = {
        "benchmark": "total",
        **totals,
        "quality_gate_yield": totals["quality_kept"] / totals["source_before_gate"],
        "usable_pool_yield": totals["usable_pool"] / totals["source_before_gate"],
        "cap_fraction_of_source": totals["treatment_cap"] / totals["source_before_gate"],
    }
    rows.append(totals_row)
    out = REPO / "data/analysis/external_significance/external_expanded_quality_gate_yield.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def write_treatment_model_summary() -> None:
    df = pd.read_csv(SIG_RUNS_CSV)
    df = df[df["arm"].eq("new_treatment")]
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"].eq(model)].copy()
        grouped = sub.groupby(["benchmark", "problem_key"])["correct"].apply(list)
        acc_values = [sum(values) / len(values) for values in grouped]
        pass_values = [1.0 if any(values) else 0.0 for values in grouped]
        acc_low, acc_high = _bootstrap_ci(acc_values, seed=20260505 + len(rows))
        pass_low, pass_high = _bootstrap_ci(pass_values, seed=20260515 + len(rows))
        rows.append(
            {
                "model": model,
                "problems": len(grouped),
                "runs": len(sub),
                "correct_runs": int(sub["correct"].sum()),
                "run_accuracy": float(sub["correct"].mean()),
                "run_accuracy_boot95_low": acc_low,
                "run_accuracy_boot95_high": acc_high,
                "pass_at_3": sum(pass_values) / len(pass_values) if pass_values else 0.0,
                "pass_at_3_boot95_low": pass_low,
                "pass_at_3_boot95_high": pass_high,
                "extraction_fail_runs": int((sub["answer_extraction_status"] != "boxed_found").sum()),
            }
        )
    out = REPO / "data/analysis/external_significance/external_treatment_model_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    make_significance_hardness_bars()
    make_generation_delta_bars()
    make_external_core_evidence_bars()
    make_generation_accuracy_by_generation()
    make_expanded_pool_yield_profile()
    make_expanded_cap_provenance_profile()
    write_provenance_summary()
    write_quality_gate_yield_summary()
    write_treatment_model_summary()
    print(
        json.dumps(
            {
                "figures": [
                    str(FIG_DIR / "external_significance_hardness_bars.pdf"),
                    str(FIG_DIR / "external_generation_delta_bars.pdf"),
                    str(FIG_DIR / "external_core_evidence_bars.pdf"),
                    str(FIG_DIR / "external_generation_accuracy_by_generation.pdf"),
                    str(FIG_DIR / "external_trace_yield_profile.pdf"),
                    str(FIG_DIR / "external_operational_intervention_profile.pdf"),
                ],
                "provenance_summary": str(REPO / "data/analysis/external_significance/external_expanded_treatment_provenance.csv"),
                "quality_gate_yield": str(REPO / "data/analysis/external_significance/external_expanded_quality_gate_yield.csv"),
                "model_summary": str(REPO / "data/analysis/external_significance/external_treatment_model_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
