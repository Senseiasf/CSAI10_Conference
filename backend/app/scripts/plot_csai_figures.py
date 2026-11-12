#!/usr/bin/env python3
"""
Автоматическая генерация графиков (300 DPI PNG + PDF) для статьи CSAI 2026
по данным из experiment_outputs/benchmark_results.json.

Запуск из корня backend/:
    python -m app.scripts.plot_csai_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
repo_root = BASE_DIR.parents[1] if (BASE_DIR.parents[1] / "docker-compose.yml").exists() else BASE_DIR.parent
OUT_DIR = repo_root / "experiment_outputs"

# Академический стиль оформления для одноколоночного шаблона ACM / CSAI
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

PALETTE = ["#64748B", "#0EA5E9", "#F59E0B", "#10B981"]


def plot_fig1_ablation(data: dict, output: Path) -> None:
    summary = data["summary"]
    configs = [r["Configuration"].replace(" (", "\n(") for r in summary]
    metrics = [
        ("Destination_Hit_at_2_pct", "Destination Hit@2 (%)"),
        ("Logistics_Feasibility_pct", "Logistics Feasibility@3 (%)"),
        ("Constraint_Satisfaction_pct", "Constraint Satisfaction@3 (%)"),
        ("Success_at_3_pct", "Success@3 (%)"),
    ]

    x = np.arange(len(configs))
    width = 0.19

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for idx, (m_key, m_label) in enumerate(metrics):
        vals = [float(r[m_key]) for r in summary]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=m_label, color=PALETTE[idx], edgecolor="white")
        for b in bars:
            h = b.get_height()
            if not np.isfinite(h):
                continue
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_ylabel("Score (%)")
    ax.set_title("Travel Planning Benchmark Across Four Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(output / "fig1_pipeline_ablation.png", dpi=300)
    fig.savefig(output / "fig1_pipeline_ablation.pdf")
    plt.close(fig)


def plot_fig2_archetypes(data: dict, output: Path) -> None:
    arch_data = data["archetype_comparison"]
    archetypes = ["pet", "family", "luxury", "budget", "solo"]
    arch_labels = ["Pet-Friendly", "Family (w/ Kids)", "Luxury", "Budget", "Solo Explorer"]

    def values(key):
        return [np.nan if arch_data[key][a]["pairwise_acc"] is None else arch_data[key][a]["pairwise_acc"] for a in archetypes]

    c1_acc = values("C1_Baseline_SQL")
    c3_acc = values("C3_Graph_Heuristic")
    c4_acc = values("C4_Full_CatBoost")

    x = np.arange(len(archetypes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    b1 = ax.bar(x - width, c1_acc, width, label="C1: Price Sort Baseline", color="#94A3B8")
    b2 = ax.bar(x, c3_acc, width, label="C3: Analytical Heuristic (_surrogate_score)", color="#0EA5E9")
    b3 = ax.bar(x + width, c4_acc, width, label="C4: CatBoostRanker (Synthetic LTR)", color="#10B981")

    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            if not np.isfinite(h):
                continue
            ax.annotate(
                f"{h:.1f}%",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_ylabel("Pairwise Preference Accuracy (%)")
    source = data["metadata"]["pair_labels"].replace("_", " ")
    ax.set_title(f"Pairwise Ranking Accuracy by Traveler Archetype ({source})")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output / "fig2_archetype_pairwise.png", dpi=300)
    fig.savefig(output / "fig2_archetype_pairwise.pdf")
    plt.close(fig)


def plot_fig3_feature_importance(data: dict, output: Path) -> None:
    fi = data.get("feature_importances")
    if not fi:
        return
    sorted_items = sorted(fi.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.barh(labels, values, color="#2563EB", height=0.62)
    for b in bars:
        w = b.get_width()
        ax.annotate(
            f"{w:.1f}%",
            xy=(w, b.get_y() + b.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlabel("Normalized Importance (%)")
    ax.set_title("CatBoostRanker Feature Importance (PredictionValuesChange)")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output / "fig3_feature_importance.png", dpi=300)
    fig.savefig(output / "fig3_feature_importance.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    output = parser.parse_args().output_dir
    data = json.loads((output / "benchmark_results.json").read_text(encoding="utf-8"))
    plot_fig1_ablation(data, output)
    plot_fig2_archetypes(data, output)
    plot_fig3_feature_importance(data, output)
    print(f"Figures written to {output}")


if __name__ == "__main__":
    main()
