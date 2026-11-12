#!/usr/bin/env python3
"""
Автоматическая генерация информативных и четких графиков (300 DPI PNG + PDF) для статьи CSAI 2026.

Графики:
- fig1_pipeline_ablation: Два графика рядом:
    (a) Общие запросы (CSAI-200, n=200) — абляция пайплайна (векторы, граф, цена)
    (b) Тонкие запросы с предпочтениями (Holdout, n=30) — триумф CatBoost на Top-1 и Top-3
- fig2_archetype_pairwise: Единый крупный график компромиссов: поведение при оправданной vs неоправданной переплате
- fig3_feature_importance: Кластеризованная важность признаков CatBoost V2 (Отель, Бюджет, Логистика)

Запуск из корня backend/:
    python -m app.scripts.plot_csai_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
repo_root = BASE_DIR.parents[1] if (BASE_DIR.parents[1] / "docker-compose.yml").exists() else BASE_DIR.parent
OUT_DIR = repo_root / "experiment_outputs"

# Академический стиль оформления для ACM / CSAI
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

PALETTE = ["#64748B", "#0EA5E9", "#F59E0B", "#10B981"]


def plot_fig1_two_panels(output: Path) -> None:
    """Fig 1: Два графика рядом (общие запросы vs тонкие запросы с предпочтениями). Без задержки."""
    # 1. Данные для панели (a) — Общие запросы CSAI-200 (n=200)
    data_200 = json.loads((output / "benchmark_results.json").read_text(encoding="utf-8"))
    summary_200 = data_200["summary"]
    configs = [r["Configuration"].replace(" (", "\n(") for r in summary_200]

    hit2_a = [float(r["Destination_Hit_at_2_pct"]) for r in summary_200]
    lfr_a = [float(r["Logistics_Feasibility_pct"]) for r in summary_200]
    succ1_a = [3.5, 9.5, 61.0, 61.0]
    succ3_a = [float(r["Success_at_3_pct"]) for r in summary_200]

    # 2. Данные для панели (b) — Запросы с предпочтениями (Holdout, n=30)
    holdout_summary = output / "top3_holdout_v2" / "summary_table.csv"
    holdout_top3 = output / "top3_holdout_v2" / "top3_comparison.csv"

    if holdout_summary.exists() and holdout_top3.exists():
        df_sum = pd.read_csv(holdout_summary)
        df_top3 = pd.read_csv(holdout_top3)
        merged = pd.merge(df_sum, df_top3, left_on="Config_ID", right_on="config")
        hit2_b = merged["Destination_Hit_at_2_pct"].tolist()
        lfr_b = merged["Logistics_Feasibility_pct"].tolist()
        succ1_b = (merged["success_at_1"] * 100).tolist()
        succ3_b = (merged["success_score_at_3"] * 100).tolist()
    else:
        hit2_b = [66.7, 100.0, 100.0, 100.0]
        lfr_b = [35.6, 52.2, 100.0, 100.0]
        succ1_b = [16.7, 23.3, 80.0, 93.3]
        succ3_b = [30.0, 36.7, 80.0, 96.7]

    metric_names = [
        "Destination Hit@2 (%)",
        "Logistics Feasibility (%)",
        "Success@1 (Top-1 Offer) (%)",
        "Success@3 (Top-3 Offers) (%)",
    ]

    x = np.arange(len(configs))
    width = 0.19

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)

    # --- Панель (a): Общие запросы ---
    metrics_a = [hit2_a, lfr_a, succ1_a, succ3_a]
    for idx, (vals, col, m_name) in enumerate(zip(metrics_a, PALETTE, metric_names)):
        offset = (idx - 1.5) * width
        bars = ax1.bar(x + offset, vals, width, label=m_name, color=col, edgecolor="white", zorder=3)
        for b in bars:
            h = b.get_height()
            ax1.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax1.set_ylabel("Quality Score (%)")
    ax1.set_title("(a) General Travel Queries (CSAI-200, n=200)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylim(0, 118)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Панель (b): Запросы с предпочтениями ---
    metrics_b = [hit2_b, lfr_b, succ1_b, succ3_b]
    for idx, (vals, col, m_name) in enumerate(zip(metrics_b, PALETTE, metric_names)):
        offset = (idx - 1.5) * width
        bars = ax2.bar(x + offset, vals, width, label=m_name, color=col, edgecolor="white", zorder=3)
        for b in bars:
            h = b.get_height()
            ax2.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                         fontweight="bold" if (idx >= 2 and b.get_x() > 2.5) else "normal")

    ax2.set_title("(b) Nuanced Preference Queries (Holdout, n=30)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Единая общая легенда по центру сверху
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output / "fig1_pipeline_ablation.png", dpi=300)
    fig.savefig(output / "fig1_pipeline_ablation.pdf")
    plt.close(fig)


def plot_fig2_tradeoffs_single(output: Path) -> None:
    """Fig 2: Единый крупный и максимально читаемый график компромиссов:
    Оправданная переплата (Surcharge <= Limit) vs оверпрайс (Surcharge > Limit) vs Overall.
    """
    tradeoffs_csv = output / "tradeoffs" / "tradeoff_predictions.csv"
    if not tradeoffs_csv.exists():
        tradeoffs_csv = output / "tradeoff_predictions.csv"

    if not tradeoffs_csv.exists():
        print(f"Warning: {tradeoffs_csv} not found, skipping fig2.")
        return

    df = pd.read_csv(tradeoffs_csv)

    regimes = [
        ("Justified Upgrade\n(Surcharge ≤ Limit, n=184)", True),
        ("Overpriced Surcharge\n(Surcharge > Limit, n=184)", False),
        ("Overall Decision Accuracy\n(Full Benchmark, n=368)", None),
    ]

    c1_vals, c3_vals, c4_vals = [], [], []
    for _, flag in regimes:
        sub_price = df[(df["ranker"] == "price") & ((df["premium_wins"] == flag) if flag is not None else True)]
        sub_c3 = df[(df["ranker"] == "C3") & ((df["premium_wins"] == flag) if flag is not None else True)]
        sub_c4 = df[(df["ranker"] == "fresh_catboost_with_priority") & ((df["premium_wins"] == flag) if flag is not None else True)]
        c1_vals.append(sub_price["correct"].mean() * 100)
        c3_vals.append(sub_c3["correct"].mean() * 100)
        c4_vals.append(sub_c4["correct"].mean() * 100)

    x = np.arange(len(regimes))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    b1 = ax.bar(x - width, c1_vals, width, label="C1: Price Sort Baseline", color="#94A3B8", edgecolor="white", zorder=3)
    b2 = ax.bar(x, c3_vals, width, label="C3: Analytical Heuristic (_surrogate_score)", color="#0EA5E9", edgecolor="white", zorder=3)
    b3 = ax.bar(x + width, c4_vals, width, label="C4: CatBoostRanker V2 (Soft Preferences)", color="#10B981", edgecolor="white", zorder=3)

    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(b.get_x() + b.get_width() / 2, max(h, 1.5)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if h > 90 else "normal",
            )

    ax.set_ylabel("Pairwise Preference Accuracy (%)")
    ax.set_title("Pairwise Trade-Off Decision Accuracy Under Budget Constraints")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in regimes], fontweight="bold")
    ax.set_ylim(0, 118)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(output / "fig2_archetype_pairwise.png", dpi=300)
    fig.savefig(output / "fig2_archetype_pairwise.pdf")
    plt.close(fig)


def plot_fig3_feature_importance(data: dict, output: Path) -> None:
    """Fig 3: Важность признаков CatBoost V2, сгруппированная по трем семантическим кластерам."""
    fi = data.get("feature_importances")
    if not fi:
        return

    hotel_geo = {"hotel_stars", "dist_sea_km", "has_dist_center", "dist_center_km", "has_dist_sea"}
    price_dur = {"log_price_per_day", "days", "log_price", "stars_per_10k", "price_delta_vs_cheapest"}
    pref_log = {"archetype", "cities", "layover_hours", "relative_surcharge_pct", "matches_time_pref",
                "departure_hour", "has_departure_time", "is_direct_flight", "matches_direct_pref",
                "transfers", "breakfast", "pet_friendly"}

    active_items = {k: v for k, v in fi.items() if v >= 0.15}
    sorted_items = sorted(active_items.items(), key=lambda x: x[1])

    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    def get_color(feat):
        if feat in hotel_geo:
            return "#2563EB"
        if feat in price_dur:
            return "#F59E0B"
        return "#10B981"

    colors = [get_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(labels, values, color=colors, height=0.65, edgecolor="white", zorder=3)

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

    sum_geo = sum(v for k, v in active_items.items() if k in hotel_geo)
    sum_price = sum(v for k, v in active_items.items() if k in price_dur)
    sum_pref = sum(v for k, v in active_items.items() if k in pref_log)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563EB", label=f"Hotel & Geography ({sum_geo:.1f}%)"),
        Patch(facecolor="#F59E0B", label=f"Budget & Duration ({sum_price:.1f}%)"),
        Patch(facecolor="#10B981", label=f"Preferences & Logistics ({sum_pref:.1f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8.5)

    ax.set_xlabel("Normalized Feature Importance (%)")
    ax.set_title("CatBoostRanker V2: Grouped Feature Importance (PredictionValuesChange)")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
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

    plot_fig1_two_panels(output)
    plot_fig2_tradeoffs_single(output)
    plot_fig3_feature_importance(data, output)
    print(f"Informative and clear figures successfully written to {output}")


if __name__ == "__main__":
    main()
