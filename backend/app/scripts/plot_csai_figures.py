#!/usr/bin/env python3
"""
Автоматическая генерация информативных графиков (300 DPI PNG + PDF) для статьи CSAI 2026.

Графики:
- fig1_pipeline_ablation: Вклад компонентов пайплайна (Destination Hit, LFR, CSR, Success) + время отклика (Latency)
- fig2_archetype_pairwise: Диагностика компромиссов (Option B): оправданная vs неоправданная переплата + срезы по атрибутам
- fig3_feature_importance: Кластеризованная важность признаков CatBoost V2 по смысловым категориям

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

# Академический стиль оформления для шаблона ACM / CSAI
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

PALETTE = ["#64748B", "#0EA5E9", "#F59E0B", "#10B981"]


def plot_fig1_ablation(data: dict, output: Path) -> None:
    """Fig 1: Сквозная абляция компонентов пайплайна + время ранжирования."""
    summary = data["summary"]
    configs = [r["Configuration"].replace(" (", "\n(") for r in summary]
    metrics = [
        ("Destination_Hit_at_2_pct", "Destination Hit@2 (%)"),
        ("Logistics_Feasibility_pct", "Logistics Feasibility@3 (%)"),
        ("Constraint_Satisfaction_pct", "Constraint Satisfaction@3 (%)"),
        ("Success_at_3_pct", "Success@3 (%)"),
    ]

    x = np.arange(len(configs))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for idx, (m_key, m_label) in enumerate(metrics):
        vals = [float(r[m_key]) for r in summary]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=m_label, color=PALETTE[idx], edgecolor="white", zorder=3)
        for b in bars:
            h = b.get_height()
            if not np.isfinite(h):
                continue
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # Вторая ось Y для времени отклика (Latency ms)
    ax2 = ax.twinx()
    latencies = [float(r.get("Ranking_Latency_ms", 0.0)) for r in summary]
    line = ax2.plot(x, latencies, color="#E11D48", marker="o", linewidth=2.0, markersize=5.5, label="Ranking Latency (ms)", zorder=4)
    for xi, yi in zip(x, latencies):
        ax2.annotate(
            f"{yi:.1f} ms",
            xy=(xi, yi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#BE123C",
        )

    ax.set_ylabel("Quality Score (%)")
    ax2.set_ylabel("Ranking Latency (ms)", color="#BE123C", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#BE123C")
    ax2.set_ylim(0, max(latencies) * 1.85)

    ax.set_title("Travel Planning Pipeline Ablation: Quality vs. Ranking Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 118)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Объединенная легенда
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, frameon=True)

    fig.tight_layout()
    fig.savefig(output / "fig1_pipeline_ablation.png", dpi=300)
    fig.savefig(output / "fig1_pipeline_ablation.pdf")
    plt.close(fig)


def plot_fig2_tradeoffs(output: Path) -> None:
    """Fig 2: Двусоставный график компромиссов (Option B):
    (a) Точность при оправданной vs неоправданной переплате
    (b) Точность в разрезе типов туристических компромиссов
    """
    tradeoffs_csv = output / "tradeoffs" / "tradeoff_predictions.csv"
    if not tradeoffs_csv.exists():
        tradeoffs_csv = output / "tradeoff_predictions.csv"

    if not tradeoffs_csv.exists():
        print(f"Warning: {tradeoffs_csv} not found, skipping fig2.")
        return

    df = pd.read_csv(tradeoffs_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))

    # --- Подграфик (a): Режимы переплаты (Justified vs Unjustified) ---
    regimes = [
        ("Surcharge ≤ Limit\n(Justified Upgrade, n=184)", True),
        ("Surcharge > Limit\n(Overpriced, n=184)", False),
        ("Overall Trade-Off\nAccuracy (n=368)", None),
    ]

    c1_vals_a, c3_vals_a, c4_vals_a = [], [], []
    for _, flag in regimes:
        sub_price = df[(df["ranker"] == "price") & ((df["premium_wins"] == flag) if flag is not None else True)]
        sub_c3 = df[(df["ranker"] == "C3") & ((df["premium_wins"] == flag) if flag is not None else True)]
        sub_c4 = df[(df["ranker"] == "fresh_catboost_with_priority") & ((df["premium_wins"] == flag) if flag is not None else True)]
        c1_vals_a.append(sub_price["correct"].mean() * 100)
        c3_vals_a.append(sub_c3["correct"].mean() * 100)
        c4_vals_a.append(sub_c4["correct"].mean() * 100)

    x_a = np.arange(len(regimes))
    w = 0.25

    b1_a = ax1.bar(x_a - w, c1_vals_a, w, label="C1: Price Sort Baseline", color="#94A3B8")
    b2_a = ax1.bar(x_a, c3_vals_a, w, label="C3: Analytical Heuristic", color="#0EA5E9")
    b3_a = ax1.bar(x_a + w, c4_vals_a, w, label="C4: CatBoostRanker V2", color="#10B981")

    for bars in (b1_a, b2_a, b3_a):
        for b in bars:
            h = b.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, max(h, 2)),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax1.set_ylabel("Pairwise Decision Accuracy (%)")
    ax1.set_title("(a) Decision Accuracy by Cost Regime")
    ax1.set_xticks(x_a)
    ax1.set_xticklabels([r[0] for r in regimes])
    ax1.set_ylim(0, 118)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", frameon=True)

    # --- Подграфик (b): Разрез по типам компромиссов ---
    benefits = [
        ("Direct Flight\n(vs 1-Stop)", "direct"),
        ("Morning Flight\n(vs Red-Eye)", "morning"),
        ("Hotel 4★/5★\n(vs 3★)", "stars"),
        ("Breakfast\nIncluded", "breakfast"),
    ]

    c1_vals_b, c3_vals_b, c4_vals_b = [], [], []
    for _, ben_key in benefits:
        sub_price = df[(df["ranker"] == "price") & (df["benefit"] == ben_key)]
        sub_c3 = df[(df["ranker"] == "C3") & (df["benefit"] == ben_key)]
        sub_c4 = df[(df["ranker"] == "fresh_catboost_with_priority") & (df["benefit"] == ben_key)]
        c1_vals_b.append(sub_price["correct"].mean() * 100)
        c3_vals_b.append(sub_c3["correct"].mean() * 100)
        c4_vals_b.append(sub_c4["correct"].mean() * 100)

    x_b = np.arange(len(benefits))

    b1_b = ax2.bar(x_b - w, c1_vals_b, w, label="C1: Price Sort", color="#94A3B8")
    b2_b = ax2.bar(x_b, c3_vals_b, w, label="C3: Heuristic", color="#0EA5E9")
    b3_b = ax2.bar(x_b + w, c4_vals_b, w, label="C4: CatBoost V2", color="#10B981")

    for bars in (b1_b, b2_b, b3_b):
        for b in bars:
            h = b.get_height()
            ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, max(h, 2)),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax2.set_ylabel("Pairwise Decision Accuracy (%)")
    ax2.set_title("(b) Accuracy by Trade-Off Feature Type")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels([b[0] for b in benefits])
    ax2.set_ylim(0, 118)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output / "fig2_archetype_pairwise.png", dpi=300)
    fig.savefig(output / "fig2_archetype_pairwise.pdf")
    plt.close(fig)


def plot_fig3_feature_importance(data: dict, output: Path) -> None:
    """Fig 3: Важность признаков CatBoost V2, сгруппированная по трем семантическим кластерам."""
    fi = data.get("feature_importances")
    if not fi:
        return

    # Категоризация признаков
    hotel_geo = {"hotel_stars", "dist_sea_km", "has_dist_center", "dist_center_km", "has_dist_sea"}
    price_dur = {"log_price_per_day", "days", "log_price", "stars_per_10k", "price_delta_vs_cheapest"}
    pref_log = {"archetype", "cities", "layover_hours", "relative_surcharge_pct", "matches_time_pref",
                "departure_hour", "has_departure_time", "is_direct_flight", "matches_direct_pref",
                "transfers", "breakfast", "pet_friendly"}

    # Фильтрация нулевых признаков (оставляем только информативные)
    active_items = {k: v for k, v in fi.items() if v >= 0.15}
    sorted_items = sorted(active_items.items(), key=lambda x: x[1])

    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Цветовое кодирование по категориям
    def get_color(feat):
        if feat in hotel_geo:
            return "#2563EB"  # Синий
        if feat in price_dur:
            return "#F59E0B"  # Янтарный
        return "#10B981"      # Изумрудный

    colors = [get_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.barh(labels, values, color=colors, height=0.65, edgecolor="white")

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

    # Суммарные доли для легенды
    sum_geo = sum(v for k, v in active_items.items() if k in hotel_geo)
    sum_price = sum(v for k, v in active_items.items() if k in price_dur)
    sum_pref = sum(v for k, v in active_items.items() if k in pref_log)

    # Кастомная легенда по кластерам
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
    plot_fig2_tradeoffs(output)
    plot_fig3_feature_importance(data, output)
    print(f"Informative figures successfully written to {output}")


if __name__ == "__main__":
    main()
