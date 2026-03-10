"""
Visualization module – generates publication-quality plots and saves them
to disk.  Uses the non-interactive Agg backend so the pipeline can run
headless (servers, CI, etc.).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402 – must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg

logger = logging.getLogger("insurance_ml.visualization")

# Global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150


# ─── Part Matching ───────────────────────────────────────────────────────────

def plot_matching_scores(combined: pd.DataFrame, save_path: Path) -> None:
    """Scatter plot: fuzzy score vs cosine score with agreement color."""
    fig, ax = plt.subplots(figsize=(10, 8))
    data = combined.dropna(subset=["fuzzy_score", "cosine_score"])

    scatter = ax.scatter(
        data["fuzzy_score"],
        data["cosine_score"],
        c=data["match_agreement"].astype(int),
        cmap="RdYlGn",
        alpha=0.6,
        s=30,
        edgecolors="gray",
        linewidths=0.3,
    )

    ax.axhline(
        y=cfg.COSINE_THRESHOLD, color="red", ls="--", alpha=0.5,
        label=f"Cosine threshold = {cfg.COSINE_THRESHOLD}",
    )
    ax.axvline(
        x=cfg.FUZZY_THRESHOLD, color="blue", ls="--", alpha=0.5,
        label=f"Fuzzy threshold = {cfg.FUZZY_THRESHOLD}",
    )

    ax.set_xlabel("Fuzzy Match Score")
    ax.set_ylabel("Cosine Similarity Score")
    ax.set_title("Part Matching: Fuzzy vs Cosine Scores", fontweight="bold")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Methods Agree")
    plt.tight_layout()
    _save(fig, save_path)


# ─── Common Parts ────────────────────────────────────────────────────────────

def plot_common_parts(
    part_counts: pd.Series, save_path: Path, top_n: int = 15,
) -> None:
    """Horizontal bar chart of the most commonly damaged parts."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = part_counts.head(top_n)
    df_plot = pd.DataFrame({"part": data.index, "count": data.values})
    sns.barplot(data=df_plot, x="count", y="part", hue="part",
                palette="viridis", ax=ax, legend=False)
    ax.set_title(f"Top {top_n} Most Commonly Damaged Parts", fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Part Name")
    plt.tight_layout()
    _save(fig, save_path)


# ─── Claim Distribution ─────────────────────────────────────────────────────

def plot_claim_distribution(
    primary_parts_df: pd.DataFrame, save_path: Path, top_n: int = 10,
) -> None:
    """Donut chart of claim distribution by primary parts."""
    fig, ax = plt.subplots(figsize=(10, 8))
    data = primary_parts_df.head(top_n)
    wedges, _, autotexts = ax.pie(
        data.iloc[:, 1],
        labels=data.iloc[:, 0],
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel", n_colors=top_n),
        pctdistance=0.85,
    )
    ax.add_artist(plt.Circle((0, 0), 0.55, fc="white"))
    ax.set_title(
        f"Claim Distribution by Primary Parts (Top {top_n})", fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, save_path)


# ─── Anomaly Distribution ───────────────────────────────────────────────────

def plot_anomaly_distribution(predictions: pd.DataFrame, save_path: Path) -> None:
    """Overlaid histogram of claim amounts for normal vs suspicious claims."""
    fig, ax = plt.subplots(figsize=(12, 6))

    normal = predictions[predictions["is_fraud"] == 0]["total_amount"]
    fraud = predictions[predictions["is_fraud"] == 1]["total_amount"]

    ax.hist(
        normal, bins=40, alpha=0.7, color="#2196F3", edgecolor="white",
        label=f"Normal ({len(normal)})",
    )
    if len(fraud) > 0:
        ax.hist(
            fraud, bins=20, alpha=0.8, color="#F44336", edgecolor="white",
            label=f"Suspicious ({len(fraud)})",
        )

    ax.axvline(
        normal.mean(), color="#1565C0", ls="--", lw=1.5,
        label=f"Normal mean: {normal.mean():,.0f}",
    )
    if len(fraud) > 0:
        ax.axvline(
            fraud.mean(), color="#C62828", ls="--", lw=1.5,
            label=f"Fraud mean: {fraud.mean():,.0f}",
        )

    ax.set_title(
        "Claim Amount Distribution: Normal vs Suspicious", fontweight="bold"
    )
    ax.set_xlabel("Total Claim Amount")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save(fig, save_path)


# ─── Similarity Heatmap ─────────────────────────────────────────────────────

def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    claim_ids: list,
    save_path: Path,
    n: int = 15,
) -> None:
    """Heatmap of the top-n claim similarity scores."""
    n = min(n, len(claim_ids))
    labels = [str(c)[:12] for c in claim_ids[:n]]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sim_matrix[:n, :n],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Similarity Score"},
    )
    ax.set_title("Claim Similarity Heatmap (SVD)", fontweight="bold")
    ax.set_xlabel("Claim ID")
    ax.set_ylabel("Claim ID")
    plt.tight_layout()
    _save(fig, save_path)


# ─── Fraud Feature Box-Plots ────────────────────────────────────────────────

def plot_fraud_feature_importance(
    predictions: pd.DataFrame,
    feature_names: list,
    save_path: Path,
) -> None:
    """Box-plots comparing feature distributions for normal vs fraud."""
    plot_features = [f for f in feature_names if f in predictions.columns][:6]
    if not plot_features:
        return

    n = len(plot_features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, feat in enumerate(plot_features):
        if i < len(axes_flat):
            sns.boxplot(
                data=predictions, x="is_fraud", y=feat,
                hue="is_fraud", ax=axes_flat[i],
                palette={0: "#2196F3", 1: "#F44336"}, legend=False,
            )
            axes_flat[i].set_title(feat.replace("_", " ").title(), fontsize=11)
            axes_flat[i].set_xlabel("Fraud (0=No, 1=Yes)")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions: Normal vs Suspicious Claims",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, save_path)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved → {path}")
