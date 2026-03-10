"""
SVD-based Collaborative Filtering Recommendation Engine.

Finds similar claims based on their part-amount profiles,
using TruncatedSVD for dimensionality reduction.

Fixes from original code:
  - Aggregates duplicates before pivoting (prevents ValueError).
  - Excludes the query claim from its own recommendation list.
  - Scales data before SVD for better latent-space geometry.
  - Guards against n_components > matrix dimensions.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger("insurance_ml.recommender")


class ClaimRecommender:
    """Recommend similar insurance claims via SVD collaborative filtering."""

    def __init__(self) -> None:
        self.svd: Optional[TruncatedSVD] = None
        self.scaler: Optional[StandardScaler] = None
        self.interaction_matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.claim_ids: List[str] = []

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, merged_df: pd.DataFrame) -> "ClaimRecommender":
        """Build the claim-part interaction matrix and fit SVD."""
        part_col = self._resolve_col(merged_df, "PART", exclude="CODE")
        amount_col = self._resolve_col(merged_df, "AMOUNT")

        # Aggregate duplicate (claim, part) pairs BEFORE pivoting
        aggregated = (
            merged_df.groupby([cfg.CLAIM_ID_COL, part_col])
            .agg({amount_col: "sum"})
            .reset_index()
        )

        self.interaction_matrix = aggregated.pivot(
            index=cfg.CLAIM_ID_COL,
            columns=part_col,
            values=amount_col,
        ).fillna(0)

        self.claim_ids = self.interaction_matrix.index.tolist()

        if len(self.claim_ids) < 2:
            raise ValueError("Need at least 2 claims for recommendations")

        # Scale
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.interaction_matrix)

        # Safe n_components
        n_components = min(
            cfg.SVD_N_COMPONENTS,
            min(scaled.shape) - 1,
            len(self.claim_ids) - 1,
        )
        n_components = max(n_components, 1)

        self.svd = TruncatedSVD(n_components=n_components, random_state=cfg.RANDOM_STATE)
        latent = self.svd.fit_transform(scaled)

        explained = self.svd.explained_variance_ratio_.sum() * 100
        logger.info(
            f"SVD fitted: {n_components} components, {explained:.1f}% variance explained"
        )

        self.similarity_matrix = cosine_similarity(latent)

        logger.info(
            f"Recommender ready – {len(self.claim_ids)} claims, "
            f"{self.interaction_matrix.shape[1]} parts"
        )
        return self

    # ── Recommend ────────────────────────────────────────────────────────

    def recommend(self, claim_id: str, top_n: Optional[int] = None) -> List[Dict]:
        """Return top-N similar claims, **excluding the query claim**."""
        top_n = top_n or cfg.SVD_TOP_N

        if claim_id not in self.claim_ids:
            raise ValueError(f"Claim ID '{claim_id}' not in training data")

        idx = self.claim_ids.index(claim_id)
        scores = self.similarity_matrix[idx]

        ranked = np.argsort(scores)[::-1]
        recs: List[Dict] = []
        for j in ranked:
            if j == idx:  # skip self
                continue
            recs.append(
                {
                    "claim_id": self.claim_ids[j],
                    "similarity_score": round(float(scores[j]), 4),
                }
            )
            if len(recs) >= top_n:
                break
        return recs

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(self) -> Dict:
        """Internal quality metrics for the recommendation model."""
        if self.similarity_matrix is None:
            return {}

        threshold = 0.5
        coverage = 0
        avg_top_sim: List[float] = []

        for i in range(len(self.claim_ids)):
            row = self.similarity_matrix[i].copy()
            row[i] = 0.0  # mask self
            if np.max(row) >= threshold:
                coverage += 1
            top_scores = np.sort(row)[::-1][: cfg.SVD_TOP_N]
            avg_top_sim.append(float(np.mean(top_scores)))

        explained = (
            self.svd.explained_variance_ratio_.sum() * 100 if self.svd else 0
        )

        metrics = {
            "num_claims": len(self.claim_ids),
            "num_parts": (
                self.interaction_matrix.shape[1]
                if self.interaction_matrix is not None
                else 0
            ),
            "svd_explained_variance_pct": round(explained, 2),
            "coverage_pct_at_0.5": round(coverage / len(self.claim_ids) * 100, 2),
            "avg_top_n_similarity": round(float(np.mean(avg_top_sim)), 4),
        }
        logger.info(f"Recommender evaluation: {metrics}")
        return metrics

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "svd": self.svd,
                    "scaler": self.scaler,
                    "interaction_matrix": self.interaction_matrix,
                    "similarity_matrix": self.similarity_matrix,
                    "claim_ids": self.claim_ids,
                },
                f,
            )
        logger.info(f"Recommender saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "ClaimRecommender":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.svd = data["svd"]
        obj.scaler = data["scaler"]
        obj.interaction_matrix = data["interaction_matrix"]
        obj.similarity_matrix = data["similarity_matrix"]
        obj.claim_ids = data["claim_ids"]
        logger.info(f"Recommender loaded ← {path}")
        return obj

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_col(df: pd.DataFrame, keyword: str, exclude: str = "") -> str:
        """Find the first column whose name contains *keyword* (case-insensitive)."""
        for c in [cfg.SURVEYOR_PART_COL, cfg.AMOUNT_COL]:
            if c in df.columns and keyword.upper() in c.upper():
                if not exclude or exclude.upper() not in c.upper():
                    return c
        for c in df.columns:
            if keyword.upper() in c.upper():
                if not exclude or exclude.upper() not in c.upper():
                    return c
        raise ValueError(f"No column matching '{keyword}' found in DataFrame")
