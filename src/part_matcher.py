"""
Part matching using Fuzzy Matching and TF-IDF Cosine Similarity.

Fixes from original code:
  - Saves the fitted TfidfVectorizer (not the sparse matrix).
  - Cosine similarity is computed *across* sources (survey vs garage),
    so a survey part can never erroneously match another survey part.
  - Provides a combined best-match output and agreement metrics.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg

logger = logging.getLogger("insurance_ml.part_matcher")


class PartMatcher:
    """Match part descriptions between surveyor and garage data."""

    def __init__(self) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.survey_parts: np.ndarray = np.array([])
        self.garage_parts: np.ndarray = np.array([])
        self.fuzzy_results: Optional[pd.DataFrame] = None
        self.cosine_results: Optional[pd.DataFrame] = None

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        survey_parts: np.ndarray,
        garage_parts: np.ndarray,
    ) -> "PartMatcher":
        """Fit the TF-IDF vectorizer on all unique part descriptions."""
        self.survey_parts = np.array([p for p in survey_parts if str(p).strip()])
        self.garage_parts = np.array([p for p in garage_parts if str(p).strip()])

        if len(self.survey_parts) == 0 or len(self.garage_parts) == 0:
            raise ValueError("Cannot fit PartMatcher with empty part lists")

        logger.info(
            f"Fitting on {len(self.survey_parts)} survey parts, "
            f"{len(self.garage_parts)} garage parts"
        )

        all_parts = np.concatenate([self.survey_parts, self.garage_parts])
        self.vectorizer = TfidfVectorizer(
            ngram_range=cfg.TFIDF_NGRAM_RANGE,
            max_features=5000,
            sublinear_tf=True,
        )
        self.vectorizer.fit(all_parts)
        return self

    # ── Fuzzy Matching ───────────────────────────────────────────────────

    def match_fuzzy(self) -> pd.DataFrame:
        """Match parts via fuzzy string similarity."""
        logger.info("Running fuzzy matching …")
        garage_list = self.garage_parts.tolist()
        results: List[Dict] = []

        for part in self.survey_parts:
            match_result = process.extractOne(
                part, garage_list, scorer=fuzz.token_sort_ratio,
            )
            if match_result:
                match, score = match_result[0], match_result[1]
                results.append(
                    {
                        "survey_part": part,
                        "garage_part": match if score >= cfg.FUZZY_THRESHOLD else None,
                        "fuzzy_score": score,
                        "matched": score >= cfg.FUZZY_THRESHOLD,
                    }
                )

        self.fuzzy_results = pd.DataFrame(results)
        match_rate = self.fuzzy_results["matched"].mean() * 100
        logger.info(
            f"Fuzzy matching complete – {match_rate:.1f}% match rate "
            f"(threshold={cfg.FUZZY_THRESHOLD})"
        )
        return self.fuzzy_results

    # ── TF-IDF Cosine Matching ───────────────────────────────────────────

    def match_cosine(self) -> pd.DataFrame:
        """Match parts via TF-IDF cosine similarity (cross-source only)."""
        if self.vectorizer is None:
            raise RuntimeError("Call fit() before match_cosine()")

        logger.info("Running TF-IDF cosine similarity matching …")

        survey_vecs = self.vectorizer.transform(self.survey_parts)
        garage_vecs = self.vectorizer.transform(self.garage_parts)

        # Cross-source similarity only – prevents self/same-source matches
        sim_matrix = cosine_similarity(survey_vecs, garage_vecs)

        results: List[Dict] = []
        for i, part in enumerate(self.survey_parts):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            results.append(
                {
                    "survey_part": part,
                    "garage_part": (
                        self.garage_parts[best_idx]
                        if best_score >= cfg.COSINE_THRESHOLD
                        else None
                    ),
                    "cosine_score": round(best_score, 4),
                    "matched": best_score >= cfg.COSINE_THRESHOLD,
                }
            )

        self.cosine_results = pd.DataFrame(results)
        match_rate = self.cosine_results["matched"].mean() * 100
        logger.info(
            f"Cosine matching complete – {match_rate:.1f}% match rate "
            f"(threshold={cfg.COSINE_THRESHOLD})"
        )
        return self.cosine_results

    # ── Combined Results ─────────────────────────────────────────────────

    def get_combined_results(self) -> pd.DataFrame:
        """Merge fuzzy and cosine results and pick the best match."""
        if self.fuzzy_results is None:
            self.match_fuzzy()
        if self.cosine_results is None:
            self.match_cosine()

        combined = pd.merge(
            self.fuzzy_results[["survey_part", "garage_part", "fuzzy_score"]],
            self.cosine_results[["survey_part", "garage_part", "cosine_score"]],
            on="survey_part",
            suffixes=("_fuzzy", "_cosine"),
        )

        def _best(row):
            f_match = row["garage_part_fuzzy"]
            c_match = row["garage_part_cosine"]
            if f_match and c_match:
                if f_match == c_match:
                    return f_match
                # Prefer fuzzy when its score (0-100) beats cosine (0-1 scaled)
                return f_match if row["fuzzy_score"] >= row["cosine_score"] * 100 else c_match
            return f_match or c_match

        combined["best_match"] = combined.apply(_best, axis=1)
        combined["match_agreement"] = (
            combined["garage_part_fuzzy"] == combined["garage_part_cosine"]
        )

        agreement_rate = combined["match_agreement"].mean() * 100
        logger.info(f"Method agreement rate: {agreement_rate:.1f}%")
        return combined

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the **fitted vectorizer** and part lists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "survey_parts": self.survey_parts,
                    "garage_parts": self.garage_parts,
                },
                f,
            )
        logger.info(f"PartMatcher saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "PartMatcher":
        """Load a previously saved PartMatcher."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.survey_parts = data["survey_parts"]
        obj.garage_parts = data["garage_parts"]
        logger.info(f"PartMatcher loaded ← {path}")
        return obj

    # ── Single-Part Prediction ───────────────────────────────────────────

    def predict(self, part_name: str) -> Dict:
        """Find the best garage match for an arbitrary part name."""
        if self.vectorizer is None:
            raise RuntimeError("PartMatcher not fitted or loaded")

        part_name = part_name.strip().lower()

        # Fuzzy
        garage_list = self.garage_parts.tolist()
        fuzzy_result = process.extractOne(
            part_name, garage_list, scorer=fuzz.token_sort_ratio,
        )
        fuzzy_match = (
            fuzzy_result[0]
            if fuzzy_result and fuzzy_result[1] >= cfg.FUZZY_THRESHOLD
            else None
        )
        fuzzy_score = fuzzy_result[1] if fuzzy_result else 0

        # Cosine
        part_vec = self.vectorizer.transform([part_name])
        garage_vecs = self.vectorizer.transform(self.garage_parts)
        scores = cosine_similarity(part_vec, garage_vecs).flatten()
        best_idx = int(np.argmax(scores))
        cosine_match = (
            self.garage_parts[best_idx]
            if scores[best_idx] >= cfg.COSINE_THRESHOLD
            else None
        )

        return {
            "input": part_name,
            "fuzzy_match": fuzzy_match,
            "fuzzy_score": fuzzy_score,
            "cosine_match": cosine_match,
            "cosine_score": round(float(scores[best_idx]), 4),
        }
