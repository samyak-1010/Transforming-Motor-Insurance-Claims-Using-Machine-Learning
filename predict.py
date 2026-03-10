"""
Motor Insurance Claims ML Pipeline – Prediction / Inference CLI
===============================================================
Interactive command-line interface for using trained models.

Usage:
    python predict.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from src.utils import setup_logging
from src.part_matcher import PartMatcher
from src.recommender import ClaimRecommender
from src.fraud_detector import FraudDetector


def load_models() -> dict:
    """Load all available trained models from the models/ directory."""
    models: dict = {}

    matcher_path = cfg.MODELS_DIR / "part_matcher.pkl"
    if matcher_path.exists():
        models["matcher"] = PartMatcher.load(matcher_path)
        print("  [OK] Part matcher loaded")
    else:
        print("  [--] Part matcher not found")

    recommender_path = cfg.MODELS_DIR / "recommender.pkl"
    if recommender_path.exists():
        models["recommender"] = ClaimRecommender.load(recommender_path)
        print("  [OK] Recommender loaded")
    else:
        print("  [--] Recommender not found")

    detector_path = cfg.MODELS_DIR / "fraud_detector.pkl"
    if detector_path.exists():
        models["detector"] = FraudDetector.load(detector_path)
        print("  [OK] Fraud detector loaded")
    else:
        print("  [--] Fraud detector not found")

    return models


# ─── Actions ─────────────────────────────────────────────────────────────────

def find_similar_part(models: dict) -> None:
    """Find matching garage parts for a user-supplied part name."""
    matcher = models.get("matcher")
    if not matcher:
        print("  Part matcher not available. Run 'python train.py' first.")
        return

    part_name = input("  Enter part name: ").strip()
    if not part_name:
        print("  No input provided.")
        return

    result = matcher.predict(part_name)
    print(f"\n  Input        : {result['input']}")
    print(f"  Fuzzy match  : {result['fuzzy_match'] or 'No match'} (score: {result['fuzzy_score']})")
    print(f"  Cosine match : {result['cosine_match'] or 'No match'} (score: {result['cosine_score']})")


def get_recommendations(models: dict) -> None:
    """Retrieve similar-claim recommendations for a given claim ID."""
    recommender = models.get("recommender")
    if not recommender:
        print("  Recommender not available. Run 'python train.py' first.")
        return

    print(f"  {len(recommender.claim_ids)} claims in database")
    print(f"  Sample IDs: {recommender.claim_ids[:5]}")

    claim_id = input("  Enter Claim ID: ").strip()
    try:
        recs = recommender.recommend(claim_id)
        print(f"\n  Top {len(recs)} similar claims:")
        for i, rec in enumerate(recs, 1):
            print(f"    {i}. {rec['claim_id']}  (similarity: {rec['similarity_score']:.4f})")
    except ValueError as exc:
        print(f"  Error: {exc}")


def check_fraud(models: dict) -> None:
    """Check whether a claim amount is suspicious."""
    detector = models.get("detector")
    if not detector:
        print("  Fraud detector not available. Run 'python train.py' first.")
        return

    try:
        total_amount = float(input("  Enter total claim amount: "))
        part_count = int(input("  Enter number of parts (default 1): ") or "1")
    except ValueError:
        print("  Invalid input.")
        return

    avg = total_amount / max(part_count, 1)
    features = pd.DataFrame(
        [
            {
                "total_amount": total_amount,
                "part_count": part_count,
                "avg_part_amount": avg,
                "max_part_amount": total_amount,
                "min_part_amount": avg,
                "std_part_amount": 0.0,
                "amount_range": total_amount - avg if part_count > 1 else 0.0,
                "coeff_variation": 0.0,
                "log_total_amount": np.log1p(total_amount),
            }
        ]
    )

    result = detector.predict(features)
    status = "SUSPICIOUS" if result["is_fraud"].iloc[0] else "NORMAL"
    score = result["anomaly_score"].iloc[0]

    print(f"\n  Status        : {status}")
    print(f"  Anomaly score : {score:.4f}")
    print(f"  (Lower score = more anomalous)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging(cfg.LOG_LEVEL)

    print("=" * 50)
    print("  Motor Insurance Claims – Prediction Interface")
    print("=" * 50)
    print()

    models = load_models()
    if not models:
        print("\n  No models found. Run 'python train.py' first.")
        sys.exit(1)

    while True:
        print("\n  ─── Menu ───")
        print("  1. Find similar part")
        print("  2. Get claim recommendations")
        print("  3. Check for fraudulent claim")
        print("  4. Exit")

        choice = input("\n  Choice (1-4): ").strip()

        if choice == "1":
            find_similar_part(models)
        elif choice == "2":
            get_recommendations(models)
        elif choice == "3":
            check_fraud(models)
        elif choice == "4":
            print("  Goodbye.")
            break
        else:
            print("  Invalid choice – enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
