"""
Motor Insurance Claims ML Pipeline – Training Script
=====================================================
Orchestrates the full training pipeline:

  1. Load & clean data
  2. Merge datasets (sampling happens AFTER merge)
  3. Part matching (Fuzzy + TF-IDF Cosine)
  4. Association rule mining (Apriori)
  5. Recommendation engine (SVD collaborative filtering)
  6. Fraud detection (multi-feature Isolation Forest with train/test split)
  7. Visualisation & reporting

Usage:
    python train.py
"""

import json
import logging
import sys
import time

import pandas as pd

import config as cfg
from src.utils import setup_logging, ensure_directories, save_dataframe
from src.data_loader import load_and_clean_data, merge_datasets, sample_data, split_data
from src.part_matcher import PartMatcher
from src.association_mining import AssociationMiner
from src.recommender import ClaimRecommender
from src.fraud_detector import FraudDetector, SupervisedFraudDetector
from src import visualization as viz


def main():
    start = time.time()

    # ── Setup ────────────────────────────────────────────────────────────
    ensure_directories(cfg.MODELS_DIR, cfg.OUTPUTS_DIR, cfg.PLOTS_DIR, cfg.LOGS_DIR)
    logger = setup_logging(cfg.LOG_LEVEL, cfg.LOGS_DIR / "training.log")

    logger.info("=" * 60)
    logger.info("  Motor Insurance Claims ML Pipeline – Training")
    logger.info("=" * 60)

    all_metrics: dict = {}

    # ── 1. Load & Clean ─────────────────────────────────────────────────
    logger.info("\n[1/6] Loading and cleaning data ...")
    try:
        surveyor_df, garage_df, primary_parts_df = load_and_clean_data()
    except FileNotFoundError as exc:
        logger.error(f"Data file not found: {exc}")
        logger.error(
            "Place surveyor_data.csv, garage_data.csv (and optionally "
            "Primary_Parts_Code.csv) in the data/ directory."
        )
        sys.exit(1)

    # ── 2. Merge & Prepare ──────────────────────────────────────────────
    logger.info("\n[2/6] Merging datasets ...")
    try:
        merged_df = merge_datasets(surveyor_df, garage_df)
    except ValueError as exc:
        logger.error(f"Merge failed: {exc}")
        sys.exit(1)

    # Sample AFTER merge to preserve join integrity
    if cfg.SAMPLE_SIZE:
        merged_df = sample_data(merged_df, cfg.SAMPLE_SIZE)

    save_dataframe(merged_df, cfg.OUTPUTS_DIR / "merged_data.csv", "Merged data")
    all_metrics["data"] = {
        "surveyor_rows": len(surveyor_df),
        "garage_rows": len(garage_df),
        "merged_rows": len(merged_df),
    }

    # ── 3. Part Matching ────────────────────────────────────────────────
    logger.info("\n[3/6] Part matching (Fuzzy + TF-IDF Cosine) ...")
    try:
        survey_parts = surveyor_df[cfg.SURVEYOR_PART_COL].dropna().unique()
        garage_parts = garage_df[cfg.GARAGE_PART_COL].dropna().unique()

        matcher = PartMatcher()
        matcher.fit(survey_parts, garage_parts)

        fuzzy_results = matcher.match_fuzzy()
        cosine_results = matcher.match_cosine()
        combined_results = matcher.get_combined_results()

        matcher.save(cfg.MODELS_DIR / "part_matcher.pkl")
        save_dataframe(
            combined_results,
            cfg.OUTPUTS_DIR / "part_matching_results.csv",
            "Part matching results",
        )

        all_metrics["part_matching"] = {
            "total_survey_parts": len(survey_parts),
            "total_garage_parts": len(garage_parts),
            "fuzzy_match_rate_pct": round(fuzzy_results["matched"].mean() * 100, 2),
            "cosine_match_rate_pct": round(cosine_results["matched"].mean() * 100, 2),
            "method_agreement_rate_pct": round(
                combined_results["match_agreement"].mean() * 100, 2
            ),
        }

        viz.plot_matching_scores(combined_results, cfg.PLOTS_DIR / "matching_scores.png")

    except Exception as exc:
        logger.error(f"Part matching failed: {exc}", exc_info=True)
        all_metrics["part_matching"] = {"error": str(exc)}

    # ── 4. Association Rule Mining ──────────────────────────────────────
    logger.info("\n[4/6] Association rule mining (Apriori) ...")
    try:
        miner = AssociationMiner()
        miner.fit(merged_df)

        all_metrics["association_rules"] = miner.get_summary()

        if miner.rules is not None and not miner.rules.empty:
            rules_export = miner.rules.copy()
            rules_export["antecedents"] = rules_export["antecedents"].apply(
                lambda x: ", ".join(x)
            )
            rules_export["consequents"] = rules_export["consequents"].apply(
                lambda x: ", ".join(x)
            )
            save_dataframe(
                rules_export,
                cfg.OUTPUTS_DIR / "association_rules.csv",
                "Association rules",
            )

        if miner.frequent_itemsets is not None and not miner.frequent_itemsets.empty:
            itemsets_export = miner.frequent_itemsets.copy()
            itemsets_export["itemsets"] = itemsets_export["itemsets"].apply(
                lambda x: ", ".join(x)
            )
            save_dataframe(
                itemsets_export,
                cfg.OUTPUTS_DIR / "frequent_itemsets.csv",
                "Frequent itemsets",
            )

    except Exception as exc:
        logger.error(f"Association mining failed: {exc}", exc_info=True)
        all_metrics["association_rules"] = {"error": str(exc)}

    # ── 5. Recommendation Engine ────────────────────────────────────────
    logger.info("\n[5/6] Building recommendation engine (SVD) ...")
    try:
        recommender = ClaimRecommender()
        recommender.fit(merged_df)

        rec_metrics = recommender.evaluate()
        all_metrics["recommendation"] = rec_metrics

        recommender.save(cfg.MODELS_DIR / "recommender.pkl")

        # Example recommendation
        if recommender.claim_ids:
            sample_id = recommender.claim_ids[0]
            recs = recommender.recommend(sample_id)
            logger.info(f"Sample recommendation for {sample_id}: {recs[:3]}")

        if recommender.similarity_matrix is not None:
            viz.plot_similarity_heatmap(
                recommender.similarity_matrix,
                recommender.claim_ids,
                cfg.PLOTS_DIR / "similarity_heatmap.png",
            )

    except Exception as exc:
        logger.error(f"Recommendation engine failed: {exc}", exc_info=True)
        all_metrics["recommendation"] = {"error": str(exc)}

    # ── 6. Fraud Detection ──────────────────────────────────────────────
    logger.info("\n[6/6] Training fraud detector ...")
    try:
        claim_features = FraudDetector.engineer_features(merged_df)

        # Attach ground-truth fraud labels if available (aggregated per claim)
        label_col = "IS_FRAUD"
        has_labels = label_col in merged_df.columns
        if has_labels:
            gt_labels = (
                merged_df.groupby(cfg.CLAIM_ID_COL)[label_col]
                .max()  # claim is fraud if ANY row is fraud
                .reset_index()
            )
            claim_features = claim_features.merge(
                gt_labels, on=cfg.CLAIM_ID_COL, how="left"
            )
            claim_features[label_col] = claim_features[label_col].fillna(0).astype(int)
            n_fraud = int(claim_features[label_col].sum())
            logger.info(
                f"Ground-truth labels: {n_fraud} fraud / "
                f"{len(claim_features)} total ({n_fraud / len(claim_features) * 100:.1f}%)"
            )

        # Proper train/test split
        train_feats, test_feats = split_data(claim_features)

        # ── A) Unsupervised – Isolation Forest (always trained) ──────
        logger.info("\n  [6a] Isolation Forest (unsupervised) ...")
        train_input = train_feats.drop(columns=[label_col], errors="ignore")
        test_input = test_feats.drop(columns=[label_col], errors="ignore")

        detector = FraudDetector()
        detector.fit(train_input)

        test_preds_if = detector.predict(test_input)
        test_metrics_if = detector.evaluate(test_input, test_preds_if)
        if has_labels:
            sup_test_if = detector.evaluate_supervised(test_preds_if, test_feats[label_col])
            test_metrics_if["supervised"] = sup_test_if

        full_input = claim_features.drop(columns=[label_col], errors="ignore")
        full_preds_if = detector.predict(full_input)
        full_metrics_if = detector.evaluate(full_input, full_preds_if)
        if has_labels:
            sup_full_if = detector.evaluate_supervised(full_preds_if, claim_features[label_col])
            full_metrics_if["supervised"] = sup_full_if

        detector.save(cfg.MODELS_DIR / "fraud_detector_iforest.pkl")

        all_metrics["fraud_isolation_forest"] = {
            "features_used": detector.feature_names,
            "train_size": len(train_feats),
            "test_size": len(test_feats),
            "test_metrics": test_metrics_if,
            "full_data_metrics": full_metrics_if,
        }

        # ── B) Supervised – Random Forest (only when labels exist) ───
        if has_labels:
            logger.info("\n  [6b] Random Forest (supervised) ...")
            sup_detector = SupervisedFraudDetector()
            sup_detector.fit(train_input, train_feats[label_col])

            test_preds_rf = sup_detector.predict(test_input)
            test_metrics_rf = sup_detector.evaluate(test_preds_rf, test_feats[label_col])

            full_preds_rf = sup_detector.predict(full_input)
            full_metrics_rf = sup_detector.evaluate(full_preds_rf, claim_features[label_col])

            sup_detector.save(cfg.MODELS_DIR / "fraud_detector_rf.pkl")

            all_metrics["fraud_random_forest"] = {
                "features_used": sup_detector.feature_names,
                "train_size": len(train_feats),
                "test_size": len(test_feats),
                "test_metrics": test_metrics_rf,
                "full_data_metrics": full_metrics_rf,
            }

            # Choose best model for final predictions output
            best_f1_if = test_metrics_if.get("supervised", {}).get("f1_score", 0)
            best_f1_rf = test_metrics_rf.get("f1_score", 0)
            if best_f1_rf >= best_f1_if:
                full_preds = full_preds_rf
                logger.info(
                    f"✓ Best model: Random Forest (test F1={best_f1_rf:.4f}) "
                    f"vs Isolation Forest (test F1={best_f1_if:.4f})"
                )
            else:
                full_preds = full_preds_if
                logger.info(
                    f"✓ Best model: Isolation Forest (test F1={best_f1_if:.4f}) "
                    f"vs Random Forest (test F1={best_f1_rf:.4f})"
                )
        else:
            full_preds = full_preds_if

        save_dataframe(
            full_preds,
            cfg.OUTPUTS_DIR / "fraud_predictions.csv",
            "Fraud predictions",
        )

        viz.plot_anomaly_distribution(
            full_preds_if, cfg.PLOTS_DIR / "anomaly_distribution.png"
        )
        viz.plot_fraud_feature_importance(
            full_preds_if, detector.feature_names, cfg.PLOTS_DIR / "fraud_features.png"
        )

    except Exception as exc:
        logger.error(f"Fraud detection failed: {exc}", exc_info=True)
        all_metrics["fraud_detection"] = {"error": str(exc)}

    # ── Common Parts Analysis ───────────────────────────────────────────
    logger.info("\nGenerating common-parts analysis ...")
    try:
        if cfg.SURVEYOR_PART_COL in surveyor_df.columns:
            part_counts = surveyor_df[cfg.SURVEYOR_PART_COL].value_counts()
            total = part_counts.sum()
            common_df = pd.DataFrame(
                {
                    "part_name": part_counts.index,
                    "count": part_counts.values,
                    "percentage": (part_counts.values / total * 100).round(2),
                }
            )
            save_dataframe(common_df, cfg.OUTPUTS_DIR / "common_parts.csv", "Common parts")
            viz.plot_common_parts(part_counts, cfg.PLOTS_DIR / "common_parts.png")

        if cfg.PRIMARY_PART_CODE_COL in surveyor_df.columns:
            primary_counts = surveyor_df[cfg.PRIMARY_PART_CODE_COL].value_counts()
            primary_df = pd.DataFrame(
                {
                    "primary_part_code": primary_counts.index,
                    "claim_count": primary_counts.values,
                }
            )
            save_dataframe(
                primary_df,
                cfg.OUTPUTS_DIR / "primary_parts_distribution.csv",
                "Primary parts distribution",
            )
            viz.plot_claim_distribution(
                primary_df, cfg.PLOTS_DIR / "claim_distribution.png"
            )
    except Exception as exc:
        logger.error(f"Common-parts analysis failed: {exc}", exc_info=True)

    # ── Save All Metrics ────────────────────────────────────────────────
    metrics_path = cfg.OUTPUTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"\nAll metrics → {metrics_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Time   : {elapsed:.1f}s")
    logger.info(f"  Models : {cfg.MODELS_DIR}")
    logger.info(f"  Results: {cfg.OUTPUTS_DIR}")
    logger.info(f"  Plots  : {cfg.PLOTS_DIR}")

    for section, metrics in all_metrics.items():
        logger.info(f"\n  --- {section.upper().replace('_', ' ')} ---")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, dict):
                    logger.info(f"    {k}:")
                    for kk, vv in v.items():
                        logger.info(f"      {kk}: {vv}")
                elif isinstance(v, list):
                    logger.info(f"    {k}: [{len(v)} items]")
                else:
                    logger.info(f"    {k}: {v}")

    return all_metrics


if __name__ == "__main__":
    main()
