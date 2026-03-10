"""
Association Rule Mining using the Apriori algorithm.

Discovers which vehicle parts are commonly damaged together within a claim.
"""

import logging
from typing import Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import config as cfg

logger = logging.getLogger("insurance_ml.association")


class AssociationMiner:
    """Build and analyse part co-occurrence rules from claim data."""

    def __init__(self) -> None:
        self.frequent_itemsets: Optional[pd.DataFrame] = None
        self.rules: Optional[pd.DataFrame] = None
        self.encoder: Optional[TransactionEncoder] = None

    def fit(self, merged_df: pd.DataFrame) -> "AssociationMiner":
        """Extract frequent itemsets and association rules."""
        # Resolve part-name column
        part_col = cfg.SURVEYOR_PART_COL
        if part_col not in merged_df.columns:
            part_cols = [
                c
                for c in merged_df.columns
                if "PART" in c.upper() and "CODE" not in c.upper()
            ]
            if part_cols:
                part_col = part_cols[0]
            else:
                raise ValueError("No part-description column found in merged data")

        # Build transactions (one list of unique parts per claim)
        transactions = (
            merged_df.groupby(cfg.CLAIM_ID_COL)[part_col]
            .apply(lambda x: list(set(x)))
            .tolist()
        )

        # Only keep multi-part transactions (single items yield no rules)
        transactions = [t for t in transactions if len(t) > 1]
        if len(transactions) < 10:
            logger.warning(
                f"Only {len(transactions)} multi-part transactions – rules may be sparse."
            )

        logger.info(f"Mining rules from {len(transactions)} transactions …")

        # One-hot encoding
        self.encoder = TransactionEncoder()
        encoded = self.encoder.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(encoded, columns=self.encoder.columns_)

        # Apriori
        self.frequent_itemsets = apriori(
            df_encoded,
            min_support=cfg.APRIORI_MIN_SUPPORT,
            use_colnames=True,
        )

        if self.frequent_itemsets.empty:
            logger.warning("No frequent itemsets found – consider lowering min_support.")
            self.rules = pd.DataFrame()
            return self

        logger.info(f"Frequent itemsets: {len(self.frequent_itemsets)}")

        # Association rules
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=cfg.APRIORI_MIN_CONFIDENCE,
        )

        if not self.rules.empty:
            self.rules = self.rules[self.rules["lift"] >= cfg.APRIORI_MIN_LIFT]
            self.rules = self.rules.sort_values("lift", ascending=False).reset_index(
                drop=True
            )

        logger.info(f"Association rules generated: {len(self.rules)}")
        return self

    def get_summary(self) -> dict:
        """Return a summary dict of the mining results."""
        summary = {
            "num_frequent_itemsets": (
                len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0
            ),
            "num_rules": len(self.rules) if self.rules is not None else 0,
        }
        if self.rules is not None and not self.rules.empty:
            summary.update(
                {
                    "avg_confidence": round(self.rules["confidence"].mean(), 4),
                    "avg_lift": round(self.rules["lift"].mean(), 4),
                    "avg_support": round(self.rules["support"].mean(), 4),
                    "max_lift": round(self.rules["lift"].max(), 4),
                }
            )
        return summary
