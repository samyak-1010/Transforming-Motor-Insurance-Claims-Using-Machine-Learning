"""
Data loading, cleaning, and merging module.
Handles all data I/O, text normalisation, and dataset merging with proper
error handling and post-merge sampling.
"""

import logging
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from src.utils import load_dataframe

logger = logging.getLogger("insurance_ml.data")


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text) -> str:
    """Normalise a part-description string."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip()


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all CSV datasets and apply text cleaning."""
    surveyor_df = load_dataframe(cfg.SURVEYOR_DATA, "Surveyor")
    garage_df = load_dataframe(cfg.GARAGE_DATA, "Garage")

    if cfg.PRIMARY_PARTS_DATA.exists():
        primary_parts_df = load_dataframe(cfg.PRIMARY_PARTS_DATA, "Primary Parts")
    else:
        logger.warning(f"Primary parts file not found: {cfg.PRIMARY_PARTS_DATA}")
        primary_parts_df = pd.DataFrame()

    # Clean text columns
    if cfg.SURVEYOR_PART_COL in surveyor_df.columns:
        surveyor_df[cfg.SURVEYOR_PART_COL] = surveyor_df[cfg.SURVEYOR_PART_COL].apply(clean_text)
    if cfg.GARAGE_PART_COL in garage_df.columns:
        garage_df[cfg.GARAGE_PART_COL] = garage_df[cfg.GARAGE_PART_COL].apply(clean_text)

    # Ensure merge-key columns are strings
    for col in cfg.MERGE_KEYS:
        if col in surveyor_df.columns:
            surveyor_df[col] = surveyor_df[col].astype(str).str.strip()
        if col in garage_df.columns:
            garage_df[col] = garage_df[col].astype(str).str.strip()

    logger.info(f"Surveyor shape: {surveyor_df.shape}  |  Garage shape: {garage_df.shape}")
    return surveyor_df, garage_df, primary_parts_df


# ─── Merging ─────────────────────────────────────────────────────────────────

def merge_datasets(
    surveyor_df: pd.DataFrame,
    garage_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-merge surveyor and garage data on shared keys."""
    available_keys = [
        k for k in cfg.MERGE_KEYS
        if k in surveyor_df.columns and k in garage_df.columns
    ]
    if not available_keys:
        raise ValueError(
            f"No common merge keys. Surveyor cols: {list(surveyor_df.columns)}, "
            f"Garage cols: {list(garage_df.columns)}"
        )

    merged_df = pd.merge(
        surveyor_df,
        garage_df,
        on=available_keys,
        how="inner",
        suffixes=("_survey", "_garage"),
    )

    if merged_df.empty:
        raise ValueError(
            "Merge produced an empty DataFrame – check that datasets share "
            "common claim records."
        )

    # Resolve TOTAL_AMOUNT naming: prefer the garage side, fall back to any match
    amount_cols = [c for c in merged_df.columns if "TOTAL_AMOUNT" in c.upper()]
    if amount_cols and cfg.AMOUNT_COL not in merged_df.columns:
        merged_df.rename(columns={amount_cols[0]: cfg.AMOUNT_COL}, inplace=True)

    # ── Clean absurd amounts (CSV parsing artefacts) ────────────────────
    if cfg.AMOUNT_COL in merged_df.columns:
        amount_col = cfg.AMOUNT_COL
    else:
        amount_col = next(
            (c for c in merged_df.columns if "AMOUNT" in c.upper()), None
        )
    if amount_col:
        before = len(merged_df)
        q99 = merged_df[amount_col].quantile(0.99)
        cap = max(q99 * 10, 100_000)  # generous cap
        merged_df = merged_df[merged_df[amount_col] <= cap].reset_index(drop=True)
        dropped = before - len(merged_df)
        if dropped:
            logger.warning(
                f"Dropped {dropped} rows with {amount_col} > {cap:,.0f} "
                f"(likely CSV parse errors)"
            )

    logger.info(
        f"Merged dataset: {merged_df.shape} ({len(available_keys)} keys: {available_keys})"
    )
    return merged_df


# ─── Sampling (post-merge) ──────────────────────────────────────────────────

def sample_data(df: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
    """Optionally sample rows *after* merging to preserve join integrity."""
    if n is not None and 0 < n < len(df):
        df = df.sample(n=n, random_state=cfg.RANDOM_STATE)
        logger.info(f"Sampled to {n} rows")
    return df


# ─── Train / Test Split ─────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, stratify_col: str = "IS_FRAUD") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test sets with optional stratification."""
    stratify = df[stratify_col] if stratify_col in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.TEST_SPLIT_RATIO,
        random_state=cfg.RANDOM_STATE,
        stratify=stratify,
    )
    logger.info(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
    return train_df, test_df
