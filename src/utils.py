"""
Utility functions: logging setup, directory creation, and common helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return the root logger for the pipeline."""
    logger = logging.getLogger("insurance_ml")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


def ensure_directories(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path, description: str = "") -> None:
    """Save a DataFrame to CSV with logging."""
    logger = logging.getLogger("insurance_ml")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {description}: {path} ({len(df)} rows)")


def load_dataframe(path: Path, description: str = "") -> pd.DataFrame:
    """Load a CSV file with error handling."""
    logger = logging.getLogger("insurance_ml")
    if not path.exists():
        raise FileNotFoundError(f"{description} data file not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {description}: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df
