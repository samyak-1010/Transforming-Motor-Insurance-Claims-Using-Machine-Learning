"""
Configuration settings for the Motor Insurance Claims ML Pipeline.
All hyperparameters, file paths, and constants are centralized here.
"""

import os
from pathlib import Path


# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"

# ─── Data Files ──────────────────────────────────────────────────────────────
SURVEYOR_DATA = DATA_DIR / "surveyor_data.csv"
GARAGE_DATA = DATA_DIR / "garage_data.csv"
PRIMARY_PARTS_DATA = DATA_DIR / "Primary_Parts_Code.csv"

# ─── Data Processing ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.2
SAMPLE_SIZE = None  # Set to an int to sample after merge; None = use full data

# ─── Merge Keys ──────────────────────────────────────────────────────────────
MERGE_KEYS = ["REFERENCE_NUM", "VEHICLE_MODEL_CODE", "CLAIMNO"]

# ─── Column Names ────────────────────────────────────────────────────────────
SURVEYOR_PART_COL = "TXT_PARTS_NAME"
GARAGE_PART_COL = "PARTDESCRIPTION"
CLAIM_ID_COL = "CLAIMNO"
AMOUNT_COL = "TOTAL_AMOUNT"
PRIMARY_PART_CODE_COL = "NUM_PART_CODE"

# ─── Part Matching ───────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 80
COSINE_THRESHOLD = 0.15
TFIDF_NGRAM_RANGE = (1, 2)

# ─── Association Rule Mining ─────────────────────────────────────────────────
APRIORI_MIN_SUPPORT = 0.01
APRIORI_MIN_CONFIDENCE = 0.1
APRIORI_MIN_LIFT = 1.0

# ─── Recommendation Engine (SVD) ────────────────────────────────────────────
SVD_N_COMPONENTS = 10
SVD_TOP_N = 5

# ─── Fraud Detection (Isolation Forest – unsupervised fallback) ─────────────
ISO_FOREST_CONTAMINATION = 0.10
ISO_FOREST_N_ESTIMATORS = 200
ISO_FOREST_MAX_FEATURES = 1.0
ISO_FOREST_BOOTSTRAP = True

# ─── Fraud Detection (Random Forest – supervised, used when labels exist) ───
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_SPLIT = 5

# ─── Cross-Validation & Grid Search ────────────────────────────────────────
CV_N_SPLITS = 5             # folds per repeat
CV_N_REPEATS = 3            # repetitions for RepeatedStratifiedKFold
GRID_SEARCH_ENABLED = True  # set False to skip hyperparameter tuning
RF_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [8, 12, 16, None],
    "min_samples_split": [2, 5, 10],
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
