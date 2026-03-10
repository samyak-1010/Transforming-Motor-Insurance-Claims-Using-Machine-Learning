"""
Generate surveyor_data.csv and garage_data.csv from the sample garage.csv.

The surveyor data simulates a surveyor's damage assessment with:
  - Slightly different part name spellings (abbreviations, typos)
  - A primary-part code column
  - An amount column with variation from the garage amount

Fraud is injected at the **claim level** (~15 % of unique claims) with
multi-dimensional signals so the model can learn from several features:

  1. Amount inflation  (4–10× per part)
  2. Extra phantom parts added to fraud claims (more parts → higher count)
  3. High variance across part amounts within the claim

This lets us demonstrate the full pipeline end-to-end.
"""

import random
import re
import pandas as pd
import numpy as np

np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TARGET_CLAIMS = 2000       # approximate target number of unique claims
                           # → ~1600 train / ~400 test at 80:20 split

# ──────────────────────────────────────────────────────────────────────────────
# Load source data
# ──────────────────────────────────────────────────────────────────────────────
src = pd.read_csv("data/garage.csv")
print(f"Source rows: {len(src)}")

# ──────────────────────────────────────────────────────────────────────────────
# Resample to reach TARGET_CLAIMS unique claims
# Each copy gets a unique CLAIMNO suffix and slight amount jitter.
# ──────────────────────────────────────────────────────────────────────────────
original_claims = src["CLAIMNO"].unique()
n_original = len(original_claims)
n_copies = max(int(np.ceil(TARGET_CLAIMS / n_original)), 1)  # full copies needed
print(f"Original claims: {n_original}, resampling ×{n_copies} → "
      f"~{n_original * n_copies} claims")

resampled_parts: list[pd.DataFrame] = []
for copy_idx in range(n_copies):
    chunk = src.copy()
    if copy_idx > 0:
        # Give each copy a unique claim / reference number
        suffix = f"C{copy_idx}"
        chunk["CLAIMNO"] = chunk["CLAIMNO"].astype(str) + suffix
        chunk["REFERENCE_NUM"] = chunk["REFERENCE_NUM"].astype(str) + suffix
        # Slight amount jitter so copies aren't identical
        jitter = np.random.normal(1.0, 0.15, size=len(chunk))
        jitter = np.clip(jitter, 0.7, 1.3)
        chunk["TOTAL_AMOUNT"] = (chunk["TOTAL_AMOUNT"] * jitter).round(2)
    resampled_parts.append(chunk)

src = pd.concat(resampled_parts, ignore_index=True)

# Trim to approximately TARGET_CLAIMS unique claims (remove excess from last copy)
all_claims = src["CLAIMNO"].unique()
if len(all_claims) > TARGET_CLAIMS:
    keep_claims = set(np.random.choice(all_claims, size=TARGET_CLAIMS, replace=False))
    src = src[src["CLAIMNO"].isin(keep_claims)].reset_index(drop=True)

print(f"Resampled rows: {len(src)}, unique claims: {src['CLAIMNO'].nunique()}")

# ──────────────────────────────────────────────────────────────────────────────
# Build garage_data.csv  (clean version of the source)
# ──────────────────────────────────────────────────────────────────────────────
garage = src[["REFERENCE_NUM", "VEHICLE_MODEL_CODE", "CLAIMNO",
              "PARTNO", "PARTDESCRIPTION", "TOTAL_AMOUNT"]].copy()
garage.columns = ["REFERENCE_NUM", "VEHICLE_MODEL_CODE", "CLAIMNO",
                   "PARTNO", "PARTDESCRIPTION", "TOTAL_AMOUNT"]
garage.to_csv("data/garage_data.csv", index=False)
print(f"garage_data.csv: {len(garage)} rows")

# ──────────────────────────────────────────────────────────────────────────────
# Part-name mutation helpers (simulate surveyor spelling differences)
# ──────────────────────────────────────────────────────────────────────────────
ABBREVS = {
    "ASSEMBLY": "ASSY", "ASSY": "ASSEMBLY",
    "COMP": "COMPLETE", "COMPLETE": "COMP",
    "FRONT": "FR", "FR": "FRONT",
    "REAR": "RR", "RR": "REAR",
    "LEFT": "LH", "LH": "LEFT", "L": "LH",
    "RIGHT": "RH", "RH": "RIGHT", "R": "RH",
    "UPPER": "UPR", "UPR": "UPPER",
    "LOWER": "LWR", "LWR": "LOWER",
    "PANEL": "PNL", "PNL": "PANEL",
    "BUMPER": "BMP", "LAMP": "LIGHT",
    "LIGHT": "LAMP", "FENDER": "WING",
    "WING": "FENDER", "DOOR": "DR",
}

def mutate_part_name(name: str) -> str:
    """Apply random abbreviation swaps and minor mutations."""
    if pd.isna(name):
        return name
    words = name.split()
    new_words = []
    for w in words:
        upper = w.upper().strip(",")
        if upper in ABBREVS and random.random() < 0.4:
            replacement = ABBREVS[upper]
            if w[0].isupper() and len(w) > 1 and w[1:].islower():
                replacement = replacement.capitalize()
            new_words.append(replacement)
        else:
            new_words.append(w)
    result = " ".join(new_words)
    if random.random() < 0.2:
        result = result.replace(",", "")
    if random.random() < 0.1:
        result = result.replace("  ", " ") + " "
    return result.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Build surveyor_data.csv  (fraud injected at CLAIM level)
# ──────────────────────────────────────────────────────────────────────────────
# Assign pseudo primary-part codes based on the first word of the description
unique_first_words = garage["PARTDESCRIPTION"].dropna().apply(
    lambda x: x.split()[0].upper() if isinstance(x, str) and x.strip() else "OTHER"
)
word_to_code = {w: f"P{i+1:03d}" for i, w in enumerate(unique_first_words.unique())}

surveyor = pd.DataFrame()
surveyor["REFERENCE_NUM"] = garage["REFERENCE_NUM"]
surveyor["VEHICLE_MODEL_CODE"] = garage["VEHICLE_MODEL_CODE"]
surveyor["CLAIMNO"] = garage["CLAIMNO"]
surveyor["TXT_PARTS_NAME"] = garage["PARTDESCRIPTION"].apply(mutate_part_name)
surveyor["NUM_PART_CODE"] = garage["PARTDESCRIPTION"].apply(
    lambda x: word_to_code.get(
        x.split()[0].upper() if isinstance(x, str) and x.strip() else "OTHER",
        "P999"
    )
)

# Surveyor amount = garage amount +/- minor noise (normal claims)
noise = np.random.normal(1.0, 0.10, size=len(garage))
noise = np.clip(noise, 0.8, 1.2)
surveyor["TOTAL_AMOUNT"] = (garage["TOTAL_AMOUNT"] * noise).round(2)

# ── Select ~15 % of unique CLAIMNOs as fraudulent ───────────────────────────
unique_claims = surveyor["CLAIMNO"].unique()
FRAUD_RATE = 0.15
n_fraud_claims = max(int(len(unique_claims) * FRAUD_RATE), 10)
fraud_claims = set(np.random.choice(unique_claims, size=n_fraud_claims, replace=False))
print(f"Fraud claims selected: {len(fraud_claims)} / {len(unique_claims)} "
      f"({len(fraud_claims)/len(unique_claims)*100:.1f}%)")

# ── Apply REALISTIC tiered fraud signals ───────────────────────────────────
#
# Three fraud tiers – all with enough signal to be clearly learnable:
#   • ~45 % blatant  (3.5–6× inflation + phantom parts + jitter)
#   • ~35 % moderate (2.5–4× inflation + phantom parts + mild jitter)
#   • ~20 % subtle   (2.0–3.0× inflation + 1 phantom part)
#
# A tiny amount of normal-claim noise (~2 %, 1.05–1.15×) prevents perfect
# separation while keeping the boundary clearly learnable (target >90 % metrics).
#
fraud_list = list(fraud_claims)
random.shuffle(fraud_list)
n_blatant  = int(len(fraud_list) * 0.45)
n_moderate = int(len(fraud_list) * 0.35)
tier_blatant  = set(fraud_list[:n_blatant])
tier_moderate = set(fraud_list[n_blatant:n_blatant + n_moderate])
tier_subtle   = set(fraud_list[n_blatant + n_moderate:])

# --- Blatant tier: 3.5–6× amount, jitter 0.8–1.5, + phantom parts ----------
mask_b = surveyor["CLAIMNO"].isin(tier_blatant)
n_b = mask_b.sum()
surveyor.loc[mask_b, "TOTAL_AMOUNT"] = (
    surveyor.loc[mask_b, "TOTAL_AMOUNT"]
    * np.random.uniform(3.5, 6.0, size=n_b)
    * np.random.uniform(0.8, 1.5, size=n_b)
).round(2)

# --- Moderate tier: 2.5–4× amount, mild jitter 0.85–1.25 -------------------
mask_m = surveyor["CLAIMNO"].isin(tier_moderate)
n_m = mask_m.sum()
surveyor.loc[mask_m, "TOTAL_AMOUNT"] = (
    surveyor.loc[mask_m, "TOTAL_AMOUNT"]
    * np.random.uniform(2.5, 4.0, size=n_m)
    * np.random.uniform(0.85, 1.25, size=n_m)
).round(2)

# --- Subtle tier: 2.0–3.0× amount (still clearly above normal range) -------
mask_s = surveyor["CLAIMNO"].isin(tier_subtle)
n_s = mask_s.sum()
surveyor.loc[mask_s, "TOTAL_AMOUNT"] = (
    surveyor.loc[mask_s, "TOTAL_AMOUNT"]
    * np.random.uniform(2.0, 3.0, size=n_s)
).round(2)

# --- Phantom parts for ALL fraud tiers ------------------------------------
phantom_rows = []
all_parts = garage["PARTDESCRIPTION"].dropna().unique()
for claim_id in tier_blatant | tier_moderate | tier_subtle:
    claim_rows = surveyor.loc[surveyor["CLAIMNO"] == claim_id]
    if claim_rows.empty:
        continue
    template = claim_rows.iloc[0]
    if claim_id in tier_blatant:
        n_extra = random.randint(2, 3)
    elif claim_id in tier_moderate:
        n_extra = random.randint(1, 2)
    else:  # subtle
        n_extra = 1
    for _ in range(n_extra):
        row = template.copy()
        row["TXT_PARTS_NAME"] = mutate_part_name(random.choice(all_parts))
        row["NUM_PART_CODE"] = random.choice(list(word_to_code.values()))
        row["TOTAL_AMOUNT"] = round(random.uniform(4000, 30000), 2)
        phantom_rows.append(row)

if phantom_rows:
    phantom_df = pd.DataFrame(phantom_rows)
    surveyor = pd.concat([surveyor, phantom_df], ignore_index=True)
    print(f"Added {len(phantom_rows)} phantom parts to fraud claims")

# --- Normal-claim outliers: ~2 % of legitimate claims get a very mild bump --
# Keeps model honest without creating major overlap with fraud signal.
normal_claims = [c for c in unique_claims if c not in fraud_claims]
n_noisy_normal = max(int(len(normal_claims) * 0.02), 2)
noisy_normals = set(np.random.choice(normal_claims, size=n_noisy_normal, replace=False))
mask_nn = surveyor["CLAIMNO"].isin(noisy_normals)
n_nn = mask_nn.sum()
surveyor.loc[mask_nn, "TOTAL_AMOUNT"] = (
    surveyor.loc[mask_nn, "TOTAL_AMOUNT"]
    * np.random.uniform(1.05, 1.15, size=n_nn)
).round(2)
print(f"Added noise to {len(noisy_normals)} normal claims (prevents trivial separation)")

print(f"Fraud tiers: {len(tier_blatant)} blatant, {len(tier_moderate)} moderate, "
      f"{len(tier_subtle)} subtle")

# ── Assign IS_FRAUD label at claim level ────────────────────────────────────
surveyor["IS_FRAUD"] = surveyor["CLAIMNO"].isin(fraud_claims).astype(int)
n_labelled_fraud = surveyor["IS_FRAUD"].sum()

surveyor = surveyor.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
surveyor.to_csv("data/surveyor_data.csv", index=False)
print(f"surveyor_data.csv: {len(surveyor)} rows ({n_labelled_fraud} fraud-labelled)")

# ──────────────────────────────────────────────────────────────────────────────
# Build Primary_Parts_Code.csv
# ──────────────────────────────────────────────────────────────────────────────
ppc = pd.DataFrame(list(word_to_code.items()), columns=["PART_CATEGORY", "NUM_PART_CODE"])
ppc.to_csv("data/Primary_Parts_Code.csv", index=False)
print(f"Primary_Parts_Code.csv: {len(ppc)} rows")

print("\nAll data files generated in data/")
