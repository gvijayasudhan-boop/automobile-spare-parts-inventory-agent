# ─────────────────────────────────────────────────────────────────────────────
# 03_processing/preprocessing.py
# This script runs INSIDE the SageMaker Processing Job container.
# Do NOT run it locally — launch it via run_processing_job.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os

INPUT_USAGE  = "/opt/ml/processing/input/usage/usage_history.csv"
INPUT_PARTS  = "/opt/ml/processing/input/parts/parts_catalog.csv"
OUTPUT_TRAIN = "/opt/ml/processing/output/train/"
OUTPUT_VAL   = "/opt/ml/processing/output/val/"
OUTPUT_INFER = "/opt/ml/processing/output/inference/"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VAL,   exist_ok=True)
os.makedirs(OUTPUT_INFER, exist_ok=True)

print("Loading data...")
df    = pd.read_csv(INPUT_USAGE, parse_dates=["service_date"])
parts = pd.read_csv(INPUT_PARTS)

# ── Aggregate to weekly per part + vehicle model ───────────────────────────
df["week_start"] = df["service_date"] - pd.to_timedelta(df["service_date"].dt.dayofweek, unit="D")
weekly = (
    df.groupby(["part_id", "vehicle_model", "variant", "week_start"])
      .agg(qty_used=("qty_used","sum"))
      .reset_index()
      .sort_values(["part_id","vehicle_model","variant","week_start"])
)

# ── Feature engineering ────────────────────────────────────────────────────
print("Engineering features...")
grp = ["part_id","vehicle_model","variant"]

weekly["lag_1w"]  = weekly.groupby(grp)["qty_used"].shift(1).fillna(0)
weekly["lag_4w"]  = weekly.groupby(grp)["qty_used"].shift(4).fillna(0)
weekly["lag_12w"] = weekly.groupby(grp)["qty_used"].shift(12).fillna(0)

weekly["rolling_mean_4w"]  = (
    weekly.groupby(grp)["qty_used"]
          .transform(lambda x: x.rolling(4,  min_periods=1).mean())
)
weekly["rolling_mean_12w"] = (
    weekly.groupby(grp)["qty_used"]
          .transform(lambda x: x.rolling(12, min_periods=1).mean())
)
weekly["rolling_std_4w"]   = (
    weekly.groupby(grp)["qty_used"]
          .transform(lambda x: x.rolling(4,  min_periods=1).std().fillna(0))
)

# Seasonality flags
weekly["month"]       = weekly["week_start"].dt.month
weekly["is_monsoon"]  = weekly["month"].isin([6,7,8,9]).astype(int)
weekly["is_summer"]   = weekly["month"].isin([3,4,5]).astype(int)
weekly["week_of_year"]= weekly["week_start"].dt.isocalendar().week.astype(int)

# Encode vehicle model
weekly["vehicle_code"] = (
    pd.factorize(weekly["vehicle_model"] + "#" + weekly["variant"])[0]
)

# Encode part_id
weekly["part_code"] = pd.factorize(weekly["part_id"])[0]

# Composite key (matches DynamoDB SK)
weekly["part_vehicle_key"] = weekly["part_id"] + "_" + weekly["vehicle_model"] + "#" + weekly["variant"]

# Drop rows with any NaN after lags
weekly = weekly.dropna().reset_index(drop=True)

# ── Target variable ────────────────────────────────────────────────────────
# Predict next 4 weeks demand (qty_used is current week — XGBoost learns to predict it from lags)
FEATURE_COLS = [
    "lag_1w","lag_4w","lag_12w",
    "rolling_mean_4w","rolling_mean_12w","rolling_std_4w",
    "is_monsoon","is_summer","week_of_year",
    "vehicle_code","part_code",
]
TARGET_COL = "qty_used"

# ── Train / Validation split (time-based — last 4 weeks as validation) ─────
print("Splitting train/validation...")
all_weeks  = sorted(weekly["week_start"].unique())
cutoff     = all_weeks[-4]  # last 4 weeks = validation

train = weekly[weekly["week_start"] <  cutoff]
val   = weekly[weekly["week_start"] >= cutoff]

# XGBoost expects: first column = label, rest = features (no header)
train_data = pd.concat([train[[TARGET_COL]], train[FEATURE_COLS]], axis=1)
val_data   = pd.concat([val[[TARGET_COL]],   val[FEATURE_COLS]],   axis=1)

# Inference input = feature columns only (no label)
infer_meta  = weekly[weekly["week_start"] >= cutoff][["part_id","vehicle_model","variant","part_vehicle_key"]].copy()
infer_feats = weekly[weekly["week_start"] >= cutoff][FEATURE_COLS].copy()

print(f"Train rows: {len(train_data)}, Validation rows: {len(val_data)}")

# ── Save outputs ───────────────────────────────────────────────────────────
train_data.to_csv(f"{OUTPUT_TRAIN}/train.csv", index=False, header=False)
val_data.to_csv(  f"{OUTPUT_VAL}/val.csv",     index=False, header=False)
infer_feats.to_csv(f"{OUTPUT_INFER}/inference.csv", index=False, header=False)

# Save metadata for mapping predictions back to part IDs
infer_meta.to_csv(f"{OUTPUT_INFER}/inference_meta.csv", index=False)

print("Feature engineering complete. Outputs written.")
print(f"  Train: {OUTPUT_TRAIN}train.csv")
print(f"  Val:   {OUTPUT_VAL}val.csv")
print(f"  Infer: {OUTPUT_INFER}inference.csv")
