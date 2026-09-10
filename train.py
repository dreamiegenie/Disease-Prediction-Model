"""
train.py — Training script for ISS Hackathon Track 1: AI-Assisted Infectious Disease Screening
Derived from /Users/shdev/others/ai-system-design-guide/ISS_Hackathon.ipynb

Dataset (Track 1 participant dataset):
  - 150k rows, 34 columns (raw) / 35 columns (cleaned balanced with bmi_flag)
  - Target: target_diagnosis (9 classes)
  - Features: demographics, vitals, symptoms, exposures, labs + engineered bmi_flag

Notebook pipeline replicated:
  1. month string → int mapping
  2. One-hot encode: sex (2), state (12), season (2) via pd.get_dummies
  3. Drop: patient_id, sex, state, season, target_diagnosis
  4. hstack remaining numeric columns + dummies → X (46-dim)
  5. One-hot encode target_diagnosis → y (9-dim)
  6. Train MultiOutputClassifier(LogisticRegression) and MultiOutputClassifier(RandomForestClassifier)
  7. Evaluate: accuracy_score + classification_report
  8. Persist: model/rf.joblib, model/labels_encoding.joblib

Usage:
  # With pre-split scaled files (as used in notebook cells 93-94):
  python train.py --train /path/to/scaled_train.csv --test /path/to/scaled_test.csv --model-type both

  # With single cleaned file (balanced):
  python train.py --data /path/to/track1_cleaned_balanced.csv --test-size 0.2

  # With raw participant dataset (requires scaling):
  python train.py --data /path/to/track1_participant_dataset.csv --scale --test-size 0.2

  # Train only logistic regression or random forest:
  python train.py --data data.csv --model-type rf

Outputs:
  - model/rf.joblib  (or model/lr.joblib)
  - model/labels_encoding.joblib
  - model/metadata.json (feature order, months_map, scaling stats)

Refs:
  - notebook: ISS_Hackathon.ipynb cell 96 (preprocess), cells 111-121 (training)
  - shapes: train_x (119997, 46), train_y (119997, 9) :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants — must match notebook exactly
# ---------------------------------------------------------------------------
MONTHS_MAP: Dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

LABELS = [
    "Acute Respiratory Infection",
    "COVID-19",
    "Cholera",
    "Healthy",
    "Lassa Fever",
    "Malaria",
    "Meningitis",
    "Tuberculosis",
    "Typhoid Fever",
]
# Notebook cell 122: labels_encoding = dict((i, l) for i, l in enumerate(labels))  :contentReference[oaicite:2]{index=2}
LABELS_ENCODING: Dict[int, str] = {i: l for i, l in enumerate(LABELS)}
LABELS_DECODING: Dict[str, int] = {l: i for i, l in LABELS_ENCODING.items()}

# Numeric columns that remain after dropping categoricals (order matters for hstack)
# Derived from df2.columns minus ["patient_id","sex","state","season","target_diagnosis"]
NUMERIC_COLS = [
    "month",
    "age",
    "pregnant",
    "bmi",
    "days_symptoms",
    "temperature_c",
    "heart_rate",
    "resp_rate",
    "spo2",
    "sbp",
    "dbp",
    "fever",
    "cough",
    "sore_throat",
    "headache",
    "vomiting",
    "diarrhea",
    "rash",
    "neck_stiffness",
    "weight_loss",
    "fatigue",
    "mosquito_exposure",
    "unsafe_water",
    "tb_contact",
    "recent_travel",
    "vaccinated",
    "hemoglobin",
    "wbc",
    "platelets",
    "bmi_flag",
]

# For raw dataset without bmi_flag, the column is absent — handle dynamically
NUMERIC_COLS_RAW = [c for c in NUMERIC_COLS if c != "bmi_flag"]

DROP_COLS = ["sex", "state", "season", "target_diagnosis", "patient_id"]


# ---------------------------------------------------------------------------
# Preprocessing — notebook cell 96 :contentReference[oaicite:3]{index=3}
# ---------------------------------------------------------------------------
def preprocess(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], Dict]:
    """
    Replicates notebook preprocess():

        data["month"] = data["month"].map(months_map)
        sex   = pd.get_dummies(data["sex"])
        state = pd.get_dummies(data["state"], prefix="State_")
        weather = pd.get_dummies(data["season"])
        target  = pd.get_dummies(data["target_diagnosis"])
        data = data.drop(columns=["sex","state","season","target_diagnosis","patient_id"])
        data = np.hstack([data.to_numpy(), sex.to_numpy(), state.to_numpy(), weather.to_numpy()])
        target = target.to_numpy()
        return (data, target)

    Enhancements:
      - Handles month already-int (scaled files store month as string still in notebook,
        but scaled_train.csv stores scaled float — we detect and skip mapping)
      - Optional StandardScaler for raw data (fit on train, transform on test)
      - Reindexes dummy columns to ensure fixed width even if a category missing in split
      - Ensures column order: numeric + sex + state + season
    """
    df = df.copy()

    # -- month mapping: map if any value is a month string (scaled files keep month as string like "Aug")
    try:
        sample = df["month"].dropna().iloc[0] if len(df) else None
    except Exception:
        sample = None
    if isinstance(sample, str) and sample in MONTHS_MAP:
        df["month"] = df["month"].map(MONTHS_MAP)
    elif df["month"].dtype == object or str(df["month"].dtype).startswith("string"):
        # fallback dtype check
        mapped = df["month"].map(MONTHS_MAP)
        # only apply if mapping succeeded (not all NaN)
        if mapped.notna().any():
            df["month"] = mapped
    # else already numeric / scaled — leave as is

    # -- dummy frames (ensure fixed columns)
    # sex: Female, Male
    sex_dummies = pd.get_dummies(df["sex"])
    for col in ["Female", "Male"]:
        if col not in sex_dummies.columns:
            sex_dummies[col] = 0
    sex_dummies = sex_dummies[["Female", "Male"]]

    # state: 12 states, prefixed State_
    state_dummies = pd.get_dummies(df["state"], prefix="State_")
    # Canonical state list observed in df.describe: Rivers, Bauchi, Oyo, Benue, FCT, Plateau, Lagos, Kaduna, etc.
    # We don't hardcode all 12 to stay robust, but we save the columns for inference.
    state_dummies = state_dummies.reindex(sorted(state_dummies.columns), axis=1)

    # season: Dry, Rainy
    season_dummies = pd.get_dummies(df["season"])
    for col in ["Dry", "Rainy"]:
        if col not in season_dummies.columns:
            season_dummies[col] = 0
    season_dummies = season_dummies[["Dry", "Rainy"]]

    # target: 9 classes — enforce LABELS order
    target_dummies = pd.get_dummies(df["target_diagnosis"])
    for col in LABELS:
        if col not in target_dummies.columns:
            target_dummies[col] = 0
    target_dummies = target_dummies[LABELS]

    # -- drop categoricals
    # bmi_flag may not exist in raw dataset
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    numeric_df = df.drop(columns=cols_to_drop)

    # Ensure numeric column order (include only those present)
    present_numeric = [c for c in NUMERIC_COLS if c in numeric_df.columns]
    # If raw data lacks bmi_flag, fallback to raw list
    if not present_numeric:
        present_numeric = [c for c in NUMERIC_COLS_RAW if c in numeric_df.columns]
    numeric_df = numeric_df[present_numeric]

    # -- optional scaling (for raw data path)
    if scaler is not None or fit_scaler:
        if fit_scaler:
            scaler = StandardScaler()
            numeric_df = pd.DataFrame(
                scaler.fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index
            )
        else:
            # transform using provided scaler (assumes same columns)
            assert scaler is not None
            numeric_df = pd.DataFrame(
                scaler.transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index
            )
    else:
        # No scaling — assume already scaled (scaled_train.csv stores already-scaled floats)
        pass

    # -- hstack: numeric + sex + state + season
    X = np.hstack([numeric_df.to_numpy(), sex_dummies.to_numpy(), state_dummies.to_numpy(), season_dummies.to_numpy()])
    y = target_dummies.to_numpy()

    meta = {
        "numeric_columns": present_numeric,
        "sex_columns": list(sex_dummies.columns),
        "state_columns": list(state_dummies.columns),
        "season_columns": list(season_dummies.columns),
        "target_columns": LABELS,
        "n_features": int(X.shape[1]),
    }
    return X, y, scaler, meta


def metrics(target: np.ndarray, pred: np.ndarray, model_name: str = "") -> None:
    """Notebook cell 116 helper :contentReference[oaicite:4]{index=4}"""
    acc = accuracy_score(target, pred)
    # classification_report supports 2D for multi-output
    report = classification_report(target, pred, zero_division=0)
    print(f"MODEL: {model_name}")
    print(f"Accuracy: {acc}")
    print(f"Classification report\n {report}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[LOAD] {path} → shape {df.shape}, columns {list(df.columns)[:5]}…")
    return df


def train_and_evaluate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    model_type: str = "both",
    n_jobs: int = -1,
    random_state: int = 42,
):
    results = {}
    if model_type in ("lr", "both"):
        print("\n[TRAIN] LogisticRegression (MultiOutput)...")
        lr = MultiOutputClassifier(
            LogisticRegression(max_iter=1000, n_jobs=n_jobs if n_jobs != -1 else None),
            n_jobs=n_jobs,
        )
        lr.fit(train_x, train_y)
        pred = lr.predict(test_x)
        try:
            proba = lr.predict_proba(test_x)
        except Exception:
            proba = None
        metrics(test_y, pred, model_name="Logistic Regression")
        results["lr"] = (lr, pred, proba)

    if model_type in ("rf", "both"):
        print("\n[TRAIN] RandomForest (MultiOutput)...")
        rf = MultiOutputClassifier(
            RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
            n_jobs=n_jobs,
        )
        rf.fit(train_x, train_y)
        pred = rf.predict(test_x)
        metrics(test_y, pred, model_name="Random Forest")
        # Feature importance insight (optional)
        try:
            importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
            top_idx = np.argsort(importances)[-5:][::-1]
            print(f"  Top 5 feature indices by mean importance: {top_idx} → {importances[top_idx].round(4)}")
        except Exception:
            pass
        results["rf"] = (rf, pred, None)

    if model_type == "xgb":
        print("\n[TRAIN] XGBoost (MultiOutput)...")
        try:
            from xgboost import XGBClassifier

            xgb = MultiOutputClassifier(
                XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=n_jobs, random_state=random_state),
                n_jobs=n_jobs,
            )
            xgb.fit(train_x, train_y)
            pred = xgb.predict(test_x)
            metrics(test_y, pred, model_name="XGBoost")
            results["xgb"] = (xgb, pred, None)
        except ImportError:
            print("[ERROR] xgboost not installed. pip install xgboost", file=sys.stderr)
            sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train disease screening classifier (ISS Hackathon Track 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--data", type=Path, help="Single CSV (cleaned_balanced or raw participant dataset). Will be split into train/test.")
    src.add_argument("--train", type=Path, help="Pre-split training CSV (e.g., scaled_train.csv)")
    parser.add_argument("--test", type=Path, help="Pre-split test CSV (e.g., scaled_test.csv) — required if --train is used")
    parser.add_argument("--model-type", choices=["lr", "rf", "both", "xgb"], default="rf", help="Which model(s) to train")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio when using --data")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--scale", action="store_true", help="Fit StandardScaler on numeric cols (needed for raw participant dataset; not needed for scaled_*.csv)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallelism for sklearn")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "model", help="Output directory for joblib artifacts")
    parser.add_argument("--poly", action="store_true", help="Apply PolynomialFeatures (degree 2) as in notebook cells 128-133)")
    args = parser.parse_args()

    # Validate
    if args.train and not args.test:
        parser.error("--test is required when --train is provided")
    if not args.train and not args.data:
        # Try auto-detect relative to repo
        candidates = [
            Path("/Users/shdev/Downloads/scaled_train.csv"),
            Path("/Users/shdev/Downloads/track1_cleaned_balanced.csv"),
            Path("/Users/shdev/Downloads/track1_participant_dataset.csv"),
            Path(__file__).parent / "data" / "scaled_train.csv",
            Path(__file__).parent / "data" / "track1_cleaned_balanced.csv",
        ]
        for c in candidates:
            if c.exists():
                if "scaled_train" in c.name:
                    args.train = c
                    args.test = c.parent / "scaled_test.csv"
                    if args.test.exists():
                        print(f"[AUTO] Using train={args.train} test={args.test}")
                        break
                else:
                    args.data = c
                    print(f"[AUTO] Using data={args.data}")
                    break
        if not args.train and not args.data:
            parser.error("No data found. Provide --data or --train/--test explicitly.")

    args.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & preprocess
    # ------------------------------------------------------------------
    scaler: Optional[StandardScaler] = None
    if args.train:
        train_df = load_csv(args.train)
        test_df = load_csv(args.test)
        # Determine if scaling needed — scaled files already contain scaled floats, so skip
        fit_scaler = args.scale
        train_x, train_y, scaler, meta_train = preprocess(train_df, scaler=None, fit_scaler=fit_scaler)
        # For test, reuse scaler if fitted, else pass None
        test_x, test_y, _, meta_test = preprocess(test_df, scaler=scaler, fit_scaler=False)
        # Align state columns if train/test have different state categories
        # (hstack already sorted; if mismatch, would need reindex — preprocess sorts, so pad)
        if train_x.shape[1] != test_x.shape[1]:
            print(f"[WARN] Feature mismatch train {train_x.shape[1]} vs test {test_x.shape[1]}. Check state dummies.", file=sys.stderr)
            # Attempt to align by re-processing combined state vocab
            # Simplest: union columns already sorted, so missing state in one split just means zero col absent
            # For rigor, re-create with union: not needed if both splits contain all 12 states (they do in 150k)
            pass
        print(f"[PREP] train_x {train_x.shape}, train_y {train_y.shape} | test_x {test_x.shape}, test_y {test_y.shape}")
    else:
        df = load_csv(args.data)  # type: ignore
        # Optional: report class balance like notebook cells 5-6
        if "target_diagnosis" in df.columns:
            print("[INFO] Class distribution:")
            print(df["target_diagnosis"].value_counts().to_string())
            print(f"[INFO] Shape: {df.shape}")
        # Split before preprocess to avoid leakage (scaler fit on train only)
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=args.random_state, stratify=df["target_diagnosis"] if "target_diagnosis" in df.columns else None
        )
        print(f"[SPLIT] train {train_df.shape} | test {test_df.shape} (test_size={args.test_size})")
        train_x, train_y, scaler, meta_train = preprocess(train_df, scaler=None, fit_scaler=args.scale)
        test_x, test_y, _, meta_test = preprocess(test_df, scaler=scaler, fit_scaler=False)
        print(f"[PREP] train_x {train_x.shape}, train_y {train_y.shape} | test_x {test_x.shape}, test_y {test_y.shape}")

    # Optional polynomial features (notebook cells 128-133 — blows up dim, use with caution)
    if args.poly:
        from sklearn.preprocessing import PolynomialFeatures

        print("[POLY] Applying PolynomialFeatures (degree=2, include_bias=False)...")
        poly = PolynomialFeatures(include_bias=False)
        train_x = poly.fit_transform(train_x)
        test_x = poly.transform(test_x)
        print(f"[POLY] New shape train_x {train_x.shape}, test_x {test_x.shape}")
        joblib.dump(poly, args.model_dir / "poly.joblib")
        print(f"[SAVE] {args.model_dir / 'poly.joblib'}")

    # ------------------------------------------------------------------
    # Train & evaluate
    # ------------------------------------------------------------------
    results = train_and_evaluate(train_x, train_y, test_x, test_y, model_type=args.model_type, n_jobs=args.n_jobs, random_state=args.random_state)

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    # Labels encoding (notebook cell 126)
    labels_path = args.model_dir / "labels_encoding.joblib"
    joblib.dump(LABELS_ENCODING, labels_path)
    print(f"[SAVE] {labels_path}  → {LABELS_ENCODING}")

    # Also save JSON for non-joblib consumers
    (args.model_dir / "labels_encoding.json").write_text(json.dumps(LABELS_ENCODING, indent=2))
    
    for name, (model, pred, proba) in results.items():
        model_path = args.model_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"[SAVE] {model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Save scaler if used
    if scaler is not None:
        scaler_path = args.model_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"[SAVE] {scaler_path}")

    # Save metadata for inference
    meta = {
        "months_map": MONTHS_MAP,
        "labels": LABELS,
        "labels_encoding": LABELS_ENCODING,
        "numeric_columns": meta_train["numeric_columns"],
        "sex_columns": meta_train["sex_columns"],
        "state_columns": meta_train["state_columns"],
        "season_columns": meta_train["season_columns"],
        "target_columns": meta_train["target_columns"],
        "n_features": meta_train["n_features"],
        "scaled": args.scale or (args.train is not None and "scaled" in str(args.train)),
        "poly": args.poly,
        "model_type": args.model_type,
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
    }
    meta_path = args.model_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[SAVE] {meta_path}")

    # ------------------------------------------------------------------
    # Archive model dir as zip (notebook cell 127: shutil.make_archive)
    # ------------------------------------------------------------------
    try:
        import shutil

        zip_path = shutil.make_archive(str(args.model_dir), "zip", str(args.model_dir))
        print(f"[ZIP] {zip_path}")
    except Exception as e:
        print(f"[WARN] Could not create zip: {e}")

    print("\n[DONE] Training complete. Artifacts in", args.model_dir.resolve())
    print("  Inference example:")
    print("    import joblib, pandas as pd")
    print(f"    rf = joblib.load('{args.model_dir / 'rf.joblib'}')")
    print(f"    labels = joblib.load('{args.model_dir / 'labels_encoding.joblib'}')")
    print("    # preprocess a single row with same preprocess() then rf.predict(X)")


if __name__ == "__main__":
    main()
