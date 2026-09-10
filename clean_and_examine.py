"""
Data cleaning + examination for track1_participant_dataset.csv

Usage:
    python clean_and_examine.py path/to/track1_participant_dataset.csv
"""

import sys
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else 'track1_participant_dataset.csv'
OUTPUT_PATH = 'track1_participant_dataset_cleaned.csv'
BALANCED_OUTPUT_PATH = 'track1_participant_dataset_balanced.csv'

BINARY_COLS = [
    'pregnant', 'fever', 'cough', 'sore_throat', 'headache', 'vomiting',
    'diarrhea', 'rash', 'neck_stiffness', 'weight_loss', 'fatigue',
    'mosquito_exposure', 'unsafe_water', 'tb_contact', 'recent_travel',
    'vaccinated',
]
NUMERIC_COLS = [
    'age', 'bmi', 'days_symptoms', 'temperature_c', 'heart_rate',
    'resp_rate', 'spo2', 'sbp', 'dbp', 'hemoglobin', 'wbc', 'platelets',
]
STRING_COLS = ['patient_id', 'state', 'month', 'sex', 'season', 'target_diagnosis']
SYMPTOM_COLS = [c for c in BINARY_COLS if c not in ('pregnant', 'vaccinated')]
CATEGORICAL_COLS = ['state', 'month', 'sex', 'season', 'bmi_flag'] + BINARY_COLS


def load(path):
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. STRUCTURAL DATA QUALITY CHECKS
# ---------------------------------------------------------------------------
def run_quality_checks(df):
    print("=" * 70)
    print("STRUCTURAL CHECKS")
    print("=" * 70)

    print("Shape:", df.shape)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    print(missing if len(missing) else "  none")

    print("\nDuplicate patient_id:", df['patient_id'].duplicated().sum())
    print("Fully duplicate rows:", df.duplicated().sum())

    print("\nPatient ID format (expected P#######):")
    bad_ids = (~df['patient_id'].str.match(r'^P\d{7}$')).sum()
    print(f"  {bad_ids} rows do not match expected format")

    print("\nWhitespace issues in text columns:")
    for c in STRING_COLS:
        n = (df[c] != df[c].str.strip()).sum()
        print(f"  {c}: {n}")

    print("\nCategorical value counts:")
    for c in ['state', 'month', 'sex', 'season', 'target_diagnosis']:
        print(f"\n--- {c} ({df[c].nunique()} unique) ---")
        print(df[c].value_counts(dropna=False))

    print("\nBinary columns with values outside {0,1}:")
    any_bad = False
    for c in BINARY_COLS:
        bad_vals = set(df[c].unique()) - {0, 1}
        if bad_vals:
            any_bad = True
            print(f"  {c}: {bad_vals}")
    if not any_bad:
        print("  none — all binary columns are clean 0/1")

    print("\nNumeric ranges:")
    print(df[NUMERIC_COLS].describe().T)

    print("\nLogical consistency checks:")
    print("  Pregnant males:", ((df['sex'] == 'Male') & (df['pregnant'] == 1)).sum())
    print("  Pregnant & age > 60:", ((df['pregnant'] == 1) & (df['age'] > 60)).sum())
    print("  Pregnant & age < 12:", ((df['pregnant'] == 1) & (df['age'] < 12)).sum())

    n_bmi_low = (df['bmi'] < 12).sum()
    print(f"\n  Rows with physiologically implausible BMI (<12): {n_bmi_low}")


# ---------------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------------
def clean(df):
    df = df.copy()

    # Trim any stray whitespace in text fields (no-op if already clean).
    for c in STRING_COLS:
        df[c] = df[c].str.strip()

    # Flag (rather than drop) physiologically implausible BMI values.
    # Rows are kept because they carry real diagnosis labels; drop or impute
    # downstream if your pipeline requires it.
    df['bmi_flag'] = (df['bmi'] < 12).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. EXAMINATION / EDA
# ---------------------------------------------------------------------------
def run_examination(df):
    print("\n" + "=" * 70)
    print("EXAMINATION")
    print("=" * 70)

    print("\nClass balance (target_diagnosis):")
    print(df['target_diagnosis'].value_counts())

    print("\nMean vitals by diagnosis:")
    print(df.groupby('target_diagnosis')[['temperature_c', 'spo2', 'heart_rate']].mean().round(1))

    print("\nSymptom/exposure prevalence by diagnosis (rate 0-1):")
    print(df.groupby('target_diagnosis')[SYMPTOM_COLS].mean().round(2).T)

    print("\nPotential label-leak features (near-perfect single-class indicators):")
    for c in SYMPTOM_COLS:
        rates = df.groupby('target_diagnosis')[c].mean()
        if rates.max() > 0.9 and rates.nsmallest(len(rates) - 1).max() < 0.15:
            top_class = rates.idxmax()
            print(f"  {c}: ~{rates.max():.0%} present in {top_class}, "
                  f"<15% elsewhere")


# ---------------------------------------------------------------------------
# 4. BALANCING (hybrid: undersample majority + SMOTENC oversample minority)
# ---------------------------------------------------------------------------
def balance(df, target_count=None, random_state=42):
    """
    Brings every target_diagnosis class to the same size:
      - classes above target_count are randomly undersampled down to it
      - classes below target_count are oversampled up to it with SMOTENC
        (SMOTE variant that handles the mixed numeric/categorical columns
        in this dataset instead of treating everything as continuous)

    Default target_count is the mean of the current class counts, i.e. a
    middle ground between full undersampling (shrink everything to the
    smallest class) and full oversampling (grow everything to the largest).
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ('patient_id', 'target_diagnosis')]
    X = df[feature_cols]
    y = df['target_diagnosis']

    counts = y.value_counts()
    if target_count is None:
        target_count = int(round(counts.mean()))
    print(f"\nBalancing target per class: {target_count}")

    # Step 1: cap oversized classes.
    under_strategy = {cls: min(n, target_count) for cls, n in counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
    X_under, y_under = rus.fit_resample(X, y)

    # Step 2: synthesize up undersized classes with SMOTENC.
    remaining_counts = y_under.value_counts()
    over_strategy = {cls: target_count for cls, n in remaining_counts.items() if n < target_count}

    if over_strategy:
        cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL_COLS if c in X.columns]
        smote = SMOTENC(categorical_features=cat_idx, sampling_strategy=over_strategy,
                         k_neighbors=5, random_state=random_state)
        X_bal, y_bal = smote.fit_resample(X_under, y_under)
    else:
        X_bal, y_bal = X_under, y_under

    df_bal = X_bal.copy()
    df_bal['target_diagnosis'] = y_bal.values

    # Re-round any columns SMOTENC interpolated that should stay integer-valued.
    int_cols = ['age', 'days_symptoms', 'heart_rate', 'resp_rate', 'spo2',
                'sbp', 'dbp', 'platelets'] + BINARY_COLS + ['bmi_flag']
    for c in int_cols:
        if c in df_bal.columns:
            df_bal[c] = df_bal[c].round().astype(int)

    # Give every row (real or synthetic) a fresh sequential patient_id.
    df_bal.insert(0, 'patient_id', [f"P{i+1:07d}" for i in range(len(df_bal))])

    print("\nClass counts after balancing:")
    print(df_bal['target_diagnosis'].value_counts())

    return df_bal


if __name__ == '__main__':
    df = load(INPUT_PATH)
    run_quality_checks(df)
    df_clean = clean(df)
    run_examination(df_clean)

    df_clean.to_csv(OUTPUT_PATH, index=False, lineterminator='\n')
    print(f"\nCleaned file written to: {OUTPUT_PATH}")

    df_balanced = balance(df_clean)
    df_balanced.to_csv(BALANCED_OUTPUT_PATH, index=False, lineterminator='\n')
    print(f"Balanced file written to: {BALANCED_OUTPUT_PATH}")
