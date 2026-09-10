import pandas as pd

SYMPTOM_COLUMNS = [
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
]

RAW_NUMERIC_FEATURES = [
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
    *SYMPTOM_COLUMNS,
    "mosquito_exposure",
    "unsafe_water",
    "tb_contact",
    "recent_travel",
    "vaccinated",
    "hemoglobin",
    "wbc",
    "platelets",
]

ENGINEERED_FEATURES = ["symptom_count", "fever_deviation", "shock_index", "pulse_pressure"]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = ["state", "month", "sex", "season"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["symptom_count"] = df[SYMPTOM_COLUMNS].sum(axis=1)
    df["fever_deviation"] = df["temperature_c"] - 37.0
    df["shock_index"] = df["heart_rate"] / df["sbp"]
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    return df
