import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from features import SYMPTOM_COLUMNS

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
LABEL_ENCODER_PATH = Path(__file__).parent / "models" / "label_encoder.joblib"

STATES = [
    "Anambra", "Bauchi", "Benue", "Borno", "Enugu", "FCT",
    "Kaduna", "Kano", "Lagos", "Oyo", "Plateau", "Rivers",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

# Column order the model was trained on: base fields (original dataset order,
# with bmi_flag appended last) followed by one-hot dummies for sex, state, season —
# reverse-engineered from the 46 input features the saved model expects.
BASE_COLUMN_ORDER = [
    "month", "age", "pregnant", "bmi", "days_symptoms", "temperature_c",
    "heart_rate", "resp_rate", "spo2", "sbp", "dbp",
    "fever", "cough", "sore_throat", "headache", "vomiting", "diarrhea",
    "rash", "neck_stiffness", "weight_loss", "fatigue",
    "mosquito_exposure", "unsafe_water", "tb_contact", "recent_travel", "vaccinated",
    "hemoglobin", "wbc", "platelets", "bmi_flag",
]

SYMPTOM_LABELS = {
    "fever": "Fever",
    "cough": "Cough",
    "sore_throat": "Sore throat",
    "headache": "Headache",
    "vomiting": "Vomiting",
    "diarrhea": "Diarrhea",
    "rash": "Rash",
    "neck_stiffness": "Neck stiffness",
    "weight_loss": "Weight loss",
    "fatigue": "Fatigue",
}

def preprocess(data: pd.DataFrame) -> np.ndarray:
    data = data.copy()
    data["month"] = data["month"].map(MONTH_TO_NUM)
    # ASSUMPTION — confirm with teammate: WHO-style abnormal-BMI flag
    # (underweight <18.5 or obese >=30). Adjust the threshold if their
    # definition differs.
    data["bmi_flag"] = ((data["bmi"] < 18.5) | (data["bmi"] >= 30)).astype(int)

    base = data[BASE_COLUMN_ORDER].to_numpy(dtype=float)
    # Reindex to the FULL fixed category list (not just what's present in this
    # row) — get_dummies on a single row only emits columns for categories
    # actually seen in that row, which silently breaks the feature width.
    sex_dummies = (
        pd.get_dummies(data["sex"]).reindex(columns=["Female", "Male"], fill_value=0).to_numpy(dtype=float)
    )
    state_dummies = (
        pd.get_dummies(data["state"]).reindex(columns=STATES, fill_value=0).to_numpy(dtype=float)
    )
    season_dummies = (
        pd.get_dummies(data["season"]).reindex(columns=["Dry", "Rainy"], fill_value=0).to_numpy(dtype=float)
    )

    return np.hstack([base, sex_dummies, state_dummies, season_dummies])

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


st.set_page_config(page_title="Infectious Disease Screening", page_icon="🩺", layout="centered")

with st.sidebar:
    st.markdown("## 🩺 Community Screening")
    st.caption(
        "AI-assisted triage aid for frontline health workers — enter what's "
        "observable at a basic clinic visit, no lab required, to get a ranked "
        "list of likely diagnoses."
    )
    st.divider()
    st.markdown(
        "**Why this exists**\n\n"
        "Fever-presenting diseases (Typhoid, Cholera, Meningitis, Lassa Fever) "
        "often get defaulted to Malaria in low-resource settings. This tool "
        "surfaces the full ranked list so rare, dangerous conditions aren't missed."
    )
    st.divider()
    st.caption("Track 1: AI-Assisted Infectious Disease Screening — ISS_IRCE Hackathon")

st.title("AI-Assisted Infectious Disease Screening")

model, label_encoder = load_model()
if model is None:
    st.warning(
        "No trained model found yet at `models/model.joblib`. "
        "The form below is fully functional — once the model artifact is added, "
        "predictions will appear automatically.",
        icon="⏳",
    )

with st.form("screening_form"):
    tab_demo, tab_vitals, tab_symptoms, tab_history, tab_labs = st.tabs(
        ["Demographics", "Vitals", "Symptoms", "History", "Labs"]
    )

    with tab_demo:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            sex = st.selectbox("Sex", ["Female", "Male"])
        with col2:
            state = st.selectbox("State", STATES)
            month = st.selectbox("Month", MONTHS)
        with col3:
            season = st.selectbox("Season", ["Rainy", "Dry"])
            pregnant = st.checkbox("Pregnant", disabled=(sex == "Male"))

    with tab_vitals:
        col1, col2, col3 = st.columns(3)
        with col1:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
            temperature_c = st.number_input(
                "Temperature (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1
            )
            heart_rate = st.number_input("Heart rate (bpm)", min_value=30, max_value=200, value=80)
        with col2:
            resp_rate = st.number_input("Respiratory rate (breaths/min)", min_value=8, max_value=50, value=18)
            spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=98)
            days_symptoms = st.number_input("Days with symptoms", min_value=0, max_value=90, value=3)
        with col3:
            sbp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=220, value=120)
            dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=78)

    with tab_symptoms:
        symptom_values = {}
        symptom_cols = st.columns(5)
        for i, sym in enumerate(SYMPTOM_COLUMNS):
            with symptom_cols[i % 5]:
                symptom_values[sym] = st.checkbox(SYMPTOM_LABELS[sym], key=f"sym_{sym}")

    with tab_history:
        col1, col2 = st.columns(2)
        with col1:
            mosquito_exposure = st.checkbox("Mosquito exposure")
            unsafe_water = st.checkbox("Unsafe water access")
        with col2:
            tb_contact = st.checkbox("TB contact history")
            recent_travel = st.checkbox("Recent travel")
        vaccinated = st.checkbox("Vaccinated (general status)")

    with tab_labs:
        col1, col2, col3 = st.columns(3)
        with col1:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=13.0, step=0.1)
        with col2:
            wbc = st.number_input("WBC (x10^9/L)", min_value=1.0, max_value=30.0, value=7.5, step=0.1)
        with col3:
            platelets = st.number_input("Platelets (x10^9/L)", min_value=10000, max_value=500000, value=250000, step=1000)

    submitted = st.form_submit_button("Screen patient", use_container_width=True)

if submitted:
    row = {
        "age": age,
        "pregnant": int(pregnant),
        "bmi": bmi,
        "days_symptoms": days_symptoms,
        "temperature_c": temperature_c,
        "heart_rate": heart_rate,
        "resp_rate": resp_rate,
        "spo2": spo2,
        "sbp": sbp,
        "dbp": dbp,
        **{sym: int(val) for sym, val in symptom_values.items()},
        "mosquito_exposure": int(mosquito_exposure),
        "unsafe_water": int(unsafe_water),
        "tb_contact": int(tb_contact),
        "recent_travel": int(recent_travel),
        "vaccinated": int(vaccinated),
        "hemoglobin": hemoglobin,
        "wbc": wbc,
        "platelets": platelets,
        "sex": sex,
        "season": season,
        "state": state,
        "month": month,
                
    }

    data = pd.DataFrame([row])

    if model is None:
        st.info("Inputs captured below — predictions will appear once a trained model is available.")
        st.dataframe(data, use_container_width=True)
    else:
        processed = preprocess(data)

        # model is a MultiOutputClassifier: 9 independent binary Random Forests,
        # one per diagnosis. predict_proba returns a LIST of 9 arrays, each
        # shaped (n_samples, 2) = [P(False), P(True)] — not one multiclass array.
        proba_per_output = model.predict_proba(processed)
        positive_probs = np.array([output_proba[0][1] for output_proba in proba_per_output])

        # label_encoder is a plain {index: diagnosis_name} dict, not a
        # sklearn LabelEncoder — build the class order from its sorted keys.
        classes = [label_encoder[i] for i in sorted(label_encoder)]

        # These are independent one-vs-rest probabilities, not a calibrated
        # softmax, so they don't sum to 1 on their own — normalize for display.
        total = positive_probs.sum()
        normalized = positive_probs / total if total > 0 else positive_probs

        results = (
            pd.Series(normalized, index=classes)
            .sort_values(ascending=False)
            .rename("Probability")
        )

        top_diagnosis = results.index[0]
        top_prob = results.iloc[0]

        rare_diseases = {"Meningitis", "Lassa Fever", "Cholera"}
        rare_hits = {d: results[d] for d in rare_diseases if d in results.index and results[d] > 0.15}

        if rare_hits:
            badge_color, badge_label = "#B3261E", "HIGH — Urgent referral suggested"
        elif top_diagnosis != "Healthy" and top_prob > 0.5:
            badge_color, badge_label = "#B58900", "MODERATE — Clinical follow-up suggested"
        else:
            badge_color, badge_label = "#0E7C7B", "LOW — Routine care"

        st.markdown(
            f"""
            <div style="background-color:{badge_color}1A;border-left:6px solid {badge_color};
                        padding:0.9rem 1rem;border-radius:6px;margin-bottom:1rem;">
                <span style="color:{badge_color};font-weight:700;letter-spacing:0.03em;">
                    RISK LEVEL: {badge_label}
                </span><br/>
                <span style="font-size:1.05rem;">
                    Most likely diagnosis: <b>{top_diagnosis}</b> ({top_prob:.1%})
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Ranked likelihood")
        st.bar_chart(results.head(5))
        st.dataframe(results.map(lambda p: f"{p:.1%}").to_frame(), use_container_width=True)

        if rare_hits:
            st.error(
                "⚠️ Elevated probability for a rare but dangerous condition — consider urgent referral: "
                + ", ".join(f"{d} ({p:.1%})" for d, p in rare_hits.items())
            )
