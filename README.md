# Track 1: AI-Assisted Infectious Disease Screening

**Smarter Screening. Earlier Action.**

## Problem

In under-resourced clinics across states like Bauchi, Kano, Rivers, and Borno, a patient presenting with fever is often defaulted to a malaria diagnosis — it's the most common cause, so it's the "safe guess" when lab tests (blood culture, sputum microscopy, RDTs) aren't available.

But fever is also the first symptom of Typhoid Fever, Cholera, COVID-19, Tuberculosis, Meningitis, and Lassa Fever — conditions that need very different treatment, and some of which (Meningitis, Lassa Fever) can turn fatal fast if that assumption goes unchallenged.

**Who's affected:** Nurses, community health workers, and clinicians conducting initial patient assessments.

**Main consequences of the status quo:**
- Delayed diagnosis and treatment, worsening patient outcomes
- Wrong conditions prioritized because symptoms overlap
- Infectious patients not isolated or referred quickly, increasing transmission risk
- Health workers spending more time manually comparing symptoms and records
- Limited tests, medicines, beds, and staff used inefficiently

**Where the current workflow breaks:** deciding which possible infection to prioritize *before* confirmatory lab testing is available.

**Our leverage point:** support the health worker at initial assessment with a consistent, evidence-based screening result that combines the available patient information — not to replace clinical judgment, but to challenge the automatic "it's probably malaria" assumption.

## Solution

An AI-assisted screening tool that takes structured patient information — symptoms, vitals, exposure history, and basic lab values — and predicts the most likely infectious disease among nine classes (Malaria, Typhoid Fever, Cholera, COVID-19, Tuberculosis, Meningitis, Lassa Fever, Acute Respiratory Infection, Healthy).

**System flow:** Patient Input → Preprocessing → ML Model → Prediction → Screening Result

## Dataset

150,000 patient records (`track1_participant_dataset.csv`), 34 columns spanning:

- **Symptoms & duration:** fever, cough, sore throat, headache, vomiting, diarrhea, rash, neck stiffness, weight loss, fatigue, days of symptoms
- **Exposure & history:** mosquito exposure, unsafe water, TB contact, recent travel, pregnancy, vaccination status
- **Clinical measurements:** age, BMI, temperature, heart rate, respiratory rate, oxygen saturation (SpO2), systolic/diastolic blood pressure, hemoglobin, white blood cell count, platelets
- **Demographics:** state, month, sex, season
- **Label:** `target_diagnosis`

Built to reflect real conditions in under-resourced clinics where lab access is limited.

## How the Data Was Cleaned

We loaded all 150,000 rows and checked for missing values, duplicate patient IDs, and duplicate rows — found none. We verified every `patient_id` matched the expected `P#######` format, and that text columns (state, month, sex, season, diagnosis) had no whitespace or casing inconsistencies. We confirmed all sixteen binary symptom/exposure columns contained only 0s and 1s, and that categorical fields held only their expected values with no stray typos. We checked logical consistency across fields — e.g. no pregnancy records for male patients, no implausible ages — and reviewed numeric ranges for age, vitals, and bloodwork, finding them clinically plausible overall.

The one real issue found was 199 rows with a BMI under 12, which is physiologically implausible even for severe malnutrition. Rather than deleting those diagnosis-labeled rows, we added a `bmi_flag` column to mark them for optional filtering. The dataset required almost no repair beyond that flag.

## Exploratory Findings

- **Class imbalance:** Malaria accounts for ~32% of records (48,194), while Meningitis and Lassa Fever each make up under 2% (~3,000 records each).
- **Label-leak features:** `neck_stiffness` is present in ~99% of Meningitis cases and under 15% elsewhere; `tb_contact` is present in ~100% of Tuberculosis cases and ~4% elsewhere; `weight_loss` shows the same near-perfect split for Tuberculosis. These are strong, almost deterministic predictors for their respective classes.
- **Weak/uninformative features:** `mosquito_exposure`, `unsafe_water`, `recent_travel`, and `vaccinated` show nearly identical rates across every diagnosis (e.g. mosquito exposure sits around 54–57% regardless of whether the diagnosis is Malaria or not), suggesting they carry little predictive signal despite intuitive relevance.
- Symptom clusters otherwise map cleanly to disease groups (e.g. fever + headache + vomiting + diarrhea tracking Typhoid/Cholera-type infections; cough + sore throat tracking respiratory infections), consistent with a dataset built with strong underlying structure.

## Preprocessing Pipeline

To avoid data leakage, steps were applied in this order:

1. **Clean** — whitespace trimming, `bmi_flag` added (see above). Cleaning is applied before splitting since it only touches row-level values, not cross-row statistics.
2. **Split** — stratified 80/20 train/test split on `target_diagnosis`, done *before* any balancing or scaling, so the test set stays a true held-out sample.
3. **Balance (train set only)** — a hybrid of undersampling and SMOTE:
   - Classes above the mean class size (Malaria, Typhoid Fever, Acute Respiratory Infection) were randomly undersampled down to it.
   - Classes below the mean (Healthy, COVID-19, Tuberculosis, Cholera, Lassa Fever, Meningitis) were oversampled up to it using **SMOTENC** — a SMOTE variant that correctly handles the mix of continuous vitals and categorical/binary symptom columns in this dataset (plain SMOTE would treat binary columns as continuous and produce invalid interpolated values).
   - The test set was left untouched, preserving the real-world class distribution for honest evaluation.
   - Synthetic rows are tagged with a `SYN` patient ID prefix so they can be identified or excluded later.
4. **Scale** — numeric columns standardized with `StandardScaler`, fit on the balanced training set only and applied to both train and test, so no test-set statistics leak into the fitting process.

**Note:** because `neck_stiffness`, `weight_loss`, and `tb_contact` are near-perfect single-class indicators, SMOTE's nearest-neighbor interpolation for Meningitis/Tuberculosis mostly reinforces that existing pattern rather than adding real feature diversity — balancing fixes the class-count imbalance but does not address the underlying label-leakage risk from these features.

## Modeling Approach

Redundant columns were dropped and categorical columns were label-encoded. Two models were evaluated:

- **Logistic Regression** — baseline
- **Random Forest**
- **XGBoost** — benchmark

## Results

| Metric | Score |
|---|---|
| Weighted Avg Precision | 0.96 |
| Weighted Avg Recall | 0.93 |
| Weighted Avg F1-score | 0.94 |
| Macro Avg F1-score | 0.91 |

The gap between the weighted F1 (0.94) and macro F1 (0.91) reflects the class imbalance in the underlying data — performance is weaker on rarer classes like Lassa Fever than the weighted average suggests.

## Real-World Impact

- Reduces triage time, which is critical in low-resource, remote facilities where lab access is limited
- Challenges the default "it's probably malaria" assumption with structured, evidence-based screening
- Supports faster isolation and referral for high-risk infectious cases
- Helps stretched clinical staff prioritize which patients need urgent attention

## Limitations

- Prototype trained on available (partly synthetic) data — not clinically validated
- Lower accuracy on rarer conditions (e.g., Lassa Fever), where more real training data is needed
- Several exposure/history features (`mosquito_exposure`, `unsafe_water`, `recent_travel`, `vaccinated`) showed little predictive value in this dataset and may need better real-world data to be useful
- Does not replace clinical judgment or lab-confirmed diagnosis

## Required Statements

- **Our biggest assumption:** the symptoms, exposure history, vital signs, and basic laboratory values available at intake are sufficient to support useful early screening.
- **Our biggest risk:** a health worker may over-trust a confident model result and delay escalation for a serious condition.
- **Our AI must never:** replace clinical judgment or be used as the sole basis for diagnosis, treatment, or discharge.
- **Before real-world deployment, we must prove:** the tool is safe, calibrated, fair across relevant patient groups, and improves screening decisions compared with standard care or a simple checklist.

## Future Improvements

- Expand training data, especially for underrepresented diseases like Lassa Fever
- Clinical validation with real patient data and healthcare partners
- Deploy as a lightweight mobile/offline tool for remote clinics

## Repository Contents

| File | Description |
|---|---|
| `track1_participant_dataset.csv` | Original raw dataset (150,000 rows) |
| `clean_and_examine.py` | Cleaning, examination, split, balancing, and scaling pipeline |
| `track1_participant_dataset_cleaned.csv` | Cleaned dataset (whitespace trimmed, `bmi_flag` added) |
| `track1_train_balanced_scaled.csv` | Training set: balanced (SMOTENC + undersampling) and scaled |
| `track1_test_scaled.csv` | Held-out test set: original class distribution, scaled using training-set statistics |

### Requirements

```
pandas
scikit-learn
imbalanced-learn
```

### Usage

```bash
pip install pandas scikit-learn imbalanced-learn
python clean_and_examine.py track1_participant_dataset.csv
```
