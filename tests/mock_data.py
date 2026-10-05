"""
MOCK DATA, not NFHS-5.

make_mock_v2(n, seed) builds an in-memory DataFrame with the exact 31 v2
columns, the category labels from config.yaml and plausible structure, so the
P2 code can be tested without the real survey files. It is never written to
disk and must never be used to produce the shipped model.
"""

import numpy as np
import pandas as pd

from models.common import MOCK_BANNER, load_config

V2_COLUMNS = [
    "_row_id", "sex", "age", "age_band", "state", "residence", "education",
    "wealth_quintile", "waist_cm", "hip_cm", "any_tobacco", "alcohol",
    "glucose_ever_checked", "told_high_glucose", "on_glucose_medicine",
    "glucose_time", "glucose_raw", "elevated_glucose_proxy", "bp_ever_checked",
    "survey_weight", "bmi", "bmi_measured", "bmi_band", "weight_kg", "height_cm",
    "systolic_avg", "diastolic_avg", "bp_measured", "on_bp_medication",
    "hypertension", "self_reported_hypertension",
]


def make_mock_v2(n: int = 5000, seed: int = 0, imputed: bool = True,
                 banner: bool = True) -> pd.DataFrame:
    """MOCK DATA, not NFHS-5. imputed=False leaves sporadic gaps in the
    DAE-imputed columns (waist, hip, tobacco, alcohol, bp_ever_checked)."""
    if banner:
        print(f"*** {MOCK_BANNER} ***")
    cfg = load_config()
    rng = np.random.default_rng(seed)

    sex = (rng.random(n) < 0.15).astype(int)                 # mostly women, as in NFHS-5
    age = np.where(sex == 1, rng.integers(15, 55, n), rng.integers(15, 50, n))
    age_band = np.empty(n, dtype=object)
    for s, key in [(0, "women"), (1, "men")]:
        labels = cfg["whatif_options"]["age_band"][f"{key}_options"]
        for (lo, hi), lab in zip(cfg["age_bands"][key], labels):
            age_band[(sex == s) & (age >= lo) & (age <= hi)] = lab
    state = rng.choice(list(range(1, 26)) + list(range(27, 38)), n)
    residence = np.where(rng.random(n) < 0.3, "urban", "rural")
    wealth = rng.integers(1, 6, n)
    education = np.clip(np.round(rng.normal(1.5 + 0.2 * (wealth - 3), 1.0)), 0, 3)
    tobacco = (rng.random(n) < np.where(sex == 1, 0.40, 0.07)).astype(float)
    alcohol = (rng.random(n) < np.where(sex == 1, 0.20, 0.01)).astype(float)

    height = np.where(sex == 1, rng.normal(164, 7, n), rng.normal(152, 6, n))
    bmi = np.clip(rng.normal(20.5 + 0.08 * (age - 15) + 0.6 * (wealth - 3), 3.5), 13, 45)
    weight = bmi * (height / 100) ** 2
    waist = 40 + 2.0 * bmi + rng.normal(0, 5, n) + 3 * sex
    hip = waist + 8 + rng.normal(0, 4, n) - 4 * sex

    systolic = 105 + 0.6 * (age - 15) + 0.8 * (bmi - 21) + 6 * sex + rng.normal(0, 12, n)
    diastolic = 70 + 0.3 * (age - 15) + 0.5 * (bmi - 21) + rng.normal(0, 8, n)
    on_bp_med = (rng.random(n) < 0.02 + 0.002 * (age - 15)).astype(float)
    hypertension = ((systolic >= 140) | (diastolic >= 90) | (on_bp_med == 1)).astype(float)

    lin = 4.55 + 0.004 * (age - 15) + 0.008 * (bmi - 21) + 0.08 * hypertension + 0.02 * sex
    log_g = lin + rng.normal(0, 0.18, n)
    tail = rng.random(n) < 0.025 + 0.002 * (age - 15) / 10 + 0.02 * hypertension
    log_g = np.where(tail, rng.normal(5.45, 0.25, n), log_g)       # rare high tail
    glucose = np.clip(np.round(np.exp(log_g)), 40, 499)

    bmi_measured = (rng.random(n) < np.where(sex == 1, 0.92, 0.96)).astype(int)
    bp_measured = (rng.random(n) < 0.95).astype(int)
    glucose_known = rng.random(n) < 0.965

    df = pd.DataFrame({
        "_row_id": np.arange(n, dtype=np.int64),
        "sex": sex.astype(np.int64),
        "age": age.astype(np.int64),
        "age_band": age_band,
        "state": state.astype(np.int64),
        "residence": residence,
        "education": education,
        "wealth_quintile": wealth.astype(np.int64),
        "waist_cm": np.round(waist, 1),
        "hip_cm": np.round(hip, 1),
        "any_tobacco": tobacco,
        "alcohol": alcohol,
        "glucose_ever_checked": (rng.random(n) < 0.3).astype(float),
        "told_high_glucose": (rng.random(n) < 0.03).astype(float),
        "on_glucose_medicine": (rng.random(n) < 0.01).astype(float),
        "glucose_time": rng.integers(700, 1900, n).astype(float),
        "glucose_raw": np.where(glucose_known, glucose, np.nan),
        "elevated_glucose_proxy": np.where(glucose_known, (glucose >= cfg["outcome"]["threshold_mg_dl"]).astype(float), np.nan),
        "bp_ever_checked": (rng.random(n) < 0.6).astype(float),
        "survey_weight": rng.uniform(0.2, 3.0, n),
        "bmi": np.where(bmi_measured == 1, np.round(bmi, 2), np.nan),
        "bmi_measured": bmi_measured.astype(np.int64),
        "bmi_band": None,
        "weight_kg": np.where(bmi_measured == 1, np.round(weight, 1), np.nan),
        "height_cm": np.where(bmi_measured == 1, np.round(height, 1), np.nan),
        "systolic_avg": np.where(bp_measured == 1, np.round(systolic), np.nan),
        "diastolic_avg": np.where(bp_measured == 1, np.round(diastolic), np.nan),
        "bp_measured": bp_measured.astype(np.int64),
        "on_bp_medication": on_bp_med,
        "hypertension": np.where((bp_measured == 1) | (on_bp_med == 1), hypertension, np.nan),
        "self_reported_hypertension": (rng.random(n) < 0.08).astype(float),
    })
    names = list(cfg["bmi"]["bands"].keys())
    edges = [cfg["bmi"]["bands"][k][0] for k in names] + [cfg["bmi"]["bands"][names[-1]][1]]
    df["bmi_band"] = pd.cut(df["bmi"], bins=edges, labels=names, right=False)
    if not imputed:
        for col in ["waist_cm", "hip_cm", "any_tobacco", "alcohol", "bp_ever_checked"]:
            df.loc[rng.random(n) < 0.04, col] = np.nan
    assert list(df.columns) == V2_COLUMNS
    df.attrs["MOCK"] = MOCK_BANNER
    return df
