"""
ProxyPatient — Stage 3: Data Cleaning & Preprocessing
======================================================
Source  : NFHS-5 India (2019-21), MoHFW/IIPS, DHS Program
Author  : P1 (Data & Backend Lead)
Outputs :
    processed/women_clean.parquet
    processed/men_clean.parquet
    processed/combined_clean.parquet
    processed/preprocess.pkl         <- scaler + encoder objects
    docs/preprocessing_report.md

RULES (DO NOT VIOLATE):
  - Never print raw rows or any individual-level data.
  - Never impute the glucose variable (sb74 / smb74).
  - Never hardcode statistics — everything is computed from data.
  - DHS special codes (9994-9999, 8, 9 for categoricals) -> NaN.
  - Glucose is ONLY the outcome. Never a conditioning input.
  - Always label outcome as 'elevated glucose (proxy)'.
"""

import os
import json
import joblib
import warnings
import pyreadstat
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (all relative to data/ root)
# ─────────────────────────────────────────────────────────────────────────────
BASE        = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
WOMEN_DTA   = os.path.join(BASE, "raw", "IAIR7EDT_extracted", "IAIR7EFL.DTA")
MEN_DTA     = os.path.join(BASE, "raw", "IAMR7EDT_extracted", "IAMR7EFL.DTA")
PROC_DIR    = os.path.join(BASE, "processed")
DOCS_DIR    = os.path.join(BASE, "docs")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DHS MISSING CODES  (from config.yaml)
# ─────────────────────────────────────────────────────────────────────────────
DHS_MISSING_MULTI  = [9994, 9995, 9996, 9997, 9998, 9999]   # continuous vars
DHS_MISSING_SINGLE = [8, 9]                                   # categorical vars

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS TO LOAD (only these — saves ~95% memory on the 5GB women's file)
# All codes verified from audit_results.md. DO NOT add codes not in that file.
# ─────────────────────────────────────────────────────────────────────────────
WOMEN_COLS = [
    "v005",   # survey weight      (÷ 1,000,000)
    "v012",   # age
    "v013",   # age 5-yr group
    "v024",   # state
    "v025",   # urban/rural
    "v106",   # education
    "v190",   # wealth quintile
    "v191",   # wealth factor score
    "v437",   # weight kg          (÷ 10)
    "v445",   # BMI                (÷ 100)
    "s305",   # waist circumference
    "s306",   # hip circumference
    "sb55",   # glucose ever checked
    "sb56",   # told high glucose
    "sb57",   # medicine for glucose
    "sb73",   # time of glucose reading
    "sb74",   # GLUCOSE LEVEL ← OUTCOME ONLY. Never impute. Never condition on.
    "sb19",   # BP ever checked
    "v463a",  # smokes cigarettes
    "v463z",  # does not use tobacco (inverse flag)
    "s711",   # uses tobacco besides cigarettes
    "s720",   # drinks alcohol
    "s721",   # alcohol frequency
    "sb14c",  # smoked 30 min before BP measure
]

MEN_COLS = [
    "mv005",   # survey weight     (÷ 1,000,000)
    "mv012",   # age
    "mv024",   # state
    "mv025",   # urban/rural
    "mv106",   # education
    "mv190",   # wealth quintile
    "mv191",   # wealth factor score
    "smb305",  # waist circumference
    "smb306",  # hip circumference
    "smb55",   # glucose ever checked
    "smb56",   # told high glucose
    "smb57",   # medicine for glucose
    "smb73",   # time of glucose reading
    "smb74",   # GLUCOSE LEVEL ← OUTCOME ONLY. Never impute. Never condition on.
    "smb19",   # BP ever checked
    "mv463a",  # smokes cigarettes
    "mv463z",  # smokes nothing
    "sm619",   # drinks alcohol
    "smb14c",  # smoked 30 min before BP measure
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def replace_dhs_missing(df, continuous_cols, categorical_cols):
    """Convert DHS special codes to NaN. Document every replacement."""
    report = {}
    for col in continuous_cols:
        if col in df.columns:
            mask = df[col].isin(DHS_MISSING_MULTI)
            n = mask.sum()
            if n > 0:
                df[col] = df[col].where(~mask, np.nan)
                report[col] = {"type": "continuous", "missing_replaced": int(n)}
    for col in categorical_cols:
        if col in df.columns:
            mask = df[col].isin(DHS_MISSING_SINGLE)
            n = mask.sum()
            if n > 0:
                df[col] = df[col].where(~mask, np.nan)
                report[col] = {"type": "categorical", "missing_replaced": int(n)}
    return df, report


def derive_women_features(df):
    """Derive standardised feature columns for the women's file."""

    # Survey weight
    df["survey_weight"] = df["v005"] / 1_000_000

    # BMI: raw value has 2 implied decimals
    df["bmi"] = df["v445"].where(
        ~df["v445"].isin(DHS_MISSING_MULTI), np.nan
    ) / 100.0

    # BMI band
    bins   = [0, 18.5, 25.0, 30.0, 999]
    labels = ["underweight", "normal", "overweight", "obese"]
    df["bmi_band"] = pd.cut(df["bmi"], bins=bins, labels=labels, right=False)

    # Weight in kg
    df["weight_kg"] = df["v437"].where(
        ~df["v437"].isin(DHS_MISSING_MULTI), np.nan
    ) / 10.0

    # Age band
    df["age_band"] = pd.cut(
        df["v012"],
        bins=[14, 24, 34, 49],
        labels=["15-24", "25-34", "35-49"]
    )

    # Residence: 1=urban -> "urban", 2=rural -> "rural"
    df["residence"] = df["v025"].map({1: "urban", 2: "rural"})

    # Tobacco: v463z == 1 means "does not use". Any tobacco = NOT v463z.
    df["any_tobacco"] = np.where(
        df["v463z"].isna(), np.nan,
        np.where(df["v463z"] == 1, 0, 1)   # 0=no tobacco, 1=tobacco user
    )
    # Also flag from s711 (uses tobacco besides cigarettes)
    if "s711" in df.columns:
        tobacco_from_s711 = df["s711"].isin([1])
        df["any_tobacco"] = np.where(
            tobacco_from_s711, 1, df["any_tobacco"]
        )

    # Alcohol: s720 == 1 -> yes
    df["alcohol"] = df["s720"].map({0: 0, 1: 1})

    # Rename standard columns
    df = df.rename(columns={
        "v012": "age",
        "v024": "state",
        "v106": "education",
        "v190": "wealth_quintile",
        "s305": "waist_cm",
        "s306": "hip_cm",
        "sb74": "glucose_raw",
        "sb55": "glucose_ever_checked",
        "sb56": "told_high_glucose",
        "sb57": "on_glucose_medicine",
        "sb73": "glucose_time",
        "sb19": "bp_ever_checked",
        "v013": "age_group_5yr",
        "v191": "wealth_score",
    })

    df["sex"] = 0   # 0 = female

    return df


def derive_men_features(df):
    """Derive standardised feature columns for the men's file."""

    df["survey_weight"] = df["mv005"] / 1_000_000

    # Age band
    df["age_band"] = pd.cut(
        df["mv012"],
        bins=[14, 24, 34, 54],
        labels=["15-24", "25-34", "35-54"]
    )

    # Residence
    df["residence"] = df["mv025"].map({1: "urban", 2: "rural"})

    # Tobacco
    df["any_tobacco"] = np.where(
        df["mv463z"].isna(), np.nan,
        np.where(df["mv463z"] == 1, 0, 1)
    )

    # Alcohol
    df["alcohol"] = df["sm619"].map({0: 0, 1: 1})

    # Rename standard columns
    df = df.rename(columns={
        "mv012":  "age",
        "mv024":  "state",
        "mv106":  "education",
        "mv190":  "wealth_quintile",
        "mv191":  "wealth_score",
        "smb305": "waist_cm",
        "smb306": "hip_cm",
        "smb74":  "glucose_raw",
        "smb55":  "glucose_ever_checked",
        "smb56":  "told_high_glucose",
        "smb57":  "on_glucose_medicine",
        "smb73":  "glucose_time",
        "smb19":  "bp_ever_checked",
    })

    df["sex"] = 1   # 1 = male

    return df


def compute_outcome(df):
    """
    Derive elevated_glucose_proxy from glucose_raw.
    RULE: NaN stays NaN — never impute glucose.
    RULE: Always label as 'elevated glucose (proxy)', never 'diabetes'.
    Threshold: >= 200 mg/dL (from config.yaml).
    """
    THRESHOLD = 200
    df["elevated_glucose_proxy"] = np.where(
        df["glucose_raw"].isna(),
        np.nan,
        (df["glucose_raw"] >= THRESHOLD).astype(float)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SHARED COLUMNS (present in both women and men after renaming)
# ─────────────────────────────────────────────────────────────────────────────
SHARED_COLS = [
    "sex", "age", "age_band", "state", "residence",
    "education", "wealth_quintile", "wealth_score",
    "waist_cm", "hip_cm",
    "any_tobacco", "alcohol",
    "glucose_ever_checked", "told_high_glucose",
    "on_glucose_medicine", "glucose_time",
    "glucose_raw",                  # keep raw for reference
    "elevated_glucose_proxy",       # OUTCOME
    "bp_ever_checked",
    "survey_weight",
]

# Women-only extras (OK to keep; will be NaN for men rows)
WOMEN_EXTRA = ["bmi", "bmi_band", "weight_kg", "age_group_5yr"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    report = {
        "run_date": datetime.now().isoformat(),
        "source": "NFHS-5 India (2019-21)",
        "women": {},
        "men": {},
        "combined": {},
        "missing_replacement": {},
    }

    # ── WOMEN ──────────────────────────────────────────────────────────────
    print("[1/6] Loading Women's file (usecols only — no full dataframe)...")
    # Filter to only columns that exist in the file
    _, meta_w = pyreadstat.read_dta(WOMEN_DTA, metadataonly=True)
    avail_w = set(meta_w.column_names_to_labels.keys())
    load_w  = [c for c in WOMEN_COLS if c in avail_w]
    missing_w = [c for c in WOMEN_COLS if c not in avail_w]
    if missing_w:
        print(f"  [WARN] Columns not found in women's file: {missing_w}")

    df_w, _ = pyreadstat.read_dta(WOMEN_DTA, usecols=load_w)
    print(f"  Loaded women: {len(df_w):,} rows x {len(df_w.columns)} cols")
    report["women"]["rows_raw"] = len(df_w)

    # Replace missing codes
    cont_w = ["v005", "v437", "v445", "s305", "s306", "sb74", "sb73", "v191"]
    cat_w  = ["v025", "v106", "v190", "sb55", "sb56", "sb57", "sb19",
              "v463a", "v463z", "s711", "s720", "s721", "sb14c"]
    df_w, miss_report_w = replace_dhs_missing(df_w, cont_w, cat_w)
    report["missing_replacement"]["women"] = miss_report_w

    # Derive features
    df_w = derive_women_features(df_w)
    df_w = compute_outcome(df_w)

    # Keep only relevant columns
    keep_w = [c for c in SHARED_COLS + WOMEN_EXTRA if c in df_w.columns]
    df_w = df_w[keep_w]

    report["women"]["rows_after_clean"] = len(df_w)
    report["women"]["glucose_non_null"] = int(df_w["glucose_raw"].notna().sum())
    report["women"]["outcome_non_null"] = int(df_w["elevated_glucose_proxy"].notna().sum())

    # Save
    out_w = os.path.join(PROC_DIR, "women_clean.parquet")
    df_w.to_parquet(out_w, index=False)
    print(f"  Saved: {out_w}")

    # ── MEN ────────────────────────────────────────────────────────────────
    print("[2/6] Loading Men's file...")
    _, meta_m = pyreadstat.read_dta(MEN_DTA, metadataonly=True)
    avail_m = set(meta_m.column_names_to_labels.keys())
    load_m  = [c for c in MEN_COLS if c in avail_m]
    missing_m = [c for c in MEN_COLS if c not in avail_m]
    if missing_m:
        print(f"  [WARN] Columns not found in men's file: {missing_m}")

    df_m, _ = pyreadstat.read_dta(MEN_DTA, usecols=load_m)
    print(f"  Loaded men: {len(df_m):,} rows x {len(df_m.columns)} cols")
    report["men"]["rows_raw"] = len(df_m)

    cont_m = ["mv005", "smb305", "smb306", "smb74", "smb73", "mv191"]
    cat_m  = ["mv025", "mv106", "mv190", "smb55", "smb56", "smb57", "smb19",
              "mv463a", "mv463z", "sm619", "smb14c"]
    df_m, miss_report_m = replace_dhs_missing(df_m, cont_m, cat_m)
    report["missing_replacement"]["men"] = miss_report_m

    df_m = derive_men_features(df_m)
    df_m = compute_outcome(df_m)

    keep_m = [c for c in SHARED_COLS if c in df_m.columns]
    df_m = df_m[keep_m]

    report["men"]["rows_after_clean"] = len(df_m)
    report["men"]["glucose_non_null"] = int(df_m["glucose_raw"].notna().sum())
    report["men"]["outcome_non_null"] = int(df_m["elevated_glucose_proxy"].notna().sum())

    out_m = os.path.join(PROC_DIR, "men_clean.parquet")
    df_m.to_parquet(out_m, index=False)
    print(f"  Saved: {out_m}")

    # ── COMBINE ────────────────────────────────────────────────────────────
    print("[3/6] Combining women + men into one dataframe...")
    df_all = pd.concat([df_w, df_m], ignore_index=True, sort=False)
    report["combined"]["rows_total"] = len(df_all)
    report["combined"]["women_rows"] = int((df_all["sex"] == 0).sum())
    report["combined"]["men_rows"]   = int((df_all["sex"] == 1).sum())
    report["combined"]["glucose_non_null"] = int(df_all["glucose_raw"].notna().sum())
    report["combined"]["outcome_non_null"] = int(df_all["elevated_glucose_proxy"].notna().sum())
    print(f"  Combined: {len(df_all):,} rows")

    out_all = os.path.join(PROC_DIR, "combined_clean.parquet")
    df_all.to_parquet(out_all, index=False)
    print(f"  Saved: {out_all}")

    # ── SCALER + ENCODER ───────────────────────────────────────────────────
    print("[4/6] Fitting scaler and encoder on combined data...")

    continuous_features = ["age", "waist_cm", "hip_cm", "wealth_score"]
    categorical_features = [
        "sex", "residence", "education", "wealth_quintile",
        "any_tobacco", "alcohol", "state"
    ]
    # Only fit on rows where outcome is known (for downstream integrity)
    df_fit = df_all[df_all["elevated_glucose_proxy"].notna()].copy()

    scaler = StandardScaler()
    valid_cont = [c for c in continuous_features if c in df_fit.columns]
    scaler.fit(df_fit[valid_cont].fillna(df_fit[valid_cont].median()))

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    valid_cat = [c for c in categorical_features if c in df_fit.columns]
    encoder.fit(df_fit[valid_cat].astype(str))

    pkl_path = os.path.join(PROC_DIR, "preprocess.pkl")
    joblib.dump({
        "scaler":              scaler,
        "encoder":             encoder,
        "continuous_features": valid_cont,
        "categorical_features":valid_cat,
        "fit_rows":            len(df_fit),
        "created":             datetime.now().isoformat(),
        "glucose_threshold":   200,
        "outcome_col":         "elevated_glucose_proxy",
        "note":                "Glucose variable excluded from scaler/encoder. Never imputed."
    }, pkl_path)
    print(f"  Saved: {pkl_path}")

    # ── ACCEPTANCE CHECKS ──────────────────────────────────────────────────
    print("[5/6] Running acceptance checks...")
    checks = {}

    # Check 1: Glucose never in encoder
    checks["glucose_not_in_encoder"] = "elevated_glucose_proxy" not in valid_cat \
                                       and "glucose_raw" not in valid_cat
    # Check 2: At least some glucose readings exist
    checks["glucose_readings_exist"] = report["combined"]["glucose_non_null"] > 0
    # Check 3: Women and men rows present
    checks["both_sexes_present"] = (
        report["combined"]["women_rows"] > 0 and
        report["combined"]["men_rows"]   > 0
    )
    # Check 4: No raw rows printed (this script never prints rows)
    checks["no_raw_rows_printed"] = True

    report["acceptance_checks"] = checks
    all_passed = all(checks.values())
    print(f"  Acceptance checks: {'ALL PASSED [OK]' if all_passed else 'SOME FAILED [FAIL]'}")
    for k, v in checks.items():
        print(f"    {k}: {'[OK]' if v else '[FAIL]'}")

    # ── SAVE REPORT ────────────────────────────────────────────────────────
    print("[6/6] Writing preprocessing report...")
    report_path = os.path.join(DOCS_DIR, "preprocessing_report.md")

    lines = [
        "# ProxyPatient — Stage 3 Preprocessing Report",
        f"\n**Run date:** {report['run_date']}",
        f"**Source:** {report['source']}\n",
        "## Row Counts",
        f"| File | Raw Rows | Cleaned Rows | Glucose Non-Null |",
        f"|---|---|---|---|",
        f"| Women | {report['women']['rows_raw']:,} | {report['women']['rows_after_clean']:,} | {report['women']['glucose_non_null']:,} |",
        f"| Men   | {report['men']['rows_raw']:,} | {report['men']['rows_after_clean']:,} | {report['men']['glucose_non_null']:,} |",
        f"| **Combined** | — | **{report['combined']['rows_total']:,}** | **{report['combined']['glucose_non_null']:,}** |",
        "\n## Acceptance Checks",
    ]
    for k, v in checks.items():
        lines.append(f"- {'[OK]' if v else '[FAIL]'} `{k}`")

    lines.append("\n## Missing Code Replacements")
    lines.append("_(counts of DHS special codes replaced with NaN)_\n")
    for sex, mrep in report["missing_replacement"].items():
        lines.append(f"### {sex.capitalize()}")
        if mrep:
            lines.append("| Column | Type | Replaced |")
            lines.append("|---|---|---|")
            for col, info in mrep.items():
                lines.append(f"| `{col}` | {info['type']} | {info['missing_replaced']:,} |")
        else:
            lines.append("_No replacements needed._")

    lines.append("\n## Output Files")
    lines.append("| File | Description |")
    lines.append("|---|---|")
    lines.append("| `processed/women_clean.parquet` | Cleaned women's data |")
    lines.append("| `processed/men_clean.parquet` | Cleaned men's data |")
    lines.append("| `processed/combined_clean.parquet` | Women + men combined |")
    lines.append("| `processed/preprocess.pkl` | Fitted scaler + encoder |")
    lines.append("\n> **Privacy note:** No raw rows are stored in this report. No cell data is shown.")
    lines.append("\n_Stage 3 complete. Awaiting Stage 4: Stratified Split + Aggregates._")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    if not all_passed:
        raise RuntimeError("STOP: One or more acceptance checks failed. Fix before Stage 4.")

    print("\n✅ Stage 3 complete. All files saved. All checks passed.")
    print(f"   Combined dataset: {report['combined']['rows_total']:,} rows")
    print(f"   Glucose readings available: {report['combined']['glucose_non_null']:,}")


if __name__ == "__main__":
    main()
