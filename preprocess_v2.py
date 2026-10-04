"""
ProxyPatient — Stage 3 v2: Preprocessing (BP + Men's BMI + bmi_measured)
=========================================================================
Changes vs v1:
  + Blood pressure variables (3 readings, averaged 2nd+3rd)
  + hypertension column (JNC-7 / NFHS-5 rule)
  + on_bp_medication column
  + bp_measured flag
  + Men's height_cm, weight_kg, bmi from household file (hb2, hb3, hb40)
  + Women's height_cm from household file (ha3)
  + bmi_measured flag (1 = actually measured, 0 = not)
  + bmi_band recomputed from measured values only
  - Removed: age_group_5yr (unreliable), wealth_score (redundant)
  Row count: MUST remain 825,954 (LEFT joins only, no row drops)
  Seed: 42
RULES: Never print rows. Glucose never imputed. No commits of data.
"""

import os, sys, hashlib, joblib, pyreadstat
import pandas as pd
import numpy as np
from datetime import datetime

BASE     = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
RAW_DIR  = os.path.join(BASE, "raw")
PROC_DIR = os.path.join(BASE, "processed")
DOCS_DIR = os.path.join(BASE, "docs")

WOMEN_DTA = os.path.join(RAW_DIR, "IAIR7EDT_extracted", "IAIR7EFL.DTA")
MEN_DTA   = os.path.join(RAW_DIR, "IAMR7EDT_extracted", "IAMR7EFL.DTA")
HH_DTA    = os.path.join(RAW_DIR, "IAPR7EDT_extracted", "IAPR7EFL.DTA")
SEED      = 42

# ── DHS special/missing codes ─────────────────────────────────────────────────
MISS_CONT = set(range(9994, 10000)) | set(range(99994, 100000))
MISS_CAT  = {8, 9}
MISS_BP   = {994, 995, 996, 997, 998, 999}   # BP readings 0-300 mmHg valid
MISS_HH   = set(range(9990, 10000)) | set(range(99990, 100000)) | {9990}
GLUCOSE_THRESHOLD = 200

# ── Column lists ──────────────────────────────────────────────────────────────
WOMEN_COLS = [
    # Demographics
    "v001", "v002", "v003", "v005", "v012", "v013", "v024", "v025", "v106", "v190",
    # Anthropometry (women)
    "v437", "v445",
    # Waist/hip
    "s305", "s306",
    # Glucose
    "sb74", "sb73", "sb55", "sb56", "sb57",
    # BP history + medication (v1)
    "sb19",
    # Tobacco / alcohol
    "v463a", "v463z", "s711", "s720",
    # NEW: BP readings (3 sets)
    "sb18s", "sb18d",   # 1st reading
    "sb25s", "sb25d",   # 2nd reading
    "sb29s", "sb29d",   # 3rd reading
    "sb21",             # on BP medicine
    # Self-reported hypertension (reference only)
    "s728b",
]

MEN_COLS = [
    # Demographics
    "mv005", "mv012", "mv024", "mv025", "mv106", "mv190",
    # Waist/hip
    "smb305", "smb306",
    # Glucose
    "smb74", "smb73", "smb55", "smb56", "smb57",
    # BP history
    "smb19",
    # Tobacco / alcohol
    "mv463a", "mv463z", "sm619",
    # Linkage keys to household file
    "mv001", "mv002", "mv003",
    # NEW: BP readings
    "smb18s", "smb18d",
    "smb25s", "smb25d",
    "smb29s", "smb29d",
    "smb21",
    # Self-reported hypertension (reference only)
    "sm627b",
]

HH_COLS = ["hv001", "hv002", "hvidx", "hb2", "hb3", "hb40", "ha3"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def replace_missing(s, codes):
    return s.apply(lambda x: np.nan if (pd.notna(x) and x in codes) else x)

def safe_load(dta_path, requested_cols):
    _, meta = pyreadstat.read_dta(dta_path, metadataonly=True)
    avail = set(meta.column_names_to_labels.keys())
    usecols = [c for c in requested_cols if c in avail]
    missing = [c for c in requested_cols if c not in avail]
    if missing:
        print(f"  WARNING: cols not in file: {missing}")
    df, _ = pyreadstat.read_dta(dta_path, usecols=usecols)
    return df

def bp_avg(r2, r3, r1=None):
    """
    Average of 2nd and 3rd reading.
    Fallback: if both missing, use 1st.
    Returns NaN if no valid reading available.
    """
    r2 = pd.to_numeric(r2, errors="coerce")
    r3 = pd.to_numeric(r3, errors="coerce")
    avg = np.where(
        r2.notna() & r3.notna(),
        (r2.values + r3.values) / 2,
        np.where(r2.notna(), r2.values, r3.values)
    )
    if r1 is not None:
        r1 = pd.to_numeric(r1, errors="coerce")
        avg = np.where(np.isnan(avg), r1.values, avg)
    return pd.Series(avg, index=r2.index, dtype="float64")

def compute_hypertension(sys_avg, dia_avg, on_med):
    """
    hypertension = 1 if sys>=140 OR dia>=90 OR on_med==1
    NaN if BP missing AND on_med != 1
    on_med==1 always gives hypertension=1
    """
    result = np.full(len(sys_avg), np.nan)
    has_bp = sys_avg.notna().values | dia_avg.notna().values
    bp_hyp = ((sys_avg.fillna(0) >= 140) | (dia_avg.fillna(0) >= 90)).values
    on_med_1 = (on_med == 1).values

    # Where BP measured: use BP rule
    result[has_bp] = bp_hyp[has_bp].astype(float)
    # Where on medicine: force 1
    result[on_med_1] = 1.0
    return pd.Series(result, index=sys_avg.index, dtype="float64")

def bp_measured_flag(sys_avg, dia_avg):
    return ((sys_avg.notna()) | (dia_avg.notna())).astype(int)

def derive_bmi_band(bmi_series):
    return pd.cut(
        bmi_series,
        bins=[0, 18.5, 25.0, 30.0, 999],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def validate_women_height(df, minimum_share=None):
    from models.common import load_config
    minimum_share = (load_config().get("preprocessing", {}).get("women_height_min_share", 0.5)
                     if minimum_share is None else minimum_share)
    if not 0 <= minimum_share <= 1:
        raise ValueError("women_height_min_share must be between 0 and 1")
    share = pd.to_numeric(df["height_cm"], errors="coerce").notna().mean()
    if not np.isfinite(share) or share < minimum_share:
        raise ValueError("Women's household height linkage failed: check v001/v002/v003 and ha3; "
                         f"non-missing share must be at least {minimum_share:.0%}")


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ── Load v1 hashes for comparison ─────────────────────────────────────────
    print("[0/7] Computing v1 split hashes for later comparison...")
    v1_hashes = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(PROC_DIR, f"{split}.parquet")
        if os.path.exists(path):
            ids = pd.read_parquet(path, columns=["_row_id"])["_row_id"].sort_values().values
            h = hashlib.sha256(ids.tobytes()).hexdigest()
            v1_hashes[split] = h
            print(f"  v1 {split} hash: {h}")
        else:
            print(f"  v1 {split}.parquet NOT FOUND")
    print()

    # ── 1. Load Women ──────────────────────────────────────────────────────────
    print("[1/7] Loading Women's file (usecols only)...")
    df_w = safe_load(WOMEN_DTA, WOMEN_COLS)
    print(f"  Loaded women: {df_w.shape}")

    # ── 2. Load Men ────────────────────────────────────────────────────────────
    print("[2/7] Loading Men's file...")
    df_m = safe_load(MEN_DTA, MEN_COLS)
    print(f"  Loaded men: {df_m.shape}")

    # ── 3. Load Household (men's + women's anthropometry) ──────────────────────
    print("[3/7] Loading Household file (usecols only)...")
    df_hh = safe_load(HH_DTA, HH_COLS)
    print(f"  Loaded household: {df_hh.shape}")

    # Verify hb40 scale: compute mean after /10 and /100, pick the one in [15,50]
    hb40_raw = pd.to_numeric(df_hh["hb40"], errors="coerce")
    hb40_raw = hb40_raw.apply(lambda x: np.nan if (pd.notna(x) and x in MISS_HH) else x)
    hb40_raw = hb40_raw[hb40_raw > 0]
    mean_div10  = hb40_raw.mean() / 10
    mean_div100 = hb40_raw.mean() / 100
    # DHS standard: hb40 = 1 implied decimal (divide by 10)
    # mean of ~21-27 expected. If div10 gives 15-50 range, use /10
    print(f"  hb40 raw mean={hb40_raw.mean():.1f} | /10={mean_div10:.2f} | /100={mean_div100:.2f}")
    hb40_divisor = 10 if 14 < mean_div10 < 60 else 100
    print(f"  Using hb40 divisor: {hb40_divisor}")

    # Similarly for hb2, hb3 (both 1 decimal = divide by 10)
    hb2_raw = pd.to_numeric(df_hh["hb2"], errors="coerce")
    hb3_raw = pd.to_numeric(df_hh["hb3"], errors="coerce")
    ha3_raw = pd.to_numeric(df_hh.get("ha3", pd.Series(dtype=float)), errors="coerce")

    # ── 4. Process Women ───────────────────────────────────────────────────────
    print("[4/7] Deriving Women's features + BP...")

    # Survey weight
    df_w["survey_weight"] = pd.to_numeric(df_w["v005"], errors="coerce") / 1_000_000

    # Age
    df_w["age"] = replace_missing(pd.to_numeric(df_w["v012"], errors="coerce"), MISS_CONT)

    # Geography
    df_w["state"]          = pd.to_numeric(df_w["v024"], errors="coerce")
    df_w["residence"]      = pd.to_numeric(df_w["v025"], errors="coerce").map({1:"urban", 2:"rural"})
    df_w["education"]      = replace_missing(pd.to_numeric(df_w["v106"], errors="coerce"), MISS_CAT)
    df_w["wealth_quintile"]= replace_missing(pd.to_numeric(df_w["v190"], errors="coerce"), MISS_CAT)

    # BMI (women's file: v445 / 100)
    bmi_w = replace_missing(pd.to_numeric(df_w["v445"], errors="coerce"), MISS_CONT)
    bmi_w = bmi_w.apply(lambda x: np.nan if (pd.notna(x) and x == 0) else x)
    df_w["bmi"] = bmi_w / 100
    df_w["bmi_measured"] = ((df_w["bmi"].notna()) & (df_w["bmi"] > 0)).astype(int)
    df_w["bmi_band"] = derive_bmi_band(df_w["bmi"])

    # Weight (v437 / 10)
    wt_w = replace_missing(pd.to_numeric(df_w["v437"], errors="coerce"), MISS_CONT)
    df_w["weight_kg"] = wt_w.apply(lambda x: np.nan if (pd.notna(x) and x == 0) else x) / 10

    # Waist / Hip
    df_w["waist_cm"] = replace_missing(pd.to_numeric(df_w["s305"], errors="coerce"), MISS_CONT)
    df_w["hip_cm"]   = replace_missing(pd.to_numeric(df_w["s306"], errors="coerce"), MISS_CONT)
    df_w.loc[df_w["waist_cm"] == 0, "waist_cm"] = np.nan
    df_w.loc[df_w["hip_cm"]   == 0, "hip_cm"]   = np.nan

    # Glucose (never imputed)
    gluc_w = replace_missing(pd.to_numeric(df_w["sb74"], errors="coerce"), MISS_CONT)
    df_w["glucose_raw"] = gluc_w.apply(lambda x: np.nan if (pd.notna(x) and x == 0) else x)
    df_w["elevated_glucose_proxy"] = df_w["glucose_raw"].apply(
        lambda x: 1.0 if (pd.notna(x) and x >= GLUCOSE_THRESHOLD) else (
                  0.0 if (pd.notna(x) and x < GLUCOSE_THRESHOLD) else np.nan)
    )
    df_w["glucose_time"]  = replace_missing(pd.to_numeric(df_w["sb73"], errors="coerce"),  MISS_CONT)
    df_w["glucose_ever_checked"] = replace_missing(pd.to_numeric(df_w["sb55"], errors="coerce"), MISS_CAT)
    df_w["told_high_glucose"]    = replace_missing(pd.to_numeric(df_w["sb56"], errors="coerce"), MISS_CAT)
    df_w["on_glucose_medicine"]  = replace_missing(pd.to_numeric(df_w["sb57"], errors="coerce"), MISS_CAT)

    # Tobacco
    v463z = pd.to_numeric(df_w["v463z"], errors="coerce")
    df_w["any_tobacco"] = v463z.apply(lambda x: 0.0 if x == 1 else 1.0 if pd.notna(x) else np.nan)
    has_s711 = "s711" in df_w.columns
    if has_s711:
        s711 = pd.to_numeric(df_w["s711"], errors="coerce")
        df_w.loc[s711 == 1, "any_tobacco"] = 1.0

    # Alcohol
    df_w["alcohol"] = replace_missing(pd.to_numeric(df_w["s720"], errors="coerce"), MISS_CAT)
    df_w["alcohol"] = df_w["alcohol"].apply(lambda x: 1.0 if x == 1 else (0.0 if pd.notna(x) else np.nan))

    # BP: replace special codes first
    for col in ["sb18s","sb18d","sb25s","sb25d","sb29s","sb29d"]:
        if col in df_w.columns:
            s = pd.to_numeric(df_w[col], errors="coerce")
            df_w[col] = s.apply(lambda x: np.nan if (pd.notna(x) and (x in MISS_BP or x == 0)) else x)

    df_w["systolic_avg"]  = bp_avg(df_w.get("sb25s", pd.Series(np.nan, index=df_w.index)),
                                    df_w.get("sb29s", pd.Series(np.nan, index=df_w.index)),
                                    df_w.get("sb18s", pd.Series(np.nan, index=df_w.index)))
    df_w["diastolic_avg"] = bp_avg(df_w.get("sb25d", pd.Series(np.nan, index=df_w.index)),
                                    df_w.get("sb29d", pd.Series(np.nan, index=df_w.index)),
                                    df_w.get("sb18d", pd.Series(np.nan, index=df_w.index)))

    med_w = pd.to_numeric(df_w.get("sb21", pd.Series(np.nan, index=df_w.index)), errors="coerce")
    df_w["on_bp_medication"] = med_w.apply(lambda x: np.nan if pd.isna(x) else (1.0 if x == 1 else 0.0))
    df_w["bp_measured"]  = bp_measured_flag(df_w["systolic_avg"], df_w["diastolic_avg"])
    df_w["hypertension"] = compute_hypertension(df_w["systolic_avg"], df_w["diastolic_avg"], df_w["on_bp_medication"])

    # Self-reported hypertension (reference only)
    sr_w = pd.to_numeric(df_w.get("s728b", pd.Series(np.nan, index=df_w.index)), errors="coerce")
    df_w["self_reported_hypertension"] = sr_w.apply(lambda x: np.nan if pd.isna(x) else float(x))

    df_w["bp_ever_checked"] = replace_missing(pd.to_numeric(df_w.get("sb19", pd.Series(np.nan, index=df_w.index)), errors="coerce"), MISS_CAT)

    # Age band (women 15-49)
    df_w["age_band"] = pd.cut(df_w["age"], bins=[14,24,34,49], labels=["15-24","25-34","35-49"], right=True)

    # Sex
    df_w["sex"] = 0

    # height_cm: try linking to household via v001, v002, v003 -> ha3
    if "ha3" in df_hh.columns:
        ha3 = ha3_raw.apply(lambda x: np.nan if (pd.notna(x) and x in MISS_HH or (pd.notna(x) and x == 0)) else x)
        df_hh_w = df_hh[["hv001","hv002","hvidx","ha3"]].copy()
        df_hh_w["ha3_cm"] = ha3 / 10
        df_hh_w = df_hh_w.rename(columns={"hv001":"v001","hv002":"v002","hvidx":"v003"})
        df_hh_w = df_hh_w.drop_duplicates(subset=["v001","v002","v003"])
        if "v001" in df_w.columns and "v002" in df_w.columns and "v003" in df_w.columns:
            df_w = df_w.merge(df_hh_w[["v001","v002","v003","ha3_cm"]], on=["v001","v002","v003"], how="left")
            df_w = df_w.rename(columns={"ha3_cm":"height_cm"})
        else:
            df_w["height_cm"] = np.nan
    else:
        df_w["height_cm"] = np.nan

    validate_women_height(df_w)

    # Select output columns
    WOMEN_OUT = [
        "sex","age","age_band","state","residence","education","wealth_quintile",
        "waist_cm","hip_cm","any_tobacco","alcohol",
        "glucose_ever_checked","told_high_glucose","on_glucose_medicine",
        "glucose_time","glucose_raw","elevated_glucose_proxy",
        "bp_ever_checked","survey_weight",
        "bmi","bmi_measured","bmi_band","weight_kg","height_cm",
        "systolic_avg","diastolic_avg","bp_measured","on_bp_medication","hypertension",
        "self_reported_hypertension",
    ]
    df_w_out = df_w[[c for c in WOMEN_OUT if c in df_w.columns]].copy()
    df_w_out.insert(0, "_row_id", range(len(df_w_out)))
    df_w_out.to_parquet(os.path.join(PROC_DIR, "women_clean_v2.parquet"), index=False)
    print(f"  Women clean v2: {df_w_out.shape}")

    # ── 5. Process Men ─────────────────────────────────────────────────────────
    print("[5/7] Deriving Men's features + BP + household BMI...")

    df_m["survey_weight"] = pd.to_numeric(df_m["mv005"], errors="coerce") / 1_000_000
    df_m["age"]           = replace_missing(pd.to_numeric(df_m["mv012"], errors="coerce"), MISS_CONT)
    df_m["state"]         = pd.to_numeric(df_m["mv024"], errors="coerce")
    df_m["residence"]     = pd.to_numeric(df_m["mv025"], errors="coerce").map({1:"urban", 2:"rural"})
    df_m["education"]     = replace_missing(pd.to_numeric(df_m["mv106"], errors="coerce"), MISS_CAT)
    df_m["wealth_quintile"]= replace_missing(pd.to_numeric(df_m["mv190"], errors="coerce"), MISS_CAT)

    df_m["waist_cm"] = replace_missing(pd.to_numeric(df_m["smb305"], errors="coerce"), MISS_CONT)
    df_m["hip_cm"]   = replace_missing(pd.to_numeric(df_m["smb306"], errors="coerce"), MISS_CONT)
    df_m.loc[df_m["waist_cm"] == 0, "waist_cm"] = np.nan
    df_m.loc[df_m["hip_cm"]   == 0, "hip_cm"]   = np.nan

    gluc_m = replace_missing(pd.to_numeric(df_m["smb74"], errors="coerce"), MISS_CONT)
    df_m["glucose_raw"] = gluc_m.apply(lambda x: np.nan if (pd.notna(x) and x == 0) else x)
    df_m["elevated_glucose_proxy"] = df_m["glucose_raw"].apply(
        lambda x: 1.0 if (pd.notna(x) and x >= GLUCOSE_THRESHOLD) else (
                  0.0 if (pd.notna(x) and x < GLUCOSE_THRESHOLD) else np.nan)
    )
    df_m["glucose_time"]  = replace_missing(pd.to_numeric(df_m["smb73"], errors="coerce"), MISS_CONT)
    df_m["glucose_ever_checked"] = replace_missing(pd.to_numeric(df_m["smb55"], errors="coerce"), MISS_CAT)
    df_m["told_high_glucose"]    = replace_missing(pd.to_numeric(df_m["smb56"], errors="coerce"), MISS_CAT)
    df_m["on_glucose_medicine"]  = replace_missing(pd.to_numeric(df_m["smb57"], errors="coerce"), MISS_CAT)

    v463z_m = pd.to_numeric(df_m["mv463z"], errors="coerce")
    df_m["any_tobacco"] = v463z_m.apply(lambda x: 0.0 if x == 1 else 1.0 if pd.notna(x) else np.nan)
    df_m["alcohol"] = replace_missing(pd.to_numeric(df_m["sm619"], errors="coerce"), MISS_CAT)
    df_m["alcohol"] = df_m["alcohol"].apply(lambda x: 1.0 if x == 1 else (0.0 if pd.notna(x) else np.nan))

    for col in ["smb18s","smb18d","smb25s","smb25d","smb29s","smb29d"]:
        if col in df_m.columns:
            s = pd.to_numeric(df_m[col], errors="coerce")
            df_m[col] = s.apply(lambda x: np.nan if (pd.notna(x) and (x in MISS_BP or x == 0)) else x)

    df_m["systolic_avg"]  = bp_avg(df_m.get("smb25s", pd.Series(np.nan, index=df_m.index)),
                                    df_m.get("smb29s", pd.Series(np.nan, index=df_m.index)),
                                    df_m.get("smb18s", pd.Series(np.nan, index=df_m.index)))
    df_m["diastolic_avg"] = bp_avg(df_m.get("smb25d", pd.Series(np.nan, index=df_m.index)),
                                    df_m.get("smb29d", pd.Series(np.nan, index=df_m.index)),
                                    df_m.get("smb18d", pd.Series(np.nan, index=df_m.index)))

    med_m = pd.to_numeric(df_m.get("smb21", pd.Series(np.nan, index=df_m.index)), errors="coerce")
    df_m["on_bp_medication"] = med_m.apply(lambda x: np.nan if pd.isna(x) else (1.0 if x == 1 else 0.0))
    df_m["bp_measured"]  = bp_measured_flag(df_m["systolic_avg"], df_m["diastolic_avg"])
    df_m["hypertension"] = compute_hypertension(df_m["systolic_avg"], df_m["diastolic_avg"], df_m["on_bp_medication"])

    sr_m = pd.to_numeric(df_m.get("sm627b", pd.Series(np.nan, index=df_m.index)), errors="coerce")
    df_m["self_reported_hypertension"] = sr_m.apply(lambda x: np.nan if pd.isna(x) else float(x))
    df_m["bp_ever_checked"] = replace_missing(pd.to_numeric(df_m.get("smb19", pd.Series(np.nan, index=df_m.index)), errors="coerce"), MISS_CAT)

    # Age band (men 15-54)
    df_m["age_band"] = pd.cut(df_m["age"], bins=[14,24,34,54], labels=["15-24","25-34","35-54"], right=True)
    df_m["sex"] = 1

    # ── Link household for men's BMI ──────────────────────────────────────────
    print("  Linking household file for men's BMI...")
    df_hh_men = df_hh[["hv001","hv002","hvidx","hb2","hb3","hb40"]].copy()
    df_hh_men["hb2"] = pd.to_numeric(df_hh_men["hb2"], errors="coerce")
    df_hh_men["hb3"] = pd.to_numeric(df_hh_men["hb3"], errors="coerce")
    df_hh_men["hb40"]= pd.to_numeric(df_hh_men["hb40"], errors="coerce")
    for col in ["hb2","hb3","hb40"]:
        df_hh_men[col] = df_hh_men[col].apply(lambda x: np.nan if (pd.notna(x) and x in MISS_HH) else x)
    df_hh_men.loc[df_hh_men["hb2"] == 0,  "hb2"]  = np.nan
    df_hh_men.loc[df_hh_men["hb3"] == 0,  "hb3"]  = np.nan
    df_hh_men.loc[df_hh_men["hb40"] == 0, "hb40"] = np.nan

    # Filter to rows with men's data (hb2 not null)
    df_hh_men_valid = df_hh_men[df_hh_men["hb2"].notna()].copy()
    # Check key uniqueness in HH file
    dup_count = df_hh_men_valid.duplicated(subset=["hv001","hv002","hvidx"]).sum()
    print(f"  HH key duplicates (with men's data): {dup_count:,}")

    df_hh_men_valid["weight_kg_hh"] = df_hh_men_valid["hb2"] / 10
    df_hh_men_valid["height_cm"]    = df_hh_men_valid["hb3"] / 10
    df_hh_men_valid["bmi_hh"]       = df_hh_men_valid["hb40"] / hb40_divisor
    df_hh_men_valid = df_hh_men_valid.drop_duplicates(subset=["hv001","hv002","hvidx"])

    n_men_before = len(df_m)
    df_m = df_m.merge(
        df_hh_men_valid[["hv001","hv002","hvidx","weight_kg_hh","height_cm","bmi_hh"]],
        left_on=["mv001","mv002","mv003"],
        right_on=["hv001","hv002","hvidx"],
        how="left"
    )
    n_men_after = len(df_m)
    matched = df_m["bmi_hh"].notna().sum()
    match_rate = matched / n_men_before * 100
    print(f"  Men rows before/after merge: {n_men_before:,} / {n_men_after:,}")
    print(f"  Men matched with BMI: {matched:,} / {n_men_before:,} ({match_rate:.1f}%)")
    assert n_men_after == n_men_before, f"FATAL: merge changed row count {n_men_before} -> {n_men_after}"

    df_m["bmi"]      = df_m["bmi_hh"]
    df_m["weight_kg"]= df_m["weight_kg_hh"]
    df_m["bmi_measured"] = (df_m["bmi"].notna() & (df_m["bmi"] > 0)).astype(int)
    df_m["bmi_band"] = derive_bmi_band(df_m["bmi"])

    # Aggregate BMI check (men)
    m_bmi_valid = df_m[df_m["bmi_measured"] == 1]["bmi"]
    print(f"  Men BMI (measured only): min={m_bmi_valid.min():.2f} mean={m_bmi_valid.mean():.2f} max={m_bmi_valid.max():.2f}")

    MEN_OUT = [
        "sex","age","age_band","state","residence","education","wealth_quintile",
        "waist_cm","hip_cm","any_tobacco","alcohol",
        "glucose_ever_checked","told_high_glucose","on_glucose_medicine",
        "glucose_time","glucose_raw","elevated_glucose_proxy",
        "bp_ever_checked","survey_weight",
        "bmi","bmi_measured","bmi_band","weight_kg","height_cm",
        "systolic_avg","diastolic_avg","bp_measured","on_bp_medication","hypertension",
        "self_reported_hypertension",
    ]
    df_m_out = df_m[[c for c in MEN_OUT if c in df_m.columns]].copy()
    df_m_out.insert(0, "_row_id", range(len(df_w_out), len(df_w_out) + len(df_m_out)))
    df_m_out.to_parquet(os.path.join(PROC_DIR, "men_clean_v2.parquet"), index=False)
    print(f"  Men clean v2: {df_m_out.shape}")

    # ── 6. Combine ─────────────────────────────────────────────────────────────
    print("[6/7] Combining and saving combined_clean_v2.parquet...")
    df_all = pd.concat([df_w_out, df_m_out], ignore_index=True)
    assert len(df_all) == 825954, f"FATAL: row count {len(df_all)} != 825954"
    df_all.to_parquet(os.path.join(PROC_DIR, "combined_clean_v2.parquet"), index=False)
    print(f"  Combined v2: {df_all.shape}")

    # ── 7. Fit and save preprocess_v2.pkl ─────────────────────────────────────
    print("[7/7] Fitting and saving preprocess_v2.pkl...")
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder

    cont_feats = ["age", "waist_cm", "hip_cm"]
    cat_feats  = ["sex", "residence", "education", "wealth_quintile",
                  "any_tobacco", "alcohol", "state"]

    # Fit only on rows with known glucose
    df_fit = df_all[df_all["elevated_glucose_proxy"].notna()].copy()
    scaler = StandardScaler()
    X_cont = df_fit[cont_feats].fillna(df_fit[cont_feats].median())
    scaler.fit(X_cont)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = df_fit[cat_feats].fillna("missing").astype(str)
    encoder.fit(X_cat)

    pkl = {
        "scaler":               scaler,
        "encoder":              encoder,
        "continuous_features":  cont_feats,
        "categorical_features": cat_feats,
        "fit_rows":             len(df_fit),
        "created":              datetime.now().isoformat(),
        "glucose_threshold":    GLUCOSE_THRESHOLD,
        "outcome_col":          "elevated_glucose_proxy",
        "note":                 "v2: BP added. wealth_score removed. BMI not scaled (use bmi_measured flag). Glucose never imputed.",
        "version":              "2",
    }
    joblib.dump(pkl, os.path.join(PROC_DIR, "preprocess_v2.pkl"))
    print(f"  preprocess_v2.pkl saved. Fit rows: {len(df_fit):,}")

    # ── Acceptance checks ──────────────────────────────────────────────────────
    print("\n=== ACCEPTANCE CHECKS ===")
    print(f"Total rows: {len(df_all):,} (expected 825,954)")
    print(f"Columns: {list(df_all.columns)}")

    # Self-reported hypertension distribution (check for all-zero coding)
    sr_dist = df_all["self_reported_hypertension"].value_counts(dropna=False).to_dict()
    print(f"\nSelf-reported hypertension distribution: {sr_dist}")
    n_sr_1 = df_all["self_reported_hypertension"].eq(1).sum()
    n_sr_0 = df_all["self_reported_hypertension"].eq(0).sum()
    n_sr_nan = df_all["self_reported_hypertension"].isna().sum()
    print(f"  =1 (reports hypertension): {n_sr_1:,}")
    print(f"  =0 (no hypertension):      {n_sr_0:,}")
    print(f"  NaN:                       {n_sr_nan:,}")
    if n_sr_0 > 0 and n_sr_1 == 0:
        print("  WARNING: s728b all zeros — likely missing coded as 0, treat as unreliable reference")
    else:
        print("  s728b appears to have real variation — ok as reference column")

    # Hypertension rates
    for sex_val, label in [(0,"Women"),(1,"Men")]:
        sub = df_all[df_all["sex"]==sex_val]
        htn_rate = sub["hypertension"].mean()
        bp_meas  = sub["bp_measured"].mean()
        bmi_meas = sub["bmi_measured"].mean()
        print(f"\n{label} (n={len(sub):,}):")
        print(f"  hypertension rate : {htn_rate:.4f} ({htn_rate*100:.2f}%)")
        print(f"  bp_measured rate  : {bp_meas:.4f} ({bp_meas*100:.2f}%)")
        print(f"  bmi_measured rate : {bmi_meas:.4f} ({bmi_meas*100:.2f}%)")
        if sub["bmi_measured"].sum() > 0:
            bmi_vals = sub[sub["bmi_measured"]==1]["bmi"]
            print(f"  BMI (measured only): min={bmi_vals.min():.2f} mean={bmi_vals.mean():.2f} max={bmi_vals.max():.2f}")

    print(f"\nOverall hypertension rate: {df_all['hypertension'].mean():.4f}")
    print(f"Overall bp_measured rate : {df_all['bp_measured'].mean():.4f}")
    print(f"NaN counts (key columns):")
    for col in ["glucose_raw","elevated_glucose_proxy","hypertension","bmi","systolic_avg","diastolic_avg"]:
        if col in df_all.columns:
            print(f"  {col}: {df_all[col].isna().sum():,}")

    print("\npreprocess_v2.py DONE.")
    return df_all

if __name__ == "__main__":
    main()
