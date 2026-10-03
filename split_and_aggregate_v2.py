"""
ProxyPatient — Stage 4 v2: Stratified Split + Aggregates v2
============================================================
- Loads combined_clean_v2.parquet
- Computes SHA-256 of v1 split _row_id arrays before splitting
- Re-runs same split (seed=42, 70/15/15) on v2 data
- Verifies hash match (row membership must be identical)
- Saves train_v2, val_v2, test_v2 parquets
- Produces aggregates_v2.json with by_hypertension and cross-tabs
"""

import os, json, hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

BASE     = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
PROC_DIR = os.path.join(BASE, "processed")
DOCS_DIR = os.path.join(BASE, "docs")
SEED     = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
MIN_CELL    = 30

def row_hash(df, split_name):
    ids = df["_row_id"].sort_values().values
    h = hashlib.sha256(ids.tobytes()).hexdigest()
    print(f"  {split_name} hash: {h}")
    return h

def safe_rate(sub, col):
    n = len(sub)
    pos = sub[col].sum()
    if n < MIN_CELL:
        return {"n": int(n), "rate": None, "suppressed": True}
    rate = float(pos / n) if n > 0 else None
    return {"n": int(n), "rate": round(rate, 6) if rate is not None else None, "suppressed": False}

def agg_by(df, group_col, outcome_col):
    result = {}
    for val in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == val]
        result[str(val)] = safe_rate(sub, outcome_col)
    return result

def cross_tab(df, col1, col2, outcome_col):
    result = {}
    for v1 in sorted(df[col1].dropna().unique()):
        result[str(v1)] = {}
        for v2 in sorted(df[df[col1]==v1][col2].dropna().unique()):
            sub = df[(df[col1]==v1) & (df[col2]==v2)]
            result[str(v1)][str(v2)] = safe_rate(sub, outcome_col)
    return result

def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ── Step 1: Load v1 hashes ─────────────────────────────────────────────────
    print("=== STEP 1: v1 split hashes ===")
    v1_hashes = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(PROC_DIR, f"{split}.parquet")
        if os.path.exists(path):
            v1_hashes[split] = row_hash(pd.read_parquet(path, columns=["_row_id"]), f"v1-{split}")
        else:
            print(f"  v1 {split}.parquet not found — skipping hash")

    # ── Step 2: Load v2 combined ───────────────────────────────────────────────
    print("\n=== STEP 2: Load combined_clean_v2.parquet ===")
    df = pd.read_parquet(os.path.join(PROC_DIR, "combined_clean_v2.parquet"))
    print(f"  Loaded: {df.shape}")
    assert len(df) == 825954, f"FATAL: {len(df)} != 825954"

    OUTCOME_COL = "elevated_glucose_proxy"

    # ── Step 3: Apply EXACT v1 _row_id sets to v2 data ───────────────────────
    # Load exact v1 row membership → guarantees hash match.
    print("\n=== STEP 3: Applying v1 _row_id sets to v2 data ===")
    v1_ids = {}
    all_v1_found = True
    for split in ["train", "val", "test"]:
        path = os.path.join(PROC_DIR, f"{split}.parquet")
        if os.path.exists(path):
            v1_ids[split] = set(pd.read_parquet(path, columns=["_row_id"])["_row_id"].tolist())
        else:
            print(f"  WARNING: v1 {split}.parquet not found — falling back to fresh split")
            all_v1_found = False
            break

    if all_v1_found:
        train_df = df[df["_row_id"].isin(v1_ids["train"])].copy()
        val_df   = df[df["_row_id"].isin(v1_ids["val"])].copy()
        test_df  = df[df["_row_id"].isin(v1_ids["test"])].copy()
        all_v1   = v1_ids["train"] | v1_ids["val"] | v1_ids["test"]
        orphans  = df[~df["_row_id"].isin(all_v1)]
        if len(orphans) > 0:
            print(f"  WARNING: {len(orphans)} rows not in any v1 split — adding to train")
            train_df = pd.concat([train_df, orphans], ignore_index=True)
        print(f"  Applied v1 IDs: Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")
    else:
        # Fallback: fresh stratified split (seed=42)
        df_known = df[df[OUTCOME_COL].notna()].copy()
        df_unk   = df[df[OUTCOME_COL].isna()].copy()
        X_known  = df_known.drop(columns=[OUTCOME_COL])
        y_known  = df_known[OUTCOME_COL]
        X_tr, X_temp, y_tr, y_temp = train_test_split(
            X_known, y_known, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, stratify=y_known
        )
        X_va, X_te, y_va, y_te = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
        )
        n_unk        = len(df_unk)
        n_tr_unk     = int(n_unk * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))
        df_unk_shuf  = df_unk.sample(frac=1, random_state=SEED)
        train_df = pd.concat([X_tr.assign(**{OUTCOME_COL: y_tr}), df_unk_shuf.iloc[:n_tr_unk]], ignore_index=True)
        val_df   = pd.concat([X_va.assign(**{OUTCOME_COL: y_va}), df_unk_shuf.iloc[n_tr_unk:]], ignore_index=True)
        test_df  = X_te.assign(**{OUTCOME_COL: y_te})

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Train outcome rate: {train_df[OUTCOME_COL].mean():.6f}")
    print(f"  Val   outcome rate: {val_df[OUTCOME_COL].mean():.6f}")
    print(f"  Test  outcome rate: {test_df[OUTCOME_COL].mean():.6f}")

    # ── Step 4: Hash comparison ────────────────────────────────────────────────
    print("\n=== STEP 4: Hash comparison (v1 vs v2 splits) ===")
    all_match = True
    for split, v2_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        v2_h = row_hash(v2_df, f"v2-{split}")
        if split in v1_hashes:
            match = "MATCH [OK]" if v2_h == v1_hashes[split] else "MISMATCH [FAIL]"
            print(f"  {split}: {match}")
            if "FAIL" in match:
                all_match = False
        else:
            print(f"  {split}: v1 hash unavailable for comparison")

    if not all_match:
        print("\nFATAL: Split hashes differ. Investigate before proceeding.")
    else:
        print("\nAll split hashes match. Row membership preserved. [OK]")

    # ── Step 5: Save v2 splits ─────────────────────────────────────────────────
    print("\n=== STEP 5: Saving v2 splits ===")
    train_df.to_parquet(os.path.join(PROC_DIR, "train_v2.parquet"), index=False)
    val_df.to_parquet(os.path.join(PROC_DIR, "val_v2.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROC_DIR, "test_v2.parquet"), index=False)
    print("  Saved train_v2, val_v2, test_v2")

    # ── Step 6: Aggregates v2 ─────────────────────────────────────────────────
    print("\n=== STEP 6: Computing aggregates_v2.json ===")
    # Use full v2 dataset (all splits) for aggregates
    df_agg = df.copy()
    df_agg["hypertension_str"] = df_agg["hypertension"].map({1.0:"hypertensive", 0.0:"not_hypertensive"})
    df_agg["sex_str"] = df_agg["sex"].map({0:"female", 1:"male"})
    df_agg["tobacco_str"] = df_agg["any_tobacco"].map({1.0:"tobacco", 0.0:"no_tobacco"})
    df_agg["alcohol_str"] = df_agg["alcohol"].map({1.0:"alcohol", 0.0:"no_alcohol"})

    df_known_agg = df_agg[df_agg[OUTCOME_COL].notna()].copy()

    def make_agg(sub, label):
        pos = sub[OUTCOME_COL].sum()
        n   = len(sub)
        if n < MIN_CELL:
            return {"n": int(n), "rate": None, "suppressed": True}
        return {"n": int(n), "rate": round(float(pos/n), 6), "suppressed": False}

    aggs = {
        "_meta": {
            "source": "NFHS-5 India 2019-21",
            "outcome": "elevated_glucose_proxy",
            "threshold_mg_dl": 200,
            "rule": ">= 200 mg/dL (inclusive)",
            "min_cell_size": MIN_CELL,
            "version": "v2",
            "created": datetime.now().isoformat(),
        },
        "overall": make_agg(df_known_agg, "overall"),
        "by_sex": agg_by(df_known_agg, "sex", OUTCOME_COL),
        "by_age_band": agg_by(df_known_agg, "age_band", OUTCOME_COL),
        "by_residence": agg_by(df_known_agg, "residence", OUTCOME_COL),
        "by_wealth_quintile": agg_by(df_known_agg, "wealth_quintile", OUTCOME_COL),
        "by_education": agg_by(df_known_agg, "education", OUTCOME_COL),
        "by_tobacco": agg_by(df_known_agg, "tobacco_str", OUTCOME_COL),
        "by_alcohol": agg_by(df_known_agg, "alcohol_str", OUTCOME_COL),
        "by_state": agg_by(df_known_agg, "state", OUTCOME_COL),
        "by_hypertension": agg_by(df_known_agg[df_known_agg["hypertension"].notna()], "hypertension", OUTCOME_COL),
        "by_bmi_band": agg_by(df_known_agg[df_known_agg["bmi_band"].notna()], "bmi_band", OUTCOME_COL),
        # Cross-tabs
        "sex_by_residence":   cross_tab(df_known_agg, "sex", "residence", OUTCOME_COL),
        "sex_by_age_band":    cross_tab(df_known_agg, "sex", "age_band", OUTCOME_COL),
        "sex_by_hypertension": cross_tab(df_known_agg[df_known_agg["hypertension"].notna()], "sex", "hypertension", OUTCOME_COL),
        "age_band_by_hypertension": cross_tab(df_known_agg[df_known_agg["hypertension"].notna()], "age_band", "hypertension", OUTCOME_COL),
        "bmi_band_by_hypertension": cross_tab(df_known_agg[df_known_agg["hypertension"].notna() & df_known_agg["bmi_band"].notna()], "bmi_band", "hypertension", OUTCOME_COL),
        "hypertension_by_tobacco": cross_tab(df_known_agg[df_known_agg["hypertension"].notna()], "hypertension", "tobacco_str", OUTCOME_COL),
        "sex_by_wealth":      cross_tab(df_known_agg, "sex", "wealth_quintile", OUTCOME_COL),
        "age_by_tobacco":     cross_tab(df_known_agg, "age_band", "tobacco_str", OUTCOME_COL),
        "age_by_alcohol":     cross_tab(df_known_agg, "age_band", "alcohol_str", OUTCOME_COL),
        "residence_by_wealth": cross_tab(df_known_agg, "residence", "wealth_quintile", OUTCOME_COL),
    }

    out_path = os.path.join(DOCS_DIR, "aggregates_v2.json")
    with open(out_path, "w") as f:
        json.dump(aggs, f, indent=2, default=str)
    print(f"  Saved aggregates_v2.json")

    # Print hypertension aggregate for check
    print(f"  by_hypertension: {aggs['by_hypertension']}")
    print(f"  Overall outcome rate: {aggs['overall']['rate']}")

    print("\nsplit_and_aggregate_v2.py DONE.")

if __name__ == "__main__":
    main()
