"""
ProxyPatient — Stage 4: Stratified Train/Val/Test Split + Aggregates
=====================================================================
Source  : NFHS-5 India (2019-21), MoHFW/IIPS, DHS Program
Author  : P1 (Data & Backend Lead)
Inputs  : processed/combined_clean.parquet
Outputs :
    processed/train.parquet
    processed/val.parquet
    processed/test.parquet           <- LOCKED. Touch only for final validation.
    docs/aggregates.json             <- Safe summary stats for P3 and P4 (frontend charts)
    docs/split_report.md

RULES:
  - Stratified split on elevated_glucose_proxy (outcome).
  - Rows where outcome is NaN are split separately (no leakage).
  - Fixed seed = 42. Never change after splits are created.
  - Test set must NOT be used until final model validation.
  - Aggregates: coarse bins only. Suppress any cell with n < 30 (set to null).
  - Never print raw rows or individual-level data.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE         = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
PROC_DIR     = os.path.join(BASE, "processed")
DOCS_DIR     = os.path.join(BASE, "docs")
INPUT_FILE   = os.path.join(PROC_DIR, "combined_clean.parquet")

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
MIN_CELL     = 30          # suppress aggregates with n < MIN_CELL
OUTCOME_COL  = "elevated_glucose_proxy"
THRESHOLD    = 200         # mg/dL (from config.yaml)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_rate(df, outcome_col=OUTCOME_COL, min_cell=MIN_CELL):
    """
    Return {n, rate, suppressed} for a group.
    If n < min_cell, rate is suppressed (None).
    """
    n = int(df[outcome_col].notna().sum())
    if n < min_cell:
        return {"n": n, "rate": None, "suppressed": True}
    rate = float(df[outcome_col].mean())
    return {"n": n, "rate": round(rate, 6), "suppressed": False}


def aggregate_by(df, group_col, outcome_col=OUTCOME_COL, min_cell=MIN_CELL):
    """Return dict of {group_value: {n, rate, suppressed}} for one grouping variable."""
    result = {}
    for val, grp in df.groupby(group_col, dropna=True):
        result[str(val)] = safe_rate(grp, outcome_col, min_cell)
    return result


def aggregate_cross(df, col1, col2, outcome_col=OUTCOME_COL, min_cell=MIN_CELL):
    """Return nested dict for a 2-way cross-tab."""
    result = {}
    for v1, grp1 in df.groupby(col1, dropna=True):
        result[str(v1)] = {}
        for v2, grp2 in grp1.groupby(col2, dropna=True):
            result[str(v1)][str(v2)] = safe_rate(grp2, outcome_col, min_cell)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1/5] Loading combined_clean.parquet...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Loaded: {len(df):,} rows")

    # ── SPLIT ─────────────────────────────────────────────────────────────
    print("[2/5] Performing stratified train/val/test split...")

    # Add stable row ID before any splitting — survives reset_index
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index

    # Separate rows with known vs unknown outcome to avoid leakage
    df_known   = df[df[OUTCOME_COL].notna()].copy()
    df_unknown = df[df[OUTCOME_COL].isna()].copy()

    print(f"  Rows with known outcome   : {len(df_known):,}")
    print(f"  Rows with unknown outcome : {len(df_unknown):,}")

    # Stratified split on known-outcome rows
    y = df_known[OUTCOME_COL].astype(int)

    # Step 1: Split off test (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df_known, y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=SEED
    )

    # Step 2: Split trainval into train (70%) and val (15%)
    # Val ratio relative to trainval: 0.15 / 0.85 = 0.1765
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        stratify=y_trainval,
        random_state=SEED
    )

    # Distribute unknown-outcome rows into train/val only (NOT test)
    # Split unknown rows 70/(70+15) = 82.4% train, 17.6% val
    if len(df_unknown) > 0:
        unk_train, unk_val = train_test_split(
            df_unknown,
            test_size=val_relative,
            random_state=SEED
        )
        X_train = pd.concat([X_train, unk_train], ignore_index=True)
        X_val   = pd.concat([X_val,   unk_val],   ignore_index=True)

    # Shuffle each split
    X_train = X_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    X_val   = X_val.sample(frac=1,   random_state=SEED).reset_index(drop=True)
    X_test  = X_test.sample(frac=1,  random_state=SEED).reset_index(drop=True)

    # Save splits
    X_train.to_parquet(os.path.join(PROC_DIR, "train.parquet"), index=False)
    X_val.to_parquet(  os.path.join(PROC_DIR, "val.parquet"),   index=False)
    X_test.to_parquet( os.path.join(PROC_DIR, "test.parquet"),  index=False)

    print(f"  Train : {len(X_train):,} rows  | outcome rate: {X_train[OUTCOME_COL].mean():.4f}")
    print(f"  Val   : {len(X_val):,}  rows  | outcome rate: {X_val[OUTCOME_COL].mean():.4f}")
    print(f"  Test  : {len(X_test):,} rows  | outcome rate: {X_test[OUTCOME_COL].mean():.4f}")
    print("  [REMINDER] Test set is now LOCKED. Do not use until final validation.")

    # Verify stratification preserved outcome rate
    base_rate = float(df_known[OUTCOME_COL].mean())
    strat_ok  = all(
        abs(split[OUTCOME_COL].mean() - base_rate) < 0.01
        for split in [X_train, X_val, X_test]
        if split[OUTCOME_COL].notna().sum() > 0
    )
    print(f"  Stratification check (rate within 1pp of base {base_rate:.4f}): {'[OK]' if strat_ok else '[WARN]'}")

    # ── AGGREGATES ─────────────────────────────────────────────────────────
    # IMPORTANT: All aggregates computed from FULL dataset (not train only).
    # These are population-representative stats for the frontend reference profiles.
    # No raw rows. No cell with n < 30.
    print("[3/5] Computing aggregates from full dataset...")

    agg = {
        "_meta": {
            "source":         "NFHS-5 India (2019-21)",
            "outcome":        "elevated glucose (proxy)",
            "threshold_mg_dl": THRESHOLD,
            "min_cell_size":  MIN_CELL,
            "note":           "Cells with n < 30 are suppressed (rate = null). Never show suppressed cells in UI.",
            "generated":      datetime.now().isoformat(),
        },

        # Overall
        "overall": safe_rate(df),

        # By sex
        "by_sex": {
            "female": safe_rate(df[df["sex"] == 0]),
            "male":   safe_rate(df[df["sex"] == 1]),
        },

        # By age band
        "by_age_band": aggregate_by(df, "age_band"),

        # By residence
        "by_residence": aggregate_by(df, "residence"),

        # By wealth quintile
        "by_wealth_quintile": aggregate_by(df, "wealth_quintile"),

        # By education
        "by_education": aggregate_by(df, "education"),

        # By tobacco use
        "by_tobacco": {
            "no_tobacco": safe_rate(df[df["any_tobacco"] == 0]),
            "tobacco":    safe_rate(df[df["any_tobacco"] == 1]),
        },

        # By alcohol use
        "by_alcohol": {
            "no_alcohol": safe_rate(df[df["alcohol"] == 0]),
            "alcohol":    safe_rate(df[df["alcohol"] == 1]),
        },

        # By state (for map/state-level reference)
        "by_state": aggregate_by(df, "state"),

        # Cross-tabs for the Explore and Scenarios pages
        "sex_by_residence":     aggregate_cross(df, "sex",      "residence"),
        "sex_by_age_band":      aggregate_cross(df, "sex",      "age_band"),
        "sex_by_wealth":        aggregate_cross(df, "sex",      "wealth_quintile"),
        "residence_by_wealth":  aggregate_cross(df, "residence","wealth_quintile"),
        "age_by_tobacco":       aggregate_cross(df, "age_band", "any_tobacco"),
        "age_by_alcohol":       aggregate_cross(df, "age_band", "alcohol"),
    }

    # BMI band (women only, since men's file had no BMI variable)
    if "bmi_band" in df.columns:
        agg["by_bmi_band_women"] = aggregate_by(
            df[df["sex"] == 0], "bmi_band"
        )

    agg_path = os.path.join(DOCS_DIR, "aggregates.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"  Saved: {agg_path}")

    # ── ACCEPTANCE CHECKS ──────────────────────────────────────────────────
    print("[4/5] Running acceptance checks...")
    checks = {
        "train_val_test_sum_equals_total": (len(X_train) + len(X_val) + len(X_test)) == len(df),
        "test_is_15pct":  abs(len(X_test) / len(df_known) - TEST_RATIO) < 0.01,
        "stratification_ok": strat_ok,
        "no_overlap_train_test": len(set(X_train["_row_id"]) & set(X_test["_row_id"])) == 0,
        "no_overlap_val_test":   len(set(X_val["_row_id"])   & set(X_test["_row_id"])) == 0,
        "aggregates_saved": os.path.exists(agg_path),
        "overall_rate_non_zero": agg["overall"]["rate"] is not None and agg["overall"]["rate"] > 0,
    }
    all_passed = all(checks.values())
    print(f"  Result: {'ALL PASSED [OK]' if all_passed else 'SOME FAILED [FAIL]'}")
    for k, v in checks.items():
        print(f"    {k}: {'[OK]' if v else '[FAIL]'}")

    # ── REPORT ─────────────────────────────────────────────────────────────
    print("[5/5] Writing split report...")

    overall_rate = agg["overall"]["rate"]
    lines = [
        "# ProxyPatient -- Stage 4: Split + Aggregates Report",
        f"\n**Run date:** {datetime.now().isoformat()}",
        f"**Seed:** {SEED}  |  **Split:** {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}\n",

        "## Split Sizes",
        "| Split | Rows | Outcome Rate |",
        "|---|---|---|",
        f"| Train | {len(X_train):,} | {X_train[OUTCOME_COL].mean():.4f} |",
        f"| Val   | {len(X_val):,}   | {X_val[OUTCOME_COL].mean():.4f} |",
        f"| Test  | {len(X_test):,}  | {X_test[OUTCOME_COL].mean():.4f} |",
        f"\n> **Test set is LOCKED.** Do not load test.parquet until final model validation.\n",

        "## Overall Aggregate",
        f"- Overall elevated glucose (proxy) rate: **{overall_rate*100:.2f}%**",
        f"- Based on {agg['overall']['n']:,} respondents with glucose readings",
        f"- Threshold: >= {THRESHOLD} mg/dL (random capillary glucose)",
        f"- Min cell size for suppression: n >= {MIN_CELL}\n",

        "## Acceptance Checks",
    ]
    for k, v in checks.items():
        lines.append(f"- {'[OK]' if v else '[FAIL]'} `{k}`")

    lines += [
        "\n## Output Files",
        "| File | Description |",
        "|---|---|",
        "| processed/train.parquet | Training split (70%) |",
        "| processed/val.parquet | Validation split (15%) |",
        "| processed/test.parquet | Test split (15%) — LOCKED |",
        "| docs/aggregates.json | Safe population aggregates for P3 and P4 |",
        "\n> Privacy: No raw rows in this report. All cells shown have n >= 30.",
        "\n_Stage 4 complete. P2 can now use train.parquet. P3 can use val.parquet + aggregates.json. P4 can use aggregates.json._"
    ]

    report_path = os.path.join(DOCS_DIR, "split_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    if not all_passed:
        raise RuntimeError("STOP: One or more acceptance checks failed.")

    print("\nStage 4 DONE. All checks passed.")


if __name__ == "__main__":
    main()
