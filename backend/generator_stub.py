"""
ProxyPatient — Generator Stub
==============================
This is a TEMPORARY stub used until P2 (Model Lead) delivers the real CVAE.

INTERFACE CONTRACT (do not change the function signature):
    generate(condition: dict, n: int, seed: int) -> pd.DataFrame

P2 must deliver a file called generator.py in this same folder that
implements the same generate() signature. To switch from stub to real model:
    1. Drop generator.py into backend/
    2. In main.py, change:
         from backend.generator_stub import generate
       to:
         from backend.generator import generate

NOTE (P2): the real generator (backend/generator.py, enabled with env var
PP_GENERATOR=real) DOES return glucose_raw (mg/dL), elevated_glucose_proxy
(derived from the generated glucose_raw) and is_synthetic=True. Glucose is a
GENERATED variable, never a conditioning input. The v2 data files are in
ORIGINAL units (not scaled). The rule below applies to this stub only.

RULES THIS STUB ENFORCES:
  - glucose_raw and elevated_glucose_proxy are NEVER generated/returned.
  - Output DataFrame never contains raw DHS rows.
  - All output is synthetic (sampled with replacement + noise).
"""

import numpy as np
import pandas as pd

# Glucose columns are NEVER output by the generator
FORBIDDEN_OUTPUT_COLS = [
    "glucose_raw", "elevated_glucose_proxy", "glucose_time",
    "glucose_ever_checked", "told_high_glucose", "on_glucose_medicine",
    "sb74", "smb74", "_row_id"
]

# Glucose is NEVER a conditioning input
FORBIDDEN_INPUT_CONDITIONS = [
    "glucose", "glucose_raw", "elevated_glucose_proxy",
    "sb74", "smb74", "hba1c"
]

# Module-level cache for the imputed combined dataset
_df_cache = None


def _load_base_data():
    """Load the imputed combined dataset once and cache it."""
    global _df_cache
    if _df_cache is None:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "processed", "dae_imputed_combined.parquet")
        _df_cache = pd.read_parquet(path)
        # Drop forbidden columns from cache
        drop = [c for c in FORBIDDEN_OUTPUT_COLS if c in _df_cache.columns]
        _df_cache = _df_cache.drop(columns=drop)
    return _df_cache


def generate(condition: dict, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    STUB: Sample n synthetic rows matching the given condition.
    Uses filtered resampling with Gaussian noise on continuous columns.

    Args:
        condition: dict of conditioning variables (NO glucose allowed)
        n:         cohort size
        seed:      random seed

    Returns:
        pd.DataFrame with n rows. Does NOT contain glucose_raw or outcome column.
        The caller (main.py / outcome_stat) will attach a synthetic glucose column.

    NOTE: Replace this stub with P2's CVAE generator.py when ready.
    """
    # Safety: reject any glucose conditioning attempt
    for key in condition:
        if key in FORBIDDEN_INPUT_CONDITIONS:
            raise ValueError(
                f"FORBIDDEN: '{key}' is the outcome variable and cannot be "
                "used as a conditioning input. ProxyPatient rule violation."
            )

    rng = np.random.default_rng(seed)
    df  = _load_base_data().copy()

    # Filter to matching rows
    if condition.get("sex") is not None:
        sex_val = 0 if condition["sex"] == "female" else 1
        df = df[df["sex"] == sex_val]

    if condition.get("residence") is not None:
        df = df[df["residence"] == condition["residence"]]

    if condition.get("age_band") is not None:
        df = df[df["age_band"].astype(str) == condition["age_band"]]

    if condition.get("wealth_quintile") is not None:
        df = df[df["wealth_quintile"] == int(condition["wealth_quintile"])]

    if condition.get("bmi_band") is not None:
        df = df[df["bmi_band"].astype(str) == condition["bmi_band"]]

    if condition.get("tobacco") is not None:
        df = df[df["any_tobacco"] == int(condition["tobacco"])]

    if condition.get("alcohol") is not None:
        df = df[df["alcohol"] == int(condition["alcohol"])]

    if len(df) < 10:
        raise ValueError(
            f"Condition too restrictive: only {len(df)} matching rows in base data. "
            "Relax one or more conditions."
        )

    # Sample with replacement
    sampled = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    # Add small Gaussian noise to continuous columns to create synthetic variation
    cont_cols = ["age", "waist_cm", "hip_cm", "wealth_score", "bmi", "weight_kg"]
    for col in cont_cols:
        if col in sampled.columns:
            std = sampled[col].std()
            if std > 0:
                noise = rng.normal(0, std * 0.05, size=len(sampled))
                sampled[col] = (sampled[col] + noise).clip(lower=0)

    # Final safety: drop any forbidden columns that may have slipped through
    drop = [c for c in FORBIDDEN_OUTPUT_COLS if c in sampled.columns]
    sampled = sampled.drop(columns=drop)

    return sampled
