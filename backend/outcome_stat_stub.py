"""
ProxyPatient — Outcome Stat Stub
==================================
TEMPORARY stub until P3 (Validation Lead) delivers the real outcome_stat.

INTERFACE CONTRACT (do not change):
    outcome_stat(df: pd.DataFrame, rule: dict) -> dict
    Returns: {rate, ci_low, ci_high, n}

To switch to P3's real implementation:
    1. Drop P3's outcome_stat.py into backend/
    2. In main.py, change:
         from backend.outcome_stat_stub import outcome_stat
       to:
         from backend.outcome_stat import outcome_stat

RULES:
  - Never hardcode any rate values.
  - Glucose threshold comes from config (rule dict), never hardcoded.
  - Bootstrap CI computed from generated cohort, never from real data.
  - If a row has missing glucose, exclude from rate calculation — never impute.
"""

import numpy as np
import pandas as pd


def _attach_synthetic_glucose(df: pd.DataFrame, rule: dict,
                               seed: int = 42) -> pd.DataFrame:
    """
    STUB: Attach a synthetic glucose column to the generated DataFrame.
    Uses a Beta distribution shaped by the conditioning context.

    P2's CVAE will generate glucose directly. This stub approximates it
    by drawing from a distribution calibrated to the overall NFHS-5 rate.

    RULE: This is only for stub testing. Real glucose comes from the CVAE.
    """
    rng = np.random.default_rng(seed)
    threshold = rule.get("threshold_mg_dl", 200)

    # Base elevated rate from NFHS-5 Phase A results: ~2.88%
    # Apply crude adjustments based on present condition columns
    base_rate = 0.0288

    if "wealth_quintile" in df.columns:
        # Higher wealth -> slightly higher rate (urban lifestyle)
        wealth_adj = df["wealth_quintile"].map(
            {1: -0.005, 2: -0.002, 3: 0.0, 4: 0.003, 5: 0.006}
        ).fillna(0)
        base_rate = (base_rate + wealth_adj.mean())

    if "bmi_band" in df.columns:
        bmi_map = {"underweight": -0.01, "normal": 0.0,
                   "overweight": 0.01, "obese": 0.025}
        bmi_adj = df["bmi_band"].astype(str).map(bmi_map).fillna(0)
        base_rate = (base_rate + bmi_adj.mean())

    if "any_tobacco" in df.columns:
        tobacco_adj = df["any_tobacco"].fillna(0) * 0.005
        base_rate = base_rate + tobacco_adj.mean()

    base_rate = float(np.clip(base_rate, 0.005, 0.30))

    # Draw binary elevated glucose labels using Bernoulli
    elevated = rng.binomial(1, base_rate, size=len(df)).astype(float)

    # Convert to pseudo mg/dL values (above or below threshold)
    glucose_vals = np.where(
        elevated == 1,
        rng.uniform(threshold, threshold + 100, size=len(df)),   # elevated
        rng.uniform(80, threshold - 1,           size=len(df)),   # normal
    )
    df = df.copy()
    df["glucose_raw"]             = glucose_vals
    df["elevated_glucose_proxy"]  = elevated
    return df


def outcome_stat(df: pd.DataFrame, rule: dict,
                 seed: int = 42, n_bootstrap: int = 1000) -> dict:
    """
    Compute elevated glucose (proxy) rate with bootstrap 95% CI.

    Args:
        df:          Generated synthetic cohort DataFrame.
                     Must contain 'elevated_glucose_proxy' column.
        rule:        {"threshold_mg_dl": 200}
        seed:        random seed for bootstrap
        n_bootstrap: number of bootstrap iterations

    Returns:
        {"rate": float, "ci_low": float, "ci_high": float, "n": int}

    RULES:
      - Never impute missing glucose.
      - Rate computed only from rows where elevated_glucose_proxy is not NaN.
      - Never hardcode rate values.
    """
    # Attach synthetic glucose if not already present (stub only)
    if "elevated_glucose_proxy" not in df.columns:
        df = _attach_synthetic_glucose(df, rule, seed)

    # Use only rows with known outcome — never impute
    outcome_series = df["elevated_glucose_proxy"].dropna()
    n = len(outcome_series)

    if n == 0:
        raise ValueError("No rows with glucose readings in generated cohort.")

    rate = float(outcome_series.mean())

    # Bootstrap 95% CI
    rng = np.random.default_rng(seed)
    boot_rates = []
    vals = outcome_series.values
    for _ in range(n_bootstrap):
        sample = rng.choice(vals, size=n, replace=True)
        boot_rates.append(sample.mean())

    boot_rates = np.array(boot_rates)
    ci_low  = float(np.percentile(boot_rates, 2.5))
    ci_high = float(np.percentile(boot_rates, 97.5))

    return {
        "rate":     round(rate,     6),
        "ci_low":   round(ci_low,   6),
        "ci_high":  round(ci_high,  6),
        "n":        n,
        "rate_pct": round(rate * 100, 4),
    }
