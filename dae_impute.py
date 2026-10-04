"""
ProxyPatient — dae_impute.py
============================
F4: DAE inference function + benchmark vs median imputation.
Provides impute_dae(df) that works on any DataFrame (test split, new data).
Also benchmarks DAE vs median on masked val data.
"""

import os, joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BASE     = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
PROC_DIR = os.path.join(BASE, "processed")


# ── Same model architecture as train_dae_v2.py ────────────────────────────────

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        enc_layers = []
        d = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        dec_layers.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


NEVER_IMPUTE = {"glucose_raw", "elevated_glucose_proxy", "bmi", "weight_kg",
                "height_cm", "systolic_avg", "diastolic_avg", "hypertension",
                "on_bp_medication", "bmi_measured", "bp_measured"}


def _normalise(df, cont_cols, cat_cols, fit_stats):
    result = []
    for col in cont_cols:
        vals = df[col].fillna(fit_stats["cont_medians"][col]).astype(float)
        norm = (vals - fit_stats["cont_medians"][col]) / (fit_stats["cont_stds"][col] or 1.0)
        result.append(norm.values.reshape(-1, 1))
    for col in cat_cols:
        cm = fit_stats["cat_maps"][col]
        mode_enc = cm.get(fit_stats["cat_modes"][col], 0)
        vals = df[col].fillna(fit_stats["cat_modes"][col]).astype(str)
        enc  = vals.map(lambda x: cm.get(x, mode_enc)).astype(float)
        n_cats = max(len(cm), 1)
        result.append((enc / n_cats).values.reshape(-1, 1))
    return torch.tensor(np.hstack(result), dtype=torch.float32)


def _denorm_cont(norm_vals, col, fit_stats):
    return norm_vals * fit_stats["cont_stds"][col] + fit_stats["cont_medians"][col]


def impute_dae(df: pd.DataFrame,
               weights_path: str = None,
               fit_stats_path: str = None) -> pd.DataFrame:
    """
    Load DAE v2 weights and impute ONLY the allowed sporadic-missing covariates.
    NEVER imputes: glucose, outcome, bmi, weight_kg, height_cm, BP, hypertension.
    Works on any DataFrame that has the required columns.

    Args:
        df:              Input DataFrame (test split, new data, etc.)
        weights_path:    Path to dae_weights_v2.pt (default: processed/dae_weights_v2.pt)
        fit_stats_path:  Path to dae_fit_stats_v2.pkl (default: processed/dae_fit_stats_v2.pkl)

    Returns:
        DataFrame with sporadic-missing covariates filled.
    """
    if weights_path is None:
        weights_path   = os.path.join(PROC_DIR, "dae_weights_v2.pt")
    if fit_stats_path is None:
        fit_stats_path = os.path.join(PROC_DIR, "dae_fit_stats_v2.pkl")

    # Load
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    fit_stats = joblib.load(fit_stats_path)

    model = DenoisingAutoencoder(ck["input_dim"], ck["hidden_dims"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    cont_cols   = ck["cont_cols"]
    cat_cols    = ck["cat_cols"]
    impute_cols = ck["impute_cols"]
    n_cont      = len(cont_cols)

    X = _normalise(df, cont_cols, cat_cols, fit_stats)
    with torch.no_grad():
        X_hat = model(X).numpy()

    df_out = df.copy()

    for i, col in enumerate(cont_cols):
        if col in NEVER_IMPUTE or col not in impute_cols:
            continue
        missing = df_out[col].isna()
        if missing.any():
            raw = _denorm_cont(X_hat[missing.values, i], col, fit_stats)
            df_out.loc[missing, col] = raw

    for i, col in enumerate(cat_cols):
        if col in NEVER_IMPUTE or col not in impute_cols:
            continue
        missing = df_out[col].isna()
        if missing.any():
            n_cats  = len(fit_stats["cat_maps"][col])
            enc_val = X_hat[missing.values, n_cont + i] * n_cats
            rounded = np.clip(np.round(enc_val), 0, n_cats - 1).astype(int)
            inv_map = {v: k for k, v in fit_stats["cat_maps"][col].items()}
            imputed_vals = [inv_map.get(r, list(inv_map.values())[-1]) for r in rounded]
            # Convert back to original dtype (e.g. '1.0' string → 1.0 float)
            orig_dtype = df_out[col].dtype
            try:
                imputed_vals = pd.to_numeric(imputed_vals, errors='raise').astype(orig_dtype)
            except (ValueError, TypeError):
                pass  # genuine string categories — leave as-is
            df_out.loc[missing, col] = imputed_vals

    # Recompute bmi_band where bmi_measured=1
    if "bmi" in df_out.columns and "bmi_measured" in df_out.columns:
        measured = df_out["bmi_measured"] == 1
        if measured.any():
            df_out.loc[measured, "bmi_band"] = pd.cut(
                df_out.loc[measured, "bmi"],
                bins=[0, 18.5, 25.0, 30.0, 999],
                labels=["underweight", "normal", "overweight", "obese"],
                right=False,
            )

    return df_out


def benchmark_dae_vs_median(*args, **kwargs):
    raise RuntimeError("Circular legacy benchmark retired; use python -m models.check_dae_benchmark")


if __name__ == "__main__":
    raise SystemExit("Library only. Development benchmark: python -m models.check_dae_benchmark. Test imputation requires the explicit final stage.")
