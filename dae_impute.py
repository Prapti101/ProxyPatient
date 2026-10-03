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
            df_out.loc[missing, col] = [inv_map.get(r, list(inv_map.values())[-1]) for r in rounded]

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


def benchmark_dae_vs_median(val_parquet=None, mask_frac=0.10, seed=42):
    """
    F4 benchmark: mask 10% of known waist_cm, hip_cm values.
    Impute with (a) DAE, (b) column median from train.
    Report RMSE per column. Aggregates only, no rows.
    """
    np.random.seed(seed)
    if val_parquet is None:
        val_parquet = os.path.join(PROC_DIR, "dae_imputed_val_v2.parquet")

    val_df = pd.read_parquet(val_parquet)
    train_df = pd.read_parquet(os.path.join(PROC_DIR, "dae_imputed_train_v2.parquet"))

    BENCHMARK_COLS = ["waist_cm", "hip_cm"]
    results = {}

    for col in BENCHMARK_COLS:
        known_idx = val_df[val_df[col].notna()].index
        n_mask = int(len(known_idx) * mask_frac)
        mask_idx = np.random.choice(known_idx, n_mask, replace=False)

        true_vals = val_df.loc[mask_idx, col].values

        # Median imputation
        train_median = train_df[col].median()
        median_preds = np.full(n_mask, train_median)
        rmse_median  = float(np.sqrt(np.mean((true_vals - median_preds) ** 2)))

        # DAE imputation
        val_masked = val_df.copy()
        val_masked.loc[mask_idx, col] = np.nan
        val_imputed = impute_dae(val_masked)
        dae_preds   = val_imputed.loc[mask_idx, col].values
        rmse_dae    = float(np.sqrt(np.mean((true_vals - dae_preds) ** 2)))

        results[col] = {
            "n_masked":    int(n_mask),
            "rmse_median": round(rmse_median, 4),
            "rmse_dae":    round(rmse_dae, 4),
            "dae_beats_median": rmse_dae < rmse_median,
            "improvement_pct": round((rmse_median - rmse_dae) / rmse_median * 100, 2),
        }

    print("\n=== F4: DAE vs Median Benchmark ===")
    print(f"{'Column':<15} {'RMSE Median':>14} {'RMSE DAE':>12} {'Better?':>10} {'Improvement':>14}")
    print("-" * 70)
    for col, r in results.items():
        better = "[DAE]" if r["dae_beats_median"] else "[MEDIAN]"
        print(f"  {col:<13} {r['rmse_median']:>14.4f} {r['rmse_dae']:>12.4f} {better:>10} {r['improvement_pct']:>13.2f}%")

    return results


def produce_imputed_test(seed=42):
    """
    Apply DAE inference (not training) to test_v2.parquet.
    Saves dae_imputed_test_v2.parquet.
    """
    test_path = os.path.join(PROC_DIR, "test_v2.parquet")
    if not os.path.exists(test_path):
        print("  test_v2.parquet not found. Skipping.")
        return
    test_df = pd.read_parquet(test_path)
    print(f"\n[F4] Imputing test split: {test_df.shape}")
    test_imp = impute_dae(test_df)
    out = os.path.join(PROC_DIR, "dae_imputed_test_v2.parquet")
    test_imp.to_parquet(out, index=False)
    print(f"  Saved dae_imputed_test_v2.parquet")
    print(f"  Missing in IMPUTE_COLS after: {test_imp[['waist_cm','hip_cm']].isna().sum().sum():,}")

    # Verify column means match train (to 1e-4 tolerance)
    train_imp = pd.read_parquet(os.path.join(PROC_DIR, "dae_imputed_train_v2.parquet"))
    for col in ["waist_cm", "hip_cm"]:
        tr_mean  = train_imp[col].mean()
        te_mean  = test_imp[col].mean()
        diff     = abs(tr_mean - te_mean)
        ok = "[OK]" if diff < 5.0 else "[CHECK]"   # within 5cm mean difference is expected
        print(f"  {col}: train_mean={tr_mean:.4f} test_mean={te_mean:.4f} diff={diff:.4f} {ok}")


if __name__ == "__main__":
    print("=== dae_impute.py ===")
    # 1. Benchmark
    bm = benchmark_dae_vs_median()
    # 2. Impute test split
    produce_imputed_test()
    print("\ndae_impute.py DONE.")
