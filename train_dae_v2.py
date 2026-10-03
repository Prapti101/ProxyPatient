"""
ProxyPatient — Stage 6 v2: Denoising Autoencoder (structural-missing aware)
============================================================================
Key changes vs v1:
  + systolic_avg, diastolic_avg added as continuous features
  + hypertension, on_bp_medication, bmi_measured, bp_measured added as binary features
  - BMI, weight_kg, height_cm NOT imputed by DAE (structural missing where bmi_measured=0)
  - Only sporadic-missing covariates (waist_cm, hip_cm, any_tobacco, alcohol) imputed
  - Masking skips bmi/weight/height for rows where bmi_measured=0
  - bmi_band recomputed after DAE from imputed bmi values
Glucose never touched.
"""

import os, sys, hashlib, joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

BASE     = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
PROC_DIR = os.path.join(BASE, "processed")
SEED     = 42
EPOCHS   = 30
BATCH    = 512
LR       = 1e-3
NOISE    = 0.20  # 20% masking rate for sporadic columns
DEVICE   = torch.device("cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# Columns the DAE is ALLOWED to impute (sporadic missing only)
IMPUTE_COLS = [
    "waist_cm", "hip_cm",          # continuous, sporadic
    "any_tobacco", "alcohol",       # binary, sporadic
    "bp_ever_checked",              # binary, sporadic
]

# Columns used as INPUT to DAE (includes structural-missing columns when present)
CONT_COLS = ["waist_cm", "hip_cm", "systolic_avg", "diastolic_avg"]
CAT_COLS  = [
    "sex", "residence", "education", "wealth_quintile",
    "any_tobacco", "alcohol", "bp_ever_checked",
    "hypertension", "on_bp_medication", "bmi_measured", "bp_measured",
]

# Structural-missing indicator columns (DAE does NOT impute these)
STRUCTURAL_COLS = ["bmi", "weight_kg", "height_cm", "systolic_avg", "diastolic_avg"]

NEVER_IMPUTE = {"glucose_raw", "elevated_glucose_proxy", "bmi", "weight_kg",
                "height_cm", "systolic_avg", "diastolic_avg", "hypertension",
                "on_bp_medication", "bmi_measured", "bp_measured"}


# ── Model ──────────────────────────────────────────────────────────────────────

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


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise(df, cont_cols, cat_cols, fit_stats=None):
    """Normalise for DAE input. Returns tensor + fit_stats."""
    df = df.copy()
    if fit_stats is None:
        fit_stats = {
            "cont_cols": cont_cols, "cat_cols": cat_cols,
            "cont_medians": {}, "cont_stds": {},
            "cat_maps": {}, "cat_modes": {},
        }
        for col in cont_cols:
            med = df[col].median() if df[col].notna().any() else 0.0
            std = df[col].std()    if df[col].notna().any() else 1.0
            std = std if (pd.notna(std) and std > 1e-6) else 1.0
            fit_stats["cont_medians"][col] = float(med)
            fit_stats["cont_stds"][col]    = float(std)
        for col in cat_cols:
            cats = sorted(df[col].dropna().astype(str).unique().tolist())
            mode = df[col].mode()
            fit_stats["cat_maps"][col]  = {c: i for i, c in enumerate(cats)}
            fit_stats["cat_modes"][col] = str(mode.iloc[0]) if len(mode) > 0 else cats[0] if cats else "0"

    result = []
    for col in cont_cols:
        vals = df[col].fillna(fit_stats["cont_medians"][col]).astype(float)
        norm = (vals - fit_stats["cont_medians"][col]) / fit_stats["cont_stds"][col]
        result.append(norm.values.reshape(-1, 1))

    for col in cat_cols:
        cm = fit_stats["cat_maps"][col]
        mode_enc = cm.get(fit_stats["cat_modes"][col], 0)
        vals = df[col].fillna(fit_stats["cat_modes"][col]).astype(str)
        enc = vals.map(lambda x: cm.get(x, mode_enc)).astype(float)
        n_cats = max(len(cm), 1)
        result.append((enc / n_cats).values.reshape(-1, 1))

    tensor = torch.tensor(np.hstack(result), dtype=torch.float32)
    return tensor, fit_stats


def denormalise_cont(norm_vals, col, fit_stats):
    med = fit_stats["cont_medians"][col]
    std = fit_stats["cont_stds"][col]
    return norm_vals * std + med


# ── Training ───────────────────────────────────────────────────────────────────

def train_dae(X_train, input_dim, fit_stats, bmi_measured_train):
    model = DenoisingAutoencoder(input_dim, hidden_dims=[128, 64, 32]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n_cont = len(fit_stats["cont_cols"])

    dataset = TensorDataset(X_train, bmi_measured_train)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    print(f"  DAE training: {input_dim} features, {len(X_train):,} rows, {EPOCHS} epochs")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, bm_batch in loader:
            # Add noise only to IMPUTE_COLS (first len(CONT_COLS) cont + relevant cat)
            mask = torch.rand_like(X_batch) < NOISE
            # Never mask structural-missing cols (they are already 0/median for missing rows)
            # mask columns 2,3 (systolic_avg, diastolic_avg index in cont_cols) -> don't noise them
            # They are indices 2 and 3 in the tensor
            for i, col in enumerate(fit_stats["cont_cols"]):
                if col in STRUCTURAL_COLS:
                    mask[:, i] = False
            X_noisy = X_batch.clone()
            X_noisy[mask] = 0.0
            X_hat = model(X_noisy)
            # Loss: only on IMPUTE_COLS, not on structural cols
            loss_mask = torch.zeros_like(X_batch, dtype=torch.bool)
            for i, col in enumerate(fit_stats["cont_cols"]):
                if col not in STRUCTURAL_COLS:
                    loss_mask[:, i] = True
            for i, col in enumerate(fit_stats["cat_cols"]):
                loss_mask[:, n_cont + i] = True
            loss = ((X_hat - X_batch) ** 2)[loss_mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(X_batch)
        avg_loss = total_loss / len(X_train)
        if epoch in {1, 5, 10, 20, 30}:
            print(f"  Epoch {epoch:2d}: loss={avg_loss:.6f}")
    return model


# ── Impute ────────────────────────────────────────────────────────────────────

def impute(df, model, fit_stats, bmi_measured_col="bmi_measured"):
    model.eval()
    X, _ = normalise(df, fit_stats["cont_cols"], fit_stats["cat_cols"], fit_stats)
    with torch.no_grad():
        X_hat = model(X).numpy()

    df_out = df.copy()
    n_cont = len(fit_stats["cont_cols"])

    for i, col in enumerate(fit_stats["cont_cols"]):
        if col in NEVER_IMPUTE:
            continue
        missing_mask = df_out[col].isna()
        if missing_mask.any():
            imputed = X_hat[missing_mask.values, i]
            orig_scale = denormalise_cont(imputed, col, fit_stats)
            df_out.loc[missing_mask, col] = orig_scale

    for i, col in enumerate(fit_stats["cat_cols"]):
        if col in NEVER_IMPUTE:
            continue
        missing_mask = df_out[col].isna()
        if missing_mask.any():
            n_cats = len(fit_stats["cat_maps"][col])
            enc_val = X_hat[missing_mask.values, n_cont + i] * n_cats
            rounded = np.clip(np.round(enc_val), 0, n_cats - 1).astype(int)
            inv_map = {v: k for k, v in fit_stats["cat_maps"][col].items()}
            df_out.loc[missing_mask, col] = [inv_map.get(r, list(inv_map.values())[-1]) for r in rounded]

    return df_out


def recompute_bmi_band(df):
    """Recompute bmi_band from bmi column where bmi_measured=1."""
    import pandas as pd
    df = df.copy()
    measured = df["bmi_measured"] == 1
    df.loc[measured, "bmi_band"] = pd.cut(
        df.loc[measured, "bmi"],
        bins=[0, 18.5, 25.0, 30.0, 999],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False
    )
    df.loc[~measured, "bmi_band"] = np.nan
    return df


def fix_dtypes(df):
    """Convert Categorical to object so parquet saves cleanly."""
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].astype(object)
    return df


def main():
    print("[1/5] Loading v2 train and val splits...")
    train_df = pd.read_parquet(os.path.join(PROC_DIR, "train_v2.parquet"))
    val_df   = pd.read_parquet(os.path.join(PROC_DIR, "val_v2.parquet"))
    combined = pd.read_parquet(os.path.join(PROC_DIR, "combined_clean_v2.parquet"))
    print(f"  Train: {train_df.shape} | Val: {val_df.shape}")

    # Verify glucose unchanged
    assert "glucose_raw" in train_df.columns
    assert "elevated_glucose_proxy" in train_df.columns

    print(f"  Missing before imputation (train): {train_df[IMPUTE_COLS].isna().sum().sum():,}")
    print(f"  Missing before imputation (val):   {val_df[IMPUTE_COLS].isna().sum().sum():,}")

    # ── Step 2: Normalise ──────────────────────────────────────────────────────
    print("\n[2/5] Normalising training data...")
    # Available columns (some may not be present)
    cont_use = [c for c in CONT_COLS if c in train_df.columns]
    cat_use  = [c for c in CAT_COLS  if c in train_df.columns]

    bmi_measured_train = torch.tensor(
        train_df["bmi_measured"].fillna(0).astype(float).values, dtype=torch.float32
    ).unsqueeze(1)
    bmi_measured_val   = torch.tensor(
        val_df["bmi_measured"].fillna(0).astype(float).values,   dtype=torch.float32
    ).unsqueeze(1)

    X_train, fit_stats = normalise(train_df, cont_use, cat_use)
    X_val,   _         = normalise(val_df,   cont_use, cat_use, fit_stats)
    input_dim = X_train.shape[1]
    print(f"  Input dim: {input_dim}")

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    print("\n[3/5] Training DAE v2...")
    model = train_dae(X_train, input_dim, fit_stats, bmi_measured_train)

    # Validation loss
    model.eval()
    with torch.no_grad():
        X_hat_val = model(X_val)
        val_loss  = nn.MSELoss()(X_hat_val, X_val).item()
    print(f"  Val loss: {val_loss:.6f}")

    # ── Step 4: Impute ─────────────────────────────────────────────────────────
    print("\n[4/5] Imputing missing covariates (IMPUTE_COLS only)...")
    train_imp = impute(train_df, model, fit_stats)
    val_imp   = impute(val_df,   model, fit_stats)
    combined_imp = impute(combined, model, fit_stats)

    # Recompute bmi_band
    train_imp = recompute_bmi_band(train_imp)
    val_imp   = recompute_bmi_band(val_imp)
    combined_imp = recompute_bmi_band(combined_imp)

    # Verify glucose unchanged
    for label, orig, imp_df in [("train", train_df, train_imp), ("val", val_df, val_imp)]:
        orig_gluc = orig["glucose_raw"].fillna(-999)
        imp_gluc  = imp_df["glucose_raw"].fillna(-999)
        assert (orig_gluc == imp_gluc).all(), f"FATAL: glucose changed in {label}"
        print(f"  glucose_unchanged_{label}: [OK]")

    # Print NaN counts after imputation
    print(f"\n  Missing after imputation (train, IMPUTE_COLS): {train_imp[IMPUTE_COLS].isna().sum().sum():,}")
    print(f"  Missing after imputation (val,   IMPUTE_COLS): {val_imp[IMPUTE_COLS].isna().sum().sum():,}")
    print(f"  bmi NaN in train (structural, OK): {train_imp['bmi'].isna().sum():,}")

    # ── Step 5: Save ───────────────────────────────────────────────────────────
    print("\n[5/5] Saving v2 imputed files...")
    # Save weights FIRST (before parquet saves that might fail)
    ck = {
        "model_state_dict": model.state_dict(),
        "input_dim":   input_dim,
        "hidden_dims": [128, 64, 32],
        "fit_stats":   fit_stats,
        "cont_cols":   cont_use,
        "cat_cols":    cat_use,
        "impute_cols": IMPUTE_COLS,
        "never_impute": list(NEVER_IMPUTE),
        "seed":    SEED,
        "created": datetime.now().isoformat(),
        "version": "v2",
        "note":    "DAE v2: BP added. BMI/weight/height/BP NOT imputed (structural). Only sporadic covariates imputed.",
    }
    torch.save(ck, os.path.join(PROC_DIR, "dae_weights_v2.pt"))
    joblib.dump(fit_stats, os.path.join(PROC_DIR, "dae_fit_stats_v2.pkl"))
    print("  Saved dae_weights_v2.pt, dae_fit_stats_v2.pkl")

    fix_dtypes(train_imp).to_parquet(os.path.join(PROC_DIR, "dae_imputed_train_v2.parquet"), index=False)
    fix_dtypes(val_imp).to_parquet(os.path.join(PROC_DIR, "dae_imputed_val_v2.parquet"), index=False)
    fix_dtypes(combined_imp).to_parquet(os.path.join(PROC_DIR, "dae_imputed_combined_v2.parquet"), index=False)
    print("  Saved dae_imputed_train_v2, dae_imputed_val_v2, dae_imputed_combined_v2")

    print("\ntrain_dae_v2.py DONE.")

if __name__ == "__main__":
    main()
