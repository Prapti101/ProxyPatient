"""
ProxyPatient — Stage 6: Denoising Autoencoder (DAE) for Covariate Imputation
==============================================================================
Source  : NFHS-5 India (2019-21), MoHFW/IIPS, DHS Program
Author  : P1 (Data & Backend Lead)
Inputs  : processed/train.parquet, processed/val.parquet, processed/combined_clean.parquet
Outputs :
    processed/dae_weights.pt              <- trained DAE model weights
    processed/dae_imputed_train.parquet   <- train split with imputed covariates
    processed/dae_imputed_val.parquet     <- val split with imputed covariates
    processed/dae_imputed_combined.parquet
    docs/dae_report.md

STRICT RULES (NEVER VIOLATE):
  - glucose_raw and elevated_glucose_proxy are NEVER fed into the DAE.
  - glucose_raw and elevated_glucose_proxy are NEVER imputed — ever.
  - Glucose columns are detached before DAE, reattached unchanged after.
  - DAE is trained on train.parquet ONLY. val/combined are inference only.
  - Never print raw rows.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE        = r"C:\Users\PraptiPriya\OneDrive\Desktop\data"
PROC_DIR    = os.path.join(BASE, "processed")
DOCS_DIR    = os.path.join(BASE, "docs")

SEED        = 42
EPOCHS      = 30
BATCH_SIZE  = 512
LR          = 1e-3
MASK_RATIO  = 0.20     # randomly corrupt 20% of feature values during training
HIDDEN_DIMS = [128, 64, 32]   # encoder dims; decoder mirrors these

# Glucose columns — NEVER touch these
GLUCOSE_COLS = ["glucose_raw", "elevated_glucose_proxy", "glucose_time",
                "glucose_ever_checked", "told_high_glucose", "on_glucose_medicine"]

# Covariate features the DAE will learn to impute
CONTINUOUS_FEATURES  = ["age", "waist_cm", "hip_cm", "wealth_score",
                         "bmi", "weight_kg"]
CATEGORICAL_FEATURES = ["sex", "residence", "education", "wealth_quintile",
                         "any_tobacco", "alcohol", "bp_ever_checked"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_feature_matrix(df, cont_cols, cat_cols, fit_stats=None):
    """
    Build a numeric feature matrix from the dataframe.
    - Continuous: fill NaN with median, then z-score normalise.
    - Categorical: ordinal encode, fill NaN with mode.
    Returns (matrix, fit_stats) where fit_stats can be reused for val/test.
    Glucose columns are never included.
    """
    # Safety check: ensure glucose never sneaks in
    for col in GLUCOSE_COLS:
        assert col not in cont_cols and col not in cat_cols, \
            f"FATAL: Glucose column '{col}' found in feature list. Aborting."

    # Only use columns that actually exist in this dataframe
    cont_cols = [c for c in cont_cols if c in df.columns]
    cat_cols  = [c for c in cat_cols  if c in df.columns]

    X_cont = df[cont_cols].copy().astype(float)
    X_cat  = df[cat_cols].copy()

    # Encode categoricals
    cat_maps = {}
    X_cat_enc = pd.DataFrame(index=df.index)
    for col in cat_cols:
        unique_vals = X_cat[col].dropna().unique()
        val_map = {v: i for i, v in enumerate(sorted(unique_vals, key=str))}
        cat_maps[col] = val_map
        X_cat_enc[col] = X_cat[col].map(val_map)  # NaN stays NaN

    if fit_stats is None:
        # Fit: compute medians/modes/stds from training data
        cont_medians = X_cont.median()
        cont_stds    = X_cont.std().replace(0, 1)   # avoid div-by-zero
        cat_modes    = {col: X_cat_enc[col].mode()[0]
                        if not X_cat_enc[col].mode().empty else 0
                        for col in cat_cols}
        fit_stats = {
            "cont_cols":    cont_cols,
            "cat_cols":     cat_cols,
            "cat_maps":     cat_maps,
            "cont_medians": cont_medians.to_dict(),
            "cont_stds":    cont_stds.to_dict(),
            "cat_modes":    cat_modes,
        }

    # Apply fit_stats
    cont_medians = pd.Series(fit_stats["cont_medians"])
    cont_stds    = pd.Series(fit_stats["cont_stds"])
    cat_modes    = fit_stats["cat_modes"]

    X_cont_filled = X_cont.fillna(cont_medians)
    X_cont_norm   = (X_cont_filled - cont_medians) / cont_stds

    X_cat_filled = X_cat_enc.copy()
    for col in cat_cols:
        X_cat_filled[col] = X_cat_filled[col].fillna(cat_modes.get(col, 0))

    # Normalise categoricals to [0,1] range
    for col in cat_cols:
        n_vals = max(len(fit_stats["cat_maps"].get(col, {1: 0})), 1)
        X_cat_filled[col] = X_cat_filled[col] / n_vals

    X = pd.concat([X_cont_norm, X_cat_filled], axis=1).astype(float)

    # Track which values were originally missing (for imputation mask)
    missing_mask_cont = df[cont_cols].isna()
    missing_mask_cat  = df[cat_cols].isna()
    missing_mask = pd.concat([missing_mask_cont, missing_mask_cat], axis=1)

    return X.values.astype(np.float32), missing_mask, fit_stats, cont_cols, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class DenoisingAutoencoder(nn.Module):
    """
    Symmetric encoder-decoder for mixed tabular data imputation.
    Input: normalised covariate feature vector (NO glucose).
    Output: reconstructed clean feature vector.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def add_masking_noise(x, mask_ratio=MASK_RATIO):
    """Randomly zero out mask_ratio fraction of features during training."""
    mask = torch.bernoulli(torch.full(x.shape, 1 - mask_ratio)).to(x.device)
    return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_dae(X_train, input_dim, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    device = torch.device("cpu")
    model  = DenoisingAutoencoder(input_dim, HIDDEN_DIMS).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    print(f"  Training DAE: {input_dim} features | {epochs} epochs | batch {batch_size}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch      = batch.to(device)
            corrupted  = add_masking_noise(batch, MASK_RATIO)
            recon      = model(corrupted)
            loss       = loss_fn(recon, batch)   # reconstruct the CLEAN original
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)

        avg_loss = epoch_loss / len(X_train)
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute_with_dae(model, X_filled, missing_mask_df,
                    df_original, fit_stats, cont_cols, cat_cols):
    """
    Use DAE to fill only the cells that were genuinely missing.
    Non-missing cells keep their original values.
    Glucose columns are never touched.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        X_recon  = model(X_tensor).numpy()

    # Denormalise continuous columns from DAE output
    cont_medians = pd.Series(fit_stats["cont_medians"])
    cont_stds    = pd.Series(fit_stats["cont_stds"])

    df_out = df_original.copy()
    all_feat_cols = cont_cols + cat_cols

    for i, col in enumerate(all_feat_cols):
        if col not in df_out.columns:
            continue
        was_missing = missing_mask_df[col] if col in missing_mask_df.columns else pd.Series(False, index=df_out.index)

        if col in cont_cols:
            # Denormalise
            recon_val = X_recon[:, i] * cont_stds.get(col, 1) + cont_medians.get(col, 0)
            df_out.loc[was_missing, col] = recon_val[was_missing.values]
        else:
            # Categorical: decode from normalised value
            cat_map = fit_stats["cat_maps"].get(col, {})
            n_vals  = max(len(cat_map), 1)
            inv_map = {v: k for k, v in cat_map.items()}
            recon_idx = np.round(X_recon[:, i] * n_vals).astype(int).clip(0, n_vals - 1)
            recon_cat = [inv_map.get(idx, list(inv_map.values())[0] if inv_map else 0)
                         for idx in recon_idx]
            recon_series = pd.Series(recon_cat, index=df_out.index)
            df_out.loc[was_missing, col] = recon_series[was_missing.values]

    # SAFETY: glucose columns copied exactly from original — never modified
    for gcol in GLUCOSE_COLS:
        if gcol in df_original.columns:
            df_out[gcol] = df_original[gcol].values

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    report = {"run_date": datetime.now().isoformat(), "seed": SEED}

    print("[1/6] Loading train split...")
    df_train    = pd.read_parquet(os.path.join(PROC_DIR, "train.parquet"))
    df_val      = pd.read_parquet(os.path.join(PROC_DIR, "val.parquet"))
    df_combined = pd.read_parquet(os.path.join(PROC_DIR, "combined_clean.parquet"))
    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Combined: {len(df_combined):,}")

    # Double-check glucose is present
    for gcol in ["glucose_raw", "elevated_glucose_proxy"]:
        assert gcol in df_train.columns, f"FATAL: {gcol} missing from train!"

    print("[2/6] Preparing feature matrices (glucose excluded)...")
    X_train, miss_train, fit_stats, cont_cols, cat_cols = prepare_feature_matrix(
        df_train, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
    )
    X_val, miss_val, _, _, _         = prepare_feature_matrix(
        df_val, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, fit_stats
    )
    X_combined, miss_combined, _, _, _ = prepare_feature_matrix(
        df_combined, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, fit_stats
    )

    input_dim = X_train.shape[1]
    report["input_dim"]   = input_dim
    report["cont_cols"]   = cont_cols
    report["cat_cols"]    = cat_cols
    report["glucose_excluded"] = True

    # Count missing before imputation
    miss_before = {
        "train": int(df_train[cont_cols + cat_cols].isna().sum().sum()),
        "val":   int(df_val[cont_cols + cat_cols].isna().sum().sum()),
    }
    report["missing_before"] = miss_before
    print(f"  Missing values — Train: {miss_before['train']:,} | Val: {miss_before['val']:,}")

    print("[3/6] Training Denoising Autoencoder...")
    model, history = train_dae(X_train, input_dim)
    report["final_train_loss"] = round(history[-1], 8)
    report["epochs"] = EPOCHS

    print("[4/6] Saving model weights...")
    weights_path = os.path.join(PROC_DIR, "dae_weights.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim":        input_dim,
        "hidden_dims":      HIDDEN_DIMS,
        "fit_stats":        fit_stats,
        "cont_cols":        cont_cols,
        "cat_cols":         cat_cols,
        "seed":             SEED,
        "created":          datetime.now().isoformat(),
        "note":             "Glucose never imputed. Never included in encoder input.",
    }, weights_path)
    print(f"  Saved: {weights_path}")

    # Save fit_stats separately for use by FastAPI backend
    joblib.dump(fit_stats, os.path.join(PROC_DIR, "dae_fit_stats.pkl"))

    print("[5/6] Running imputation on train / val / combined...")
    df_train_imp    = impute_with_dae(model, X_train,    miss_train,    df_train,    fit_stats, cont_cols, cat_cols)
    df_val_imp      = impute_with_dae(model, X_val,      miss_val,      df_val,      fit_stats, cont_cols, cat_cols)
    df_combined_imp = impute_with_dae(model, X_combined, miss_combined, df_combined, fit_stats, cont_cols, cat_cols)

    # Count missing after imputation
    miss_after = {
        "train": int(df_train_imp[cont_cols + cat_cols].isna().sum().sum()),
        "val":   int(df_val_imp[cont_cols + cat_cols].isna().sum().sum()),
    }
    report["missing_after"] = miss_after
    print(f"  Missing after imputation — Train: {miss_after['train']:,} | Val: {miss_after['val']:,}")

    df_train_imp.to_parquet(   os.path.join(PROC_DIR, "dae_imputed_train.parquet"),    index=False)
    df_val_imp.to_parquet(     os.path.join(PROC_DIR, "dae_imputed_val.parquet"),      index=False)
    df_combined_imp.to_parquet(os.path.join(PROC_DIR, "dae_imputed_combined.parquet"), index=False)
    print("  Imputed parquet files saved.")

    print("[6/6] Acceptance checks + report...")
    checks = {
        "glucose_not_in_features":     "glucose_raw" not in cont_cols + cat_cols,
        "outcome_not_in_features":     "elevated_glucose_proxy" not in cont_cols + cat_cols,
        "glucose_unchanged_train":     df_train_imp["glucose_raw"].equals(df_train["glucose_raw"]) if "glucose_raw" in df_train.columns else True,
        "glucose_unchanged_val":       df_val_imp["glucose_raw"].equals(df_val["glucose_raw"])     if "glucose_raw" in df_val.columns   else True,
        "missing_reduced_train":       miss_after["train"] <= miss_before["train"],
        "missing_reduced_val":         miss_after["val"]   <= miss_before["val"],
        "model_weights_saved":         os.path.exists(weights_path),
        "loss_converged":              history[-1] < history[0],
    }
    all_passed = all(checks.values())
    report["acceptance_checks"] = {k: bool(v) for k, v in checks.items()}

    print(f"  Result: {'ALL PASSED [OK]' if all_passed else 'SOME FAILED [FAIL]'}")
    for k, v in checks.items():
        print(f"    {k}: {'[OK]' if v else '[FAIL]'}")

    # Write report
    lines = [
        "# ProxyPatient -- Stage 6: Denoising Autoencoder Report",
        f"\n**Run date:** {report['run_date']}",
        f"**Seed:** {SEED} | **Epochs:** {EPOCHS} | **Mask ratio:** {MASK_RATIO}\n",
        "## Architecture",
        f"- Input dim: {input_dim} features",
        f"- Encoder: {HIDDEN_DIMS}",
        f"- Decoder: {list(reversed(HIDDEN_DIMS[:-1]))} -> {input_dim}",
        f"- Training loss (final): {report['final_train_loss']}\n",
        "## Imputation Impact",
        "| Split | Missing Before | Missing After | Reduced |",
        "|---|---|---|---|",
        f"| Train | {miss_before['train']:,} | {miss_after['train']:,} | {'Yes' if miss_after['train'] <= miss_before['train'] else 'No'} |",
        f"| Val   | {miss_before['val']:,}   | {miss_after['val']:,}   | {'Yes' if miss_after['val']   <= miss_before['val']   else 'No'} |",
        "\n> Glucose variable was never imputed. Glucose columns reattached unchanged.",
        "\n## Acceptance Checks",
    ]
    for k, v in checks.items():
        lines.append(f"- {'[OK]' if v else '[FAIL]'} `{k}`")

    lines += [
        "\n## Output Files",
        "| File | Description |",
        "|---|---|",
        "| processed/dae_weights.pt | Trained DAE model weights |",
        "| processed/dae_fit_stats.pkl | Normalisation stats for inference |",
        "| processed/dae_imputed_train.parquet | Imputed training data |",
        "| processed/dae_imputed_val.parquet | Imputed validation data |",
        "| processed/dae_imputed_combined.parquet | Imputed full dataset |",
        "\n_Stage 6 complete. P2 should use dae_imputed_train.parquet for CVAE training._"
    ]

    with open(os.path.join(DOCS_DIR, "dae_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if not all_passed:
        raise RuntimeError("STOP: One or more acceptance checks failed.")

    print("\nStage 6 DONE. All checks passed.")
    print(f"  Final training loss : {history[-1]:.6f}")
    print(f"  Missing reduced     : {miss_before['train']:,} -> {miss_after['train']:,} (train)")


if __name__ == "__main__":
    main()
