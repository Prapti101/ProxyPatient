# ProxyPatient -- Stage 6: Denoising Autoencoder Report

**Run date:** 2026-10-02T18:39:05.629938
**Seed:** 42 | **Epochs:** 30 | **Mask ratio:** 0.2

## Architecture
- Input dim: 13 features
- Encoder: [128, 64, 32]
- Decoder: [64, 128] -> 13
- Training loss (final): 0.03765021

## Imputation Impact
| Split | Missing Before | Missing After | Reduced |
|---|---|---|---|
| Train | 241,720 | 0 | Yes |
| Val   | 51,913   | 0   | Yes |

> Glucose variable was never imputed. Glucose columns reattached unchanged.

## Acceptance Checks
- [OK] `glucose_not_in_features`
- [OK] `outcome_not_in_features`
- [OK] `glucose_unchanged_train`
- [OK] `glucose_unchanged_val`
- [OK] `missing_reduced_train`
- [OK] `missing_reduced_val`
- [OK] `model_weights_saved`
- [OK] `loss_converged`

## Output Files
| File | Description |
|---|---|
| processed/dae_weights.pt | Trained DAE model weights |
| processed/dae_fit_stats.pkl | Normalisation stats for inference |
| processed/dae_imputed_train.parquet | Imputed training data |
| processed/dae_imputed_val.parquet | Imputed validation data |
| processed/dae_imputed_combined.parquet | Imputed full dataset |

_Stage 6 complete. P2 should use dae_imputed_train.parquet for CVAE training._