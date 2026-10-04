# Running the P2 model code on real NFHS-5 data (Colab or Kaggle)

The respondent-level files may only live on your machine or in a PRIVATE
Drive folder / PRIVATE Kaggle dataset that only the team can open. Never put
them in the repo, a public dataset, or an AI chat. Every script below prints
and writes **aggregates only** (counts, rates, quantiles; cells with n < 30
suppressed).

## 0. Upload (once)
Put these in a private folder, e.g. `MyDrive/proxypatient_private/processed/`
(or a private Kaggle dataset):

```
train_v2.parquet  val_v2.parquet  test_v2.parquet
dae_imputed_train_v2.parquet  dae_imputed_val_v2.parquet  dae_imputed_test_v2.parquet
dae_weights_v2.pt  dae_fit_stats_v2.pkl  MANIFEST.json
```

## 1. Set up the runtime
Colab: Runtime → Change runtime type → GPU (T4 is enough).

```bash
from google.colab import drive; drive.mount('/content/drive')      # Colab only (Python cell)
!git clone -b claude/new-session-sj0fv1 https://github.com/Prapti101/ProxyPatient.git   # use main once the P2 PR is merged
%cd ProxyPatient
# Keep Colab's preinstalled torch/pandas/numpy; add the rest:
!pip install -q ctgan==0.12.1 rdt==1.22.0 pyyaml fastapi httpx pytest
import os; os.environ["PP_DATA_DIR"] = "/content/drive/MyDrive/proxypatient_private/processed"
# Kaggle instead: os.environ["PP_DATA_DIR"] = "/kaggle/input/<your-private-dataset>"
```
On a local machine: `pip install -r requirements.txt` and `export PP_DATA_DIR=/path/to/processed`.

Sanity check that the code works before touching real data (MOCK data only):
```bash
!python -m pytest -q
```

## 2. Run, in this order

| step | command | writes | rough time (T4) |
|---|---|---|---|
| 1 | `!python -m models.step0_checks` | `docs/p2_step0_checks.md/.json` | 2-5 min |
| 2 | `!python -m models.check_dae_benchmark` | `docs/p2_dae_benchmark.md/.json` | 1-3 min |
| 3 | `!python -m models.train_cvae` | `models/cvae_weights.pt` (PRIVATE), `models/cvae_preproc.json`, `models/condition_marginals.json`, `models/cvae_train_log.json`, `models/model_card.json` | 10-40 min |
| 4 | `!python -m models.train_baselines` | `outputs/baselines/{tvae,ctgan}.pkl` (PRIVATE), `docs/baselines_train_log.json` | 30-90 min |
| 5 | `!python -m models.eval_dev` | `docs/model_comparison_dev.json/.md` | 5-20 min |

Read `docs/p2_step0_checks.md` after step 1 before going on. Stop and tell P2
if: any MANIFEST status is not OK, the glucose unit verdict is not mg/dL, the
any_tobacco rate is higher in women than in men, or the in-scope share is far
below ~85%.

Useful flags: `--max-rows 50000` (quick trial of train_cvae), `--epochs N`,
`--no-state`, `--glucose-head gaussian`, `--generate-bp`, `--cpu`.
`train_baselines --epochs 30` if the GPU session is short.

### Optional ablations (T8) and head comparison
```bash
!python -m models.train_cvae --variant gru --out-dir outputs/cvae_gru
!python -m models.train_cvae --variant cnn --out-dir outputs/cvae_cnn
!python -m models.train_cvae --glucose-head gaussian --out-dir outputs/cvae_gauss
!python -m models.eval_dev --extra-cvae CVAE-gru=outputs/cvae_gru/cvae_weights.pt \
     --extra-cvae CVAE-cnn=outputs/cvae_cnn/cvae_weights.pt \
     --extra-cvae CVAE-gaussian=outputs/cvae_gauss/cvae_weights.pt
```
(`--out-dir outputs/...` keeps the main model's files in `models/` untouched.)

## 3. What to keep, download and commit
* **Keep PRIVATE (never commit):** `models/cvae_weights.pt`, `outputs/` (baseline
  pickles, any exported samples), every parquet. Copy `cvae_weights.pt` to the
  private Drive folder so the backend machine can use it.
* **Download and commit (aggregate-only):**
  `models/cvae_preproc.json`, `models/condition_marginals.json`,
  `models/model_card.json`, `models/cvae_train_log.json`,
  `docs/p2_step0_checks.md`, `docs/p2_step0_checks.json`,
  `docs/p2_dae_benchmark.md`, `docs/p2_dae_benchmark.json`,
  `docs/baselines_train_log.json`,
  `docs/model_comparison_dev.json`, `docs/model_comparison_dev.md`.

```bash
git add models/cvae_preproc.json models/condition_marginals.json models/model_card.json \
        models/cvae_train_log.json docs/p2_step0_checks.* docs/p2_dae_benchmark.* \
        docs/baselines_train_log.json docs/model_comparison_dev.*
git status                     # must show NO .parquet/.csv/.pt/.pkl files
git ls-files | grep -E '\.(parquet|csv|dta|sav|zip|pt)$'     # must print nothing new
git commit -m "P2: real-data aggregate results (step0, DAE benchmark, CVAE, baselines, dev eval)"
```
Open each file before committing and check it contains only counts, rates,
quantiles and settings.

## 4. Use the real generator in the backend
```bash
export PP_GENERATOR=real                        # otherwise the stub is used
export PP_CVAE_WEIGHTS=/private/path/cvae_weights.pt   # default models/cvae_weights.pt
uvicorn backend.main:app --port 8000
```

## 5. FINAL test (run ONCE, at the very end, after the model is frozen)
```bash
!python -m models.eval_dev --final-test        # -> docs/model_comparison_final_test.json/.md
!python -m models.step0_checks --final-test --out docs/p2_step0_checks_with_test.md   # optional
```
Do not change the model after looking at these numbers; if you do, say so in
the report.
