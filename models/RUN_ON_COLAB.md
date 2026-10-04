# Private-holder one-command workflow

Updated 2026-10-05 (Asia/Kolkata). No real NFHS run or GPU timing is verified by the cloud implementation task.

Do not upload respondent data to this task, an AI chat, or public hosting. Colab/Kaggle or another private cloud is permitted only if the actual DHS research agreement authorizes that environment and every recipient. The repository cannot determine that permission. Until confirmed, run on the approved local holder's machine.

## Prepare

Use the fork's `main` after the presentation-slice branch is merged. During review use `p2-presentation-slice`; do not clone an obsolete development branch. Python 3.11/CPU has been verified:

```bash
git clone https://github.com/dhruvvvgg/proxypatient-fork.git
cd proxypatient-fork
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest -q
python -m models.run_all --mock --quick --out-dir outputs/mock-smoke
```

The private processed folder needs un-imputed `train_v2.parquet` and `val_v2.parquet`, imputed `dae_imputed_train_v2.parquet` and `dae_imputed_val_v2.parquet`, and a `MANIFEST.json` with matching SHA-256 entries. The DAE benchmark additionally requires trusted `dae_weights_v2.pt` and `dae_fit_stats_v2.pkl`. Keep raw/preprocessed respondent files private. Final-test files stay locked until the final command. Only metadata hashes for train/val are examined in development.

P1 must repair/rebuild affected women's height linkage before the private run, verify glucose units and official codebook maps, and review the deferred medication/BP rules. The current P2 workflow fits its own TRAIN-only preprocessor; do not use the historical all-data `preprocess.pkl`/`preprocess_v2.pkl`.

## Quick, then full

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/pp-quick --quick
python -m models.run_all --data-dir /private/processed --out-dir /private/pp-full --full
```

Use separate output directories so the quick trial remains inspectable. Quick keeps all integrity/unit/per-sex/state guards: it does not lower the real 5,000 complete rows per sex requirement. The runner is CPU-only; no T4 runtime or memory promise is made. Full uses config defaults. Set epochs/batch size/seed/optional generated variables in `config.yaml` before freezing it. Individual training-stage flags override config only when explicitly supplied.

Order: step0 -> DAE benchmark -> CVAE -> TVAE/CTGAN -> validation evaluation -> package. Stage failures stop the run. If an optional benchmark or baseline cannot be run, explicitly use `--skip-dae-benchmark` or `--skip-baselines`; omissions are recorded and are not successes. Inspect requested/retained condition coverage, clipping, tail fidelity, subgroup metrics, and the comparison's differing data/state use. Do not claim a winning architecture from this unequal setup alone.

Each stage records runtime and peak RSS. `safe_outputs/` has aggregate JSON/Markdown and a hash manifest. `private_outputs/` has weights, preprocessor, marginals, supported profiles and baseline pickles. **Never commit the private directory or data/weights.** A real package updates `models/model_card.json` from computed real outputs; mock packaging never overwrites the tracked pending template. Review aggregate outputs and licensing/disclosure questions before committing even safe files. The package does not invent a formal P3 validation report.

## Serve and freeze

```bash
PP_MODEL_DIR=/private/pp-full/private_outputs python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Do not set `PP_DEMO_MOCK` for real serving. Missing/mock/incompatible artifacts fail closed. Choose full supported `/profiles` and use `/options` for actual state codes. `/validation` remains pending unless a separate explicit-status `validation_report.json` is supplied in `PP_MODEL_DIR`.

After selecting one model/config on VAL, freeze all artifacts. Only then:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/pp-full --final-test --i-understand-this-is-the-single-final-run
```

The exclusive final-stage marker is created before evaluation and is not removed on failure. Investigate failures without repeating model selection or deleting the marker. This guard is per output directory; copies of files cannot be globally controlled by code. See `docs/TEST_SPLIT_POLICY.md` for prior aggregate/DAE exposure and the honest held-out description.
