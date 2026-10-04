# ProxyPatient

Synthetic Scenario Exploration for Elevated Glucose (Proxy)

ProxyPatient generates new synthetic cohorts with a conditional VAE trained on Indian NFHS-5 survey data. The displayed outcome is **elevated glucose (proxy)**: generated random capillary glucose at or above the threshold in `config.yaml` (currently 200 mg/dL). Scenario differences are descriptive, not causal effects. This is not a diagnosis, individual prediction, or treatment recommendation.

## What we present / known limitations

The presentation slice runs an explicitly labelled MOCK checkpoint, full baseline profiles, what-if changes, decoder generation, and computed outcome summaries. Every demo response carries a banner; mock numbers are not NFHS-5 results. Real mode refuses mock checkpoints, missing artifacts, and incompatible model/configuration fingerprints.

The fitted model scope is complete encoded TRAIN respondents with measured BMI, known blood-pressure status and known glucose, including both sexes. Modeling is unweighted; neither the fitted model nor its synthetic rate is a survey population estimate. Height and derived weight are optional if per-sex measurement support is inadequate. Historical reference aggregates have a broader all-known-glucose scope and are not used as full scenario presets.

The Wilson Monte Carlo interval describes variation in a generated cohort conditional on a fixed fitted model. It does not measure model or survey uncertainty. Rejection and clipping diagnostics are visible. Fidelity, tail behavior, disclosure risk, the clinical meaning of units, and scenario support require private real-data validation. P3's formal report is pending; the optional parser is rule-based and demo-only.

## CPU development quickstart

Python 3.11 is tested. Install pinned requirements and select CPU PyTorch explicitly:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest -q
python -m models.make_demo_checkpoint --out-dir outputs/demo
PP_DEMO_MOCK=1 PP_MODEL_DIR=outputs/demo python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Use `/profiles` to choose one of three supported FULL demo profiles, then submit its complete condition dictionary to `/generate`. All eight non-state conditions are required: sex, age_band, residence, wealth_quintile, bmi_band, hypertension, tobacco and alcohol. State is optional and its supported raw codes come from `/options`. Sex accepts 0/1 or female/male and normalizes to 0/1. Unknown fields, glucose-related input keys, and unsupported values return HTTP 422.

The only partial-profile option is explicit demo/development `PP_ALLOW_PARTIAL_PROFILE=1`; it warns that independent marginals are used. Default real serving never resamples real rows or fabricates glucose.

| Endpoint | Purpose |
|---|---|
| GET `/health` | Real/demo/unavailable readiness and model fingerprint |
| GET `/schema`, `/options` | Raw-variable dictionary and supported model options |
| GET `/profiles` | Full supported TRAIN profiles, each backed by at least 500 encoded rows |
| POST `/generate`, `/compare` | Synthetic outcome summaries, provenance, scope and diagnostics |
| GET `/validation` | Explicit pending status until a real-run report exists |
| POST `/parse` | Demo-only proposed conditions, negation handling, no confidence claim; requires confirmation |

## One-command model workflow

The private NFHS files must remain on the authorized holder's machine and must never be uploaded to this cloud task, an AI chat or the public repository. No private data is bundled. `preprocess.pkl` is a historical all-data artifact and is not used by the current P2 workflow.

Smoke the full pipeline using in-memory MOCK data:

```bash
python -m models.run_all --mock --quick --out-dir outputs/mock-run
```

On the authorized private machine, after checking the review's codebook/licence questions and rebuilding affected preprocessing:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-quick --quick
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-full --full
```

Outputs are separated into `safe_outputs/` (aggregate JSON/Markdown and checksum manifest) and `private_outputs/` (weights, preprocessor, marginals, supported profiles and baseline pickles). Never commit the private directory. Review even aggregate outputs before public release. Each stage records runtime and peak memory. `--skip-baselines` and `--skip-dae-benchmark` are explicit optional omissions recorded in the manifest.

Real serving: `PP_MODEL_DIR=/private/proxypatient-full/private_outputs python -m uvicorn backend.main:app`. No demo flag. The formal validation report must be supplied separately through the documented adapter; a passing software smoke run is not a fidelity certificate.

Freeze the selected model/configuration before the single final evaluation:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-full --final-test --i-understand-this-is-the-single-final-run
```

See [test split policy](docs/TEST_SPLIT_POLICY.md). The held-out split's aggregate statistics were previously computed; it is not an untouched test set. The historical documented test size is 119,794, not independently verified here. V1 rates, DAE losses and missingness statistics are historical and are not current model results.

## Files

- `preprocess_v2.py`, `split_and_aggregate_v2.py`, `train_dae_v2.py`: private-holder preparation; do not run against survey data in this sandbox.
- `models/run_all.py`: guarded orchestration and packaging.
- `models/data.py`, `artifacts.py`, `privacy.py`: TRAIN-only encoding, artifact identity and suppression.
- `models/train_cvae.py`, `train_baselines.py`, `eval_dev.py`, `check_dae_benchmark.py`: model stages.
- `backend/`: strict version 2 API and decoder-only serving.
- `tests/`: in-memory mock tests; no respondent files.
- `legacy/`: historical v1 executables; do not run.
- `docs/`: review, presentation scope, test policy, implementation log, backlog and open questions.

DHS raw data is subject to the team's approved access agreement. The repository's research statement does not establish a software licence or authorize cloud sharing/model redistribution; those remain team decisions.
