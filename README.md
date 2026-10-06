# ProxyPatient

Synthetic Scenario Exploration for Elevated Glucose (Proxy)

ProxyPatient explores synthetic scenarios for **elevated glucose (proxy)**. Its main model is a trained conditional variational autoencoder (CVAE), designed to fit NFHS-5 India survey data. It generates new profiles and glucose readings; this checkout demonstrates the software with mock-trained weights, and claims no verified real-data model result.

The user selects eight conditions: sex, age band, residence, wealth quintile, BMI band, hypertension, tobacco and alcohol. Supported state is optional. Starting from a full baseline, the user changes conditions and requests a synthetic cohort. Glucose is generated as the outcome, never entered as a condition or imputed.

The API returns the elevated glucose (proxy) percentage with a 95% Wilson Monte Carlo interval, comparison differences from baseline, up to five labelled representative synthetic examples per scenario, run banners and validation-panel data when a compatible measured report exists. Real-mode examples are withheld unless a packaged nearest-record heuristic passes; mock examples carry a MOCK label. A client can display "Synthetic Cohort Generated", the total count and example cards. This repository contains the API and model workflow; a frontend/card renderer is not included.

The outcome is generated glucose at or above the configured threshold (currently 200 mg/dL). The interval describes generated-cohort variation conditional on the fitted model. This is not a diagnosis, individual prediction, LLM or agent workflow, causal analysis, or model of change over time. GRU/CNN variants are optional ablations, and local token annotations are optional; the main model is the MLP CVAE. [Specification decisions](docs/SPEC_VS_BUILD.md) record differences from the supplied writeup excerpts.

## What we present / known limitations

The presentation slice runs an explicitly labelled MOCK checkpoint, full baseline profiles, what-if changes, decoder generation, and computed outcome summaries. Every demo response carries a banner; mock numbers are not NFHS-5 results. Real mode refuses mock checkpoints, missing artifacts, and incompatible model/configuration fingerprints.

The fitted model scope is complete encoded TRAIN respondents with measured BMI, known blood-pressure status and known glucose, including both sexes. Modeling is unweighted; neither the fitted model nor its synthetic rate is a survey population estimate. Height and derived weight are optional if per-sex measurement support is inadequate. Historical reference aggregates have a broader all-known-glucose scope and are not used as full scenario presets.

The Wilson Monte Carlo interval describes variation in a generated cohort conditional on a fixed fitted model. It does not measure model or survey uncertainty. Rejection and clipping diagnostics are visible. Fidelity, tail behavior, disclosure risk, the clinical meaning of units, and scenario support require private real-data validation. The validation panel reads measured packaged TRAIN/VAL reports; private-data quality is unverified. P3's schemas/phrases are reconciled, and the parser remains rule-based and demo-only.


| run_type | Banner | Meaning |
|---|---|---|
| mock | DEMO (mock data) | Constructed mock software demonstration; not NFHS-5 evidence |
| quick | PRELIMINARY (quick run) | Reduced, untuned private TRAIN/VAL run; indicative only |
| full | FULL RUN | Configured full training budget; scientific validation still required |

Every JSON response carries run_type, preliminary, demo and status_banner; no compatible checkpoint means unavailable readiness. Real mode rejects mock/missing-run-type checkpoints; explicit demo mode accepts only mock. Quick reserves eligible rows per sex and stratifies by outcome: 2,000 complete rows per sex versus 5,000 for full. Configured quick defaults are 100,000 CVAE rows/six epochs, 20,000 baseline rows/15 epochs and 10,000 requested evaluation rows. TVAE/CTGAN are included unless explicitly skipped; GRU/CNN ablations are excluded. timings.json records measured stage seconds/peak RSS.

The preset demo builder creates three supported full mock profiles. The pipeline mock benchmark package independently demonstrates measured report ingestion; it need not provide three supported profiles. Never attach a different fitted model's reports to a preset demo. Follow the runbook for both flows.

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
| POST `/generate`, `/compare` | Synthetic outcome summaries, bounded labelled examples, provenance and diagnostics |
| GET `/validation`, `/model-comparison` | Verified packaged evaluation/model table, or explicit pending with null metrics |
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

Real serving: `PP_MODEL_DIR=/private/proxypatient-quick/private_outputs PP_REPORT_DIR=/private/proxypatient-quick/safe_outputs python -m uvicorn backend.main:app`. Unset demo and artifact-override variables first. Reports come from the matching package; a passing software smoke run is not a fidelity certificate. Missing/stale reports remain pending. See [API contract](docs/API_CONTRACT.md), [demo runbook](docs/DEMO_RUNBOOK.md) and [viva notes](docs/VIVA_NOTES.md).

Freeze the selected model/configuration before the single final evaluation:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-full --final-test --i-understand-this-is-the-single-final-run
```

See [test split policy](docs/TEST_SPLIT_POLICY.md). The held-out split's aggregate statistics were previously computed; it is not an untouched test set. Completed final runs cannot repeat. If a previous run started and wrote no metrics, retry only after confirming the crash with `--confirm-previous-final-run-crashed`; unchanged frozen artifact hashes and an exclusive directory lock are required. Never delete the marker to tune models.

The historical documented test size is 119,794, not independently verified here. V1 rates, DAE losses and missingness statistics are historical and are not current model results.

## Frontend

The React + TypeScript client lives in `frontend/`. From that directory:

```bash
npm install
npm run dev
```

Copy `frontend/.env.example` to `frontend/.env.local` to configure the client:

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_DATA_MODE=demo
```

`VITE_DATA_MODE=demo` uses the clearly labelled local interface demo and does not require FastAPI. Set `VITE_DATA_MODE=api` to call the backend; API errors are shown directly and do not fall back to demo results. `VITE_API_BASE_URL` sets the FastAPI origin. In API mode, start FastAPI as described above; model artifacts and supported profiles must be available for generation. The frontend does not need private survey paths or files.

Routes: `/` (Home), `/explore` (profile builder and generation), `/scenarios` (session history), `/compare` (what-if comparison), `/validation` (measured reports), and `/how-it-works` (pipeline and scope).

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
