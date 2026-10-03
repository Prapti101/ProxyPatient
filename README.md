# ProxyPatient 🇮🇳
### Synthetic Patient Scenario Generation for Diabetes Risk Awareness

> **Strictly Generative AI. NOT a diagnostic tool. NOT agentic.**
> All results are labelled "elevated glucose (proxy)" — never "diabetes diagnosis".

---

## Project Overview

ProxyPatient learns patterns from the **NFHS-5 India (2019-21)** national health survey,
generates synthetic patient cohorts under **"what-if" conditions**, and computes
outcome statistics for health awareness and research purposes.

**It is NOT:**
- A clinical diagnosis tool
- An individual prediction system
- An agentic system (no LangChain, no AutoGen, no tool-calling)

**It IS:**
- A strictly generative AI pipeline (CVAE + baselines)
- A population-level scenario comparison tool
- An SDG-3 health awareness research project

---

## Team Roles

| Member | Role | Key Deliverables |
|---|---|---|
| **P1 (Prapti)** | Data & Backend Lead | Dataset, preprocessing, DAE, FastAPI |
| **P2** | Model Lead | CVAE, CTGAN, TVAE baselines |
| **P3** | Validation & NLP Lead | Fidelity checks, outcome_stat, BERT parser |
| **P4** | Frontend Lead | React UI, typed service layer |

---

## Project Pipeline (v2 — current)

```
NFHS-5 India Dataset + Household Member File (IAPR7EDT)
       ↓
  Preprocessing v2 (preprocess_v2.py)
  [BP variables, men's BMI from household file, bmi_measured flag]
       ↓
  Stratified Split + Aggregates v2 (split_and_aggregate_v2.py)
  [SHA-256 verified: same row membership as v1]
       ↓
  Denoising Autoencoder v2 (train_dae_v2.py)
  [Sporadic covariates only. BMI/BP not imputed.]
       ↓
  CVAE Generative Model ← P2 builds this
       ↓
  User picks baseline profile + what-if conditions
       ↓
  Outcome stat computed from generated cohort ← P3 builds this
       ↓
  Fidelity Validation ← P3 builds this
       ↓
  Scenario Cards + Comparison UI ← P4 builds this
```

---

## Dataset

- **Primary:** NFHS-5 India (2019-21), MoHFW/IIPS, DHS Program
  - Women: 7,24,115 respondents (15–49)
  - Men: 1,01,839 respondents (15–54)
  - **Outcome variable:** `sb74` / `smb74` — random capillary glucose (mg/dL)
  - **Outcome threshold:** ≥ 200 mg/dL → "elevated glucose (proxy)"

> ⚠️ Raw `.DTA` files and processed `.parquet` splits are **NOT in this repo**.
> They are shared privately with team members. DHS data cannot be redistributed publicly.
> See [DHS Program](https://dhsprogram.com) for access.

**FORBIDDEN datasets (never use):**
- Diabetes 130-US Hospitals
- Pima Indians
- UCI Early Stage Diabetes (Sylhet, Bangladesh)

---

## Repository Structure

```
ProxyPatient/
│
├── config.yaml                      # Project-wide settings (thresholds, seeds, paths)
├── preprocess.py                    # Stage 3: Data cleaning pipeline
├── split_and_aggregate.py           # Stage 4: Stratified split + aggregates
├── train_dae.py                     # Stage 6: Denoising Autoencoder training
│
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app (7 endpoints)
│   ├── schemas.py                   # Pydantic request/response models
│   ├── generator_stub.py            # STUB → P2 replaces with generator.py
│   └── outcome_stat_stub.py         # STUB → P3 replaces with outcome_stat.py
│
├── docs/
│   ├── schema.json                  # ALL confirmed DHS variable codes + roles
│   ├── aggregates.json              # Safe population aggregates (n≥30 only)
│   ├── audit_results.md             # Stage 1: Metadata audit findings
│   ├── preprocessing_report.md      # Stage 3: Cleaning report
│   ├── split_report.md              # Stage 4: Split sizes + stratification
│   └── dae_report.md               # Stage 6: DAE training report
│
└── processed/
    ├── dae_weights.pt               # Trained DAE model weights
    ├── dae_fit_stats.pkl            # Normalisation stats for inference
    └── preprocess.pkl               # Fitted StandardScaler + OrdinalEncoder
```

---

## Shared Contracts (Critical — Read Before Coding)

### `generate(condition, n, seed) -> pd.DataFrame`
```python
# P2 must implement this exact signature in backend/generator.py
# condition: dict of what-if variables (NO glucose allowed)
# n: cohort size (int)
# seed: random seed (int)
# Returns: DataFrame without glucose columns
```

### `outcome_stat(df, rule) -> dict`
```python
# P3 must implement this exact signature in backend/outcome_stat.py
# df: generated DataFrame
# rule: {"threshold_mg_dl": 200}
# Returns: {"rate": float, "ci_low": float, "ci_high": float, "n": int}
```

### API Endpoints (FastAPI running on port 8000)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health check |
| GET | `/schema` | Full variable schema |
| GET | `/profiles` | Baseline reference profiles |
| POST | `/generate` | Generate cohort + outcome stat |
| POST | `/compare` | Compare multiple scenarios |
| GET | `/validation` | P3's validation report |
| POST | `/parse` | NLP condition parser (P3) |

Interactive docs at: `http://localhost:8000/docs`

---

## Global Rules (Everyone Must Follow)

1. **No agentic code** — No LangChain, LangGraph, AutoGen, or tool-calling loops
2. **Indian data only** — NFHS-5 primary, no forbidden datasets
3. **Never commit raw data** — No `.DTA`, `.dta`, `.sav`, `.zip`, `.parquet` files
4. **Never guess column names** — Use `docs/schema.json` as single source of truth
5. **Never impute glucose** — `sb74`/`smb74` is ONLY the outcome, never a feature
6. **Never hardcode numbers** — All stats computed from data via code
7. **Always label correctly** — "elevated glucose (proxy)", never "diabetes"
8. **Suppress small cells** — Any aggregate with n < 30 → suppress to null
9. **Fixed seed = 42** — For all splits, training, and generation

---

## Quick Start (P2, P3, P4)

### 1. Clone the repo
```bash
git clone https://github.com/Prapti101/ProxyPatient.git
cd ProxyPatient
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn pydantic pandas pyarrow scikit-learn joblib torch scipy pyreadstat
```

### 3. Get data files from P1 (Prapti — shared privately)
Place these in the `processed/` folder:
- `dae_imputed_train.parquet` ← P2 needs this
- `dae_imputed_val.parquet`   ← P3 needs this
- `preprocess.pkl`            ← P2 and P3 need this

### 4. Start the backend API
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000/docs` to see all endpoints.

### 5. Swap stubs when your code is ready

**P2 — when CVAE is ready:**
In `backend/main.py`, change line 35:
```python
# FROM:
from backend.generator_stub import generate
# TO:
from backend.generator import generate
```

**P3 — when outcome_stat is ready:**
In `backend/main.py`, change line 39:
```python
# FROM:
from backend.outcome_stat_stub import outcome_stat
# TO:
from backend.outcome_stat import outcome_stat
```

---

## Key Numbers (from Stage 4)

| Metric | Value |
|---|---|
| Total respondents | 8,25,954 |
| With glucose readings | 7,98,622 (96.7%) |
| Elevated glucose proxy rate | **2.88%** |
| Train split | 5,81,542 rows |
| Val split | 1,24,618 rows |
| Test split (LOCKED) | 1,19,798 rows |
| DAE final training loss | 0.037650 |
| Missing values after DAE | **0** |

---

## Disclaimer

> This tool generates SYNTHETIC patient scenarios for health-awareness and research
> purposes ONLY. It is NOT a medical diagnosis, clinical assessment, or treatment
> recommendation. "Elevated glucose (proxy)" refers to a random capillary glucose
> reading ≥ 200 mg/dL and cannot distinguish Type 1 from Type 2 diabetes.
> Always consult a qualified healthcare professional.

---

*Source: NFHS-5 India (2019-21), MoHFW/IIPS, DHS Program*
*License: DHS data used under approved research agreement. Raw data not redistributed.*
