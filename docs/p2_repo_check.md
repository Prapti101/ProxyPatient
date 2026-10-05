> Historical snapshot, reviewed and superseded 2026-10-05 (Asia/Kolkata).
> Current implementation and resolved findings: OVERNIGHT_REPORT.md and PRESENTATION_SLICE.md.
> Claims below about missing hypertension/state fields, v2 aggregates, threshold,
> health version, and derived BP definitions are stale; the preflight review confirmed they already existed.

# P2 repo check (T0)

Checked on `main` at commit `1fe0530` (merge of PR #1, P1's v2 work). Order of
authority used: code > `config.yaml` > `docs/schema.json` > P1 handoff > `README.md`.

## Interfaces found in the code

### Generator (`backend/generator_stub.py`)
```python
def generate(condition: dict, n: int = 1000, seed: int = 42) -> pd.DataFrame
```
* The parameter is **`condition`** (singular), not `conditions` as in the P1
  handoff. `backend/main.py` calls it by keyword: `generate(condition=..., n=..., seed=...)`.
  `backend/generator.py` therefore uses `condition` (anything else would break the API).
* `FORBIDDEN_INPUT_CONDITIONS = ["glucose", "glucose_raw", "elevated_glucose_proxy", "sb74", "smb74", "hba1c"]`
  is imported by `main.py` from the stub.
* The stub filters `sex == "female"` (string), resamples REAL rows with
  replacement and adds 5% Gaussian noise (a jittered copy of real rows, which
  is acceptable only as a stub), and loads the v1 file `processed/dae_imputed_combined.parquet`.

### Outcome statistic (`backend/outcome_stat_stub.py`)
```python
def outcome_stat(df: pd.DataFrame, rule: dict, seed: int = 42, n_bootstrap: int = 1000) -> dict
# returns {"rate", "ci_low", "ci_high", "n", "rate_pct"}
```
If `elevated_glucose_proxy` is absent the stub attaches a fake glucose column;
the real generator now supplies the column, so the stub uses it directly.

### Where the stubs are imported (`backend/main.py`, before this PR)
| line | import |
|---|---|
| 35 | `from backend.generator_stub import generate` (36: commented real import) |
| 39 | `from backend.outcome_stat_stub import outcome_stat` (40: commented real import) |
| 42-48 | `from backend.schemas import (...)`. There is **no parser import at line 43**: the `/parse` endpoint (line 282) is an inline rule-based stub. |
| 49 | `from backend.generator_stub import FORBIDDEN_INPUT_CONDITIONS` |

This PR replaces lines 34-36 with an env-var switch: `PP_GENERATOR=real` →
`backend.generator`, otherwise the stub (unchanged default). `model_used` in
`/generate` now reports which one ran.

### Request/response schemas (`backend/schemas.py`)
* `Condition`: `sex` ("female"/"male"), `age_band` (str), `residence`
  ("urban"/"rural"), `wealth_quintile` ("1".."5"), `bmi_band`, `tobacco` ("0"/"1"),
  `alcohol` ("0"/"1"), all optional. Validators exist for sex, residence,
  wealth and bmi_band only (age_band, tobacco, alcohol are unvalidated).
  **No `hypertension` and no `state` field.**
* `GenerateRequest`: `condition: Condition`, `n` (100-10,000, default 1000), `seed` (42).
* `GenerateResponse`: `condition`, `outcome_stat: OutcomeStat{rate, ci_low, ci_high, n, rate_pct}`,
  `disclaimer`, `model_used`, `note`.
* `CompareRequest`: `scenarios: List[CompareScenario{label, condition, n, seed}]` (2-5).
* `CompareResponse`: `scenarios: List[ScenarioResult{label, condition, outcome_stat, delta_pp}]`, `disclaimer`, `note`.

### `docs/schema.json` keys
Top level: `_meta`, `women`, `men`, `derived_columns`. `women.variables`:
v005, v012, v013, v024, v025, v106, v190, v191, v437, v445, s305, s306, sb19,
sb55, sb56, sb57, sb73, sb74, v463a, v463z, s707, s708, s711, s720, s721,
sb14c, sb16. `men.variables`: mv005, mv012, mv024, mv025, mv106, mv190, smb305,
smb306, smb19, smb55, smb56, smb57, smb73, smb74, mv463a, mv463z, sm619, smb14c.
`derived_columns`: elevated_glucose_proxy, any_tobacco, age_band, bmi_band,
hypertension (still "Exact logic TBD"). No v2 columns (bmi_measured,
bp_measured, systolic/diastolic, on_bp_medication) are described.

### `git ls-files processed/`
```
processed/dae_fit_stats.pkl   (v1 DAE normalisation stats: medians/SDs/category maps)
processed/dae_weights.pt      (v1 DAE weights, 118 KB)
processed/preprocess.pkl      (v1 StandardScaler + OrdinalEncoder, fit_rows count)
```
I loaded all three: they hold fitted parameters and aggregate statistics, **no
respondent-level rows**. They are listed in `.gitignore` but were committed
before that, so they stay tracked. `dae_weights.pt` matches the leak-check
pattern `\.pt$`; whether to `git rm --cached` these is P1's call (not changed here).

## `.gitignore` fix
Added wildcards: `*.parquet`, `*.csv`, `*.dta`, `*.DTA`, `*.sav`, `*.zip`,
`*.feather`, `processed/*.parquet`, `processed/*.pkl`, `processed/MANIFEST.json`,
`models/*.pt`, `models/*.pkl`, `outputs/`.

## Discrepancies for P1 / P3 / P4
1. **API cannot send `hypertension` or `state`.** `Condition` lacks both fields,
   so the hypertension what-if (a config conditioning variable) is unreachable
   from the UI. The generator accepts them; P1 should add `hypertension: "0"/"1"`
   (and optional `state: int`) to `Condition`.
2. **Sex encoding differs:** config `whatif_options.sex.options` is `["0","1"]`;
   `schemas.py`, the stub and `/parse` use `"female"/"male"`; `aggregates_v2.json`
   `by_sex` uses keys `"0"/"1"` while `/profiles` reads `"female"/"male"`.
   The generator accepts both forms.
3. **`main.py` loads `docs/aggregates.json` (v1)** while config says
   `aggregates_file: docs/aggregates_v2.json`. Switching to v2 breaks `/profiles`
   (key mismatch above).
4. `main.py` hard-codes `RULE = {"threshold_mg_dl": 200}` instead of reading config.
5. `age_band` is not validated in `schemas.py`; both `35-49` and `35-54` are
   allowed for either sex. The generator harmonises them (35+) and echoes the
   sex-appropriate label.
6. README (stale, v1): generator "returns a DataFrame without glucose columns"
   (fixed in this PR); `outcome_stat(df, rule)` signature lacks `seed`/`rate_pct`;
   quick-start lists v1 files (`dae_imputed_train.parquet`, `preprocess.pkl`);
   test split 1,19,798 (handoff: 119,794); DAE loss 0.037650; repository tree
   lists only v1 scripts.
7. The P2 prompt quotes the stub docstring as saying "Scale: MODEL-READY"; no
   such text exists in the repo. The stub docstring's wrong claim was that the
   generator must not return glucose; a note was added there.
8. `config.yaml` `external_validation` (NMB-2017) is stale: there is no data
   file, so there is no external validation (comment added).
9. `docs/schema.json`: hypertension logic still "TBD" though `DECISIONS.md`
   defines it; v2 columns are undocumented.
10. P1's `dae_impute.benchmark_dae_vs_median` reads the already DAE-imputed
    validation file (circular) and depends on a hard-coded Windows path;
    replaced for P2 purposes by `models/check_dae_benchmark.py`.
11. `HealthResponse.version` is "1.0.0" while config `project.version` is "2.0.0".
12. Deprecations (warnings only): FastAPI `on_event`, Pydantic v1 `validator`
    and `.dict()` under Pydantic 2.
