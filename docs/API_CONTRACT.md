# API contract

Version 2.0.0. Request/response fields below follow backend/schemas.py and backend/main.py. `/openapi.json` is the machine-readable schema. Mock examples were captured from executed mock runs on 2026-10-05; these numbers are software examples, not NFHS-5 results. Demo profiles and pipeline evaluation use separate fitted mock checkpoints; fingerprints identify which one produced each response.

## Run provenance on every JSON response

| run_type | preliminary | demo | status_banner |
|---|---|---|---|
| mock | false | true | DEMO (mock data) |
| quick | true | false | PRELIMINARY (quick run) |
| full | false | false | FULL RUN |
| null (no compatible model) | false | false | UNAVAILABLE |

Quick responses include `run_note: "reduced, untuned run; results are indicative only"`. Mock JSON also includes the existing `banner`. Explicit demo mode labels even unavailable/error responses as mock demonstrations, with readiness reported separately. `X-ProxyPatient-Status` carries the banner for non-JSON responses. Status fields also appear on errors, options and profiles. A banner does not replace checking `/health.status`.

## Conditioning and requests

All eight conditions are REQUIRED for generation and every comparison scenario: `sex`, `age_band`, `residence`, `wealth_quintile`, `bmi_band`, `hypertension`, `tobacco`, `alcohol`. `state` is optional.

| Condition | Accepted values |
|---|---|
| sex | 0/1 or female/male, normalized to 0/1 |
| age_band | Configured labels: women 15-24 / 25-34 / 35-49; men 15-24 / 25-34 / 35-54; output harmonizes the oldest label by sex |
| residence | urban / rural |
| wealth_quintile | Integer 1..5 |
| bmi_band | underweight / normal / overweight / obese |
| hypertension, tobacco, alcohol | Integer 0/1 |
| state | Supported raw integer codes returned by `/options`; no invented contiguous mapping |

Unknown keys and glucose/HbA1c keys are rejected. Glucose is the outcome, never a condition. Values are not silently truncated. `n` is an integer 100..10,000, default from configuration; `seed` is an integer, default from configuration. Objects reject extra fields. Full profiles remain required unless the explicit demo-only partial-profile option is enabled; the UI must always submit all eight conditions.

## Endpoint reference

| Method/path | Request | Response |
|---|---|---|
| GET /health | None | HealthResponse: status ok/unavailable, mode real/demo/unavailable, model_fingerprint (nullable), detail (nullable), version, model, disclaimer_present and run provenance |
| GET /schema | None | Raw-variable dictionary from docs/schema.json plus supported_state_codes and run provenance |
| GET /options | None | Allowed values, sex-specific age options, supported states, required_profile_keys and run provenance |
| GET /profiles | None | ProfilesResponse: profiles [{label, condition, description, n_train}], source, reference_scope, disclaimer and run provenance. Each full preset has >=500 encoded TRAIN rows; fewer than three presets returns 503 |
| POST /generate | {condition, n?, seed?, n_examples?} | GenerateResponse described below |
| POST /compare | {scenarios: [{label, condition, n?, seed?, n_examples?}, ...]} (2..5) | CompareResponse: scenarios [GenerateResponse fields + label + delta_pp], model_fingerprint, disclaimer, weighting, uncertainty_note, note and run provenance |
| GET /validation | None | ReportResponse described below: complete measured report or pending with null metrics |
| GET /model-comparison | None | ReportResponse with measured per-model table; metrics null, models contains canonical metric groups when complete. No ranking |
| POST /parse | {text, baseline?}; text 1..2,000 chars, baseline allows a partial Condition | ParseResponse: parsed_condition (proposal/changes only), raw_text, confidence null, parser, requires_confirmation true, unresolved [], optional_token_hook and run provenance. Demo-only |
| GET /openapi.json | None | Machine-readable schemas; JSON also carries run provenance |
| GET /docs, /redoc | None | Interactive HTML schema documentation; status header |

`GenerateResponse` includes condition, effective_conditions, outcome_stat, model_fingerprint, sampling_diagnostics, synthetic, model_used, scope, weighting, uncertainty_note, disclaimer, note and run provenance. No real respondent rows or bulk synthetic exports are returned; bounded synthetic examples are described below. `outcome_stat` has rate (fraction), rate_pct, ci_low/ci_high (fractions), n, suppressed dropped_nonfinite, and monte_carlo_interval metadata. The Wilson interval measures finite-cohort Monte Carlo variation conditional on the fitted model, not survey or fitted-model uncertainty. Sampling diagnostics disclose constraint rejection, clipping, seed and any independent-marginal fill. `delta_pp` is percentage-point change from the first scenario, not a causal effect.

`ReportResponse` has status, metrics, models, evaluation_scope, real_reference, model_fingerprint, note and run provenance. `metrics` contains the canonical `eval_dev` object for complete validation: `_meta`, `real`, `models`, and run provenance. Each model contains generation/coverage plus marginals (KS/Wasserstein/TVD), correlation, tail, conditional_rate_fidelity, direction, consistency, nearest_record, tstr, real_vs_synthetic, subgroup_fidelity. Metrics use a shared retained subset; small respondent cells are null. Classification/disclosure metrics are diagnostics, not guarantees. Optional CVAE ablation reports appear only when measured artifacts exist.

The report reader uses `PP_REPORT_DIR` (default `PP_MODEL_DIR/safe_outputs`). Normal packages have sibling safe_outputs/private_outputs directories, so set both paths explicitly. It verifies the manifest checksums, served model fingerprint, run type, mock mode and VAL split. Missing, incompatible or insufficient-support reports return `status: pending`, `metrics: null`, `models: null`; no numbers are invented. P3 templates under docs/p3 are never treated as measured outputs.

## Representative synthetic examples

`n_examples` is a strict integer 1..5 per scenario, default 3. Unknown fields and `export_sample` are rejected. The internal generator's export option remains disabled. Each scenario returns `examples` (a list or null), `examples_status`, and `examples_note`. `SyntheticExample` contains `example_id`, `synthetic: true`, `label`, `effective_conditions`, `generated_features`, integer `glucose_raw` in mg/dL, boolean `elevated_glucose_proxy`, and boolean `illustrative_elevated`.

Generated features include integer age in years, coded education and BP-checked flag, and BMI, waist/hip in cm and available height in cm/derived weight in kg to one decimal. Optional generated BP readings are in mmHg. Conditions reflect each selected row, including sampled state/any explicit demo marginal fills. Glucose is rounded for display; the elevated flag is calculated from unrounded generated glucose at the configured threshold, so rounded glucose alone can appear to straddle the threshold. Examples are selected from the freshly generated cohort, never from respondent tables or a precomputed example bank.

Stable glucose sorting selects nearest ranked rows at the 25th/50th/75th percentiles (K=3), adding the 10th/90th for K=5. K=1 selects the median, K=2 the quartiles, K=4 adds the 10th percentile. If any row is elevated, the least elevated generated row receives `illustrative_elevated: true`; it replaces the last selection if absent. This resolves the conflicting extra-example and hard-cap requirements: the count never exceeds K. This illustration changes neither the full-cohort rate nor its interval; the examples are not a statistical sample for estimating rates.

Statuses are `available: MOCK`, `available: near-copy heuristic passed`, or `withheld: near-copy check missing or not passed` (with `examples: null`). MOCK labels include both MOCK and "SYNTHETIC example, not a real person". Real labels retain the synthetic warning.

In real mode the existing checksum-verified, matching-checkpoint/run-type, VAL package reader must return a complete report with at least 30 retained evaluation rows. Its primary CVAE `nearest_record` diagnostic must have at least 30 real and generated queries, finite positive real-to-TRAIN median, generated median at least 0.5 times that median, generated minimum distance above 0.00001, and zero exact-copy share. Thresholds are in `privacy.examples_near_copy` in config.yaml. Missing/suppressed/nonfinite fields and older reports without `minimum` withhold examples. The evaluator measures distances in TRAIN-standardised numeric space using sampled TRAIN references and queries; it does not measure every displayed example or its rounded representation. **This is a heuristic, not a privacy guarantee** or proof of scenario support. Checksums bind the supplied report to artifacts; they do not authorize disclosure. Rates remain available when examples are withheld.

The API is synchronous. While waiting, clients may show an indeterminate loading indicator, then "Synthetic Cohort Generated", `outcome_stat.n`, labelled cards (or the withholding status), comparison `delta_pp`, banners and validation. Do not fabricate percent-complete progress. No client code exists in this repository; rendering cards remains the frontend team's task.

## UI flow and errors

Read `/health`, then `/options` and `/profiles`. Use a preset's full condition dictionary as baseline; copy it and apply confirmed changes. Do not invent missing baseline conditions. A parse response never starts generation: show the proposed changes, retain the complete baseline, resolve any unresolved terms, then require explicit user confirmation before calling generation/comparison. Rule-based parsing remains the actual parser; optional offline token annotations do not create conditions.

Show status_banner, scope, unweighted label, uncertainty, clipping/rejection diagnostics and disclaimer with results. Handle 422 by displaying field/detail messages and retaining user inputs. Handle 503 as model/profile unavailable; do not substitute fake rates. `/health` can return HTTP 200 with unavailable status. `/parse` returns 403 outside explicit demo mode. Runtime failures may return 500. API errors use a `detail` string or Pydantic detail array plus run provenance. Report absence is HTTP 200 pending. Client API types/UI alignment SKIPPED: no package.json, frontend/ or src/ client exists in this repository.

## Captured MOCK examples

Responses below are abridged to selected actual fields where noted; `/openapi.json` gives every field/type. All displayed numeric values were captured from executed mock requests. No real-data result is implied.


### GET /health

```json
{
  "run_type": "mock",
  "preliminary": false,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "demo": true,
  "status": "ok",
  "mode": "demo",
  "model_fingerprint": "8b6406c967b5cc0ff07b23cb819579ac8709085538f2f426d673e16ca9e31b11",
  "detail": null,
  "version": "2.0.0",
  "model": "CVAE",
  "disclaimer_present": true,
  "banner": "MOCK DEMO — not NFHS-5; synthetic rates are not research results"
}
```


### GET /options (abridged)

```json
{
  "sex": [
    0,
    1
  ],
  "residence": [
    "urban",
    "rural"
  ],
  "bmi_band": [
    "underweight",
    "normal",
    "overweight",
    "obese"
  ],
  "required_profile_keys": [
    "sex",
    "age_band",
    "residence",
    "wealth_quintile",
    "bmi_band",
    "hypertension",
    "tobacco",
    "alcohol"
  ],
  "run_type": "mock",
  "status_banner": "DEMO (mock data)"
}
```


### GET /schema (abridged)

```json
{
  "supported_state_codes": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37
  ],
  "run_type": "mock",
  "status_banner": "DEMO (mock data)"
}
```


### GET /profiles (one of three captured presets)

```json
{
  "source": "Encoded TRAIN model scope; unweighted sample",
  "run_type": "mock",
  "status_banner": "DEMO (mock data)",
  "profiles": [
    {
      "label": "Supported TRAIN profile 1",
      "condition": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0
      },
      "description": "Observed joint TRAIN cell; model scope, unweighted sample",
      "n_train": 1242
    }
  ]
}
```


### POST /generate request

```json
{
  "condition": {
    "sex": 1,
    "age_band": "25-34",
    "residence": "rural",
    "wealth_quintile": 3,
    "bmi_band": "obese",
    "hypertension": 0,
    "tobacco": 0,
    "alcohol": 0
  },
  "n": 1000,
  "seed": 42,
  "n_examples": 3
}
```


### POST /generate response (captured MOCK run)

```json
{
  "run_type": "mock",
  "preliminary": false,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "demo": true,
  "examples": [
    {
      "example_id": "S1-A",
      "synthetic": true,
      "label": "MOCK \u2014 SYNTHETIC example, not a real person",
      "effective_conditions": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "alcohol": 0,
        "state": 32,
        "tobacco": 0
      },
      "generated_features": {
        "age": 29,
        "bmi": 33.7,
        "waist_cm": 85.0,
        "hip_cm": 107.1,
        "height_cm": 138.9,
        "weight_kg": 65.0,
        "education": 3,
        "bp_ever_checked": 1
      },
      "glucose_raw": 88,
      "elevated_glucose_proxy": false,
      "illustrative_elevated": false
    },
    {
      "example_id": "S1-B",
      "synthetic": true,
      "label": "MOCK \u2014 SYNTHETIC example, not a real person",
      "effective_conditions": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "alcohol": 0,
        "state": 29,
        "tobacco": 0
      },
      "generated_features": {
        "age": 30,
        "bmi": 32.0,
        "waist_cm": 74.5,
        "hip_cm": 92.8,
        "height_cm": 151.3,
        "weight_kg": 73.3,
        "education": 2,
        "bp_ever_checked": 0
      },
      "glucose_raw": 105,
      "elevated_glucose_proxy": false,
      "illustrative_elevated": false
    },
    {
      "example_id": "S1-C",
      "synthetic": true,
      "label": "MOCK \u2014 SYNTHETIC example, not a real person",
      "effective_conditions": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "alcohol": 0,
        "state": 10,
        "tobacco": 0
      },
      "generated_features": {
        "age": 29,
        "bmi": 31.7,
        "waist_cm": 87.9,
        "hip_cm": 98.1,
        "height_cm": 155.5,
        "weight_kg": 76.6,
        "education": 1,
        "bp_ever_checked": 0
      },
      "glucose_raw": 204,
      "elevated_glucose_proxy": true,
      "illustrative_elevated": true
    }
  ],
  "examples_status": "available: MOCK",
  "examples_note": "Nearest-record screening is a heuristic, not a privacy guarantee. Examples do not determine the cohort statistic.",
  "condition": {
    "sex": 1,
    "age_band": "25-34",
    "residence": "rural",
    "wealth_quintile": 3,
    "bmi_band": "obese",
    "hypertension": 0,
    "tobacco": 0,
    "alcohol": 0,
    "state": null
  },
  "effective_conditions": {
    "sex": 1,
    "age_band": "25-34",
    "residence": "rural",
    "wealth_quintile": 3,
    "bmi_band": "obese",
    "hypertension": 0,
    "tobacco": 0,
    "alcohol": 0
  },
  "outcome_stat": {
    "rate": 0.01,
    "ci_low": 0.005441,
    "ci_high": 0.018309,
    "n": 1000,
    "rate_pct": 1.0,
    "dropped_nonfinite": null,
    "monte_carlo_interval": {
      "low": 0.005441,
      "high": 0.018309,
      "level": 0.95,
      "method": "Wilson",
      "note": "Conditional on fitted model; not model or survey uncertainty"
    }
  },
  "model_fingerprint": "8b6406c967b5cc0ff07b23cb819579ac8709085538f2f426d673e16ca9e31b11",
  "sampling_diagnostics": {
    "n": 1000,
    "clipped_share": 0.026,
    "glucose_clipped_share": 0.0,
    "first_pass_inconsistent_share": 0.806,
    "redrawn_rows": 3969,
    "clipped_after_max_rounds": null,
    "consistent_without_clipping_share": 0.999,
    "seconds": 0.0308,
    "filled_from_marginals": [
      "state"
    ],
    "seed": 42,
    "label": "SYNTHETIC",
    "warning": null,
    "rejection_rate": 0.806
  },
  "synthetic": true,
  "banner": "MOCK DEMO \u2014 not NFHS-5; synthetic rates are not research results",
  "model_used": "CVAE",
  "scope": "Measured BMI, known blood-pressure status and finite glucose; both sexes; complete encoded TRAIN rows.",
  "weighting": "unweighted sample",
  "uncertainty_note": "Monte Carlo variation of the synthetic cohort conditional on the fitted model; does not measure model or survey uncertainty.",
  "disclaimer": "SYNTHETIC scenario exploration for elevated glucose (proxy). Not a medical diagnosis, individual prediction, treatment recommendation, or causal effect. Consult a qualified healthcare professional.",
  "note": "Descriptive synthetic scenario comparison; what-if is not causal."
}
```


### POST /compare request

```json
{
  "scenarios": [
    {
      "label": "baseline",
      "condition": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0
      },
      "n": 1000,
      "seed": 42,
      "n_examples": 3
    },
    {
      "label": "alternative",
      "condition": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 4,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0
      },
      "n": 1000,
      "seed": 42,
      "n_examples": 3
    }
  ]
}
```


### POST /compare response (captured MOCK run)

```json
{
  "run_type": "mock",
  "preliminary": false,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "demo": true,
  "scenarios": [
    {
      "run_type": "mock",
      "preliminary": false,
      "status_banner": "DEMO (mock data)",
      "run_note": null,
      "demo": true,
      "examples": [
        {
          "example_id": "S1-A",
          "synthetic": true,
          "label": "MOCK \u2014 SYNTHETIC example, not a real person",
          "effective_conditions": {
            "sex": 1,
            "age_band": "25-34",
            "residence": "rural",
            "wealth_quintile": 3,
            "bmi_band": "obese",
            "hypertension": 0,
            "alcohol": 0,
            "state": 32,
            "tobacco": 0
          },
          "generated_features": {
            "age": 29,
            "bmi": 33.7,
            "waist_cm": 85.0,
            "hip_cm": 107.1,
            "height_cm": 138.9,
            "weight_kg": 65.0,
            "education": 3,
            "bp_ever_checked": 1
          },
          "glucose_raw": 88,
          "elevated_glucose_proxy": false,
          "illustrative_elevated": false
        },
        {
          "example_id": "S1-B",
          "synthetic": true,
          "label": "MOCK \u2014 SYNTHETIC example, not a real person",
          "effective_conditions": {
            "sex": 1,
            "age_band": "25-34",
            "residence": "rural",
            "wealth_quintile": 3,
            "bmi_band": "obese",
            "hypertension": 0,
            "alcohol": 0,
            "state": 29,
            "tobacco": 0
          },
          "generated_features": {
            "age": 30,
            "bmi": 32.0,
            "waist_cm": 74.5,
            "hip_cm": 92.8,
            "height_cm": 151.3,
            "weight_kg": 73.3,
            "education": 2,
            "bp_ever_checked": 0
          },
          "glucose_raw": 105,
          "elevated_glucose_proxy": false,
          "illustrative_elevated": false
        },
        {
          "example_id": "S1-C",
          "synthetic": true,
          "label": "MOCK \u2014 SYNTHETIC example, not a real person",
          "effective_conditions": {
            "sex": 1,
            "age_band": "25-34",
            "residence": "rural",
            "wealth_quintile": 3,
            "bmi_band": "obese",
            "hypertension": 0,
            "alcohol": 0,
            "state": 10,
            "tobacco": 0
          },
          "generated_features": {
            "age": 29,
            "bmi": 31.7,
            "waist_cm": 87.9,
            "hip_cm": 98.1,
            "height_cm": 155.5,
            "weight_kg": 76.6,
            "education": 1,
            "bp_ever_checked": 0
          },
          "glucose_raw": 204,
          "elevated_glucose_proxy": true,
          "illustrative_elevated": true
        }
      ],
      "examples_status": "available: MOCK",
      "examples_note": "Nearest-record screening is a heuristic, not a privacy guarantee. Examples do not determine the cohort statistic.",
      "condition": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0,
        "state": null
      },
      "effective_conditions": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 3,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0
      },
      "outcome_stat": {
        "rate": 0.01,
        "ci_low": 0.005441,
        "ci_high": 0.018309,
        "n": 1000,
        "rate_pct": 1.0,
        "dropped_nonfinite": null,
        "monte_carlo_interval": {
          "low": 0.005441,
          "high": 0.018309,
          "level": 0.95,
          "method": "Wilson",
          "note": "Conditional on fitted model; not model or survey uncertainty"
        }
      },
      "model_fingerprint": "8b6406c967b5cc0ff07b23cb819579ac8709085538f2f426d673e16ca9e31b11",
      "sampling_diagnostics": {
        "n": 1000,
        "clipped_share": 0.026,
        "glucose_clipped_share": 0.0,
        "first_pass_inconsistent_share": 0.806,
        "redrawn_rows": 3969,
        "clipped_after_max_rounds": null,
        "consistent_without_clipping_share": 0.999,
        "seconds": 0.026,
        "filled_from_marginals": [
          "state"
        ],
        "seed": 42,
        "label": "SYNTHETIC",
        "warning": null,
        "rejection_rate": 0.806
      },
      "synthetic": true,
      "banner": "MOCK DEMO \u2014 not NFHS-5",
      "model_used": "CVAE",
      "scope": "Measured BMI, known blood-pressure status and finite glucose; both sexes; complete encoded TRAIN rows.",
      "weighting": "unweighted sample",
      "uncertainty_note": "Monte Carlo variation of the synthetic cohort conditional on the fitted model; does not measure model or survey uncertainty.",
      "disclaimer": "SYNTHETIC scenario exploration for elevated glucose (proxy). Not a medical diagnosis, individual prediction, treatment recommendation, or causal effect. Consult a qualified healthcare professional.",
      "note": "Descriptive synthetic scenario comparison; what-if is not causal.",
      "label": "baseline",
      "delta_pp": 0.0
    },
    {
      "run_type": "mock",
      "preliminary": false,
      "status_banner": "DEMO (mock data)",
      "run_note": null,
      "demo": true,
      "examples": [
        {
          "example_id": "S2-A",
          "synthetic": true,
          "label": "MOCK \u2014 SYNTHETIC example, not a real person",
          "effective_conditions": {
            "sex": 1,
            "age_band": "25-34",
            "residence": "rural",
            "wealth_quintile": 4,
            "bmi_band": "obese",
            "hypertension": 0,
            "alcohol": 0,
            "state": 36,
            "tobacco": 0
          },
          "generated_features": {
            "age": 31,
            "bmi": 30.6,
            "waist_cm": 89.7,
            "hip_cm": 90.0,
            "height_cm": 166.7,
            "weight_kg": 85.0,
            "education": 2,
            "bp_ever_checked": 1
          },
          "glucose_raw": 89,
          "elevated_glucose_proxy": false,
          "illustrative_elevated": false
        },
        {
          "example_id": "S2-B",
          "synthetic": true,
          "label": "MOCK \u2014 SYNTHETIC example, not a real person",
          "effective_conditions": {
            "sex": 1,
            "age_band": "25-34",
            "residence": "rural",
            "wealth_quintile": 4,
            "bmi_band": "obese",
            "hypertension": 0,
            "alcohol": 0,
            "state": 29,
            "tobacco": 0
          },
          "generated_features": {
            "age": 29,
            "bmi": 31.3,
            "waist_cm": 87.0,
            "hip_cm": 83.3,
            "height_cm": 160.2,
            "weight_kg": 80.3,
            "education": 1,
            "bp_ever_checked": 0
          },
          "glucose_raw": 104,
          "elevated_glucose_proxy": false,
          "illustrative_elevated": false
        },
        {
          "example_id": "S2-C",
          "synthetic": true,
          "label": "MOCK \u2014 SYNTHETIC example, not a real person",
          "effective_conditions": {
            "sex": 1,
            "age_band": "25-34",
            "residence": "rural",
            "wealth_quintile": 4,
            "bmi_band": "obese",
            "hypertension": 0,
            "alcohol": 0,
            "state": 27,
            "tobacco": 0
          },
          "generated_features": {
            "age": 30,
            "bmi": 30.7,
            "waist_cm": 76.5,
            "hip_cm": 89.6,
            "height_cm": 154.7,
            "weight_kg": 73.5,
            "education": 2,
            "bp_ever_checked": 1
          },
          "glucose_raw": 208,
          "elevated_glucose_proxy": true,
          "illustrative_elevated": true
        }
      ],
      "examples_status": "available: MOCK",
      "examples_note": "Nearest-record screening is a heuristic, not a privacy guarantee. Examples do not determine the cohort statistic.",
      "condition": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 4,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0,
        "state": null
      },
      "effective_conditions": {
        "sex": 1,
        "age_band": "25-34",
        "residence": "rural",
        "wealth_quintile": 4,
        "bmi_band": "obese",
        "hypertension": 0,
        "tobacco": 0,
        "alcohol": 0
      },
      "outcome_stat": {
        "rate": 0.008,
        "ci_low": 0.004059,
        "ci_high": 0.015706,
        "n": 1000,
        "rate_pct": 0.8,
        "dropped_nonfinite": null,
        "monte_carlo_interval": {
          "low": 0.004059,
          "high": 0.015706,
          "level": 0.95,
          "method": "Wilson",
          "note": "Conditional on fitted model; not model or survey uncertainty"
        }
      },
      "model_fingerprint": "8b6406c967b5cc0ff07b23cb819579ac8709085538f2f426d673e16ca9e31b11",
      "sampling_diagnostics": {
        "n": 1000,
        "clipped_share": 0.031,
        "glucose_clipped_share": 0.0,
        "first_pass_inconsistent_share": 0.851,
        "redrawn_rows": 5766,
        "clipped_after_max_rounds": null,
        "consistent_without_clipping_share": 0.992,
        "seconds": 0.0274,
        "filled_from_marginals": [
          "state"
        ],
        "seed": 42,
        "label": "SYNTHETIC",
        "warning": null,
        "rejection_rate": 0.851
      },
      "synthetic": true,
      "banner": "MOCK DEMO \u2014 not NFHS-5",
      "model_used": "CVAE",
      "scope": "Measured BMI, known blood-pressure status and finite glucose; both sexes; complete encoded TRAIN rows.",
      "weighting": "unweighted sample",
      "uncertainty_note": "Monte Carlo variation of the synthetic cohort conditional on the fitted model; does not measure model or survey uncertainty.",
      "disclaimer": "SYNTHETIC scenario exploration for elevated glucose (proxy). Not a medical diagnosis, individual prediction, treatment recommendation, or causal effect. Consult a qualified healthcare professional.",
      "note": "Descriptive synthetic scenario comparison; what-if is not causal.",
      "label": "alternative",
      "delta_pp": -0.2
    }
  ],
  "model_fingerprint": "8b6406c967b5cc0ff07b23cb819579ac8709085538f2f426d673e16ca9e31b11",
  "disclaimer": "SYNTHETIC scenario exploration for elevated glucose (proxy). Not a medical diagnosis, individual prediction, treatment recommendation, or causal effect. Consult a qualified healthcare professional.",
  "weighting": "unweighted sample",
  "uncertainty_note": "Monte Carlo variation of the synthetic cohort conditional on the fitted model; does not measure model or survey uncertainty.",
  "note": "Differences describe synthetic scenarios, not causal effects.",
  "banner": "MOCK DEMO \u2014 not NFHS-5; synthetic rates are not research results"
}
```


### POST /parse request

```json
{
  "text": "improved BMI",
  "baseline": {
    "sex": 1,
    "age_band": "25-34",
    "residence": "rural",
    "wealth_quintile": 3,
    "bmi_band": "obese",
    "hypertension": 0,
    "tobacco": 0,
    "alcohol": 0
  }
}
```


### POST /parse response

```json
{
  "parsed_condition": {
    "sex": null,
    "age_band": null,
    "residence": null,
    "wealth_quintile": null,
    "bmi_band": "overweight",
    "hypertension": null,
    "tobacco": null,
    "alcohol": null,
    "state": null
  },
  "raw_text": "improved BMI",
  "confidence": null,
  "parser": "rule-based demo parser",
  "requires_confirmation": true,
  "unresolved": [],
  "optional_token_hook": {
    "status": "disabled",
    "used_for_conditions": false
  },
  "run_type": "mock",
  "preliminary": false,
  "demo": true,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "banner": "MOCK DEMO — not NFHS-5; synthetic rates are not research results"
}
```


### GET /validation pending

```json
{
  "run_type": "mock",
  "preliminary": false,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "demo": true,
  "status": "pending",
  "metrics": null,
  "models": null,
  "evaluation_scope": null,
  "real_reference": null,
  "model_fingerprint": null,
  "note": "No compatible packaged TRAIN/VAL evaluation supplied",
  "banner": "MOCK DEMO — not NFHS-5; synthetic rates are not research results"
}
```


### GET /validation complete (abridged, separate mock pipeline)

```json
{
  "status": "complete",
  "run_type": "mock",
  "status_banner": "DEMO (mock data)",
  "model_fingerprint": "bfd663115eba4562416df946a1e19d30298b81218d3d4edd13c4ec92d071ab0a",
  "note": "Measured data only; no model ranking. Retained subset, unweighted sample.",
  "metrics": {
    "CVAE_marginal_glucose": {
      "ks": 0.06423,
      "wasserstein": 3.70803,
      "wasserstein_over_sd": 0.11721
    },
    "CVAE_real_vs_synthetic": {
      "status": "complete",
      "roc_auc": 0.58733,
      "n_real": 685,
      "n_synthetic": 685,
      "n_classifier_train": 959,
      "n_classifier_heldout": 411,
      "note": "Held-out source classification on matched retained VAL cohorts; 0.5 means weak distinguishability for this classifier only, not proof of privacy or fidelity."
    }
  }
}
```


### GET /model-comparison complete (abridged)

```json
{
  "status": "complete",
  "run_type": "mock",
  "status_banner": "DEMO (mock data)",
  "model_fingerprint": "bfd663115eba4562416df946a1e19d30298b81218d3d4edd13c4ec92d071ab0a",
  "note": "Measured data only; no model ranking. Retained subset, unweighted sample.",
  "models": {
    "CVAE": {
      "correlation": {
        "frobenius": 2.57066,
        "max_abs": 0.86991
      },
      "tail": {
        "p50": 104.0,
        "p90": 147.0,
        "p95": 167.0,
        "p99": 228.15999999999997,
        "share_ge_threshold": 0.018978
      }
    },
    "TVAE": {
      "correlation": {
        "frobenius": 3.72549,
        "max_abs": 1.0
      },
      "tail": {
        "p50": 106.0,
        "p90": 119.0,
        "p95": 124.79999999999995,
        "p99": 131.0,
        "share_ge_threshold": 0.0
      }
    },
    "CTGAN": {
      "correlation": {
        "frobenius": 3.37759,
        "max_abs": 0.9169
      },
      "tail": {
        "p50": 111.0,
        "p90": 169.0,
        "p95": 219.5999999999999,
        "p99": 349.9199999999996,
        "share_ge_threshold": 0.064234
      }
    }
  }
}
```


### 422 missing conditions

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": [
        "body"
      ],
      "msg": "Value error, Missing full profile conditions: age_band, residence, wealth_quintile, bmi_band, hypertension, tobacco, alcohol",
      "input": {
        "condition": {
          "sex": 0
        }
      },
      "ctx": {
        "error": {}
      }
    }
  ],
  "run_type": "mock",
  "preliminary": false,
  "demo": true,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "banner": "MOCK DEMO — not NFHS-5; synthetic rates are not research results"
}
```


### 503 missing artifacts

```json
{
  "detail": "CVAE artifacts unavailable. Train the model first or explicitly create a MOCK demo checkpoint.",
  "run_type": "mock",
  "preliminary": false,
  "demo": true,
  "status_banner": "DEMO (mock data)",
  "run_note": null,
  "banner": "MOCK DEMO — not NFHS-5; synthetic rates are not research results"
}
```
