# Presentation runbook

Every demonstrated number is MOCK software output unless the API explicitly serves privately trained quick/full weights. Display the API's status_banner, scope, unweighted label, uncertainty note and disclaimer beside results. Follow [API_CONTRACT.md](API_CONTRACT.md).

## Prepare and start the preset demo

From the repository root, with the pinned environment installed:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m models.make_demo_checkpoint --out-dir outputs/demo
PP_DEMO_MOCK=1 PP_MODEL_DIR=outputs/demo python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

This builder creates labelled mock joint cells solely to demonstrate three supported full profiles. It writes fitted artifacts, never row samples. `/health` must say ok/demo, run_type mock and `DEMO (mock data)`. It is not a private-data model. The preset demo has no measured packaged comparison report; the validation panel honestly remains pending.

In another shell, use a complete returned preset and compare a descriptive change:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/options
python - <<'PY'
import json, urllib.request
base = 'http://127.0.0.1:8000'
with urllib.request.urlopen(base+'/profiles') as response:
    profile = json.load(response)['profiles'][0]['condition']
def post(path, payload):
    request = urllib.request.Request(base+path, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request) as response:
        print(json.dumps(json.load(response), indent=2))
post('/generate', {'condition': profile, 'n': 1000, 'seed': 42, 'n_examples': 3})
post('/compare', {'scenarios': [
    {'label':'baseline', 'condition':profile, 'n':1000, 'seed':42},
    {'label':'descriptive BMI change', 'condition':{**profile, 'bmi_band':'normal'}, 'n':1000, 'seed':42}
]})
post('/parse', {'text':'improved BMI', 'baseline':profile})
PY
```

If the preset already has normal BMI, choose another preset for that comparison. A parser result is a proposal only: merge its confirmed changes into the full baseline before submitting a generation request. Unresolved phrases must be clarified in the UI. Never turn glucose text into a condition. The parser is rule-based; optional local token annotations do not determine conditions. No temporal model or automatic action is implied by input wording.

## Demonstrate the measured mock report adapter

Stop the first API before binding the same port. The full mock pipeline has its own fitted fingerprint and measured software-smoke metrics:

```bash
python -m models.run_all --mock --quick --out-dir outputs/mock-run
PP_DEMO_MOCK=1 PP_MODEL_DIR=outputs/mock-run/private_outputs PP_REPORT_DIR=outputs/mock-run/safe_outputs python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
curl -s http://127.0.0.1:8000/validation
curl -s http://127.0.0.1:8000/model-comparison
```

Both endpoints show the mock banner. Do not attach these reports to the separate preset-demo checkpoint: checksum/fingerprint checks correctly reject that mismatch. Random mock pipeline data need not supply three joint cells of 500 rows, so `/profiles` may return 503 in this report demonstration. This is intentional; the software never invents supported presets.

## Private quick run and real serving

Only the authorized private-data holder runs these commands, after resolving the codebook/access questions and rebuilding affected preprocessing. Do not upload survey files to this environment.

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-quick --quick
PP_MODEL_DIR=/private/proxypatient-quick/private_outputs PP_REPORT_DIR=/private/proxypatient-quick/safe_outputs python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Unset `PP_DEMO_MOCK`, `PP_ALLOW_PARTIAL_PROFILE`, `PP_CVAE_WEIGHTS` and `PP_CONDITION_MARGINALS` inherited from a demo shell first. A quick checkpoint must show `PRELIMINARY (quick run)`, preliminary true and the reduced/untuned note. It is real-data computation but remains an indicative run. Full weights show `FULL RUN`; that banner does not establish scientific adequacy. Check readiness, profile support, clipping and measured validation before presenting any result. Missing/invalid artifacts and insufficient presets return 503; do not substitute mock numbers or guessed conditions.

Quick defaults are in config.yaml: 100,000 stratified CVAE TRAIN rows, six epochs; 20,000 baseline rows, 15 epochs; 10,000 requested evaluation rows. Actual scope/complete encoding and rejection sampling can retain fewer rows. Both sexes require 2,000 complete rows in quick mode and 5,000 in full mode. timings.json records stage seconds and Linux peak RSS for planning. CPU mock timing is not a forecast of private-data performance.

## Explain the numbers

- rate and rate_pct describe the share of finite generated glucose readings at/above the configured threshold: elevated glucose (proxy). They are unweighted synthetic sample summaries.
- ci_low/ci_high are Wilson Monte Carlo bounds conditional on this fitted model. They exclude survey and fitted-model uncertainty.
- delta_pp is the percentage-point difference from the first synthetic scenario. It is descriptive, not causal.
- Fidelity metrics describe the shared retained subset; read requested/retained coverage before comparing models. Baseline data/state use differs. TSTR is matched-condition transductive utility.
- Real-versus-synthetic AUC tests one classifier's distinguishability. Nearest-record distances are limited disclosure diagnostics. Neither certifies privacy or fidelity.
- Counts/statistics below 30 are null; sufficient privacy support does not guarantee stable estimates. Presets require 500 encoded TRAIN respondents, without validating arbitrary changes.

Do not claim a diagnosis, individual prediction, causal effect, survey-representative rate, untouched test set or finished model from a quick run. Never present mock values as NFHS-5 results or choose a winner from architecture alone. Real-data fidelity, tail calibration, disclosure and legal release remain human review tasks.

## Present representative examples

After the synchronous response, show "Synthetic Cohort Generated", `outcome_stat.n`, the cohort percentage/interval and, for comparisons, `delta_pp`. Render each returned example as a card with its ID, full SYNTHETIC label, effective conditions, generated features in original units and generated elevated glucose (proxy) outcome. MOCK labels must remain visible. Show an indeterminate waiting indicator only; no invented progress percentage.

Default cards use glucose quartiles; five use the 10th/25th/50th/75th/90th percentile ranks. An elevated cohort includes its least elevated row as an explicit illustration, replacing the last slot if needed. Never exceed n_examples. Cards illustrate the generated cohort; calculate no rate from the cards. Glucose display rounding does not determine the elevated flag.

Real mode returns null examples and a withholding status unless the bound/checksummed VAL report's CVAE nearest-record diagnostic passes configured median-ratio, minimum-distance, zero-exact-copy and support checks. The new minimum field requires a fresh development evaluation/package for older runs. Do not rerun final TEST to enable cards. This sampled numeric check is a heuristic, not a privacy guarantee, and does not certify each card's distance or scientific validity. Continue showing summaries when examples are withheld. No bulk export or persisted samples are enabled.
