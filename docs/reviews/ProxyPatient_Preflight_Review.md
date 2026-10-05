# ProxyPatient: read-only pre-flight review

Repository: https://github.com/Prapti101/ProxyPatient  
Branch: `main`  
Reviewed commit: `627bcc95685b036abd19c6d7e70656b7d822d814`  
Review date: 4 October 2026

All 47 tracked files at this commit were read, including the older P1 scripts, metadata audit, both aggregate JSON files, and all tests. No repository files were edited, no PRs were opened, and nothing was pushed. Git status remained clean. No NFHS respondent data was requested or used. Additional runtime checks used mock frames only.

**Evidence labels:** **Verified** means directly established by source inspection, git metadata, or a stated mock execution. **Assumption / inference** means a consequence or statistical risk whose magnitude needs local real-data checks. Repository statistics are treated as committed claims, not independently verified survey results.

## 1. Verdict: No-go

1. Women’s height linkage is broken; the default model preparation would exclude every woman produced by the current preprocessing script.
2. The default API serves jittered real-row resampling and manually calibrated glucose, violating the strictly generative contract.
3. State codes are treated as contiguous 1–36 despite the committed aggregates containing code 37 and omitting code 26.
4. Older executable paths bypass the locked-test policy, and aggregate suppression is inconsistent.
5. All 36 mock tests pass, but they bypass the preprocessing defect and do not establish real-data fidelity, privacy, or licence compliance.

Dependency installation succeeded with `python -m pip install -r requirements.txt`. Tests ran on Python 3.12, CPU, with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

**Result: 36 passed, 18 warnings in 14.38 seconds; no test failures.** Warnings include Pydantic legacy validators/field arguments/`.dict()`, FastAPI startup events, and Starlette’s httpx TestClient integration. This verifies the installed environment, not the README’s Python 3.11 claim or a Colab/Kaggle GPU run.

## 2. Critical issues: fix before real-data use

### C1. Women’s height is always missing, so women are excluded from model training

**Verified.** `preprocess_v2.py:42–62` omits `v001`, `v002`, and `v003` from `WOMEN_COLS`. `safe_load` loads only requested columns (`preprocess_v2.py:94–102`). The household join requires those three keys (`preprocess_v2.py:296–300`), so the code always takes the branch that assigns missing `height_cm`.

Height is a required generated continuous variable (`config.yaml:193`), and `models/data.py:212–223` removes every row with a non-finite continuous value. The DAE explicitly does not fill height (`dae_impute.py:41–43`; `train_dae_v2.py:52–54`). A mixed-sex checkpoint can consequently train only on men while still exposing female generation.

**Mock confirmation:** after reproducing this missing-height pattern, 2,239 women remained inside the initial scope, but **zero women** survived encoding; 380 men survived. This confirms the downstream exclusion mechanism, not actual survey row counts.

**Minimal fix:** load the three women’s linkage keys, validate the household join, regenerate affected private artifacts, and require nonzero and adequately supported per-sex counts **after** `make_arrays`. Stop before training if either sex disappears. Owners: **P1 + P2**.

### C2. Default API violates genuine generation and computed-number rules

**Verified.** `backend/main.py:35–40` defaults to `generator_stub`. That stub loads `processed/dae_imputed_combined.parquet` (`backend/generator_stub.py:49–59`), samples real records with replacement and adds 5% noise (`backend/generator_stub.py:119–129`). It returns no synthetic provenance flag and no generated glucose. It also ignores hypertension and state because neither has a filter in `backend/generator_stub.py:90–111`.

The always-imported outcome stub (`backend/main.py:43`) then invents glucose when the proxy column is absent (`backend/outcome_stat_stub.py:100–102`). Its probability is manually chosen as `0.0288`, with wealth/BMI/tobacco adjustments (`backend/outcome_stat_stub.py:42–66`). Computing a final average from these draws does not make the underlying scenario rates learned from data. The default private v1 combined file also contains the held-out population.

Without that private file, default `/generate` fails; with it, it produces rule-breaking results. With `PP_GENERATOR=real`, missing weights or marginals raise at request time (`backend/generator.py:60–74`), without silently falling back to the stub, which is good.

**Minimal fix:** default to the real generator and fail closed if its approved artifacts are unavailable. Restrict demo mode to explicitly labelled, in-memory mock data with the same output schema; remove real-row sampling and fake-glucose fallback from a real-data deployment. Owners: **P1 + P2 + P3**.

### C3. State-code mapping excludes a documented real code and exposes an unsupported one

**Verified.** `docs/aggregates_v2.json:311–315` contains state `37` with a committed count of 2,607 known-glucose respondents. Its state keys omit `26`. Yet `backend/schemas.py:29,78–81`, `config.yaml:185–187`, `models/data.py:139–141`, and `backend/generator.py:118–127` assume a contiguous 1–36 range.

Code 37 becomes `-1`, then is dropped during state-enabled training (`models/data.py:219–221`). The API rejects `"37"` with HTTP 422. Code 26 is accepted even though its training support is not established. During evaluation invalid codes are silently reassigned to state index zero (`models/eval_dev.py:77–79`), which misconditions those rows rather than reporting the mismatch. Unspecified state sampling also enumerates only 1–36 (`backend/generator.py:178–183`).

**Assumption:** the same raw codes occur in the current private files; the committed aggregates strongly support checking this, but private files were not inspected.

**Minimal fix:** validate actual codebook codes locally and store an explicit raw-code-to-embedding mapping in the checkpoint. Use the same mapping in API validation, training, marginals, and evaluation; reject unknown codes instead of substituting another state. Owners: **P1 + P2**.

### C4. The locked-test rule is only enforced in P2’s helpers

**Verified.** P2’s `split_path`/`read_parquet` block named test files without the flag (`models/common.py:58–78`), and P2 training reads train/val only (`models/train_cvae.py:53–56`; `models/train_baselines.py:137`). However:

| Unguarded path | Evidence | What it reads or exposes |
|---|---|---|
| P1 preprocessing | `preprocess_v2.py:160–166` | Existing test membership, without a final-test stage |
| Split v2 | `split_and_aggregate_v2.py:64–87,123–125,154–161` | Test IDs, test outcome rate, and full-data aggregates after splitting |
| DAE v2 | `train_dae_v2.py:237–239,280–287` | Combined data containing test respondents, then imputes it |
| DAE inference executable | `dae_impute.py:199–223,226–231` | Automatically reads/imputes test and prints train/test mean comparisons |
| Handover executable | `generate_handover_v2.py:13–26,59–62` | Reads test parquet to extract shape and columns |
| Default generator | `backend/generator_stub.py:55–56` | Combined v1 data rather than a training-only source |
| Legacy DAE | `train_dae.py:271–289,328–343` | Combined data including held-out respondents |

`split_and_aggregate.py:143–159` also exposes test rates and full-data aggregates. Initial split construction necessarily handles the rows being assigned, but downstream inspection/inference and reusable full-data profiles are not covered by the promised final-test gate. Full-data reference outcome rates are already public; the claim that the test is wholly untouched is therefore too strong. No evidence establishes that P2 tuned on test.

**Minimal fix:** distinguish one-time split creation from later access, centralize guarded reads, remove combined-data inference from development paths, generate reference profiles from train, and move test imputation/evaluation into explicit final-test commands. Document prior exposure honestly. Owners: **P1 + P2 + P3**.

### C5. Small-cell suppression is incomplete in outputs recommended for publication

**Verified.** `split_and_aggregate_v2.py:33–39,163–168` and legacy `split_and_aggregate.py:50–59` suppress rates but keep exact counts below 30. This disagrees with the requested null suppression of small aggregate cells.

P2 `apply_scope` returns unsuppressed sex-specific exclusion counts (`models/data.py:89–98`), and CVAE training copies them into the log (`models/train_cvae.py:115–119`). The corrected DAE benchmark publishes counts, medians, SDs and RMSEs without any minimum-support guard (`models/check_dae_benchmark.py:39–59`). Evaluation publishes marginal, tail, and nearest-distance metrics without an overall minimum-size guard (`models/eval_dev.py:105–142,235–247,317–340`), even though rejected baseline requests can leave a tiny evaluation set.

**Scope of confirmation:** all 108 v1 and 134 v2 committed rate cells have denominators at least 30. I found **no current sub-30 rate cell** in those files. The defect is in reachable output logic, not a confirmed respondent leak in the current JSON files.

**Minimal fix:** centralize suppression for counts and statistics, apply it to every publishable JSON/Markdown/log output, and enforce minimum evaluation coverage. Inspect all proposed public artifacts after applying the policy. Owners: **P1 + P2 + P3**.

## 3. Major issues, minor issues, and nits

### Major

| ID | Evidence and issue | Minimal fix / owner |
|---|---|---|
| M1 | **Verified:** `backend/schemas.py:36–82` does not forbid extra fields. `glucose_raw`, `log_glucose`, and unknown keys are silently discarded before `backend/main.py:213–222` can reject them. Age, tobacco and alcohol have no validators; constants exist but are unused. Direct generator validation is stronger. Mock construction accepted `age_band="nonsense"`, tobacco/alcohol `"2"` and silently removed glucose keys. | Use strict enum fields and `extra="forbid"`; test forbidden outcome inputs through HTTP, not just direct Python calls. **P1** |
| M2 | **Verified:** `backend/main.py:174–183` looks up `female`/`male`; v2 aggregates use `0`/`1` (`docs/aggregates_v2.json:16–27`). `/profiles` returned only Overall/Urban/Rural. Profiles are partial conditions, despite independent-marginal fill requiring full profiles (`backend/generator.py:168–184`). Real reference profiles cover all known-glucose respondents, while the model has a narrower scope. | Normalize keys; provide supported full baseline profiles and clearly separate reference scope from model scope. **P1 + P4** |
| M3 | **Verified:** `backend/outcome_stat_stub.py:100–105` trusts the supplied proxy column, not glucose and the requested threshold. `backend/generator.py:213` uses checkpoint threshold; `backend/main.py:99–100` uses current config. A changed threshold can silently disagree. A mock frame with glucose 250, proxy 1 and rule 300 still returned rate 1.0. Missing glucose with a nonmissing label is counted. | Derive valid outcomes from finite generated glucose each time; validate checkpoint/config compatibility and remove fallback glucose creation. **P3 + P2** |
| M4 | **Verified:** `backend/main.py:129–132` always returns healthy `CVAE`, even when the stub is active and no model weights exist. Mock-checkpoint metadata is saved (`models/train_cvae.py:157–160`) but never checked by serving (`models/sampling.py:24–36`; `backend/generator.py:60–74`); a mock model can be reported simply as CVAE. `/compare` lacks model provenance (`backend/schemas.py:120–133`). | Report loaded mode/readiness; reject mock artifacts in real mode; add provenance/config fingerprint to both endpoints. **P1 + P2** |
| M5 | **Verified:** `preprocess_v2.py:452–474` still fits and writes the all-data scaler/encoder; v1 does likewise (`preprocess.py:386–408`). P2 correctly avoids them, but README quick start still tells consumers to use `preprocess.pkl` (`README.md:191–195`). | Retire those artifacts from current instructions/handover; fit only on train if another consumer truly needs them. **P1** |
| M6 | **Verified:** medication codes 8/9 become zero (`preprocess_v2.py:272–273,363–364`), despite schema missing-code rules (`docs/schema.json:477–484`). Self-reported codes 8/9 are retained (`preprocess_v2.py:278–279,368–369`) despite the schema claiming 8 becomes missing (`docs/schema.json:492`). Tobacco treats any nonmissing inverse-flag value other than 1 as a user (`preprocess_v2.py:247–253,346–347`); it does not validate binary codes or follow the documented full tobacco-variable OR (`docs/schema.json:425–429`). | Use codebook-specific value maps with explicit missing/invalid cases; verify all binary encodings locally. **P1** |
| M7 | **Verified:** `compute_hypertension` declares non-hypertension when either BP dimension is present and below threshold, even if the other dimension and medication are unknown (`preprocess_v2.py:128–136`). `bp_measured` includes first-reading fallback (`preprocess_v2.py:104–120,139–140`), whereas schema says a valid second/third reading (`docs/schema.json:471–475`). | Define a three-valued rule: known positive always wins; negative requires sufficient evidence; otherwise missing. Agree/document the fallback protocol. **P1 + P3** |
| M8 | **Verified design risk:** unspecified conditions, including state, are drawn independently (`backend/generator.py:168–184`). This destroys their empirical joint distribution and can request unseen combinations. Marginals are calculated before complete-row rejection (`models/train_cvae.py:99–103,164`; `models/data.py:226–239`), so they can represent rows the model never trained on. If all marginal counts are suppressed/missing, the generator switches to uniform probabilities (`backend/generator.py:147–152`). | Require a supported complete profile or sample a privacy-safe joint profile distribution from the actual encoded training scope; fail on unavailable marginals. **P2 + P4** |
| M9 | **Verified design risk:** rejection sampling fills each condition combination until quotas or a finite budget are reached (`models/train_baselines.py:75–106`); evaluation drops unfilled rows for every model (`models/eval_dev.py:317–320`). This can hide rare/difficult cells. CVAE trains on full scoped train with state; baselines use ~100K without state (`models/train_baselines.py:139–141`). | Report retained versus requested condition coverage; do not describe retained-subset metrics as full-scope fidelity. Add equal-data/equal-state comparisons, or explicitly label resource/architecture differences. **P2 + P3** |
| M10 | **Verified:** the corrected DAE benchmark uses un-imputed truth, but `np.nanmean` excludes failed DAE predictions (`models/check_dae_benchmark.py:48–57`) while the median comparator uses all masked targets. It can declare DAE better on unequal coverage. Legacy benchmark remains circular (`dae_impute.py:153–179`). | Fail or score missing predictions explicitly; compare identical valid targets and coverage. Retire the legacy benchmark. **P2 + P1** |
| M11 | **Verified statistical limitation:** `backend/outcome_stat_stub.py:113–123` bootstraps one generated cohort only. Increasing synthetic n reduces this interval without reducing survey/model uncertainty. The API calls it a bootstrap 95% CI (`backend/schemas.py:94–95`) but never explains that distinction. At zero positives it returns [0,0], confirmed with a mock n=100 frame. | Label it as synthetic-cohort/Monte Carlo uncertainty conditional on a fixed model. Use a suitable rare-event interval and separately assess model/data uncertainty. **P3 + P4** |
| M12 | **Verified:** unweighted modeling is documented in model cards (`models/train_cvae.py:192`; `models/model_card.json:17`), but `/profiles` describes an overall population without stating unweighted scope (`backend/main.py:163–169`). Generated responses lack an explicit weighting/scope field; old aggregate code calls the outputs population-representative (`split_and_aggregate.py:158–159`). | Add visible “unweighted sample / synthetic model” labels and the measured-BMI/known-BP/glucose scope wherever rates appear. **P1 + P3 + P4** |
| M13 | **Verified:** reachable git history retains v1 model/scaler binaries. Commit `a131c60` added them; `512ddd3` removed them from the current tree only. See §6 for exact object sizes. No tracked respondent files were found. Historical binary contents were not independently deserialized. | Audit historic artifacts locally and decide whether approved history cleanup is required; do not equate untracking with erasure. **P1** |
| M14 | **Verified:** split v2 logs “FATAL” on hash mismatch but still overwrites the split files (`split_and_aggregate_v2.py:140–149`). A missing v1 split triggers a fresh split with a different construction order (`:103–120`), risking loss of the locked membership contract. | Abort before writes on any mismatch or missing locked membership; require explicit one-time initialization separately. **P1** |
| M15 | **Verified runtime design / inferred scale risk:** DAE inference runs the entire frame through CPU layers in one call (`dae_impute.py:100–102`; `train_dae_v2.py:174–176`); step0 retains both raw and imputed frames for train/val (`models/step0_checks.py:200–204`). CVAE evaluation samples on CPU in batches up to 200K and redraws up to 30 times (`models/sampling.py:78–110`). Baseline evaluation may draw ~1.5M rows for 50K requests. | Batch DAE and evaluation, reduce unnecessary frame copies, log peak memory/coverage and benchmark runtime locally. Existing T4 time estimates are unverified. **P1 + P2** |

### Minor

| ID | Evidence and issue | Fix / owner |
|---|---|---|
| m1 | **Verified:** CLI defaults override `model.state_embedding`, `model.glucose_head`, and model seed (`models/train_cvae.py:101,107,214–218`); condition names are fixed in `build_spec` rather than taken from `conditioning_variables` (`models/data.py:54–71`). | Resolve CLI/config precedence consistently and validate the supported config schema. **P2** |
| m2 | **Verified:** bootstrap count/CI level/default n are hardcoded instead of honoring `config.yaml:164–167` (`backend/outcome_stat_stub.py:80–81,122–123`; `backend/schemas.py:87–89`). P1 threshold/BMI/privacy/split constants also bypass config (`preprocess_v2.py:39,142–147`; `split_and_aggregate_v2.py:21–25,174–175`). | Read canonical config or assert matching fixed project constants. **P1 + P3** |
| m3 | **Verified:** preprocessor SD handling fails for all-missing or singleton columns: `float(v.std()) or 1.0` leaves NaN because NaN is truthy (`models/data.py:184–193`). Empty training/validation arrays are not rejected before training (`models/train_cvae.py:102–105,123–138`). Invalid ages are placed into outer bands (`models/common.py:108–112`); fractional sex/wealth/state values are truncated (`models/data.py:119–123,139–141`). | Validate finite scales, sufficient rows, valid domains and integral codes before fitting. **P2** |
| m4 | **Verified:** validation saves/restores only CPU RNG, although `torch.manual_seed(0)` also resets CUDA RNG (`models/train_cvae.py:72–80`). Deterministic GPU algorithms are not requested (`:32–33`). | Use RNG isolation for the actual devices and document determinism limits. CPU generation determinism is tested. **P2** |
| m5 | **Verified statistical risk:** age/BMI rejection conditions the whole generated row on compliance; fallback clipping creates boundary mass (`models/sampling.py:95–124`). Rejection counts are recorded, which is good, but API responses discard them (`backend/main.py:239–243`). Optional BP is capped before quantile clipping and rounding (`models/sampling.py:115–129`); no generated medication explains controlled hypertensive BP. | Expose diagnostics and refuse excessive clipping; evaluate tail-rate changes before/after constraints. Recheck BP consistency after all postprocessing. Hypertension=1 with normal BP can be legitimate medication use, not automatically a bug. **P2 + P3** |
| m6 | **Verified:** nearest-record distance uses standardized numeric/ordinal encoding for nominal categories, ignores state and samples only up to 50K train references/2K queries (`models/eval_dev.py:120–128,228–247`). It misses exact copies outside that reference sample and is not a privacy certificate. | Use mixed-type distances and matched-condition/state baselines; describe sampling limits and add disclosure tests locally. **P3** |
| m7 | **Verified:** consistency metric only checks age/BMI (`models/eval_dev.py:212–225`); categorical correlation coding and zero-filled undefined correlations can hide failures (`:120–135`); missing glucose is treated as outcome zero if it reaches `outcome` (`:67–68`). | Validate outcomes/finite inputs, report undefined metrics, and add weight/BMI/height and optional BP checks. **P3** |
| m8 | **Verified:** TSTR uses synthetic cohorts conditioned on the held-out rows’ feature mix (`models/eval_dev.py:301–305,339`), and chooses feature columns from held-out categories (`:262–268`). Glucose is correctly excluded from predictor inputs. | Describe this as matched-condition/transductive utility; add conventional TSTR from an independently generated frozen training cohort. **P3** |
| m9 | **Verified:** parser ignores negation and matches substrings (`backend/main.py:308–326`). Mock “do not smoke and drink no alcohol” produces tobacco=1/alcohol=1. Confidence is fixed at 0.6/0.1 (`:336`). | Mark parser as demo-only, handle negation or require confirmation of parsed conditions, and remove fake confidence. **P3 + P4** |
| m10 | **Verified:** manifest errors are reported but do not stop step0 (`models/step0_checks.py:50–70,195–204`). Baseline all-empty rejection results can lack columns before `drop(columns=...)` (`models/train_baselines.py:101`; `models/eval_dev.py:95`); empty retained sets also break later metrics. | Fail integrity checks, require minimum coverage and handle typed empty samples. **P2** |
| m11 | **Verified:** DAE training learns median/mode-filled missing cells as reconstruction targets, uses MSE for ordinal-coded categoricals, and its masking/loss includes all categorical inputs rather than only allowed imputation columns (`train_dae_v2.py:103–117,142–159`). Training-time `impute` fills education/wealth/residence when missing; inference-time `impute_dae` restricts to `impute_cols` (`train_dae_v2.py:190–206`; `dae_impute.py:114–116`). V1 also rebuilds category maps on each frame instead of applying train maps (`train_dae.py:84–90,108–123`). | Carry observed-cell masks into loss, use categorical heads if warranted, and share one consistent imputation implementation. Retire v1 paths. **P1** |
| m12 | **Verified:** all direct dependencies are pinned, and installed successfully, but transitive dependencies are not locked; README and Colab commands retain unpinned packages (`README.md:188`; `models/RUN_ON_COLAB.md:27`). `requirements.txt:18` pins a CUDA torch build by default; local CPU setup guidance does not guarantee that subsequent requirements installation preserves it. | Provide separate tested CPU/GPU installation paths and a resolved environment record. **P1 + P2** |
| m13 | **Verified:** old P1 scripts use personal absolute Windows roots (`preprocess.py:37`; `preprocess_v2.py:24`; `split_and_aggregate.py:33`; `split_and_aggregate_v2.py:18`; `train_dae.py:35`; `train_dae_v2.py:22`; `dae_impute.py:15`; `generate_handover_v2.py:8`). Some create folders at import (`preprocess.py:43–44`; `generate_handover_v2.py:11`). | Accept configurable paths and avoid import-time execution. These scripts are not portable unchanged. **P1** |
| m14 | **Verified:** glucose is rounded and clipped 20–600 at sampling (`models/sampling.py:59–62`), and age/BMI are rounded/floored. This changes the learned continuous distribution near boundaries, including the glucose threshold. Neither tail clipping nor overflow/nonfinite sampling has a dedicated check. | Quantify clipping/rounding rates; validate finite samples and threshold fidelity. **P2 + P3** |

### Nits

- **Verified:** API metadata says version 1.0.0 (`backend/main.py:71`), while health/config say 2.0.0. Align versions. **P1**.
- **Verified:** legacy Pydantic/FastAPI calls generate deprecations; modernize after correctness fixes. **P1**.
- **Verified:** `models/RUN_ON_COLAB.md:24` still clones a development branch and says use main after merge. The review target is already main. Update it. **P2**.
- **Verified:** `README.md:234` says test size 119,798; `docs/split_report.md:11` says 119,794. README also presents v1 DAE loss/zero-missingness as current (`README.md:235–236`). Label historical metrics and refresh from computed artifacts. **P1**.
- **Verified:** `docs/p2_repo_check.md:41–46,88–94,105–110` and `docs/p2_next_steps_for_humans.md:8–11` describe issues partly fixed since their earlier snapshot: hypertension/state fields, v2 aggregate loading, threshold loading, health version, and schema BP definitions. Preserve the snapshot date or update status. **P1 + P2**.

## 4. Contract mismatch table

| Item | Where it differs | Which source should win |
|---|---|---|
| `generate` name | Current README, stub, generator, tests and main all use singular `condition`; old P2 document mentions a private handoff using plural `conditions` (`docs/p2_repo_check.md:12–14`). That handoff is not tracked. | Keep current callable `generate(condition, n, seed)`; no verified live plural-signature bug. |
| Sex | Config options `0/1` (`config.yaml:83–86`); HTTP female/male (`backend/schemas.py:21,38`); real generator accepts both and outputs integers (`backend/generator.py:102–105,195`); aggregates v2 numeric-string keys; profiles reads words. | Define one HTTP contract, with explicit internal/aggregate conversions. |
| Age | Sex-specific external labels vs internal 35+; real generator harmonizes, but API echoes original request (`backend/main.py:240`) even if generated male rows use 35–54 (`backend/generator.py:189–196`). API does not validate age labels. | Preserve documented harmonization but return effective conditions and validate HTTP labels. |
| Hypertension/state types | HTTP string only (`backend/schemas.py:45–46`); direct generator parses integer/string forms. Earlier P2 docs claim fields absent. Stub ignores both. | Current HTTP schema is the actual API contract; update docs and real-mode behavior. |
| State codes | Contiguous 1–36 assumptions vs aggregate raw code 37 and absent 26. | Verified codebook/raw-code mapping, stored with checkpoint. |
| Tobacco | HTTP/input `tobacco`; output/data `any_tobacco` (`backend/generator.py:201`); schema describes full raw-variable OR, preprocessing uses inverse flag and optional s711. | Explicit API-to-data mapping plus validated codebook derivation. |
| BMI | Config upper obese bound 9999 (`config.yaml:143`); P1/DAE pd.cut upper bound 999 (`preprocess_v2.py:145`; `dae_impute.py:138`). Both use left-closed bands. | Canonical config with explicit plausible-range validation. |
| Returned row schema | Real includes generated glucose/proxy/is_synthetic; stub excludes them and lacks synthetic flag. | Real generator contract; mock demo must implement it consistently. |
| `outcome_stat` contract | README only df/rule and four returned fields (`README.md:141–146`); main supplies `seed`; response requires `rate_pct` (`backend/schemas.py:92–97`); actual stub supports both. | Document and implement the actual required signature and response fields. |
| Threshold | Current backend reads config, generator uses checkpoint snapshot, P1 hardcodes 200; statistic trusts precomputed labels. | Frozen, fingerprinted outcome definition; recompute labels from generated glucose under that rule. |
| Suppression | Config says null; P1 keeps small n; P2 helpers suppress n/rate but not every log/statistic. | One shared, explicit publication policy. |
| Model config | `conditioning_variables`/seed/state/glucose-head do not all control training; CLI wins even when not explicitly supplied. | Tested config schema and explicit precedence. |
| `/schema` | Returns DHS metadata (`backend/main.py:135–144`), not `whatif_options`; education is marked conditioning in metadata (`docs/schema.json:63–74,290–302`) but generated in CVAE. Household/BP raw metadata is incomplete despite v2 derived definitions being present. | Separate API scenario schema from raw-variable dictionary; include effective model capabilities. |
| Aggregate keys/labels | Sex changed female/male→0/1, hypertension is 0.0/1.0, tobacco cross-tabs changed numeric strings→word labels; v1 women-only BMI→v2 both-sex BMI. V2 meta uses machine key instead of display label (`docs/aggregates_v2.json:4`). | Versioned aggregate contract with explicit labels/conversions. |
| Reference scope | Public profiles use full known-glucose sample; CVAE uses complete conditions plus complete generated variables/state. | Label both scopes explicitly; baseline generation must use actual model scope. |
| Validation output | P2 writes `model_comparison_dev/final_test.json`; `/validation` reads only absent `validation_report.json` (`backend/main.py:115–122`). | P3 must supply a defined adapter/report contract; until then return explicit pending. |
| Wording/version | Risk-awareness titles conflict with proxy wording; API version differs; DAE reports/tree/quickstart are v1. | Current approved proxy terminology and reproducible v2 artifacts. |
| Test access guidance | `config.yaml:226–227` and `DECISIONS.md:125` recommend test for all validation; P2’s workflow correctly uses val for development. | Validation for development; one explicit frozen final-test evaluation. |

Residence and wealth level meanings otherwise match across the real generator, config and HTTP adapter. No `conditions` keyword bug exists in the reviewed executable path. Current schema **does** contain hypertension, BP and BMI-measured derived definitions; older P2 documents claiming otherwise are stale.

## 5. Rule-compliance checklist

| Rule | Result | Evidence |
|---|---|---|
| 1. No agents/tool loops/LLM rows | **Pass** | No such executable imports or generation found. Mentions are prohibitions; CVAE/TVAE/CTGAN are ordinary generative models. |
| 2. Decoder generation, not real-row jitter | **Fail overall; real CVAE passes** | `models/cvae.py:161–181` draws new latent/head samples; default stub resamples real rows (`backend/generator_stub.py:119–129`). |
| 3. Generated glucose, never condition/impute | **Pass real path; fail default contract/enforcement** | Log-glucose is generated, not a condition (`models/data.py:54–69`; `models/cvae.py:138–141,172–178`). DAE excludes it. Default invents glucose downstream; HTTP silently discards forbidden keys. |
| 4. No respondent data in git; suppress <30 | **Fail output policy; current tree clean** | No tracked data-like/model binary files; suppression defects C5. Historic binaries remain; no confirmed raw-row history leak. |
| 5. Always proxy wording, no diagnosis/risk claim | **Fail titles; disclaimers pass** | Risk-awareness titles in README/config/API; other diagnosis/prediction mentions prohibit use. No “diabetes prevalence” or “risk score” hit. |
| 6. Test locked until explicit final stage | **Fail repository-wide; P2 helper passes** | C4. Named-file helper does not protect combined-data or older direct readers. |
| 7. Train-only normalization, no old scaler use | **Pass current P2 normalization; fail whole-repo production policy** | P2 fits train only and does not load old scaler (`models/train_cvae.py:99–103`). P1 still produces all-data scaler and stale quickstart recommends v1 artifact. |
| 8. Synthetic labelled and never saved to git | **Pass real generator; fail stub labeling** | `backend/generator.py:214–221`, `.gitignore:76–91`. Baseline evaluation frames also lack row provenance; they are private intermediate synthetic frames. |
| 9. Computed app numbers, no hardcoding | **Fail** | Default glucose probabilities (`backend/outcome_stat_stub.py:44–61`) and parser confidence (`backend/main.py:336`). Real CVAE rates/bootstrap are computed. |

## 6. What I verified is OK, and coverage of review areas

### Model mathematics and generation

- **CVAE loss is mathematically coherent:** per-row Gaussian/mixture NLL and categorical cross-entropy are summed, then averaged; KL is the standard diagonal-normal-to-unit-normal expression, summed over latent dimensions then averaged (`models/cvae.py:133–157`). There is no verified KL sign, averaging, or head-shape bug.
- **Mixture likelihood/sampling agree:** stabilized log-softmax/logsumexp likelihood, clamped log variances, categorical component draw and Gaussian noise (`models/cvae.py:138–144,172–178`). K=1 supports the Gaussian ablation. This is a model in standardized log-glucose space; constant change-of-variable terms do not affect fitting for the same transform.
- **No outcome conditioning in current Spec:** glucose enters generated arrays/encoder as the reconstruction target; that is legitimate CVAE training, not a user-supplied outcome condition. Condition embeddings and optional state embeddings are shape-compatible in exercised mock paths.
- **Training:** beta increases linearly to configured beta over annealing epochs (`models/train_cvae.py:123–125`); stopping compares validation ELBO at beta=1 after annealing, with a deliberate short-smoke-run exception (`:136–149`). The selected state is loaded before saving. Latent posterior collapse remains an unmeasured risk; epoch KL logs are available, but no active-unit/latent-use analysis exists.
- **Checkpoints match loading:** saved state_dict/hparams/preproc/inference cfg (`models/train_cvae.py:157–160`) correspond to `CVAEBundle` fields (`models/sampling.py:24–36`). Real generator deliberately requires separate marginals. CPU seed reproducibility and mlp/gru/cnn/Gaussian/BP smoke paths passed.
- **Consistency:** age/BMI constraints and approximate derived weight identity are tested; weight is BMI × height² rounded to 0.1 kg, so exact equality is not promised. Constraint redraw/clipping diagnostics are stored in frame attrs. Clipping can improve consistency while worsening distributional fidelity; both must be assessed.
- **Baselines:** installed `ctgan==0.12.1` constructors accept the used `enable_gpu`, epochs and batch size arguments. TVAE/CTGAN fit and evaluation smoke tests passed. Condition rejection is valid for the fitted model within successfully filled combinations; finite-budget retained-scope bias and unequal data/state remain concerns. Proportional per-stratum sampling is approximate and may round tiny strata to zero (`models/train_baselines.py:45–50`).

### Metrics and statistical interpretation

- KS, Wasserstein and categorical TVD formulas are correct (`models/eval_dev.py:105–117`); Wasserstein normalization uses the real comparison SD. That metric scaling is evaluation, not training leakage.
- Conditional rate errors use aligned real/generated rows, percentage-point differences, and the configured evaluation minimum (default 500, lower-bounded by privacy minimum in CLI) (`models/eval_dev.py:145–180,389–399`). They cover **single variables and pairs**, not every eight-variable/state scenario. Direction checks compare aggregate associations, not causal effects; endpoint signs alone are a weak test of ordinal trends.
- Tail quantiles and share above threshold are computed (`models/eval_dev.py:138–142`). DCR uses a held-out-real-to-train comparator, which is a sensible starting point; its distance/copy-detection scope is limited as noted above.
- TSTR excludes glucose from inputs (`models/eval_dev.py:250–275`). This classifier is an internal utility evaluation, not an individual-facing prediction endpoint.
- The corrected benchmark masks truly known cells from un-imputed validation and uses un-imputed train medians (`models/check_dae_benchmark.py:39–49,107–116`). It avoids the earlier circular truth issue. Its mock mode uses a toy linear imputer, so that test does not validate the actual DAE weights.
- **Rare-tail assumption:** 2.88% is a committed broad-sample rate, not an independently established model-scope rate. If p=0.0288, a cell of n=500 has about 14.4 expected positives and a binomial standard-error illustration gives a 95% half-width around **1.47 percentage points**; n=100 gives about 2.9 positives and half-width around **3.28 points**. These normal approximations are illustrations, not recommended small-event intervals. At n=30, the chance of no positives is approximately `(1−0.0288)^30 ≈ 41.6%`. Privacy eligibility is not statistical adequacy.
- Mixed-sex pooled unweighted training is dominated by the larger women’s survey sample if C1 is fixed. This is not a balanced or survey-representative combined adult population. The measured-BMI/known-glucose/known-hypertension scope and complete generated variables add selection. Pregnancy exclusion is only inferred in `DECISIONS.md:88`; there is no explicit pregnancy filter proving all pregnant respondents are excluded.
- Threshold consistency is currently 200 in committed config/metadata. Actual raw glucose units and special-code meanings still need official codebook verification; a median heuristic (`models/step0_checks.py:135–137`) does not prove units. This report does not give a clinical diagnosis or validate a diagnostic threshold.

### Backend endpoints

| Endpoint | Verified behavior |
|---|---|
| `/generate` | Pydantic request→condition dict→selected generator→outcome stub→typed response. Real output comes from decoder; exceptions become 400/500. Response echoes requested conditions and does not expose rows. |
| `/compare` | Repeats same generation/statistic path for 2–5 scenarios and computes differences against first rate in percentage points. Differences have no interval or model provenance. Same seed is allowed; no proof of a calibrated paired statistical comparison. |
| `/profiles` | Reads v2 assets, suppresses flagged profiles; numeric-key mismatch removes women/men entries. No raw rows returned. |
| `/schema` | Returns raw metadata dictionary; not a ready frontend option schema. Returns 503 when absent. |
| `/validation` | Returns explicit pending because P3’s `validation_report.json` is absent. No formal fidelity report is served yet. |
| `/parse` | Inline rule-based stub; no BERT/LLM invocation. Negation/confidence problems noted above. No declared decorator response_model, although function returns ParseResponse. |
| `/health` | Typed but optimistic/static, as described in M4. |

Current backend config loading silently falls back to `{}` and threshold 200 on any exception (`backend/main.py:89–100`); asset loading logs failures instead of aborting (`:103–113`). HTTP errors can expose local artifact paths (`:229–231`). These are reproducibility/readiness issues; fail closed for a real deployment. No respondent-row `head`, row-level print, or sample-log output was found. Request condition dictionaries are logged (`backend/main.py:225`); these are user-selected scenarios, not loaded respondent rows. P1 prints unsuppressed aggregate distributions/extrema, so “no raw rows” does not mean all printed summaries satisfy the publication policy.

### Privacy, git and licence

Current tracked `.parquet`, `.csv`, `.dta/.DTA`, `.sav`, `.pkl`, `.pt`: **none**. Required wildcard guards exist (`.gitignore:76–91`). Model/pickle guards are scoped to specified directories rather than every possible arbitrary output path; CLI custom paths must remain private.

Reachable historical objects:

| Path | Size in bytes | Blob SHA |
|---|---:|---|
| `processed/dae_fit_stats.pkl` | 1,174 | `30230083ee1fcb3caeccfd952dbe6cb0c2a63b46` |
| `processed/dae_weights.pt` | 118,496 | `e1e7da7b5e1f467bf8774704588e8c718e32f706` |
| `processed/preprocess.pkl` | 3,525 | `d8f6b0e511f0964d43c3582abcb989b5fe7570a3` |

This was a non-shallow clone; reachable objects and history were inspected. No respondent-data extensions were found in that reachable history. The earlier `docs/p2_repo_check.md:69–72` claims those binary artifacts contained parameters/aggregates only; that is the earlier author’s claim, not a new independent binary-content audit. No history rewrite is automatically justified merely because model/scaler files existed. Hosting caches, deleted remote refs, and forks were not inspected.

The repo has no tracked code LICENSE file. README’s DHS-approved-research statement does not establish a software licence or prove the team’s data authorization. `models/RUN_ON_COLAB.md:3–5` recommends private cloud folders/datasets; approval for that environment and each recipient must be checked against the team’s actual access agreement. I tried to retrieve official DHS terms; the direct terms page returned HTTP 403. **No definitive legal finding about permitted cloud processing, team sharing, or model redistribution is made.**

Read-only commands for a local repeat scan (they inspect metadata, not respondent rows):

```bash
git fetch --all --tags
git ls-files | rg -i '\.(parquet|csv|tsv|dta|sav|pkl|pt|zip|feather)$'
git log --all --oneline --name-status -- '*.parquet' '*.csv' '*.tsv' '*.dta' '*.DTA' '*.sav' '*.pkl' '*.pt' '*.zip'
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | rg '^blob ' | sort -k3nr | head -50
git rev-list --objects --all | rg -i '\.(parquet|csv|tsv|dta|sav|pkl|pt|zip|feather)$'
```

### Command/output verification and likely real-data run behavior

Every Python module and flag shown in `models/RUN_ON_COLAB.md` and `docs/p2_next_steps_for_humans.md` exists, including `--final-test`, `--variant gru|cnn`, `--max-rows`, `--extra-cvae`, `--no-state`, `--generate-bp`, and `--glucose-head gaussian`. The `--marginals M` flag in the **eval_dev module docstring** (`models/eval_dev.py:6`) does **not** exist; evaluation uses checkpoint data and aligned real conditions instead. The two requested run-guide files do not use that nonexistent flag. `--out-dir` ablations write the named checkpoint files. Step0 and benchmark write JSON+Markdown; eval writes JSON+Markdown; CVAE writes weights/preproc/marginals/log/card; baselines write pickles/log. These were inspected in source, and corresponding mock smoke outputs were exercised.

| Stage | Verified static behavior / likely stopping point |
|---|---|
| `step0_checks` | Needs private raw+imputed train/val and manifest. Reports checks but does not enforce all findings. Women’s height missingness should be visible, yet scoped share may still look acceptable before complete-array drops. Reads full frames into RAM. |
| `check_dae_benchmark` | Needs private DAE weights/stats plus raw train/val. Explicit paths avoid the legacy Windows default. CPU whole-frame inference may be memory-heavy. No numerical real-data result verified. |
| `train_cvae` | Will discard rows with missing height and invalid state; may silently train a men-only model. Other all-missing columns/empty scopes can produce NaN preprocessing/loss errors. CUDA used when available; P1 DAE itself is CPU-only. |
| `train_baselines` | Same height missingness removes women. Training ignores state and subsamples. Exact combination rejection can have poor coverage; actual convergence/runtime unknown. |
| `eval_dev` | Aligns eight condition values to held-out rows, but mishandles invalid state; uses CPU CVAE sampling. Results may reflect only common combinations filled by both baselines. Empty coverage lacks a clean failure path. |
| `eval_dev --final-test` | Correctly requests the explicit test gate in P2, but cannot undo earlier repository-wide exposure. Freeze model/config and document prior exposure before this stage. |

`docs/p2_next_steps_for_humans.md:47–53` proposes architecture/KL adjustments based on dev diagnostics. These are hypotheses, not proven remedies for tail collapse or memorization. A failed tail metric should first trigger checks for scope, units and constraints.

### What the 36 tests prove and do not prove

They prove that constructed 31-column mock frames fit the P2 adapters; two-epoch mock CVAE training produces finite validation ELBOs/artifacts; variants load/sample; direct real-generator signatures/units/labels/age-BMI constraints/determinism and invalid-input guards work on those fixtures; missing weights produce an informative error; generation normally writes no files; a 10K generation speed check passed in this thread-limited CPU environment; real-generator HTTP wiring works; step0, toy DAE benchmark, both baseline fits and evaluation can run with mock data.

They do **not** exercise NFHS preprocessing or women’s household joins, real state codes, medication missing codes, DAE real-weight quality, actual privacy output edge cases, forbidden HTTP fields, default-stub semantics, profile key consistency, config/checkpoint drift, real tail fidelity, unseen combinations, or survey-weight representativeness. The mock builder supplies women’s height and uniform contiguous states (`tests/mock_data.py:42,49,95`), hiding C1 and C3. There is no objective real-data acceptance threshold, GPU determinism/latency evidence, or licence test.

## 7. NOT REVIEWED / could not verify

- Private NFHS files, private handover contents, raw value labels, biomarker codebook/units, manifest identity, actual linkage coverage, actual sex/state exclusions, pregnancy status, and consent/access terms: deliberately not obtained.
- Real CVAE/DAE/baseline checkpoints and their calibration, rare-tail performance, memorization, disclosure risk, convergence or posterior collapse: not available; mock results are not substitutes.
- Historic binary payload contents: object identities/sizes/history inspected, not independently deserialized. Full historical source revisions were not line-by-line reviewed; current 47 tracked files were.
- P3’s outcome implementation, formal validation report/BERT parser, and P4 frontend: not tracked on reviewed main. No frontend behavior or future PR was invented.
- Official DHS terms page could not be retrieved (403); current research authorization, sharing/cloud/model-output permissions remain team questions. No unsupported legal verdict.
- Python 3.11, Colab/Kaggle, T4/CUDA training, long-run memory/runtime, and precise timing claims: not executed. Installed runtime was Python 3.12/CPU tests.
- Unreachable GitHub history, deleted refs/caches/forks or other untracked local materials: not inspected.

## 8. Prioritized fix list

1. **P1:** repair women’s height linkage and validate required columns/codebook mappings; rebuild private v2 preprocessing artifacts. **P2:** enforce post-encoding per-sex/state support before any training.
2. **P1 + P2:** replace contiguous state assumptions with a checkpointed raw-code mapping; remove invalid-state substitution during evaluation.
3. **P1 + P3:** disable real-row stubs in real-data mode; reject missing/mock models, remove manually calibrated glucose fallback, and recompute outcome from generated glucose under a frozen threshold.
4. **P1 + P3:** enforce strict HTTP condition validation; fix `/profiles` keys, mode/readiness reporting and `/compare` provenance. Add targeted mock regression tests for C1/C3/M1/M2/M3/M4.
5. **P1 + P2 + P3:** centralize test access; retire combined-data development readers and abort split/hash failures before saving. Record already exposed test aggregates.
6. **P1 + P2 + P3:** apply one suppression/publication policy to every log/report; stop on too little evaluation support. Audit historic binary artifacts and document licence/authorization decisions.
7. **P1:** correct medication/tobacco missing encodings, partial-BP logic and DAE imputation consistency. **P2:** benchmark DAE against median on identical truly known targets with complete prediction coverage.
8. **P2 + P3:** run step0 on the holder’s machine after fixes, verify units/manifest/domain values, then train CVAE/baselines using train/val. Evaluate clipping, actual conditioned-tail rates, matched coverage and fair comparison settings.
9. **P3:** define proper uncertainty/validation semantics and provide `validation_report.json` or its explicit adapter. **P4:** show synthetic/unweighted/scope labels, full supported profiles, effective conditions and uncertainty limitations; avoid clinical/causal wording.
10. **P1 + P2:** update quickstart/tree/run guides/config authority/dependency paths and remove stale v1 statistics/risk-awareness titles. Freeze model/config before the single final test. Treat public serving as a later readiness check, not proof from passing smoke tests.

## 9. Questions for the team

1. Were the private v2 files produced from this exact preprocessing commit? If women’s height is present locally, what unpublished patch or different script produced it?
2. What are the actual NFHS state raw-code/value-label mappings, and is state 37 present after scope? Which codes have adequate post-encoding support?
3. Which codebook passages confirm glucose units/special missing codes, raw tobacco/medication mappings, and the intended partial-BP/fallback rule?
4. Is the intended reference population the unweighted observed sample or a survey-representative population? How should sex-specific sampling designs and scope restrictions be represented?
5. What counts as sufficient joint scenario support, and what response should unsupported or heavily clipped scenarios return?
6. Should reported intervals describe only Monte Carlo variation or also fitted-model/data uncertainty? What validation tolerance is required for conditional elevated-glucose rates?
7. Which test data/statistics have already been examined, by whom, and what model choices followed? Can final results honestly be described as untouched-test evaluation?
8. Does the actual research agreement authorize each team recipient and the proposed private cloud/model-serving environment? What output/model redistribution is permitted?
9. Are historical binaries confirmed aggregate/parameter-only, and does the team intend to release its source under a stated software licence?
10. When P3/P4 merge, what is the agreed versioned HTTP/aggregate/validation contract and what happens when real weights or verified validation are absent?

### Appendix: complete wording/agentic search-hit classification

Case-insensitive search covered all tracked code/docs/comments for `diabetes risk`, `prevalence`, `diagnos`, `predict`, `risk score`, `causes`, word `will`, `intervention`, agent/LLM/tool-calling and named agent libraries. Only the three risk-awareness titles below violate the requested wording. No affirmative diagnosis/individual-risk-score/causal-intervention claim, executable agentic workflow, or LLM row generation was found. No `prevalence`, `risk score`, `causes`, `LLM`, or LangGraph/AutoGen executable import hit was found.

| File:line | Hit / classification |
|---|---|
| `README.md:2` | “Diabetes Risk Awareness”: **violates proxy-only project wording**. |
| `config.yaml:11` | Same title: **violation**. |
| `backend/main.py:67` | Same title displayed in API metadata: **violation**. |
| `README.md:4` | “NOT a diagnostic tool”: acceptable prohibition. |
| `README.md:5` | “never diabetes diagnosis”: acceptable prohibition. |
| `README.md:16` | Clinical diagnosis under “It is NOT”: acceptable. |
| `README.md:17` | Individual prediction under “It is NOT”: acceptable. |
| `README.md:18` | Agentic/LangChain/AutoGen/tool-calling under “It is NOT”: acceptable. |
| `README.md:166` | No-agentic-library rule: acceptable. |
| `README.md:243` | NOT diagnosis disclaimer: acceptable. |
| `config.yaml:12` | NOT diagnostic/clinical: acceptable. |
| `config.yaml:13` | Never say diabetes risk: acceptable rule. |
| `backend/main.py:5` | No agents/LangChain: acceptable declaration; verified imports contain no such workflow. |
| `backend/main.py:68` | NOT diagnostic: acceptable disclaimer. |
| `backend/schemas.py:13` | Not individual predictions: acceptable disclaimer. |
| `backend/schemas.py:14` | NOT diagnosis/clinical assessment: acceptable disclaimer. |
| `backend/schemas.py:106` | NOT causal intervention: acceptable. |
| `backend/generator_stub.py:75` | “will attach” describes code behavior; not causal health language. The described stub behavior itself is C2. |
| `backend/outcome_stat_stub.py:34` | “CVAE will generate” describes implementation; not causal. |
| `preprocess.py:276` | “will be NaN” describes data schema; acceptable. |
| `train_dae.py:50` | “will learn” describes model training; acceptable. |
| `docs/p2_next_steps_for_humans.md:45` | “will apply” describes future review rules; acceptable. |
| `models/train_cvae.py:182` | Not diagnosis/individual risk: acceptable. |
| `models/train_cvae.py:183` | Diagnosis/individual prediction in forbidden uses: acceptable. |
| `models/train_cvae.py:196` | Not causal intervention: acceptable. |
| `models/model_card.json:4` | Not diagnosis/individual risk: acceptable. |
| `models/model_card.json:5` | Diagnosis/individual prediction in forbidden uses: acceptable. |
| `models/model_card.json:22` | Not causal intervention: acceptable. |
| `models/eval_dev.py:271` | `predict_proba` in internal TSTR AUC evaluation: acceptable scientific metric, not an app diagnosis or individual risk score. |

There were no additional wording-search hits in the metadata audit or aggregate files. Machine keys such as `elevated_glucose_proxy` are legitimate internal columns; display labels should use the approved phrase.

### Appendix: complete current-file inventory

The inventory below is the complete `git ls-files` set at the reviewed commit, not a sampled selection. Empty/package-marker files were also inspected.

```text
.gitignore
DECISIONS.md
README.md
backend/__init__.py
backend/generator.py
backend/generator_stub.py
backend/main.py
backend/outcome_stat_stub.py
backend/schemas.py
config.yaml
dae_impute.py
docs/aggregates.json
docs/aggregates_v2.json
docs/audit_results.md
docs/dae_report.md
docs/p2_next_steps_for_humans.md
docs/p2_repo_check.md
docs/preprocessing_report.md
docs/schema.json
docs/split_report.md
generate_handover_v2.py
models/RUN_ON_COLAB.md
models/__init__.py
models/check_dae_benchmark.py
models/common.py
models/cvae.py
models/data.py
models/eval_dev.py
models/model_card.json
models/sampling.py
models/step0_checks.py
models/train_baselines.py
models/train_cvae.py
preprocess.py
preprocess_v2.py
requirements.txt
split_and_aggregate.py
split_and_aggregate_v2.py
tests/__init__.py
tests/conftest.py
tests/mock_data.py
tests/test_cvae.py
tests/test_data_and_guards.py
tests/test_generator.py
tests/test_scripts.py
train_dae.py
train_dae_v2.py
```
