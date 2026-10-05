# Overnight implementation report

Completed 2026-10-05 (Asia/Kolkata) on `p2-presentation-slice`, based on fork main `c8a92a6`. The uploaded specification was evaluated against the committed source; incorrect findings were corrected rather than implemented blindly. No private survey data was downloaded, decoded or saved.

## Delivery status

| Item | Status | Result |
|---|---|---|
| Baseline/P2 merge | SKIPPED | P2 tip is already an ancestor of main; merging would add no commits. Older branch contents/binaries were not restored. |
| 1.1 | DONE | Women's household linkage, optional height/derived weight, per-sex complete-encoding support guards. |
| 1.2 | DONE | Supported raw-state mappings learned from eligible TRAIN rows and persisted consistently. |
| 1.3 | DONE | Validated CVAE serving; explicit labelled mock demo; unavailable artifacts return 503; resampling stub removed. |
| 1.4 | DONE | Strict full profiles, supported presets and scenario provenance; explicit partial-profile development opt-in. |
| 1.5 | DONE | Wilson Monte Carlo intervals, including zero/all-positive outcomes; uncertainty limitations disclosed. |
| 1.6 | DONE | Shared small-cell suppression and honest finite-target DAE benchmarking. |
| 1.7 | DONE | Locked split integrity, centralized readers, retired v1 scripts and documented historical test exposure. |
| 2.1 | DONE | Guarded one-command pipeline, measured stages, separated private artifacts and aggregate reports, acknowledged final evaluation. |
| 2.2 | DONE | Invalid/non-finite model inputs fail, empty frames preserve schema, benchmark failures cannot improve scores silently. |
| 2.3 | DONE | Retained/shared condition coverage and comparison limitations reported. |
| 2.4 | DONE | Bounded inference batches, measured peak RSS and clipping diagnostics. |
| 2.5 | DONE | Configuration defaults honored; only explicit flags override them. |
| 2.6 | DONE | Current artifact packages, complete checksums, fitted-model identities and real-run model cards. |
| 2.7 | DONE | Private quick/full/final instructions and reviewed artifact handling. |
| Phase 3 | DONE | Proxy-only presentation wording, current API documentation and honest rule-based demo parser. |
| Phase 3B | SKIPPED | Only main/P2 remote heads exist; no P3 artifacts available. Validation remains explicitly pending. |
| Phase 4 | DONE | Requested deferred IDs/owners and verbatim review questions recorded. |
| Final self-review | DONE | Additional artifact, support, configuration and publication edge cases corrected and tested. |

## Decisions and verified corrections

A full profile requires all eight listed non-state conditions; the specification says seven but enumerates eight. State is optional. Real training requires 5,000 encoded rows per sex; mock checks explicitly use 30. Optional measurement coverage defaults to 80% per sex, women's linkage guard to 50%, state support to 30 and preset joint support to 500. Privacy suppression is a minimum, not scientific adequacy or release authorization.

Serving requires compatible fitted weights, preprocessing and marginals. MOCK requires `PP_DEMO_MOCK=1` and remains labelled in every JSON response. Outcomes derive from finite generated glucose only. Reports describe unweighted synthetic proxy outcomes and Monte Carlo uncertainty. Row export is disabled. Model identity includes fitted parameters, configuration, scope and preprocessing.

Several review observations were already fixed at the fetched baseline, including existing household/state fields, configuration thresholds, v2 aggregate behavior, health version and blood-pressure schema details; they were rechecked rather than blindly reverted. The executable legacy DAE trains on TRAIN and imputes combined data: the specification's claim of training on combined is unsupported by that source. Historical private training remains unverified. Previously computed full-data outcome aggregates and combined inference still prevent claiming untouched-test evaluation.

Self-review additionally fixed state support counted before complete encoding, optional derived weight retained without sufficient coverage, stale repeated-run artifacts, incomplete package hashes, configuration-only model identities, missing scope/finite-parameter checks, unsupported preset/marginal acceptance, rare-cell direction statistics and fractional condition coercion. Configured preprocessing thresholds/split ratios and TRAIN crosstabs are preserved.

The removed `backend/generator_stub.py` resampled real rows. V1 preprocessing, split and DAE executables moved to `legacy/` with do-not-run guidance. No tracked data/weight files were introduced; historical Git binaries were not rewritten. Scientific/private-data/licensing questions remain in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), and deliberate deferrals in [BACKLOG.md](BACKLOG.md).

## Verification

- Final complete suite: **83 passed, 0 failed, 0 skipped**, 49.47 seconds. One upstream Starlette/httpx TestClient deprecation warning remains.
- Separate six-stage `--mock --quick` run completed successfully; the suite also exercises the pipeline end to end.
- Live API smoke passed health, schema/options, three supported mock profiles, generation, deterministic outcomes, comparison, parser negation/confirmation and rejected invalid/partial inputs. Validation correctly reports pending.
- Package checksum and small-cell scans passed; mock runs saved no row-level CSV/Parquet. Tracked-file scan found no data/weights. Whitespace and commit author/message hygiene passed.
- CPU-only Python 3.11.16 with pinned dependencies; two numerical threads. No CUDA or private-data quality/performance claim is made.

Local execution evidence is under `/workspace/proxypatient-onboarding/`: `final-verified-suite.log`, `final-quick.log`, `final-mock-run/` and `smoke.py`. These are local environment artifacts, not repository data products.

## Reusable commands

```bash
.venv/bin/python -m models.run_all --mock --quick --out-dir /tmp/proxypatient-mock
.venv/bin/python -m models.run_all --data-dir /private/data --out-dir /private/run --quick
.venv/bin/python -m models.run_all --data-dir /private/data --out-dir /private/run --full
.venv/bin/python -m models.run_all --data-dir /private/data --out-dir /private/run --final-test --i-understand-this-is-the-single-final-run
```

Final evaluation uses the same frozen output directory; its exclusive marker survives failures. Do not remove it or duplicate private inputs to repeat model selection. Private fitted artifacts require recipient/release review even when reports are aggregate-only.

The cloud environment configuration draft was saved with pinned CPU installation, explicit mock checkpoint creation, smoke checks and startup instructions. It still requires review/publishing in environment settings. The current API runs on port 8000 with an explicit mock checkpoint outside the repository.

## Implementation commits

```text
6d3dbed Guard household height linkage and complete training support
666f545 Learn and reuse supported raw state mappings
9104e93 Serve validated decoder artifacts and explicitly labelled mock demos
0bd397b Enforce full validated profiles and disclose scenario provenance
e8a5bb8 Report Wilson Monte Carlo intervals for generated outcomes
29144b8 Centralize suppression and refuse unsupported benchmark metrics
aa8ba9f Preserve locked splits and retire legacy development readers
a2ffceb Add guarded one-command training and private artifact packaging
cdce1df Clarify presentation scope and add honest demo parsing
d5a59c7 Reject invalid model inputs and preserve empty-sample schemas
a5e45b0 Disclose retained condition coverage and comparison limitations
aabd9d4 Batch inference and expose measured memory and clipping diagnostics
19f29b1 Honor model configuration unless flags explicitly override it
3cfdbe6 Package current artifacts with fitted-model identity and real-run cards
a3303bb Document one-command private runs and reviewed output handling
224fac1 Record deferred review work and private-data questions
931f908 Close remaining artifact support and publication edge cases
```

The report and delivery receipt are committed separately; `git log origin/main..HEAD` provides their identifiers.

## Push outcome

Pending the single authorized fork push attempt. The final local receipt will record its outcome without a second push.
