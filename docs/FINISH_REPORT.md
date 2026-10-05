# Finish report

Completed 2026-10-05 (Asia/Kolkata). Branch `p2-finish` starts from `origin/p2-presentation-slice` (`a75c72c`), because that delivery is not an ancestor of fork main (`c8a92a6`). No P3 branch merge and no main/upstream push. No private survey data was obtained or processed.

## R1–R8 status

| Item | Status | Result |
|---|---|---|
| R1 | DONE | Mock/quick/full checkpoint, card, package and API provenance; preliminary notes/banners; incompatible/missing labels rejected. All serving combinations tested. |
| R2 | DONE | Eligible TRAIN sex/outcome-stratified quick sampling; configurable 2,000 quick versus unchanged 5,000 full minimum; configured budgets, included baselines, no quick ablations; timings.json. |
| R3 | DONE | Typed validation/model-comparison endpoints consume checksum/fingerprint-bound packaged VAL reports; absent/stale/malformed/insufficient reports are explicitly pending. |
| R4 | DONE | Exactly three P3 additions reconciled without merging branch; one guarded parser, two pending templates, shared canonical metrics plus source classifier/subgroup suppression. |
| R5 | DONE | Started/completed marker; acknowledged crash-before-metrics recovery, frozen hashes, atomic state publication and directory lock; completed cannot repeat. |
| R6 | DONE | Actual API contract and captured labelled mock examples. Client update SKIPPED: no frontend/src/package.json exists. |
| R7 | DONE | Demo/private-quick runbook and evidence-based viva notes; no invented real-data scores or winner. |
| R8 | DONE | Current README/backlog/questions/guides aligned; eight full conditions; explicit unresolved private DAE provenance/test-exposure question. |

No item is blocked. Private-data scientific/release decisions remain open, not silently treated as successful software checks.

## P3 material and files

The three-dot diff `origin/main...origin/p3-files` adds only root `parser.py` (4,710 bytes), `validation_report.json` (2,430 bytes), and misspelled `model_comparision.json` (1,805 bytes). All were checked out individually. Neither added files nor the inspected P3 tree contains tracked data-like extensions; no LEAK WARNING was found. No large/data/weight file was copied.

Used git mv to relocate the parser's untouched original to `legacy/p3_parser_original.py` (never imported), validation schema to `docs/p3/validation_report.template.json`, and corrected comparison schema to `docs/p3/model_comparison.template.json`. Active phrase handling lives only in `backend/parser.py`; P3's aliases/relative BMI and optional token hook were reconciled with the existing safeguards. The lazy hook accepts an existing explicit local model directory, uses local_files_only and does not determine conditions. No offline BERT files were present, so normal serving is labelled rule-based with no confidence claim. Glucose/blood-sugar inputs reject; improved BMI resolves against baseline or remains unresolved; unchanged wording proposes no changes and never implies temporal modelling.

P3 files contained no executed real-data metrics. Templates remain pending/null, remove uploaded-file identifiers and unsupported bootstrap claims, and label GRU/CNN as CVAE tabular ablations. Existing KS/Wasserstein/correlation/TSTR/nearest-record implementations remain canonical in eval_dev. Added held-out real-versus-synthetic source classification with TRAIN-fitted numeric scaling and subgroup fidelity with the shared disclosure threshold. Neither metric certifies privacy or scientific validity.

New implementation modules: `models/run_status.py`, `models/quick.py`, `backend/parser.py`, `backend/reports.py`, `tests/test_finish.py`. New docs: API_CONTRACT.md, DEMO_RUNBOOK.md, VIVA_NOTES.md, FINISH_LOG.md, FINISH_REPORT.md and the two P3 templates. Existing pipeline, trainers, schemas, evaluation and regression tests were adjusted. No dependency updates or unrelated redesign.

## Decisions and additional bugs

- Eight required non-state conditions; optional supported raw state code. Missing conditions never receive silent defaults in real serving.
- Run type is explicit and included in fitted identity. Mock always remains mock even with --quick; quick is preliminary; full denotes configured budget, not a quality certificate. Old checkpoints need retraining, not guessed provenance.
- Quick defaults: 100,000 eligible CVAE rows/six epochs, 20,000 baseline rows/15 epochs and 10,000 requested evaluation rows. Complete-row filtering precedes sex/outcome sampling. Insufficient counts are suppressed in errors; thresholds are never lowered to pass tests.
- The preset mock demonstration and random mock benchmark have separate fitted identities. The preset panel stays pending without a compatible report; benchmark ingestion works but may not provide three >=500-row profiles. This is documented rather than inventing support.
- Recovery allows only explicit confirmation of a started run before metrics, with identical artifacts. A metrics-present started run is conservatively refused; completed always blocks; retraining after a marker remains prohibited.
- Self-review corrected full baseline training including invalid rows rejected by encoding, parsed unsupported state returning 500 instead of 422, inconsistent package run labels, missing central checkpoint provenance validation, malformed model-table acceptance, and empty-marker bypass. Locks now release explicitly even when callers retain exceptions. Final quick evaluations keep their original quick label.
- The local fetch refspec originally tracked only main, so push success alone did not create origin/p2-finish. Added that branch's fetch entry and verified remote identity without resetting source. No force push.

## Verification

Final complete suite: **104 passed, 0 failed, 0 skipped**, 42.48 seconds, CPU with OMP/MKL/OPENBLAS threads=2. One upstream Starlette/httpx TestClient deprecation warning remains; no warning was silenced and no guard/test weakened. All packages needed for these tests were installed; optional transformer/model use was not claimed or required.

Each numbered item passed the full suite before commit: R1 93, R2 95, R3 96, R4 98, R5–R8 100, final self-review 104. No item required rollback or five failed attempts.

The exact `.venv/bin/python -m models.run_all --mock --quick` command completed in the sandbox with all six stages, including TVAE and CTGAN. Its packaged reports were read by both endpoints: complete, run_type mock, status_banner DEMO (mock data), with source/subgroup additions. Live HTTP preset-demo checks passed health, supported profiles, generation and honest pending reports. Captured mock requests also exercise parse, compare, invalid full profiles and unavailable artifacts.

Checksum scans passed. No saved row CSV/Parquet and no unsuppressed declared respondent-backed count below 30 in mock JSON outputs. Structural model settings, timing values, state/category codes and epoch numbers are not respondent counts. Git tracked-file leak scan found no parquet/csv/dta/sav/zip/pt/pkl. Commit messages/identity passed attribution/session/trailer checks. Active wording is proxy-only; matches are prohibitions/technical diagnostics, user input echoes or expressly historical material. P3's untouched legacy parser is audit-only and retains historical wording; it is never imported.

Local evidence: `/tmp/finish-final-tests.log`, `/tmp/finish-dod-quick.log`, `/tmp/finish-api-capture.log`, `/workspace/proxypatient-finish/api_examples.json` and ignored `outputs/run/`. These are mock execution artifacts, not survey results or public row products. The live API on port 8000 serves the explicit preset MOCK model. Cloud startup draft is saved with current instructions; review/save/publish in environment settings is still required to activate that draft.

## Human private-data commands

Resolve P1 provenance/unit/codebook and recipient permissions first. On the authorized holder's machine, quick first:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-quick --quick
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-full --full
unset PP_DEMO_MOCK PP_ALLOW_PARTIAL_PROFILE PP_CVAE_WEIGHTS PP_CONDITION_MARGINALS
PP_MODEL_DIR=/private/proxypatient-quick/private_outputs PP_REPORT_DIR=/private/proxypatient-quick/safe_outputs python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Use supported full `/profiles`, display the preliminary banner, inspect retained coverage/clipping and matching measured reports. Freeze selected artifacts/configuration before one final run:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/proxypatient-full --final-test --i-understand-this-is-the-single-final-run
```

Only after a confirmed crash before any metrics, on that same unchanged frozen directory, add `--confirm-previous-final-run-crashed`. Never remove markers or repeat completed evaluation. Keep private_outputs and respondent files private; review aggregate outputs before any public release. Read DEMO_RUNBOOK.md and API_CONTRACT.md for request examples.

## Push receipt

All implementation commits were immediately pushed once to the fork's p2-finish branch; nine successes, no failures/retries. Times below are the persisted commit timestamps immediately preceding each push; exact push-completion seconds were not separately recorded.

| Attempt | Time (Asia/Kolkata; commit immediately before push) | Branch | Tip | Outcome |
|---|---|---|---|---|
| 1 | 2026-10-05T10:25:54+05:30 | p2-finish | 2f51c59 | SUCCESS |
| 2 | 2026-10-05T10:28:30+05:30 | p2-finish | c3b4f88 | SUCCESS |
| 3 | 2026-10-05T10:30:30+05:30 | p2-finish | cc82ca2 | SUCCESS |
| 4 | 2026-10-05T10:34:03+05:30 | p2-finish | 85ceafc | SUCCESS |
| 5 | 2026-10-05T10:37:50+05:30 | p2-finish | 5de5d48 | SUCCESS |
| 6 | 2026-10-05T10:41:32+05:30 | p2-finish | bcbeeb3 | SUCCESS |
| 7 | 2026-10-05T10:44:10+05:30 | p2-finish | c2302b9 | SUCCESS |
| 8 | 2026-10-05T10:46:35+05:30 | p2-finish | 5c3fa70 | SUCCESS |
| 9 | 2026-10-05T10:51:01+05:30 | p2-finish | 0465917 | SUCCESS |

`p2-finish` was synchronized through `0465917` when this report was prepared. This report's own publication necessarily follows its commit; the final response confirms its push and the remote-tracking SHA equality check. `p2-finish-wip` was never needed/created: all committed snapshots passed, and no item exceeded 30 minutes. Every observed implementation push receipt is also in FINISH_LOG.md. No push to main/upstream, no force push and no PR was created.

## Human decisions still needed

Which private preprocessing/DAE script and commit produced the imputed files, and did the actual run train/infer on test rows? Actual units/missing/categorical codes, state support, weighting policy, sufficient joint scenario support, real conditional/tail performance, disclosure criteria and licences/recipient permissions remain unverified. Full-budget execution, CUDA behavior and external validation are not claimed. External P4 must implement the documented API contract. The team must decide how to review a started final run that already wrote metrics without enabling model-selection reruns. See OPEN_QUESTIONS.md and BACKLOG.md.
