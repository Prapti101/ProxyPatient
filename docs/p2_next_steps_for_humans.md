# P2 next steps for the private data holder

Updated 2026-10-05 (Asia/Kolkata). Supersedes the historical manual stage list and speculative architecture adjustments. Software smoke results use MOCK data only.

1. Read `OPEN_QUESTIONS.md`, the committed preflight review and `TEST_SPLIT_POLICY.md`. Confirm authorization for the intended private runtime; do not upload NFHS rows to this task.
2. P1 repairs/regenerates affected private v2 files and verifies the women's household join, official glucose units, state code labels, medication/tobacco encodings, BP fallback and the locked membership. These facts are not established by mock tests. P2 guard failures must be investigated, not bypassed.
3. Use Python 3.11 and the pinned CPU dependency path in `models/RUN_ON_COLAB.md`. Run `pytest`, then the mock one-command smoke.
4. Run on the approved private machine:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/pp-quick --quick
python -m models.run_all --data-dir /private/processed --out-dir /private/pp-full --full
```

5. Inspect stage timings/peak memory, manifest/unit/support failures, dropped optional variables, per-sex encoding exclusions, supported state/profile cells, requested versus retained condition coverage, finite-target DAE comparison, conditional-rate fidelity, tail/clipping behavior, and disclosure limitations. Compare models only with the stated different data/state setup. A privacy denominator of 30 is not a scientific adequacy certificate.
6. Keep `private_outputs/`, respondent files and model weights private. Only reviewed aggregate `safe_outputs/*.json/*.md` may be candidates for public commit, subject to the actual agreement. The real package populates `models/model_card.json`; never replace pending fields with manually invented numbers. The API reads matching safe_outputs through PP_REPORT_DIR; private-data validation remains unverified. See API_CONTRACT.md and DEMO_RUNBOOK.md.
7. Serve through `PP_MODEL_DIR=/private/pp-full/private_outputs`, require full profiles, and inspect health/fingerprint/mode. The explicit mock demo requires `PP_DEMO_MOCK=1`; real mode rejects it.
8. Freeze model, config and selection decisions. Execute the acknowledged final command once on the frozen full-run directory:

```bash
python -m models.run_all --data-dir /private/processed --out-dir /private/pp-full --final-test --i-understand-this-is-the-single-final-run
```

Use the honest held-out description in the test policy. Record prior local test examination; do not tune after the final outcome.

Do not change mixture components, architecture or thresholds merely to obtain a desired outcome rate. Real validation can identify specific problems, but this cloud task supplies no real-data verdict or model winner.
