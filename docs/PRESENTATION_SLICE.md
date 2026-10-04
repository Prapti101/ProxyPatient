# Presentation slice

Updated 2026-10-05 (Asia/Kolkata).

The demo shows how a full baseline profile and a what-if change feed a conditional VAE decoder. It generates new rows in memory, recomputes elevated glucose (proxy) from generated glucose, and displays the synthetic share and descriptive scenario difference. It never resamples survey respondents, calls an agent, or uses an LLM to generate rows.

**Every demo number comes from MOCK training and is not an NFHS-5 result.** Demo responses show a visible banner. Real serving requires private TRAIN-fitted artifacts, rejects mock artifacts and fails closed if artifacts/configuration disagree.

Eight profile inputs are required: sex, age band, residence, wealth quintile, BMI band, hypertension, tobacco and alcohol. State is optional; supported codes are learned from encoded TRAIN rather than a guessed range. Presets come from observed joint TRAIN cells with at least 500 rows. A preset does not validate every possible what-if combination.

The outcome is the share of the synthetic cohort with generated glucose at/above the configured threshold. The Monte Carlo interval reflects generated-cohort variation conditional on the fitted model, not model or survey uncertainty. The result is an **unweighted sample**, with measured-BMI, known-BP-status, known-glucose and complete-encoding scope, including both sexes. Historical reference aggregates have a broader scope. Scenario changes are associations in a fitted model, not causal interventions.

Rejection and clipping diagnostics accompany results. Real-data unit/codebook checks, per-sex support, manifest integrity, state mapping, model fidelity, rare-tail quality and disclosure review remain necessary. Small respondent-backed counts/statistics below 30 are null. Passing mock tests demonstrates software wiring, not scientific validity.

The rule-based parser is demo-only, handles simple negation, has no claimed confidence score and proposes changes for user confirmation. Formal P3 validation is pending. The held-out test split has prior aggregate exposure; final evaluation is a separate acknowledged frozen-model stage.
