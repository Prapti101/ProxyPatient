# Deferred work

Updated 2026-10-05 (Asia/Kolkata). These are deliberate scope deferrals, not failed mock tests. Review IDs refer to `reviews/ProxyPatient_Preflight_Review.md`.

| ID | Deferred work / why | Owner |
|---|---|---|
| M5 | Retire all-data scaler artifacts and any remaining consumers. Current P2 docs no longer use them, but removing P1 artifacts requires private reruns/consumer agreement. | P1 |
| M6 | Official-codebook medication/tobacco/missing encodings; no raw data or codebook evidence is available here. Do not guess maps. | P1 |
| M7 | Three-valued partial-BP/medication rule and first-reading fallback protocol; needs codebook/team agreement and private regeneration. | P1, P3 |
| M13 | Historical binary audit and any approved history cleanup. Current tree is clean; untracking does not erase history. No history rewriting is authorized. | Maintainer, P1 |
| m4 | GPU RNG isolation/deterministic algorithms and CUDA validation. This implementation/test environment is CPU-only. | P2 |
| m6 | Mixed-type, state/condition-matched nearest-record disclosure evaluation beyond the sampled numeric metric. The existing metric is not a privacy certificate. | P3 |
| m7 | Extended weight/height/BP consistency and undefined-correlation reporting. Non-finite outcome guards are fixed; scientific metric design remains separate. | P3 |
| m8 | Conventional independent-cohort TSTR and equal-data/equal-state architecture comparisons. Current matched-condition transductive/retained metrics are explicitly labelled. | P2, P3 |
| m11 | Observed-cell DAE loss masks, categorical heads and a fully shared training/inference imputation contract. Batching/test safety is fixed; DAE redesign needs private validation. | P1 |
| m12 | Transitive dependency lock and tested CPU/GPU installation matrix. Direct pins and uv CPU install are verified; no CUDA claim or generated dependency updates are made. | P1, P2 |
| m13 | Remaining personal Windows paths/import portability in raw preprocessing/DAE training. Critical split/handover entry points were made portable; comprehensive P1 path migration is deferred. | P1 |
| Legal/licence | Software licence, recipient authorization, private cloud use and public model/output redistribution. Repository statements cannot establish permissions. | Principal investigator, maintainer |
| Review §9 | Resolve every factual question in OPEN_QUESTIONS.md before private reruns/public serving. No private/codebook facts are guessed. | Team leads |

Additional scoped limits: real-data model quality, rare-tail calibration, supported what-if extrapolation policy, survey weighting, posterior collapse, disclosure release criteria and formal P3 validation are unverified. Scientific adequacy may require support far above the privacy minimum. P4 must carry the full profile, demo/scope/weighting/uncertainty/provenance labels into its UI; no frontend is included in this repository.
