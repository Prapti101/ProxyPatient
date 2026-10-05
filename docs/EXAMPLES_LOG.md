# Examples implementation log

Base: origin/p2-finish; fork main does not contain the finish branch. CPU and mock data only.

X1: implemented bounded representative cohort examples. The contradictory extra-elevated requirement is resolved by reserving/replacing the last slot, never exceeding n_examples. K=1 uses the median; K=2 uses quartiles; K=4 adds the 10th percentile. Elevated illustration uses the least elevated generated row. Real mode is conservatively withheld until X2.
No client or PROJECT_WRITEUP.md exists; alignment uses the supplied excerpts. No real-data results are claimed.

X1 validation attempt 1: 106 passed, one new assertion incorrectly compared wall-clock sampling diagnostics. Determinism is asserted on examples and statistics; elapsed duration is intentionally variable. Mock-copy regression uses the actual fixture training size/seed (4,000, seed 1).

X1 DONE: full suite 107 passed, zero failed/skipped, one existing dependency warning. Push follows this commit.

X2: configured median ratio >=0.5, minimum standardised numeric distance >0.00001, zero exact-copy share and >=30 query rows in each reference. Reports must pass existing checksum, fingerprint, mode and VAL checks. Added actual DCR minimum measurement; older reports lacking it fail closed. This sampled numeric TRAIN/VAL diagnostic does not check each displayed profile or prove privacy, and rounded display values can coincide with real values. Real-data validation unavailable.

X2 validation attempt 1: 108 passed; constructed DCR test table had insufficient state support. Increased mock fixture size to 3,000; preserved state/privacy guards. Actual quick/full API regressions also assert withholding with absent reports.

X2 validation attempt 2: 108 passed; exact-copy regression found floating-point cancellation in the existing nearest-neighbor distances (only 71% of identical rows classified as exact). Recompute distances from matched coordinate differences to fix the root cause; exact-copy assertion retained.

X1 push: aea5b74 succeeded on attempt 1 to origin/p2-examples. X2 DONE: full suite 109 passed, zero failed/skipped, one existing dependency warning. Push follows this commit.
