# Examples implementation log

Base: origin/p2-finish; fork main does not contain the finish branch. CPU and mock data only.

X1: implemented bounded representative cohort examples. The contradictory extra-elevated requirement is resolved by reserving/replacing the last slot, never exceeding n_examples. K=1 uses the median; K=2 uses quartiles; K=4 adds the 10th percentile. Elevated illustration uses the least elevated generated row. Real mode is conservatively withheld until X2.
No client or PROJECT_WRITEUP.md exists; alignment uses the supplied excerpts. No real-data results are claimed.

X1 validation attempt 1: 106 passed, one new assertion incorrectly compared wall-clock sampling diagnostics. Determinism is asserted on examples and statistics; elapsed duration is intentionally variable. Mock-copy regression uses the actual fixture training size/seed (4,000, seed 1).

X1 DONE: full suite 107 passed, zero failed/skipped, one existing dependency warning. Push follows this commit.
