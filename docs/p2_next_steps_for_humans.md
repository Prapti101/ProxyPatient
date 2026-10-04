# P2: next steps for the humans (Half B)

The P2 code was written and tested on MOCK data only (no NFHS-5 data was ever
in the cloud environment). Nothing about real-data performance is known yet.

## Who does what
* **Data holder (P1 or P2's human):** follow `models/RUN_ON_COLAB.md` steps 0-5.
* **P1:** add `hypertension` (and optional `state`) to `Condition` in
  `backend/schemas.py`; reconcile the sex encoding and `aggregates_v2.json`
  keys (see `docs/p2_repo_check.md`); decide whether to untrack the v1
  `processed/*.pkl|.pt` files.
* **P3:** `outcome_stat` can rely on the generator's `elevated_glucose_proxy`
  and `glucose_raw` columns and on `is_synthetic=True`. Formal validation should
  use the test split once, via `models/eval_dev.py --final-test` or your own code.
* **P4:** always send a FULL baseline profile plus the what-if changes; any
  key left out is filled from independent population marginals.

## Commands (in order)
```bash
export PP_DATA_DIR=/path/to/private/processed
python -m models.step0_checks
python -m models.check_dae_benchmark
python -m models.train_cvae
python -m models.train_baselines
python -m models.eval_dev
# at the very end, once:
python -m models.eval_dev --final-test
```

## What to paste back to P2 if tuning is needed (aggregates only)
1. `docs/p2_step0_checks.md`, these sections: File integrity, Caveat checks
   (train), Training scope, Condition cell sizes.
2. `docs/p2_dae_benchmark.md` (the table).
3. From `models/cvae_train_log.json`: `n_train`, `n_val`, `best_val_elbo`,
   `train_seconds`, and the last 3 entries of `epochs`.
4. `docs/model_comparison_dev.md` (the summary table) and from the JSON, for
   each model: `tail`, `conditional_rate_fidelity.overall`,
   `conditional_rate_fidelity.by_sex`, `direction`, `consistency`,
   `generation` (CVAE: `sampling.first_pass_inconsistent_share`; baselines:
   `rejection_sampling`).
5. `docs/baselines_train_log.json`.

Never paste rows, row samples, or anything from the parquet files themselves.

## Decision rules P2 will apply to the results
* If CVAE `% >= threshold` or P95/P99 is well below REAL: compare
  `--glucose-head gaussian` vs `mixture` (default) and raise components to 5.
* If `first_pass_inconsistent_share` > 0.2 for typical conditions: raise
  `latent_dim`/epochs or model age/BMI within band.
* If conditional MAE (pp) is worse than TVAE/CTGAN: try `beta` 0.5, more epochs,
  `--no-state`.
* If nearest-record distance for generated rows is far below the real-to-real
  baseline (or `share_exact_copy` > 0): stronger KL (`beta` > 1) and report it.
