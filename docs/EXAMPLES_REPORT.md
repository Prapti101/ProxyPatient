# Representative examples and writeup alignment report

2026-10-05. Branch: `p2-examples`, based on `origin/p2-finish` at `52fe146` because fork main did not contain the finish work. All computation used CPU and constructed mock data. No private survey rows, real-data results, final TEST evaluation, optional BERT inference or external validation were used.

| Item | Status | Delivered / evidence |
|---|---|---|
| X1 representative examples | DONE | Default three, strict configurable 1..5 per scenario; deterministic glucose-rank selection from freshly generated cohort; original-unit features, generated outcome and visible synthetic/mock labels. Stable scenario IDs. Full-cohort statistic unchanged; bulk export remains disabled |
| X2 real-mode gate | DONE | Verified packaged VAL report bound to served checkpoint; >=30 query support, generated/real median ratio >=0.5, minimum >0.00001, zero exact-copy share. Missing/failed/stale/old reports withhold examples with explicit status. Added measured minimum and corrected exact-distance floating-point cancellation |
| X3 contract / plain-language docs | DONE | Executed mock generate/compare captures; schemas, selection/gate/statuses, README, demo runbook, viva notes and stale private-holder instructions aligned |
| Client types / rendered cards | SKIPPED | No client exists in this repository. API supplies card data; actual UI integration unverified |
| X4 specification table | DONE | Available writeup excerpts compared with code; optional BP/ablations/token hook, Wilson interval, eight required inputs, synchronous generation, missing external validation, naming and UI gaps documented |
| Full test suite | DONE | Final suite 111 passed, 0 failed, 0 skipped; one existing Starlette/httpx deprecation warning |
| Current-tree leak / active wording / new-commit hygiene checks | DONE | No tracked parquet/csv/dta/sav/zip/pt/pkl; active backend/README/contract/demo/viva have no prohibited outcome wording. All new commits use required identity and plain messages; no forbidden attribution, trailers, URLs or generated-by footers |
| Literal whole-history clean check | BLOCKED (inherited) | Seven inherited Claude author entries and three historical binary paths remain reachable. No shared-history rewrite was authorized or performed. This is not a clean-history claim |

## Decisions and limits

The prompt requests an extra elevated illustration and also forbids returning more than n_examples. The hard cap wins: the least elevated generated row replaces the last selection when absent and is explicitly flagged illustrative_elevated. K=1 uses the median, K=2 quartiles, K=4 adds the 10th percentile; default K=3 and K=5 follow the specified ranks. Selection is deterministic for the cohort's seed. Display rounding does not determine the outcome flag or conditional bands.

The report gate is a **heuristic, not a privacy guarantee**. It uses sampled numeric TRAIN references and generated/VAL queries; it does not check every card, categorical equivalence or rounded display values. Checksums and artifact identity do not constitute release authorization. Older reports lacking the minimum field fail closed and need a refreshed VAL evaluation/package, never a repeated final TEST run. Missing gate evidence withholds examples while retaining outcome summaries.

No frontend or full PROJECT_WRITEUP.md was available. The supplied excerpts are the only verified specification source. The model is designed for private NFHS-5 training; mock weights and software metrics are not verified NFHS-5 findings. Optional generated BP remains off; clinical coding/measurement questions remain unresolved. Main model remains MLP CVAE; GRU/CNN are optional ablations, the rule parser is demo-only, local token annotations do not determine conditions, and the earlier proposed optional BERT similarity patch remains unapplied.

The explicit old title quoted in SPEC_VS_BUILD.md is a naming discrepancy, not an outcome claim. Archived review/legacy documents and inherited history retain historical wording; active claims use elevated glucose (proxy). Historical binary paths are processed/dae_weights.pt, processed/dae_fit_stats.pkl and processed/preprocess.pkl; they are absent from the current tracked tree and were not loaded.

## Validation and push log

Every implementation item was committed only after its complete suite passed and pushed to origin/p2-examples immediately. No main/upstream/WIP push, merge or PR was made.

| Item | Commit | Passing full suite | Push |
|---|---|---|---|
| X1 | aea5b74 | 107 passed | Attempt 1 succeeded |
| X2 | 890cef5 | 109 passed | Attempt 1 succeeded |
| X3 | d09e1e2 | 110 passed | Attempt 1 succeeded |
| X4 | 52e54db | 111 passed | Attempt 1 succeeded |

The report itself is committed and pushed after its final suite; branch-tip synchronization is checked after that push. Detailed attempts/root causes are in EXAMPLES_LOG.md. Initial failures were corrected without weakening guards: variable timing is excluded from deterministic content comparison, mock state-support fixture enlarged, exact-distance cancellation fixed, and literal JSON replacement fixed for document captures.

Regression coverage includes demo examples/labels, fixed-seed equality, percentile ranks, elevated illustration/caps, unchanged cohort statistics, invalid counts/export requests, compare IDs, generated examples distinct from the actual mock training fixture, missing/failing/stale/malformed/suppressed diagnostics, actual quick/full API withholding with absent reports, identical-row distance detection and captured contract validation.

## Human decisions

- Resolve the writeup title to match elevated glucose (proxy) terminology.
- Integrate cards/statuses into the frontend; use an indeterminate wait indicator, no fake progress.
- Verify clinical units/coding, real-data fidelity/tails/support, disclosure authorization and release review on the approved holder's machine.
- Decide separately whether inherited Git history requires coordinated cleanup; no history was rewritten here.
