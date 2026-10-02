# ProxyPatient -- Stage 4: Split + Aggregates Report

**Run date:** 2026-10-02T18:22:11.549588
**Seed:** 42  |  **Split:** 70/15/15

## Split Sizes
| Split | Rows | Outcome Rate |
|---|---|---|
| Train | 581,542 | 0.0288 |
| Val   | 124,618   | 0.0288 |
| Test  | 119,794  | 0.0288 |

> **Test set is LOCKED.** Do not load test.parquet until final model validation.

## Overall Aggregate
- Overall elevated glucose (proxy) rate: **2.88%**
- Based on 798,622 respondents with glucose readings
- Threshold: >= 200 mg/dL (random capillary glucose)
- Min cell size for suppression: n >= 30

## Acceptance Checks
- [OK] `train_val_test_sum_equals_total`
- [OK] `test_is_15pct`
- [OK] `stratification_ok`
- [OK] `no_overlap_train_test`
- [OK] `no_overlap_val_test`
- [OK] `aggregates_saved`
- [OK] `overall_rate_non_zero`

## Output Files
| File | Description |
|---|---|
| processed/train.parquet | Training split (70%) |
| processed/val.parquet | Validation split (15%) |
| processed/test.parquet | Test split (15%) — LOCKED |
| docs/aggregates.json | Safe population aggregates for P3 and P4 |

> Privacy: No raw rows in this report. All cells shown have n >= 30.

_Stage 4 complete. P2 can now use train.parquet. P3 can use val.parquet + aggregates.json. P4 can use aggregates.json._