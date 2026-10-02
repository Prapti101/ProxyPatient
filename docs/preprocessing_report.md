# ProxyPatient -- Stage 3 Preprocessing Report
**Run date:** 2026-10-02T18:10:56.419472
**Source:** NFHS-5 India (2019-21)

## Row Counts
| File | Rows | Glucose Non-Null |
|---|---|---|
| Women | 724,115 | 702,492 |
| Men   | 101,839 | 96,130 |
| **Combined** | **825,954** | **798,622** |

## Acceptance Checks
- [OK] `glucose_not_in_encoder`
- [OK] `glucose_readings_exist`
- [OK] `both_sexes_present`
- [OK] `no_raw_rows_printed`

## Output Files
| File | Description |
|---|---|
| processed/women_clean.parquet | Cleaned women data |
| processed/men_clean.parquet | Cleaned men data |
| processed/combined_clean.parquet | Combined dataset |
| processed/preprocess.pkl | Fitted scaler + encoder |

> Privacy: No raw rows stored. No individual data shown.

_Stage 3 complete. Proceed to Stage 4: Stratified Split + Aggregates._