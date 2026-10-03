# DECISIONS.md — ProxyPatient

All explicit decisions made during P1 (data and backend). Referenced in HANDOFF_P1_v2.md.

---

## D1. Glucose outcome threshold
**Decision:** `elevated_glucose_proxy = 1` if `glucose_raw >= 200 mg/dL`, else 0.
**Why:** WHO/ADA random glucose screening threshold. DHS docs confirm units are mg/dL.
**Confirmed:** v1 and v2 both use this definition.
**Caveats:** P3 should verify mg/dL vs mmol/L against the NFHS-5 biomarker fieldwork report.

---

## D2. Blood pressure: 3-reading protocol, average of 2nd + 3rd

**Rule implemented (v2):**
- NFHS-5 took 3 BP readings per respondent (15 min apart).
- `systolic_avg = (sb25s + sb29s) / 2` for women, `(smb25s + smb29s) / 2` for men.
- `diastolic_avg = (sb25d + sb29d) / 2` for women, `(smb25d + smb29d) / 2` for men.
- **Fallback:** if only one of the 2nd/3rd is valid → use it alone. If neither valid → try 1st reading. If no reading at all → NaN.
- **Rationale:** NFHS-5 Final Report (FR375) and DHS Biomarker Protocol specify averaging the 2nd and 3rd readings, discarding the 1st as an adaptation reading.

**Special/missing codes for BP:**
- Codes 994, 995, 996, 997, 998, 999 → NaN
- Code 0 → NaN (not measured)
- Valid range: 1–993 mmHg (unrealistically high values should be flagged by P3)

**Hypertension definition (JNC-7 / WHO-ISH, used in NFHS-5 report):**
```
hypertension = 1 if:
    systolic_avg >= 140 OR
    diastolic_avg >= 90 OR
    on_bp_medication == 1
```
If all three are NaN → hypertension = NaN.
If `on_bp_medication == 1` and BP is missing → hypertension = 1 (medication use implies history).

**Coverage (v2):** ~96.5% women, ~94.3% men have ≥1 valid BP reading.

**`on_bp_medication`** (sb21 / smb21): 1 = yes, 0 = no, NaN = not asked/missing.

**`bp_measured`**: 1 if at least one of 2nd or 3rd BP reading is valid, else 0.

**`self_reported_hypertension`** (s728b / sm627b): REFERENCE COLUMN ONLY. Not a conditioning variable. Value distribution checked: both 0 and 1 present → real variation, not all-zero coding. Values 8 (don't know) treated as NaN.

---

## D3. Men's BMI: household member file linkage

**Problem:** NFHS-5 men's IR file has no height/weight/BMI. These are in the Household Member (PR) file.

**Source:** `IAPR7EFL.DTA` (household member file, ~2.8M rows, extracted from IAPR7EDT.zip).

**Variables:**
- `hb2` = man's weight (kg, **1 implied decimal**, divide by 10)
- `hb3` = man's height (cm, **1 implied decimal**, divide by 10)
- `hb40` = man's BMI (**2 implied decimals**, divide by 100)

**hb40 divisor confirmed:** raw mean ~2247 → /100 → ~22.5 kg/m² (plausible). Verified against hb2/hb3.

**Linkage keys:**
- Men's IR: `mv001` (cluster) + `mv002` (household) + `mv003` (line number)
- Household PR: `hv001` + `hv002` + `hvidx`
- JOIN TYPE: LEFT join from men's file. Unmatched men → missing BMI.
- Key uniqueness in HH file: 0 duplicates (confirmed).
- Match rate (v2): **94.2%** (95,908 / 101,839 men matched).

**Men's BMI range after linking:** min=12.00, mean=22.47, max=59.93 — plausible, not DAE artifact.

**Women's height:** `ha3` from household file, same linkage via v001/v002/v003 → hv001/hv002/hvidx. Where link fails, `height_cm` = NaN.

**Previous v1 bug:** DAE imputed men's BMI from a near-constant median (range 21.02-22.55). This is corrected in v2 by linking to household file.

---

## D4. bmi_measured flag

**Column name:** `bmi_measured` (int, 0 or 1).
**Set to 1 if:** BMI was actually measured and recorded (from v445 for women, from hb40 for men).
**Set to 0 if:** BMI is missing (structural, not sporadic).
**Computed BEFORE** any imputation.

**Purpose:** Allows models to distinguish "BMI unknown because not measured" from "BMI unknown because of data error". Prevents DAE from fabricating BMI for unmeasured people.

**bmi_band:** Computed only where `bmi_measured == 1`. NaN where `bmi_measured == 0`.

**Women's missingness:** ~3.4% of women lack measured BMI. Likely pregnant women excluded from NFHS-5 anthropometric measurement protocol.
**Men's missingness:** ~5.8% of men not matched in household file.

---

## D5. DAE: structural vs. sporadic missing

**DAE is allowed to impute:** `waist_cm`, `hip_cm`, `any_tobacco`, `alcohol`, `bp_ever_checked`.
**DAE is NOT allowed to impute:** `bmi`, `weight_kg`, `height_cm`, `systolic_avg`, `diastolic_avg`, `hypertension`, `on_bp_medication`, `bmi_measured`, `bp_measured`, `glucose_raw`, `elevated_glucose_proxy`.

**Rationale:** Structural missingness (BMI not measured for some respondents, BP not measured for ~4-6%) should not be imputed because the missingness itself is informative and imputing would fabricate data. Sporadic missingness (waist/hip not recorded for a small random subset) is safe to impute.

---

## D6. Survey weights

**Decision:** Survey weights stored as `survey_weight` (v005/1,000,000 for women, mv005/1,000,000 for men) but NOT applied in training. P2 may optionally use them as sample weights in CVAE training, but this is their decision to make.

---

## D7. Split stability

**Strategy:** v2 splits use EXACT same `_row_id` sets as v1, loaded from v1 parquets.
**Verification:** SHA-256 hashes of sorted `_row_id` arrays match between v1 and v2 for all three splits.
**Rationale:** P2 and P3 have already started working with v1 split indices. Changing the split would invalidate their work.

---

## D8. wealth_score and age_group_5yr dropped in v2

- `wealth_score` (v191): raw DHS integer in [-2.6M, +2.6M]. Redundant with `wealth_quintile`. Dropped.
- `age_group_5yr`: 72K+ NaN values in v1 due to processing bug. Replaced by `age_band` (derived from age). Dropped.

---

## D9. NMB-2017 external validation — SKIPPED

Mendeley DOI 10.17632/twp8xw6p25.1 contains only a PDF, no CSV/data file. External validation on NMB-2017 is permanently NOT DONE. P3 should plan validation on the held-out `test_v2.parquet` only.

---

## D10. State encoding

State uses the raw DHS `v024`/`mv024` integer codes (1-36, one per state/UT). No label encoding was applied. The integer codes are stable across NFHS-5 files and match the DHS state numbering. P2 should treat state as a categorical conditioning variable with cardinality 36.

---

*Last updated: v2 pipeline. See HANDOFF_P1_v2.md for full details.*
