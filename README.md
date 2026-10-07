# ProxyPatient

**Synthetic scenario exploration for elevated glucose**

ProxyPatient generates synthetic cohorts from a conditional variational autoencoder (CVAE) trained on NFHS-5 India survey data. A user chooses a reference profile, changes its conditions, and compares the generated glucose summaries. The project includes the preprocessing and training workflow, a FastAPI backend, and a React frontend.

The outcome is **generated glucose ≥ 200 mg/dL**, labelled **elevated glucose** throughout the application. Glucose is generated, never entered as a condition or filled in by the imputer. A what-if comparison describes a difference between synthetic cohorts. It does not estimate the effect of changing someone's behaviour, diagnose diabetes, or predict an individual's outcome.

## Project overview

A cohort is a group of people described by shared characteristics. ProxyPatient lets a user define those characteristics—such as age band, residence and BMI band—and generate a new synthetic cohort. The model learns patterns from NFHS-5 data, then samples new measurements for the chosen conditions.

For example, a user can select an urban reference profile, generate its cohort, change the BMI band and compare the resulting elevated-glucose percentages. The screen shows both summaries, their sampling intervals and the difference in percentage points. A session notebook keeps the generated scenarios available for comparison.

### How the system works

1. **Prepare the data:** harmonise women's and men's survey variables, establish the data splits, and handle selected missing non-outcome measurements with a DAE.
2. **Train the generator:** fit a CVAE that takes the eight conditions and optional state, then generates age, BMI, waist, hip, glucose and categorical attributes.
3. **Evaluate the models:** compare CVAE, TVAE and CTGAN against matched real validation rows using distribution, subgroup, utility and generation-consistency metrics.
4. **Serve scenarios:** FastAPI validates requests, samples the decoder, calculates the elevated-glucose summary and returns its provenance.
5. **Explore in the browser:** React presents the profile builder, generated results, comparisons, notebook and evaluation reports.

| Component | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend | FastAPI, Pydantic, Uvicorn |
| Main model | PyTorch conditional variational autoencoder |
| Baseline models | TVAE and CTGAN |
| Data processing | pandas, NumPy, PyArrow |
| Evaluation | SciPy and scikit-learn |
| Source data | NFHS-5 India, 2019–21 |

## Contents

- [Evaluation results](#evaluation-results)
- [Inputs and outputs](#what-the-user-can-set-and-see)
- [Data scope and preparation](#data-scope-and-preparation)
- [CVAE architecture and training](#cvae-architecture-and-training)
- [Detailed model evaluation](#detailed-model-evaluation)
- [Generation coverage and diagnostics](#generation-coverage-and-sampling-diagnostics)
- [DAE benchmark](#dae-imputation-benchmark)
- [Runtime and memory](#runtime-and-memory)
- [Frontend](#frontend)
- [Local setup](#run-locally)
- [Model workflow](#model-workflow)
- [Source reports](#result-files-and-provenance)
- [Repository map](#repository-map)

## Evaluation results

The project evaluation was recorded on **5 October 2026**. The MLP CVAE trained for six epochs on 100,000 TRAIN rows. CVAE, TVAE and CTGAN were evaluated on a shared validation cohort of 6,596 rows. The tables below give the recorded model scores, training history, imputation results and runtime measurements.

The CVAE achieved the lowest mean continuous KS distance (**0.03581**), lowest average conditional rate error (**0.6307 percentage points**) and highest synthetic-to-real outcome-prediction ROC AUC (**0.66501**) among the three evaluated models.

### Evaluation setup

| Item | Recorded value |
| --- | --- |
| Source | NFHS-5 India, 2019–21 (DHS) |
| Evaluation split | VAL |
| Random seed | 42 |
| Requested evaluation rows | 10,000 |
| Rows filled by every model and retained for scoring | 6,596 (65.96%) |
| Rows excluded from the shared comparison | 3,404 (34.04%) |
| Outcome threshold | Glucose ≥ 200 mg/dL |
| Minimum real support for conditional-error and direction metrics | 500 rows per evaluated subgroup |
| Aggregate disclosure threshold | Counts below 30 and their associated statistics are suppressed |
| Survey weighting | Not used in model training; reported rates are unweighted |

Each model generates a row for the same selected VAL conditions. TVAE and CTGAN use rejection sampling to find matching conditions. A row that either baseline cannot fill is removed from the scoring cohort for **all three models**. These scores therefore describe the retained subset, not all VAL respondents or all 10,000 requested conditions.

The CVAE used 100,000 TRAIN rows with state conditioning. TVAE and CTGAN each used a 20,000-row TRAIN sample stratified by sex, age band, BMI band and hypertension, without state conditioning.

### Main comparison

Smaller distribution errors are better. Higher outcome-prediction AUC is better. Source-classifier AUC has a different meaning: values near 0.5 mean this particular classifier had difficulty distinguishing real from synthetic rows.

| Metric | CVAE | TVAE | CTGAN |
| --- | --- | --- | --- |
| Mean continuous KS | 0.03581 | 0.12996 | 0.21686 |
| Correlation difference, Frobenius norm | 1.56256 | 2.34711 | 2.97074 |
| Largest absolute correlation difference | 0.46037 | 0.52168 | 0.94133 |
| Elevated glucose | 1.2432% | 0.2426% | 2.2135% |
| Conditional rate mean absolute error (pp) | 0.6307 | 1.3643 | 1.1106 |
| Conditional rate maximum absolute error (pp) | 4.2065 | 3.7275 | 2.8681 |
| Direction matches | 3/5 | 4/5 | 3/5 |
| Generated age within selected band | 100.0000% | 81.1856% | 30.4730% |
| Generated BMI within selected band | 100.0000% | 76.9406% | 39.4633% |
| TSTR outcome-prediction ROC AUC | 0.66501 | 0.63508 | 0.48567 |
| TRTR outcome-prediction ROC AUC | 0.75646 | 0.75646 | 0.75646 |
| Real-versus-synthetic source ROC AUC | 0.49851 | 0.74818 | 0.80041 |

“pp” means percentage points. The real retained cohort's elevated-glucose rate is **1.4554%**, compared with **1.2432%** for CVAE, **0.2426%** for TVAE and **2.2135%** for CTGAN. The signed differences from real are −0.2122, −1.2128 and +0.7581 pp, respectively. These are cohort rates, not classification accuracy scores or national prevalence estimates.

### Accuracy, precision, recall and F1

ProxyPatient's main model generates measurements rather than assigning a binary class to an input person. Its evaluation uses distribution fidelity, subgroup outcome rates and classifier utility. The supplied evaluation reports save ROC AUC for the auxiliary classifiers; they do not save thresholded predictions or confusion matrices.

| Classifier metric | CVAE | TVAE | CTGAN |
| --- | --- | --- | --- |
| Synthetic-to-real ROC AUC | 0.66501 | 0.63508 | 0.48567 |
| Real-to-real ROC AUC | 0.75646 | 0.75646 | 0.75646 |
| Real-versus-synthetic source ROC AUC | 0.49851 | 0.74818 | 0.80041 |

Accuracy and F1 cannot be recovered from AUC or aggregate glucose rates. They require labelled classifier predictions at a specified decision threshold. The 100% age-band and BMI-band consistency values below measure constraint satisfaction during generation; they are not classification accuracy.

## What the user can set and see

A baseline requires all eight conditions. State is optional and must be a supported code returned by `/options`.

| Condition | Accepted values / interpretation |
| --- | --- |
| Sex | 0 / female; 1 / male |
| Age band | 15–24, 25–34, 35–49 for women; 15–24, 25–34, 35–54 for men |
| Residence | urban / rural |
| Wealth quintile | 1–5, poorest to richest |
| BMI band | underweight, normal, overweight, obese |
| Hypertension | 0 / 1; measured BP status or BP medication under the configured definition |
| Tobacco | 0 / 1, any tobacco use |
| Alcohol | 0 / 1 |
| State (optional) | Supported raw integer codes from the fitted model; 36 states in this run |

The model internally combines the oldest sex-specific age bands into `35+`; the API retains the appropriate public label. Unknown fields, glucose-related condition keys and unsupported values are rejected.

Generation returns a synthetic cohort count, elevated-glucose percentage, a **95% Wilson Monte Carlo interval**, conditions, sampling diagnostics, and run provenance. Comparison also returns the difference from the accepted baseline. The interval describes sampling variation conditional on a fixed fitted model; it does not measure uncertainty in the model, survey design, preprocessing or treatment effects.

The API can return one to five labelled representative synthetic examples. Real-mode examples are withheld when the matching nearest-record heuristic is unavailable or fails. That heuristic does not establish anonymity or differential privacy. The frontend shows the returned screening note even when examples are withheld.

## Data scope and preparation

The private workflow prepares women's and men's files, links household measurements, creates TRAIN/VAL/TEST splits, and applies a denoising autoencoder (DAE) to selected non-outcome missing values. The current model fits its own TRAIN-only encoder and normalisation. Historical all-data `preprocess.pkl` / `preprocess_v2.pkl` files are not the current CVAE preprocessor.

Eligibility requires known glucose, measured BMI and known hypertension status, followed by valid complete encoded conditions and generated variables. Exclusion counts below are sequential: each rule is applied after the previous one.

| Scope step | TRAIN | VAL |
| --- | --- | --- |
| Input rows | 581,542 | 124,618 |
| Excluded: glucose unknown | 22,508 | 4,824 |
| Excluded next: BMI not measured | 6,422 | 1,413 |
| Excluded next: hypertension unknown | 1,689 | 330 |
| Rows in scope before model-training subsampling | 550,923 | 118,051 |
| In-scope share | 94.7349% | 94.7303% |
| Rows used by CVAE | 100,000 | 118,051 |
| Women / men in CVAE TRAIN sample | 86,723 / 13,277 | Not separately reported in training log |

The 118,051 VAL rows in the training log are used for model validation. The model-comparison metrics use the separate 6,596-row shared retained subset. They should not be confused.

TRAIN contained 509,493 women and 72,049 men before scope filtering; VAL contained 109,145 women and 15,473 men. The sex imbalance remains in the fitted sample and becomes stronger in the retained comparison: 6,010 women and 586 men. Male subgroup conclusions have much less support.

### Scope exclusions by sex and age

| Sex / harmonised age band | TRAIN excluded | VAL excluded |
| --- | --- | --- |
| Women, 15-24 | 5.2590% | 5.2300% |
| Women, 25-34 | 4.1537% | 4.2277% |
| Women, 35+ | 5.1089% | 5.1972% |
| Men, 15-24 | 7.8911% | 8.1081% |
| Men, 25-34 | 8.1199% | 7.5266% |
| Men, 35+ | 8.1356% | 7.7220% |

### Support for complete profiles

Step 0 records 1,920 possible full eight-condition combinations, excluding optional state. Of these, 1,696 were observed and 224 were never observed. Among observed combinations, 957 had fewer than 30 TRAIN rows, 577 had 30–499, and 162 had at least 500. The serving preset builder uses the ≥500 support rule. Broad marginal support does not guarantee support for an arbitrary joint profile.

### Missingness and coding checks

Selected raw missing counts are shown below. `Not reported` is the exported `null`; the package suppresses small counts, including zero, so it must not be read as a zero or as proof of a failed check.

| Variable | TRAIN raw missing | TRAIN after DAE | VAL raw missing | VAL after DAE |
| --- | --- | --- | --- | --- |
| waist_cm | 17552 | Not reported | 3744 | Not reported |
| hip_cm | 17549 | Not reported | 3744 | Not reported |
| bp_ever_checked | 24412 | Not reported | 5218 | Not reported |
| glucose_raw | 22508 | 22508 | 4824 | 4824 |
| bmi | 24147 | 24147 | 5214 | 5214 |
| bmi_band | 24147 | 24147 | 5214 | 5214 |
| height_cm | 514182 | 514182 | 110131 | 110131 |
| weight_kg | 23377 | 23377 | 5039 | 5039 |
| hypertension | 24397 | 24397 | 5215 | 5215 |
| on_bp_medication | 24477 | 24477 | 5232 | 5232 |

The fitted generator produces age, BMI, waist, hip, glucose, education and BP-ever-checked. Height and weight are not outputs of this fitted model. The encoder determines the output set from available per-sex measurement support.

The input manifest records 11 available files as `OK`. Held-out test files remain separate from the reported VAL evaluation. The linked step-0 reports contain the full missingness, quantile and consistency checks.

## CVAE architecture and training

The main generator is an MLP CVAE. Its encoder learns a latent distribution from the generated variables and selected conditions during training. At inference, the decoder samples from the latent prior and generates a new row for the requested conditions. It does not retrieve a respondent to supply the outcome.

| Setting | Recorded in the supplied model card / training log |
| --- | --- |
| Variant | MLP |
| Latent dimension | 32 |
| Hidden dimensions | 128, 64 |
| Condition cardinalities | 2, 3, 2, 5, 4, 2, 2, 2 |
| State embedding | 36 states; dimension 8 |
| Continuous outputs | Age, BMI, waist_cm, hip_cm, log_glucose |
| Categorical outputs | Education (4 categories), BP ever checked (2 categories) |
| Glucose head | 3-component mixture on log glucose |
| Device | CPU |
| Seed | 42 |
| Epochs completed | 6 |
| Minimum encoded rows per sex | 2,000 |
| Best validation ELBO | −4.40243 |
| Training-loop time | 10.1 seconds |
| Whole CVAE stage | 31.125 seconds |
| CVAE stage peak RSS | 1,605.06 MiB |

The repository configuration uses learning rate 0.001, batch size 1,024 and ten-epoch KL annealing. The logged training schedule increases β from 0.1 to 0.6 across the six completed epochs. β sets the weight of the latent-distribution regularisation term in the training objective.

### All six training epochs

Reconstruction is a negative log-likelihood term and can be negative for a continuous density. ELBO is a likelihood-based objective, not a percentage accuracy. The elapsed-seconds column is cumulative from the start of the training loop.

| Epoch | β | TRAIN loss | TRAIN reconstruction | TRAIN KL | VAL reconstruction | VAL KL | VAL ELBO | Elapsed seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 5.58049 | 5.20797 | 3.72524 | 2.75065 | 5.76203 | -8.51268 | 2.0 |
| 2 | 0.2 | 2.74930 | 1.60482 | 5.72239 | 0.12965 | 8.13147 | -8.26112 | 3.6 |
| 3 | 0.3 | 1.95210 | -0.17723 | 7.09777 | -1.00143 | 8.04616 | -7.04473 | 5.2 |
| 4 | 0.4 | 1.66680 | -1.22128 | 7.22021 | -1.95623 | 7.74293 | -5.78671 | 6.8 |
| 5 | 0.5 | 1.72098 | -1.77083 | 6.98364 | -2.07472 | 6.91227 | -4.83755 | 8.5 |
| 6 | 0.6 | 1.97753 | -1.96007 | 6.56266 | -2.20832 | 6.61075 | -4.40243 | 10.1 |

TVAE and CTGAN each trained for 15 epochs on 20,000 TRAIN rows stratified by sex, age band, BMI band and hypertension. Their table has the eight conditions plus age, BMI, waist, hip, log glucose, education and BP-ever-checked. State is excluded. The baseline log records `gpu=false`; no GPU benchmark is supplied. Training took 25.1 seconds for TVAE and 41.7 seconds for CTGAN. It does not contain per-epoch baseline losses, so none are claimed here.

GRU/CNN CVAE variants are optional ablations in the code. Neither was evaluated in this package.

## Detailed model evaluation

### Continuous distributions

KS compares cumulative distributions; Wasserstein measures the amount of displacement between distributions. Both are better when smaller. Wasserstein retains the variable's units: years, BMI units, cm or mg/dL. Dividing it by the real standard deviation makes the scale easier to compare across variables. The mean KS above averages the five continuous columns listed here.

| Variable | Model | KS | Wasserstein | Wasserstein / real SD |
| --- | --- | --- | --- | --- |
| age | CVAE | 0.03153 | 0.56049 | 0.05652 |
| age | TVAE | 0.08899 | 1.39706 | 0.14089 |
| age | CTGAN | 0.05518 | 1.31079 | 0.13219 |
| bmi | CVAE | 0.04624 | 0.30436 | 0.07804 |
| bmi | TVAE | 0.05928 | 0.31248 | 0.08013 |
| bmi | CTGAN | 0.36446 | 3.61420 | 0.92674 |
| waist_cm | CVAE | 0.03123 | 1.56710 | 0.04575 |
| waist_cm | TVAE | 0.02608 | 1.68125 | 0.04909 |
| waist_cm | CTGAN | 0.15161 | 4.39931 | 0.12845 |
| hip_cm | CVAE | 0.02395 | 1.44286 | 0.04308 |
| hip_cm | TVAE | 0.10249 | 3.54833 | 0.10595 |
| hip_cm | CTGAN | 0.20406 | 5.69410 | 0.17002 |
| glucose_raw | CVAE | 0.04609 | 3.59961 | 0.05060 |
| glucose_raw | TVAE | 0.37295 | 17.01516 | 0.23920 |
| glucose_raw | CTGAN | 0.30898 | 17.44027 | 0.24518 |

### Categorical distributions

Total variation distance (TVD) measures disagreement between category proportions; zero means identical proportions.

| Variable | CVAE | TVAE | CTGAN |
| --- | --- | --- | --- |
| education | 0.01971 | 0.35461 | 0.02047 |
| bp_ever_checked | 0.00106 | 0.11901 | 0.15540 |

### Glucose quantiles and elevated-glucose share

All quantiles are in mg/dL. These reference values belong to the retained comparison cohort; the step-0 statistics describe a broader dataset.

| Cohort | P50 | P90 | P95 | P99 | Glucose ≥200 |
| --- | --- | --- | --- | --- | --- |
| Real | 106.00 | 133.00 | 145.25 | 272.40 | 1.4554% |
| CVAE | 107.00 | 135.00 | 145.00 | 254.15 | 1.2432% |
| TVAE | 97.00 | 109.00 | 113.00 | 124.00 | 0.2426% |
| CTGAN | 120.00 | 155.00 | 175.00 | 242.20 | 2.2135% |

The CVAE matches the middle of the glucose distribution reasonably closely but its P99 is lower than real. TVAE produces too little upper-tail glucose in this run. CTGAN shifts the median and P95 upward while still having a lower P99.

### Conditional outcome-rate errors

These summaries score overlapping one-variable and two-variable subgroups with at least 500 real rows in the retained cohort. They are not 105 independent full-profile scenarios. Errors are unweighted subgroup averages in percentage points.

| Model | Slice | Scored cells | Mean absolute error (pp) | Maximum (pp) | One-variable mean (pp) | Two-variable mean (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| CVAE | Overall | 105 | 0.6307 | 4.2065 | 0.5779 | 0.6416 |
| CVAE | Women | 81 | 0.6178 | 3.3871 | 0.5998 | 0.6222 |
| CVAE | Men | 6 | 0.3115 | 0.5455 | 0.3642 | 0.2589 |
| TVAE | Overall | 105 | 1.3643 | 3.7275 | 1.3909 | 1.3587 |
| TVAE | Women | 81 | 1.3803 | 3.6649 | 1.4557 | 1.3617 |
| TVAE | Men | 6 | 1.3196 | 1.7208 | 1.2140 | 1.4251 |
| CTGAN | Overall | 105 | 1.1106 | 2.8681 | 0.9682 | 1.1400 |
| CTGAN | Women | 81 | 1.2102 | 3.0593 | 1.1189 | 1.2327 |
| CTGAN | Men | 6 | 0.5012 | 0.9728 | 0.6116 | 0.3908 |

The men's summary scores only six eligible cells, compared with 81 for women. Its smaller error does not demonstrate that the model performs better for men generally.

### Marginal subgroup rates

These rates come from the subgroup-fidelity table, which uses the ≥30 disclosure threshold. That is different from the ≥500 threshold used for conditional error and direction checks. A visible rate from 49 or 145 respondents is not a stable estimate. Each model has the same subgroup row counts in this matched retained cohort.

| Variable | Level | Rows | Real ≥200 | CVAE | TVAE | CTGAN |
| --- | --- | --- | --- | --- | --- | --- |
| sex | Women (0) | 6010 | 1.4309% | 1.2479% | 0.1997% | 2.2130% |
| sex | Men (1) | 586 | 1.7065% | 1.1945% | 0.6826% | 2.2184% |
| age_band | 15-24 | 2354 | 0.4248% | 0.5523% | 0.0000% | 1.9541% |
| age_band | 25-34 | 2046 | 1.2708% | 1.3685% | 0.0000% | 2.8837% |
| age_band | 35+ | 2196 | 2.7322% | 1.8670% | 0.7286% | 1.8670% |
| residence | urban | 618 | 3.3981% | 2.4272% | 0.1618% | 3.2362% |
| residence | rural | 5978 | 1.2546% | 1.1208% | 0.2509% | 2.1077% |
| wealth_quintile | 1 | 1482 | 0.7422% | 1.6194% | 0.0000% | 2.2267% |
| wealth_quintile | 2 | 1615 | 1.1765% | 0.4954% | 0.0619% | 1.7337% |
| wealth_quintile | 3 | 1355 | 0.9594% | 0.8856% | 0.0000% | 2.3616% |
| wealth_quintile | 4 | 997 | 2.3069% | 1.5045% | 0.5015% | 2.8084% |
| wealth_quintile | 5 | 1147 | 2.6155% | 2.0052% | 0.8718% | 2.1796% |
| bmi_band | underweight | 1302 | 0.8449% | 1.2289% | 0.0000% | 2.1505% |
| bmi_band | normal | 4200 | 0.7381% | 1.0238% | 0.0000% | 2.1667% |
| bmi_band | overweight | 875 | 4.3429% | 1.0286% | 1.1429% | 2.8571% |
| bmi_band | obese | 219 | 7.3059% | 6.3927% | 2.7397% | 0.9132% |
| hypertension | 0 | 6451 | 1.3331% | 1.2091% | 0.0000% | 2.2167% |
| hypertension | 1 | 145 | 6.8966% | 2.7586% | 11.0345% | 2.0690% |
| tobacco | 0 | 6247 | 1.4087% | 1.2326% | 0.2401% | 2.2411% |
| tobacco | 1 | 349 | 2.2923% | 1.4327% | 0.2865% | 1.7192% |
| alcohol | 0 | 6547 | 1.4358% | 1.2525% | 0.2444% | 2.2300% |
| alcohol | 1 | 49 | 4.0816% | 0.0000% | 0.0000% | 0.0000% |

For example, the overweight subgroup's real rate is 4.3429%, while the CVAE produces 1.0286%. Among respondents with hypertension the corresponding rates are 6.8966% and 2.7586%, although that subgroup has only 145 retained rows. The subgroup table makes these differences visible alongside the overall score.

### Direction checks

A match means that the sign of the rate difference between the last and first supported levels agrees with real. It does not mean every intermediate level is correctly ordered or that the size of the difference is accurate. For ordinal variables the report also records Spearman correlation between the real and generated level rates.

| Variable | CVAE | TVAE | CTGAN |
| --- | --- | --- | --- |
| age_band | Match | Match | Mismatch |
| bmi_band | Mismatch | Match | Match |
| wealth_quintile | Match | Match | Mismatch |
| residence | Match | Mismatch | Match |
| sex | Mismatch | Match | Match |
| tobacco | Not scored (<500 support) | Not scored (<500 support) | Not scored (<500 support) |
| alcohol | Not scored (<500 support) | Not scored (<500 support) | Not scored (<500 support) |
| hypertension | Not scored (<500 support) | Not scored (<500 support) | Not scored (<500 support) |

| Ordinal variable | CVAE Spearman | TVAE Spearman | CTGAN Spearman |
| --- | --- | --- | --- |
| age_band | 1.0000 | 0.8660 | -0.5000 |
| bmi_band | 0.5000 | 0.8660 | 0.5000 |
| wealth_quintile | 0.3000 | 0.9747 | -0.1000 |

The obese subgroup is omitted from this run's BMI direction check because its 219 rows fall below 500. The CVAE BMI mismatch concerns the supported underweight, normal and overweight levels. Tobacco, alcohol and hypertension directions are not scored because their positive groups lack 500 retained rows.

### Utility and source classification

TSTR means training an outcome classifier on synthetic rows and evaluating it on real rows; TRTR uses real TRAIN rows instead. This implementation uses logistic regression with standardised inputs. Features are the eight conditions plus available generated age, BMI, waist, hip, education and BP-ever-checked. Glucose itself is excluded as a predictor; height is absent in this run.

The synthetic training rows use held-out VAL conditions. The report therefore describes this as **transductive utility**, not independent-cohort TSTR. TRTR uses an equally sized real TRAIN sample. The shared TRTR AUC is 0.75646; TSTR AUCs are 0.66501 (CVAE), 0.63508 (TVAE), and 0.48567 (CTGAN). These scores were computed on VAL.

The separate real-versus-synthetic classifier uses 6,596 real and 6,596 synthetic rows per model, with 9,234 rows for classifier training and 3,958 held out. Its ROC AUCs are 0.49851, 0.74818 and 0.80041 for CVAE, TVAE and CTGAN. The CVAE's near-0.5 value means weak distinguishability for this classifier and cohort only. It is not a privacy guarantee or proof that every distribution has been reproduced.

### Nearest-record checks

Distances are measured in standardised numeric space against 50,000 in-scope TRAIN rows. Each query set has 2,000 rows. The real VAL-to-TRAIN reference is the same for all three models.

| Query set | Minimum distance | P5 distance | Median distance | Exact-copy share | Queries |
| --- | --- | --- | --- | --- | --- |
| Real VAL → TRAIN | 0.03932 | 0.13252 | 0.29458 | 0.0000% | 2000 |
| CVAE synthetic → TRAIN | 0.05694 | 0.14210 | 0.31739 | 0.0000% | 2000 |
| TVAE synthetic → TRAIN | 0.04219 | 0.13604 | 0.29334 | 0.0000% | 2000 |
| CTGAN synthetic → TRAIN | 0.10944 | 0.42174 | 1.36370 | 0.0000% | 2000 |

No exact copies were detected by the reported numerical threshold in these query samples. That does not exclude near copies, identify every disclosure risk, or establish anonymity. The CVAE generated-to-TRAIN median / real-to-TRAIN median ratio is approximately 1.07743. The API's example-screening rule checks that ratio and the generated minimum distance against configured thresholds; it remains a heuristic.

## Generation coverage and sampling diagnostics

Coverage is measured before removing rows for the common scoring subset. A model can fill a condition combination but still generate numeric age or BMI outside the requested band; the consistency scores measure that separately.

| Model | Requested | Filled | Fill rate | Candidate draws | Accepted / drawn | Generation seconds |
| --- | --- | --- | --- | --- | --- | --- |
| CVAE | 10,000 | 10,000 | 100.00% | Latent decoding + consistency redraws | Not a baseline rejection sampler | 0.07 |
| TVAE | 10000 | 6598 | 65.98% | 300000 | 2.1993% | 4.07 |
| CTGAN | 10000 | 9968 | 99.68% | 300000 | 3.3227% | 6.37 |

TVAE coverage is the main restriction on this comparison: it filled 6,598 rows; intersecting both baselines' filled rows leaves 6,596. The CVAE's age/BMI consistency reflects explicit redraw and constraint handling, not an unconstrained one-pass result.

| CVAE diagnostic, all 10,000 requested rows | Reported value |
| --- | --- |
| Any clipping | 0.3700% |
| Glucose clipping | 0.3500% |
| First-pass age/BMI inconsistency | 25.6500% |
| Redrawn rows (logged count) | 3763 |
| Consistent without final consistency clipping | 100.0000% |
| Clipped after maximum consistency rounds | Not reported / suppressed (`null`) |
| Sampler time | 0.0659 seconds |

The repository config clips generated glucose to 20–600 mg/dL. The table reports how often clipping and consistency redraws were used during generation.

### Baseline training probes

The baseline log also records a separate 2,000-row rejection-sampling probe. These are not the later 10,000-row evaluation coverage numbers.

| Model | Requested | Filled | Fill rate | Draws | Acceptance rate | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| TVAE | 2000 | 1260 | 63.00% | 40000 | 3.1500% | 0.62 |
| CTGAN | 2000 | 1953 | 97.65% | 40000 | 4.8825% | 1.13 |

## DAE imputation benchmark

The benchmark masks 10% of finite known VAL values and compares DAE predictions with the TRAIN median on the same masked targets. It evaluates waist and hip, not glucose. Each variable has 120,874 known VAL values and 12,087 masked targets. Glucose remains un-imputed throughout the workflow.

| Variable | TRAIN median (cm) | VAL known SD (cm) | DAE RMSE (cm) | Median-fill RMSE (cm) | Reported RMSE improvement |
| --- | --- | --- | --- | --- | --- |
| waist_cm | 77.2 | 100.8551 | 15.5759 | 102.1253 | 84.75% |
| hip_cm | 89.4 | 99.478 | 13.9399 | 103.3214 | 86.51% |

The report records `dae_beats_median=true` for both variables. `n_dae_returned_nan` is suppressed as `null`, not an explicit public zero; the benchmark implementation refuses non-finite predictions before scoring.

The 84.75% and 86.51% improvements measure the reduction in RMSE relative to TRAIN-median filling on the same masked targets. The benchmark scores known values that were temporarily masked; glucose is excluded from imputation. The complete benchmark inputs and scores are recorded in the linked aggregate reports.

## Runtime and memory

These measurements come from `run_metrics.json` and are duplicated in `timings.json`. Peak RSS is a per-stage process measurement; peaks should not be added. CVAE and baseline logs separately record training-loop times, so those should not replace whole-stage runtimes.

| Stage | Wall time (seconds) | Peak RSS (MiB) |
| --- | --- | --- |
| step0 | 105.786 | 1039.34 |
| dae_benchmark | 4.891 | 1046.71 |
| train_cvae | 31.125 | 1605.06 |
| train_baselines | 78.105 | 1266.56 |
| eval_dev | 31.957 | 2046.48 |
| package | 1.851 | 580.02 |

The six recorded stages total **253.715 seconds** (about 4.23 minutes); the largest recorded peak is **2,046.48 MiB**, during evaluation. No stages were skipped. These timings do not include environment installation or earlier preprocessing/DAE training, and the package does not identify a CPU model or promise the same runtime elsewhere.

## Frontend

The React + TypeScript application lives in `frontend/`. It uses a consistent light notebook theme: warm paper surfaces, dark green text, serif headings, and green/rust comparison series. The browser's dark-mode preference does not change individual controls to a second theme.

| Route | Screen |
| --- | --- |
| `/` | Project overview and run context |
| `/explore` | Full-profile builder and baseline generation |
| `/compare` | Reference / what-if cards, intervals and comparison chart |
| `/scenarios` | Chronological session notebook |
| `/validation` | Independently loaded validation and model-comparison reports |
| `/how-it-works` | Method and interpretation |

A successful generation accepts the reference conditions and result together. A failed request retains the last successful reference, and accepting a new reference clears its old comparison. Requests disable the relevant inputs while pending. The layout supports mobile widths, keyboard navigation, visible focus and reduced-motion preferences.

Set `VITE_DATA_MODE=api` to use model-backed generation and the packaged evaluation reports. The interface's default local preview uses illustrative values; it can be used to explore the screens without a backend. API mode displays request errors directly and does not replace them with preview values.

### Software verification

| Check | Result |
| --- | --- |
| TypeScript and production build | Passed |
| Chromium browser regression suite | 10 tests passed |
| Axe WCAG A/AA checks | 0 reported violations across 9 pages/states |
| Responsive layouts | Checked at 320, 390, 768, 1,024 and 1,440 pixels |
| Theme consistency | All six routes checked with light and dark browser preferences |

These checks cover the application's build, workflows and interface. See the [frontend review](frontend/FRONTEND_REVIEW.md) for reproduction instructions.

## Run locally

### Backend

Python 3.11 is the tested backend environment. Install the pinned requirements using CPU PyTorch:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

On the machine holding the trained package, point the backend to its matching model artifacts and aggregate reports:

```bash
PP_MODEL_DIR=/private/proxypatient/private_outputs \
PP_REPORT_DIR=/private/proxypatient/safe_outputs \
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend verifies checkpoint identity and packaged report checksums before serving model-backed results. Use `/health` to check readiness and `/profiles` to select a complete supported reference profile. Private artifacts stay on the approved holder's machine.

### Frontend

In a second terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Set these values in `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
VITE_DATA_MODE=api
```

Restart Vite after changing environment values. Run `npm run build` for the production build. Browser routes need an index-page fallback when deploying the production frontend.

### API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Real/demo/unavailable readiness and model identity |
| `GET /schema`, `GET /options` | Variable definitions and supported values |
| `GET /profiles` | Supported full TRAIN profiles with ≥500 encoded rows |
| `POST /generate` | Generated outcome summary, examples, interval and diagnostics |
| `POST /compare` | Baseline / what-if generation and percentage-point difference |
| `GET /validation`, `GET /model-comparison` | Compatible packaged VAL reports or explicit pending/null metrics |
| `POST /parse` | Rule-based condition proposal for the preview interface; requires user confirmation |

The active parser is `backend/parser.py`; it is not an LLM or a verified BERT pipeline. Root-level P3 parser/report copies are separate handover material, not the active serving contract. See [API contract](docs/API_CONTRACT.md).

## Model workflow

The pipeline performs data checks → DAE benchmarking → CVAE training → TVAE/CTGAN training → validation evaluation → report packaging. Its entry point is `models/run_all.py`; model and outcome settings live in `config.yaml`.

Training requires the authorised processed TRAIN/VAL files, their DAE-imputed counterparts and the matching input manifest. The DAE benchmark also uses the fitted DAE weights and fit statistics. The trained package separates two output directories:

| Directory | Contents |
| --- | --- |
| `safe_outputs/` | Aggregate model scores, training logs, model card, runtime measurements and checksums |
| `private_outputs/` | CVAE weights, fitted preprocessor, condition marginals, supported profiles and baseline artifacts |

The reported training used 100,000 CVAE rows / six epochs, 20,000 baseline rows / 15 epochs and 10,000 requested evaluation rows. All six pipeline stages completed. The detailed commands and data-access requirements are in the [private-holder runbook](models/RUN_ON_COLAB.md).

## Result files and provenance

The supplied safe reports are preserved below without changing their contents. All 12 entries in the supplied manifest were SHA-256 checked and matched. This verifies attachment integrity, not the underlying respondent data, training environment or scientific correctness. The manifest itself is not a signed attestation.

| Source file | Contents |
| --- | --- |
| [`README.md`](docs/results/quick-2026-10-05/safe_outputs/README.md) | Original package note: aggregate-only outputs and suppression policy |
| [`manifest.json`](docs/results/quick-2026-10-05/safe_outputs/manifest.json) | SHA-256 checksums for the 12 packaged files |
| [`model_card.json`](docs/results/quick-2026-10-05/safe_outputs/model_card.json) | Architecture, fitted scope, training summary, fingerprints and embedded evaluation |
| [`cvae_train_log.json`](docs/results/quick-2026-10-05/safe_outputs/cvae_train_log.json) | All six epochs, scope exclusions and encoded counts |
| [`baselines_train_log.json`](docs/results/quick-2026-10-05/safe_outputs/baselines_train_log.json) | TVAE/CTGAN training settings, times and probe coverage |
| [`model_comparison_dev.json`](docs/results/quick-2026-10-05/safe_outputs/model_comparison_dev.json) | Full per-column, per-cell, per-sex, coverage, utility and distance measurements |
| [`model_comparison_dev.md`](docs/results/quick-2026-10-05/safe_outputs/model_comparison_dev.md) | Exported short comparison table |
| [`dae_benchmark.json`](docs/results/quick-2026-10-05/safe_outputs/dae_benchmark.json) | Masked-target DAE and TRAIN-median scores |
| [`dae_benchmark.md`](docs/results/quick-2026-10-05/safe_outputs/dae_benchmark.md) | Exported DAE benchmark summary |
| [`step0.json`](docs/results/quick-2026-10-05/safe_outputs/step0.json) | Input checks, missingness, scope and aggregate rates |
| [`step0.md`](docs/results/quick-2026-10-05/safe_outputs/step0.md) | Readable step-0 report |
| [`run_metrics.json`](docs/results/quick-2026-10-05/safe_outputs/run_metrics.json) | Stage runtimes, peak RSS and skipped-stage list |
| [`timings.json`](docs/results/quick-2026-10-05/safe_outputs/timings.json) | Stage timing copy |

Model fingerprint:

```text
00109c3a2d9ed5a21d5bfeb38297b22db4b20e3d8d45a6d8751596d5f3ddeecb
```

Configuration fingerprint:

```text
eeced21543d206fda600567cf943bb746ea0664c1c2dac3d9599a8bb1258f320
```

The 734 requested-condition coverage entries, all 105 scored conditional cells per model, detailed direction rates, and step-0 aggregate tables remain available in the linked JSON files. Their suppressed `null` values are preserved. The model card's embedded evaluation matches `model_comparison_dev.json`. `run_metrics.json` and `timings.json` have the same stage measurements.

No private-output files are needed for this documentation. Reproducing inference or checking fitted artifacts requires the matching private package on its approved machine; uploading that package here is unnecessary.

## Repository map

| Path | Role |
| --- | --- |
| `preprocess_v2.py`, `split_and_aggregate_v2.py`, `train_dae_v2.py` | Private-holder data preparation and DAE workflow |
| `config.yaml` | Outcome, conditions, scope, training and sampling settings |
| `models/run_all.py` | Guarded orchestration, final-evaluation controls and packaging |
| `models/data.py`, `models/artifacts.py`, `models/privacy.py` | TRAIN-only encoding, artifact identity and suppression |
| `models/train_cvae.py`, `models/train_baselines.py` | CVAE, TVAE and CTGAN training |
| `models/eval_dev.py`, `models/check_dae_benchmark.py` | Development evaluation and masked-value benchmark |
| `backend/` | Active FastAPI contracts, decoder serving, reports, examples and parser |
| `frontend/` | React application, styles and frontend checks |
| `tests/` | Backend/model tests using constructed fixtures |
| `docs/results/quick-2026-10-05/safe_outputs/` | Recorded aggregate result package |
| `docs/` | API contract, review, runbooks, scope decisions and implementation records |
| `legacy/` | Historical v1 executables; retained for audit, not the current workflow |

For presentation and implementation context, see the [API contract](docs/API_CONTRACT.md), [viva notes](docs/VIVA_NOTES.md) and [specification decisions](docs/SPEC_VS_BUILD.md).

DHS respondent data is governed by the team's approved access agreement. Public source code does not grant permission to redistribute that data or trained artifacts. No raw rows, processed splits, model weights or private-output files are included in this documentation update.
