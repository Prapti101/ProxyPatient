# Open questions for private reruns and publication

Updated 2026-10-05 (Asia/Kolkata). No answers are inferred from MOCK software checks. The following questions are carried forward verbatim from review section 9:

1. Were the private v2 files produced from this exact preprocessing commit? If women’s height is present locally, what unpublished patch or different script produced it?
2. What are the actual NFHS state raw-code/value-label mappings, and is state 37 present after scope? Which codes have adequate post-encoding support?
3. Which codebook passages confirm glucose units/special missing codes, raw tobacco/medication mappings, and the intended partial-BP/fallback rule?
4. Is the intended reference population the unweighted observed sample or a survey-representative population? How should sex-specific sampling designs and scope restrictions be represented?
5. What counts as sufficient joint scenario support, and what response should unsupported or heavily clipped scenarios return?
6. Should reported intervals describe only Monte Carlo variation or also fitted-model/data uncertainty? What validation tolerance is required for conditional elevated-glucose rates?
7. Which test data/statistics have already been examined, by whom, and what model choices followed? Can final results honestly be described as untouched-test evaluation?
8. Does the actual research agreement authorize each team recipient and the proposed private cloud/model-serving environment? What output/model redistribution is permitted?
9. Are historical binaries confirmed aggregate/parameter-only, and does the team intend to release its source under a stated software licence?
10. When P3/P4 merge, what is the agreed versioned HTTP/aggregate/validation contract and what happens when real weights or verified validation are absent?


Additional implementation decisions needing private evidence:

- Can the post-encoding TRAIN scope supply at least three full joint profile cells with 500 rows each, and what support should be required for each what-if change? The software returns unavailable profiles when that support is absent.
- Which optional generated variables have adequate per-sex measurement coverage after preprocessing/encoding? Height/weight cannot be assumed available because a mock supplies them.
- Which state codes survive complete encoding with privacy-safe support? Unseen/unsupported validation states fail instead of being reassigned.
- Does a DAE prediction failure reflect invalid private artifacts, out-of-domain codes or the fitted model? Benchmark comparison cannot silently drop failed predictions.
- P1: Which DAE script produced the private imputed files: legacy v1 or train_dae_v2.py, and from which commit/configuration? Did that actual DAE run ever train on, infer on or otherwise examine test rows? The committed legacy source trains on TRAIN and imputes combined; the private execution history is unknown.
- What human review is needed after a started final run that already wrote metrics? Automatic crash recovery is allowed only before metrics and with unchanged frozen artifacts; completed always blocks. Never delete the marker for model selection.
- Which publication/disclosure review applies to fitted preprocessing, marginals, supported profiles, weights and aggregate reports? safe_outputs means structurally aggregate-only, not automatically authorized for release.
