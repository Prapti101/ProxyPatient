"""Bounded illustrative records selected only from the freshly decoded cohort."""
import math
import numpy as np
from models.common import load_config
from models.privacy import MIN_CELL_SIZE

WITHHELD = "withheld: near-copy check missing or not passed"


def representative_examples(df, count, scenario_id):
    if not 1 <= count <= 5:
        raise ValueError("n_examples must be 1 to 5")
    order = np.argsort(df.glucose_raw.to_numpy(), kind="stable")
    quantiles = {1: [.5], 2: [.25, .75], 3: [.25, .5, .75],
                 4: [.1, .25, .5, .75], 5: [.1, .25, .5, .75, .9]}[count]
    positions = [int(order[int(round(q * (len(df)-1)))]) for q in quantiles]
    threshold = float(load_config()['outcome']['threshold_mg_dl'])
    elevated = [int(i) for i in order if df.iloc[int(i)].glucose_raw >= threshold]
    illustrative = None
    if elevated:
        # Reserve the last slot rather than exceeding the requested hard cap.
        illustrative = elevated[0]  # least elevated row; avoid selecting an extreme
        if illustrative not in positions:
            positions[-1] = illustrative
    results = []
    for j, i in enumerate(positions):
        row = df.iloc[i]
        conditions = {key: row[key].item() if hasattr(row[key], 'item') else row[key]
                      for key in ('sex', 'age_band', 'residence', 'wealth_quintile', 'bmi_band',
                                  'hypertension', 'alcohol', 'state') if key in row}
        conditions['tobacco'] = int(row.any_tobacco)
        features = {}
        for key in ('age', 'bmi', 'waist_cm', 'hip_cm', 'height_cm', 'weight_kg',
                    'education', 'bp_ever_checked', 'systolic_avg', 'diastolic_avg'):
            if key in row:
                features[key] = int(round(float(row[key]))) if key in ('age', 'education', 'bp_ever_checked') else round(float(row[key]), 1)
        results.append(dict(example_id=f"{scenario_id}-{chr(65+j)}", synthetic=True,
                            label=('MOCK — ' if df.attrs['demo'] else '') + 'SYNTHETIC example, not a real person',
                            effective_conditions=conditions, generated_features=features,
                            glucose_raw=int(round(float(row.glucose_raw))),
                            elevated_glucose_proxy=bool(row.glucose_raw >= threshold),
                            illustrative_elevated=i == illustrative))
    return results


def near_copy_passed(report, fingerprint):
    """Fail closed on missing, suppressed, malformed, stale or inadequate diagnostics."""
    try:
        if report['status'] != 'complete' or report['model_fingerprint'] != fingerprint or report['demo']:
            return False
        metric = report['models']['CVAE']['nearest_record']
        real = metric['real_val_to_train']
        generated = metric['generated_to_train']
        cfg = load_config()['privacy']['examples_near_copy']
        ratio, epsilon = cfg['min_median_ratio'], cfg['min_distance_epsilon']
        values = [real['median'], generated['median'], generated['minimum'],
                  generated['share_exact_copy'], ratio, epsilon]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            return False
        for query in (real, generated):
            n = query['n_query']
            if isinstance(n, bool) or not isinstance(n, int) or n < MIN_CELL_SIZE:
                return False
        return (ratio > 0 and epsilon > 0 and real['median'] > 0
                and generated['median'] >= ratio * real['median']
                and generated['minimum'] > epsilon and generated['share_exact_copy'] == 0)
    except (KeyError, TypeError, ValueError):
        return False


def example_fields(df, count, scenario_id):
    if not df.attrs['demo']:
        from backend.reports import packaged_report
        if not near_copy_passed(packaged_report(comparison=True), df.attrs['fingerprint']):
            return dict(examples=None, examples_status=WITHHELD)
        return dict(examples=representative_examples(df, count, scenario_id),
                    examples_status='available: near-copy heuristic passed')
    return dict(examples=representative_examples(df, count, scenario_id), examples_status='available: MOCK')
