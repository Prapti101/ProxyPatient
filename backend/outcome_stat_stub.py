"""Computed elevated glucose (proxy) statistics; historical filename retained for P3."""
import numpy as np
import pandas as pd
from models.common import load_config


def outcome_stat(df, rule, seed=42, n_bootstrap=None):
    cfg = load_config()
    threshold = float(cfg['outcome']['threshold_mg_dl'])
    if float(rule['threshold_mg_dl']) != threshold:
        raise ValueError('Outcome threshold does not match current configuration')
    if 'glucose_raw' not in df:
        raise ValueError('Generated glucose_raw is required; glucose is never fabricated or imputed')
    glucose = pd.to_numeric(df['glucose_raw'], errors='coerce').to_numpy(float)
    finite = np.isfinite(glucose)
    values = (glucose[finite] >= threshold).astype(float)
    if len(values) < 30:
        raise ValueError('Insufficient finite generated glucose readings (minimum 30)')
    from scipy.stats import norm
    level = float(cfg['outcome_stat']['ci_level'])
    if not 0 < level < 1:
        raise ValueError('ci_level must lie strictly between zero and one')
    n = len(values)
    rate = float(values.mean())
    z = float(norm.ppf((1+level)/2))
    denominator = 1 + z*z/n
    centre = (rate + z*z/(2*n))/denominator
    half = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/denominator
    low, high = max(0., centre-half), min(1., centre+half)
    from models.privacy import suppress_count
    return dict(rate=round(rate, 6), ci_low=round(low, 6), ci_high=round(high, 6), n=n,
                rate_pct=round(rate*100, 4), dropped_nonfinite=suppress_count(int((~finite).sum())),
                monte_carlo_interval={'low': round(low, 6), 'high': round(high, 6), 'level': level,
                                      'method': 'Wilson', 'note': 'Conditional on fitted model; not model or survey uncertainty'})
