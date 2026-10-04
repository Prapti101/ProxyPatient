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
    rng = np.random.default_rng(seed)
    iterations = n_bootstrap if n_bootstrap is not None else int(cfg['outcome_stat']['bootstrap_iterations'])
    level = float(cfg['outcome_stat']['ci_level'])
    rates = rng.binomial(len(values), values.mean(), size=iterations) / len(values)
    low, high = np.quantile(rates, [(1-level)/2, (1+level)/2])
    return dict(rate=round(float(values.mean()), 6), ci_low=round(float(low), 6),
                ci_high=round(float(high), 6), n=len(values), rate_pct=round(float(values.mean())*100, 4))
