"""Deterministic sex/outcome-stratified TRAIN subsampling, without row exports."""
import numpy as np
import pandas as pd
from models.privacy import suppress_count


def stratified_train_sample(frame, n, minimum, seed, threshold):
    if minimum < 30 or n < 2 * minimum:
        raise ValueError('Quick sample size must accommodate the per-sex minimum (at least 30)')
    support = {s: int((frame.sex == s).sum()) for s in (0, 1)}
    if any(value < minimum for value in support.values()):
        counts = {str(s): suppress_count(value) for s, value in support.items()}
        raise ValueError(f'Insufficient complete rows per sex: quick minimum {minimum}; encoded_per_sex={counts}')
    if len(frame) <= n:
        return frame.copy()
    # Reserve each sex's minimum, then proportionally allocate remaining capacity.
    capacity = np.array([support[s] - minimum for s in (0, 1)])
    shares = capacity / capacity.sum() * (n - 2 * minimum)
    targets = np.floor(shares).astype(int) + minimum
    for s in np.argsort(-(shares - np.floor(shares)))[:n - int(targets.sum())]:
        targets[s] += 1
    parts = []
    for sex in (0, 1):
        group = frame[frame.sex == sex]
        outcome = group.glucose_raw.astype(float) >= threshold
        groups = [group[~outcome], group[outcome]]
        allocations = np.array([len(g) for g in groups]) * targets[sex] / len(group)
        sizes = np.floor(allocations).astype(int)
        for k in np.argsort(-(allocations - sizes))[:targets[sex] - int(sizes.sum())]:
            sizes[k] += 1
        parts.extend(g.sample(int(size), random_state=seed) for g, size in zip(groups, sizes) if size)
    return pd.concat(parts).sample(frac=1, random_state=seed).copy()
