"""Preserve locked membership; initialize only with explicit --init-split.

Test membership access is ID-only through the guarded reader. Current reference
aggregates use TRAIN only; no test outcomes are inspected here.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.common import load_config, read_parquet, rate_cell, write_json, min_cell
from models.privacy import suppress_count

MIN_CELL = 30


def row_hash(df, split_name=''):
    return hashlib.sha256(np.asarray(sorted(df['_row_id']), dtype=np.int64).tobytes()).hexdigest()


def safe_rate(sub, col):
    return rate_cell(sub[col], min_cell(load_config()))


def agg_by(df, group_col, outcome_col):
    return {str(key): safe_rate(group, outcome_col) for key, group in df.groupby(group_col, observed=True)}


def cross_tab(df, col1, col2, outcome_col):
    return {str(key): agg_by(group, col2, outcome_col) for key, group in df.groupby(col1, observed=True)}


def build_splits(df, memberships=None, init_split=False, cfg=None):
    cfg = cfg or load_config()
    if df['_row_id'].duplicated().any() or df['_row_id'].isna().any():
        raise ValueError('Split requires unique nonmissing row IDs')
    if memberships is not None:
        if set(memberships) != {'train', 'val', 'test'}:
            raise ValueError('Missing locked membership; abort before writing')
        sets = {key: set(frame['_row_id']) for key, frame in memberships.items()}
        if any(len(sets[key]) != len(memberships[key]) for key in sets):
            raise ValueError('Duplicate locked membership IDs')
        if any(sets[a] & sets[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
            raise ValueError('Overlapping locked membership; abort before writing')
        if set.union(*sets.values()) != set(df['_row_id']):
            raise ValueError('Locked membership mismatch; abort before writing')
        splits = {key: df[df['_row_id'].isin(ids)].copy() for key, ids in sets.items()}
        if any(row_hash(splits[key]) != row_hash(memberships[key]) for key in splits):
            raise ValueError('Split hash mismatch; abort before writing')
        return splits
    if not init_split:
        raise ValueError('Missing locked membership; initial creation requires --init-split')
    ratios = cfg['split']
    if any(float(ratios[key]) <= 0 for key in ('train', 'val', 'test')) or not np.isclose(sum(ratios[key] for key in ('train', 'val', 'test')), 1):
        raise ValueError('Invalid configured split ratios')
    seed = int(ratios['seed'])
    known = df[df['glucose_raw'].notna()]
    unknown = df[df['glucose_raw'].isna()].sample(frac=1, random_state=seed)
    train, remainder = train_test_split(known, test_size=ratios['val']+ratios['test'], random_state=seed,
                                       stratify=(known['glucose_raw'] >= cfg['outcome']['threshold_mg_dl']))
    val, test = train_test_split(remainder, test_size=ratios['test']/(ratios['val']+ratios['test']), random_state=seed,
                               stratify=(remainder['glucose_raw'] >= cfg['outcome']['threshold_mg_dl']))
    n = int(len(unknown)*ratios['train']/(ratios['train']+ratios['val']))
    return {'train': pd.concat([train, unknown.iloc[:n]]),
            'val': pd.concat([val, unknown.iloc[n:]]), 'test': test}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--aggregates-out', required=True)
    parser.add_argument('--init-split', action='store_true')
    args = parser.parse_args(argv)
    directory = Path(args.data_dir)
    paths = {split: directory/f'{split}.parquet' for split in ('train', 'val', 'test')}
    present = [p.exists() for p in paths.values()]
    if any(present) and not all(present):
        raise ValueError('Partial locked membership; abort before writing')
    if args.init_split and (any(present) or any((directory/f'{s}_v2.parquet').exists() for s in paths)):
        raise ValueError('--init-split cannot overwrite existing split membership')
    memberships = ({split: read_parquet(str(path), columns=['_row_id'], membership_only=True)
                    for split, path in paths.items()} if all(present) else None)
    frame = read_parquet(str(directory/'combined_clean_v2.parquet'))
    splits = build_splits(frame, memberships, args.init_split)
    # All guards complete before the first write.
    for split, data in splits.items():
        data.to_parquet(directory/f'{split}_v2.parquet', index=False)
    train = splits['train'].copy()
    train['elevated_glucose_proxy'] = (train.glucose_raw >= load_config()['outcome']['threshold_mg_dl']).where(train.glucose_raw.notna())
    cfg = load_config()
    aggregates = {'_meta': {'source': 'TRAIN reference scope, unweighted sample', 'test_used': False,
                            'threshold_mg_dl': cfg['outcome']['threshold_mg_dl'], 'min_cell_size': min_cell(cfg)},
                  'overall': safe_rate(train, 'elevated_glucose_proxy')}
    for output, column in [('by_sex', 'sex'), ('by_age_band', 'age_band'), ('by_residence', 'residence'),
                           ('by_wealth_quintile', 'wealth_quintile'), ('by_education', 'education'),
                           ('by_tobacco', 'any_tobacco'), ('by_alcohol', 'alcohol'), ('by_state', 'state'),
                           ('by_hypertension', 'hypertension'), ('by_bmi_band', 'bmi_band')]:
        aggregates[output] = agg_by(train, column, 'elevated_glucose_proxy')
    for name, first, second in [('sex_by_residence', 'sex', 'residence'), ('sex_by_age_band', 'sex', 'age_band'),
                                ('sex_by_hypertension', 'sex', 'hypertension'),
                                ('age_band_by_hypertension', 'age_band', 'hypertension'),
                                ('bmi_band_by_hypertension', 'bmi_band', 'hypertension'),
                                ('age_band_by_tobacco', 'age_band', 'any_tobacco'),
                                ('age_band_by_alcohol', 'age_band', 'alcohol'),
                                ('residence_by_wealth', 'residence', 'wealth_quintile')]:
        aggregates[name] = cross_tab(train, first, second, 'elevated_glucose_proxy')
    write_json(aggregates, args.aggregates_out)
    print({split: suppress_count(len(data)) for split, data in splits.items()})


if __name__ == '__main__':
    main()
