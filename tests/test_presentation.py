"""Regression checks using constructed MOCK data only."""
import numpy as np
import pytest
from models.common import load_config
from models.data import apply_scope, build_spec, fit_preproc, raw_generated, make_arrays, require_sex_support
from tests.mock_data import make_mock_v2


def test_missing_women_height_keeps_women_or_stops():
    cfg = load_config()
    df, _ = apply_scope(make_mock_v2(3000), cfg)
    df.loc[df.sex == 0, 'height_cm'] = np.nan
    spec = build_spec(cfg, training_df=df)
    assert 'height_cm' not in spec.cont_cols and not spec.weight_derived
    arr = make_arrays(df, fit_preproc(raw_generated(df, spec), spec))
    assert arr.support['0'] > 30 and arr.support['1'] > 30
    require_sex_support(arr, cfg, mock=True)
    with pytest.raises(ValueError, match='per sex'):
        require_sex_support(arr, cfg)


def test_height_linkage_guard_and_columns():
    from preprocess_v2 import WOMEN_COLS, validate_women_height
    assert {'v001', 'v002', 'v003'} <= set(WOMEN_COLS)
    df = make_mock_v2(100)
    df['height_cm'] = np.nan
    with pytest.raises(ValueError, match='linkage failed'):
        validate_women_height(df)
    df['height_cm'] = 150.
    validate_women_height(df)


def test_noncontiguous_state_mapping_and_unknown_rejection(real_generator_env):
    from backend import generator
    from models.data import state_index
    generator._BUNDLES.clear()
    bundle, _ = generator._load()
    assert 37 in bundle.spec.state_codes and 26 not in bundle.spec.state_codes
    frame = make_mock_v2(100)
    frame['state'] = 37
    assert (state_index(frame, bundle.spec) == bundle.spec.state_codes.index(37)).all()
    assert (generator.generate({'state': 37}, 100)['state'] == 37).all()
    with pytest.raises(ValueError, match='state'):
        generator.generate({'state': 26}, 100)
