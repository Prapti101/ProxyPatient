"""Representative example regressions using constructed mock artifacts only."""
import numpy as np
import pytest
from tests.test_generator import FULL


def test_demo_examples_api(real_generator_env):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        body = {'condition': FULL, 'n': 300, 'seed': 12}
        a = client.post('/generate', json=body).json()
        b = client.post('/generate', json=body).json()
        assert a['examples'] == b['examples'] and len(a['examples']) == 3
        assert a['outcome_stat'] == b['outcome_stat']
        assert all(e['synthetic'] and 'MOCK' in e['label'] and 'not a real person' in e['label'] for e in a['examples'])
        for count in range(1, 6):
            result = client.post('/generate', json={**body, 'n_examples': count})
            assert result.status_code == 200, result.text
            assert len(result.json()['examples']) == count
            assert result.json()['outcome_stat'] == a['outcome_stat']
        for count in (0, 6, True, 2.5):
            assert client.post('/generate', json={**body, 'n_examples': count}).status_code == 422
        assert client.post('/generate', json={**body, 'export_sample': True}).status_code == 422
        result = client.post('/compare', json={'scenarios': [{'label': label, **body} for label in ('a', 'b')]}).json()
        assert result['scenarios'][0]['examples'][0]['example_id'] == 'S1-A'
        assert result['scenarios'][1]['examples'][0]['example_id'] == 'S2-A'


def test_percentile_selection_and_elevated_cap(real_generator_env):
    from backend.generator import generate
    from backend.examples import representative_examples
    frame = generate(FULL, 100, 4)
    frame['glucose_raw'] = np.arange(100) + 50
    for count, values in [(3, [75, 100, 124]), (5, [60, 75, 100, 124, 139])]:
        assert [e['glucose_raw'] for e in representative_examples(frame, count, 'S1')] == values
    frame.loc[99, 'glucose_raw'] = 250
    examples = representative_examples(frame, 3, 'S1')
    assert len(examples) == 3 and examples[-1]['illustrative_elevated']
    assert examples[-1]['elevated_glucose_proxy']


def test_selected_generated_rows_not_mock_training_rows(real_generator_env):
    from backend.generator import generate
    from backend.examples import representative_examples
    from tests.mock_data import make_mock_v2
    frame = generate(FULL, 300, 12)
    training = make_mock_v2(4000, seed=1)
    for example in representative_examples(frame, 5, 'S1'):
        f = example['generated_features']
        assert not ((training.bmi.round(1) == f['bmi']) &
                    (training.waist_cm.round(1) == f['waist_cm']) &
                    (training.glucose_raw.round() == example['glucose_raw'])).any()


def passing_report():
    return {'status': 'complete', 'model_fingerprint': 'same', 'demo': False,
            'models': {'CVAE': {'nearest_record': {
                'real_val_to_train': {'median': 1., 'n_query': 100},
                'generated_to_train': {'median': .6, 'minimum': .01, 'share_exact_copy': 0., 'n_query': 100}}}}}


def test_real_gate_present_missing_failed_and_stale(real_generator_env, monkeypatch):
    import copy
    from backend.examples import example_fields, near_copy_passed, WITHHELD
    from backend.generator import generate
    from backend import reports
    frame = generate(FULL, 100, 2)
    frame.attrs.update(demo=False, fingerprint='same')
    report = passing_report()
    assert near_copy_passed(report, 'same')
    monkeypatch.setattr(reports, 'packaged_report', lambda comparison: report)
    assert len(example_fields(frame, 3, 'S1')['examples']) == 3
    for key, value in [('median', .1), ('minimum', 0.), ('share_exact_copy', .01),
                       ('median', float('nan')), ('minimum', None), ('n_query', 29),
                       ('n_query', True)]:
        broken = copy.deepcopy(report)
        broken['models']['CVAE']['nearest_record']['generated_to_train'][key] = value
        monkeypatch.setattr(reports, 'packaged_report', lambda comparison: broken)
        result = example_fields(frame, 3, 'S1')
        assert result == {'examples': None, 'examples_status': WITHHELD}
    for broken in ({}, {'status': 'pending'}, {**report, 'model_fingerprint': 'stale'},
                   {**report, 'demo': True}):
        assert not near_copy_passed(broken, 'same')
    broken = copy.deepcopy(report)
    del broken['models']['CVAE']['nearest_record']['generated_to_train']['minimum']
    assert not near_copy_passed(broken, 'same')


def test_dcr_minimum_detects_exact_copy():
    from models.eval_dev import dcr, real_units, complete_rows
    from models.common import load_config
    from models.data import apply_scope, build_spec
    from tests.mock_data import make_mock_v2
    cfg = load_config()
    raw, _ = apply_scope(make_mock_v2(3000, seed=1), cfg)
    spec = build_spec(cfg, training_df=raw)
    frame = real_units(raw, spec)
    frame = frame[complete_rows(frame, spec)]
    result = dcr(frame, frame, frame, spec, 42)
    assert result['generated_to_train']['minimum'] == 0
    assert result['generated_to_train']['share_exact_copy'] == 1
