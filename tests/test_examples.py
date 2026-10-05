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
