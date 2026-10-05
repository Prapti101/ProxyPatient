"""Finish regressions use only constructed mock rows and fitted mock artifacts."""
import copy
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('run_type', ['mock', 'quick', 'full', None, 'bad'])
@pytest.mark.parametrize('demo', [False, True])
def test_checkpoint_serving_run_combinations(mock_model_dir, tmp_path, monkeypatch, run_type, demo):
    import torch
    from backend import generator
    from models.artifacts import model_fingerprint
    checkpoint = torch.load(mock_model_dir/'cvae_weights.pt', weights_only=False)
    checkpoint['run_type'] = run_type
    checkpoint['is_mock'] = checkpoint['mock'] = run_type == 'mock'
    checkpoint['fingerprint'] = model_fingerprint(checkpoint)
    torch.save(checkpoint, tmp_path/'cvae_weights.pt')
    (tmp_path/'cvae_preproc.json').write_text(json.dumps(checkpoint['preproc']))
    marg = json.loads((mock_model_dir/'condition_marginals.json').read_text())
    marg['_meta']['fingerprint'] = checkpoint['fingerprint']
    (tmp_path/'condition_marginals.json').write_text(json.dumps(marg))
    monkeypatch.setenv('PP_MODEL_DIR', str(tmp_path))
    monkeypatch.delenv('PP_CVAE_WEIGHTS', raising=False)
    monkeypatch.delenv('PP_CONDITION_MARGINALS', raising=False)
    monkeypatch.setenv('PP_DEMO_MOCK', '1' if demo else '0')
    if (demo and run_type == 'mock') or (not demo and run_type in ('quick', 'full')):
        bundle, _ = generator._load()
        assert bundle.ckpt['run_type'] == run_type
        from backend.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            for endpoint in ('/health', '/validation'):
                result = client.get(endpoint).json()
                assert result['run_type'] == run_type
                assert result['preliminary'] == (run_type == 'quick')
                assert result['status_banner'] == {'mock': 'DEMO (mock data)', 'quick': 'PRELIMINARY (quick run)', 'full': 'FULL RUN'}[run_type]
    else:
        with pytest.raises(generator.ModelUnavailable):
            generator._load()
