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


def test_quick_imbalanced_encoded_train_preserves_sex_and_outcome():
    from models.quick import stratified_train_sample
    from models.common import load_config
    from models.data import apply_scope, build_spec, fit_preproc, raw_generated, make_arrays, require_sex_support
    from tests.mock_data import make_mock_v2
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(24000), cfg)
    spec = build_spec(cfg, training_df=frame)
    eligible = make_arrays(frame, fit_preproc(raw_generated(frame, spec), spec))
    source = frame.loc[eligible.retained_mask]
    sample = stratified_train_sample(source, 6500, cfg['quick']['min_rows_per_sex'], 42, cfg['outcome']['threshold_mg_dl'])
    assert len(sample) == 6500
    assert sample.index.is_unique and set(sample.index) <= set(source.index)
    selected = make_arrays(sample, fit_preproc(raw_generated(sample, spec), spec))
    require_sex_support(selected, cfg, quick=True)
    with pytest.raises(ValueError, match='5000'):
        require_sex_support(selected, cfg)
    assert selected.support['1'] >= 2000
    for sex in (0, 1):
        real = source[source.sex == sex]
        subset = sample[sample.sex == sex]
        assert abs((real.glucose_raw >= 200).mean() - (subset.glucose_raw >= 200).mean()) < .002
    assert sample.equals(stratified_train_sample(source, 6500, 2000, 42, 200))
    with pytest.raises(ValueError, match='encoded_per_sex.*None'):
        stratified_train_sample(source[source.sex == 0], 6500, 2000, 42, 200)


def test_quick_settings_and_pipeline_stage_flags(monkeypatch, tmp_path):
    from models import run_all, train_cvae, train_baselines, eval_dev
    cfg = run_all.load_config()
    calls = []
    monkeypatch.setattr(train_cvae, 'main', lambda argv: calls.append(argv))
    monkeypatch.setattr(train_baselines, 'main', lambda argv: calls.append(argv))
    monkeypatch.setattr(eval_dev, 'main', lambda argv: calls.append(argv))
    for stage in ('train_cvae', 'train_baselines', 'eval_dev'):
        args = run_all.parse_args(['--data-dir', '/unread-private', '--quick', '--out-dir', str(tmp_path), '--_stage', stage])
        run_all.stage(args)
    assert '--extra-cvae' not in sum(calls, [])
    assert str(cfg['quick']['cvae_train_rows']) in calls[0]
    assert str(cfg['quick']['cvae_epochs']) in calls[0]
    assert '--quick' in calls[1] and str(cfg['quick']['baseline_train_rows']) in calls[1]
    assert str(cfg['quick']['baseline_epochs']) in calls[1]
    assert str(cfg['quick']['evaluation_rows']) in calls[2]


def test_reports_pending_missing_or_invalid_package(real_generator_env, tmp_path, monkeypatch):
    from backend.reports import packaged_report
    monkeypatch.setenv('PP_REPORT_DIR', str(tmp_path))
    for comparison in (False, True):
        payload = packaged_report(comparison)
        assert payload['status'] == 'pending' and payload['metrics'] is None and payload['models'] is None
        assert payload['run_type'] == 'mock'
    (tmp_path/'manifest.json').write_text('{bad')
    assert packaged_report()['status'] == 'pending'
