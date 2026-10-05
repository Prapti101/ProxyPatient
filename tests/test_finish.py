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
            for endpoint in ('/health', '/validation', '/model-comparison'):
                result = client.get(endpoint).json()
                assert result['run_type'] == run_type
                assert result['preliminary'] == (run_type == 'quick')
                assert result['status_banner'] == {'mock': 'DEMO (mock data)', 'quick': 'PRELIMINARY (quick run)', 'full': 'FULL RUN'}[run_type]
            condition = dict(sex=0, age_band='25-34', residence='urban', wealth_quintile=3,
                             bmi_band='normal', hypertension=0, tobacco=0, alcohol=0)
            request = {'condition': condition, 'n': 100}
            generated = client.post('/generate', json=request)
            assert generated.status_code == 200, generated.text
            result = generated.json()
            assert result['run_type'] == run_type and result['preliminary'] == (run_type == 'quick')
            compared = client.post('/compare', json={'scenarios': [{'label': 'a', **request}, {'label': 'b', **request}]})
            assert compared.status_code == 200, compared.text
            assert compared.json()['run_type'] == run_type
            assert all(scenario['run_type'] == run_type for scenario in compared.json()['scenarios'])

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


def test_p3_phrases_and_offline_hook_are_guarded(real_generator_env, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend import parser
    monkeypatch.delenv('PP_BERT_MODEL_DIR', raising=False)
    with TestClient(app) as c:
        assert c.post('/parse', json={'text': 'elevated glucose'}).status_code == 422
        assert 'outcome' in c.post('/parse', json={'text': 'high blood sugar'}).text
        data = c.post('/parse', json={'text': 'improved body mass index', 'baseline': {'bmi_band': 'obese'}}).json()
        assert data['parsed_condition']['bmi_band'] == 'overweight'
        assert data['requires_confirmation'] and data['confidence'] is None
        assert data['optional_token_hook']['status'] == 'disabled'
        assert c.post('/parse', json={'text': 'improved BMI'}).json()['unresolved']
        data = c.post('/parse', json={'text': 'unchanged trajectory'}).json()
        assert not any(v is not None for v in data['parsed_condition'].values())
        assert 'temporal' not in data['parser'] and data['parser'] == 'rule-based demo parser'
        monkeypatch.setenv('PP_BERT_MODEL_DIR', str(tmp_path/'missing'))
        assert c.post('/parse', json={'text': 'no tobacco'}).json()['optional_token_hook']['status'] == 'unavailable'
        monkeypatch.setenv('PP_BERT_MODEL_DIR', str(tmp_path))
        monkeypatch.setattr(parser, '_offline_pipeline', lambda directory: lambda text: [{'word': 'urban', 'entity_group': 'residence'}])
        data = c.post('/parse', json={'text': 'urban women do not smoke'}).json()
        assert data['parser'] == 'rule-based demo parser' and data['parsed_condition']['tobacco'] == 0
        assert data['optional_token_hook']['used_for_conditions'] is False
    assert not Path('parser.py').exists()
    assert not Path('validation_report.json').exists()
    assert not Path('model_comparision.json').exists()
    for path in Path('docs/p3').glob('*.json'):
        assert json.loads(path.read_text())['status'] == 'pending'


def test_p3_subgroup_suppression_and_classifier_partitions():
    from models.eval_dev import subgroup_fidelity, real_vs_synthetic, real_units, complete_rows
    from models.data import apply_scope, build_spec
    from models.common import load_config
    from models.privacy import small_count_paths
    from tests.mock_data import make_mock_v2
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(2500), cfg)
    spec = build_spec(cfg, training_df=frame)
    real = real_units(frame, spec)
    real = real[complete_rows(real, spec)].reset_index(drop=True)
    rare = real.iloc[:20]
    report = subgroup_fidelity(rare, rare, spec, 200)
    assert not small_count_paths(report)
    assert all(row['real_rate'] is None and row['synthetic_rate'] is None and row['warning'] for row in report['cells'])
    assert real_vs_synthetic(real, rare, rare, spec, 42)['roc_auc'] is None
    result = real_vs_synthetic(real, real, real.copy(), spec, 42)
    assert result['status'] == 'complete' and 0 <= result['roc_auc'] <= 1
    assert not small_count_paths(result)


def test_final_run_crash_requires_confirmation_and_frozen_artifacts(tmp_path, mock_model_dir, monkeypatch):
    import shutil
    from models import run_all
    private = tmp_path/'private_outputs'
    shutil.copytree(mock_model_dir, private)
    (tmp_path/'safe_outputs').mkdir()
    args = ['--mock', '--out-dir', str(tmp_path), '--final-test', '--i-understand-this-is-the-single-final-run']
    def crash(command, **kwargs):
        raise RuntimeError('constructed crash before metrics')
    monkeypatch.setattr(run_all.subprocess, 'run', crash)
    with pytest.raises(RuntimeError, match='constructed crash') as retained_exception:
        run_all.main(args)
    assert retained_exception.value.args[0] == 'constructed crash before metrics'
    marker = tmp_path/'FINAL_TEST_STARTED.json'
    assert json.loads(marker.read_text())['status'] == 'started'
    with pytest.raises(RuntimeError, match='confirm-previous'):
        run_all.main(args)
    def success(command, **kwargs):
        (tmp_path/'stage_metrics').mkdir(exist_ok=True)
        (tmp_path/'stage_metrics/final_test.json').write_text(json.dumps({'stage': 'final_test', 'runtime_seconds': .1, 'peak_memory_mib': 100.}))
        (tmp_path/'safe_outputs/model_comparison_final_test.json').write_text(json.dumps({'status': 'test double'}))
    monkeypatch.setattr(run_all.subprocess, 'run', success)
    (private/'changed.md').write_text('modified frozen package')
    with pytest.raises(RuntimeError, match='artifacts changed'):
        run_all.main(args + ['--confirm-previous-final-run-crashed'])
    (private/'changed.md').unlink()
    run_all.main(args + ['--confirm-previous-final-run-crashed'])
    record = json.loads(marker.read_text())
    assert record['status'] == 'completed' and len(record['recoveries']) == 1
    with pytest.raises(RuntimeError, match='completed'):
        run_all.main(args + ['--confirm-previous-final-run-crashed'])


def test_started_marker_with_metrics_refuses_retry(tmp_path, mock_model_dir, monkeypatch):
    import shutil
    from models import run_all
    shutil.copytree(mock_model_dir, tmp_path/'private_outputs')
    (tmp_path/'safe_outputs').mkdir()
    (tmp_path/'FINAL_TEST_STARTED.json').write_text(json.dumps({'status': 'started'}))
    (tmp_path/'safe_outputs/model_comparison_final_test.json').write_text('{}')
    with pytest.raises(RuntimeError, match='metrics exist'):
        run_all.main(['--mock', '--out-dir', str(tmp_path), '--final-test', '--i-understand-this-is-the-single-final-run', '--confirm-previous-final-run-crashed'])


def test_invalid_parsed_state_returns_validation_error(real_generator_env):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        r = c.post('/parse', json={'text': 'state 26'})
        assert r.status_code == 422 and 'Unsupported parsed condition' in r.text


def test_empty_final_marker_cannot_bypass_lock(tmp_path):
    from models.run_all import main
    (tmp_path/'FINAL_TEST_STARTED.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Invalid final marker'):
        main(['--mock', '--final-test', '--i-understand-this-is-the-single-final-run', '--confirm-previous-final-run-crashed', '--out-dir', str(tmp_path)])


def test_package_cannot_relabel_checkpoint(tmp_path, mock_model_dir):
    import shutil
    import argparse
    from models.run_all import package
    shutil.copytree(mock_model_dir, tmp_path/'working/model')
    with pytest.raises(ValueError, match='run_type'):
        package(argparse.Namespace(out_dir=str(tmp_path), mock=True, run_type='full', skipped_stages=[]))
    assert not (tmp_path/'private_outputs').exists()


def test_full_baselines_use_only_encoding_retained_rows(tmp_path, monkeypatch):
    from models import train_baselines
    from tests import mock_data
    frame = mock_data.make_mock_v2(3000)
    frame.loc[frame.index[:100], 'any_tobacco'] = 2
    frame.loc[frame.index[100:200], 'education'] = 99
    monkeypatch.setattr(mock_data, 'make_mock_v2', lambda *args, **kwargs: frame.copy())
    def fit(kind, table, columns, epochs, seed, gpu):
        assert set(table.tobacco) <= {'0', '1'} and set(table.education) <= {'0', '1', '2', '3'}
        class Fitted:
            def save(self, path):
                pass
        return Fitted(), 0.
    monkeypatch.setattr(train_baselines, 'fit_one', fit)
    monkeypatch.setattr(train_baselines, 'rejection_sample', lambda *args, **kwargs: (None, {'acceptance_rate': 1., 'fill_rate': 1.}))
    result = train_baselines.main(['--mock', '--cpu', '--epochs', '1', '--out-dir', str(tmp_path), '--log-out', str(tmp_path/'log.json')])
    assert result['n_train_rows_used'] < len(frame)-100
