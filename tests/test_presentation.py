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


def test_real_mode_rejects_mock_and_demo_is_visible(real_generator_env, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        assert client.get('/health').json()['mode'] == 'demo'
        assert client.get('/schema').json()['demo'] is True
        monkeypatch.delenv('PP_DEMO_MOCK')
        assert client.get('/health').json()['mode'] == 'unavailable'
        response = client.post('/generate', json={'condition': FULL, 'n': 100})
        assert response.status_code == 503 and 'MOCK' in response.text


def test_outcome_uses_finite_generated_glucose_only():
    import pandas as pd
    from backend.outcome_stat_stub import outcome_stat
    df = pd.DataFrame({'glucose_raw': [250.]*100 + [np.nan, np.inf], 'elevated_glucose_proxy': [0]*102})
    assert outcome_stat(df, {'threshold_mg_dl': 200})['rate'] == 1.
    assert outcome_stat(df, {'threshold_mg_dl': 200})['n'] == 100
    with pytest.raises(ValueError, match='threshold'):
        outcome_stat(df, {'threshold_mg_dl': 300})
    with pytest.raises(ValueError, match='required'):
        outcome_stat(df.drop(columns='glucose_raw'), {'threshold_mg_dl': 200})


def test_checkpoint_config_mismatch_is_rejected(real_generator_env, monkeypatch):
    from backend import generator
    from models import artifacts
    cfg = load_config()
    cfg['outcome']['threshold_mg_dl'] = 300
    monkeypatch.setattr('models.common.load_config', lambda: cfg)
    with pytest.raises(generator.ModelUnavailable, match='fingerprint'):
        generator._load()

FULL = dict(sex=0, age_band='25-34', residence='urban', wealth_quintile=3,
            bmi_band='normal', hypertension=0, tobacco=0, alcohol=0)


@pytest.mark.parametrize('extra', [{'glucose_raw': 200}, {'log_glucose': 5}, {'hba1c': 6}, {'colour': 'red'},
                                  {'age_band': 'nonsense'}, {'tobacco': 2}, {'alcohol': 2}, {'wealth_quintile': 1.5}, {'state': 26}])
def test_strict_http_conditions(real_generator_env, extra):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        r = c.post('/generate', json={'condition': {**FULL, **extra}, 'n': 100})
        assert r.status_code == 422, r.text
        if any('glucose' in k or k == 'hba1c' for k in extra):
            assert 'glucose is the outcome, not an input' in r.text


def test_full_profile_and_response_provenance(real_generator_env, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    monkeypatch.delenv('PP_ALLOW_PARTIAL_PROFILE')
    with TestClient(app) as c:
        assert c.post('/generate', json={'condition': {'sex': 0}}).status_code == 422
        r = c.post('/generate', json={'condition': {**FULL, 'sex': 'male', 'age_band': '35-49'}, 'n': 100})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d['effective_conditions']['sex'] == 1 and d['effective_conditions']['age_band'] == '35-54'
        assert d['synthetic'] and d['demo'] and d['model_fingerprint']
        assert d['weighting'] == 'unweighted sample' and 'survey uncertainty' in d['uncertainty_note']
        assert 'clipped_share' in d['sampling_diagnostics']
        assert 37 in c.get('/options').json()['state']


def test_demo_builder_supports_three_full_profiles(tmp_path, monkeypatch):
    from models.make_demo_checkpoint import main
    from fastapi.testclient import TestClient
    from backend.main import app
    main(['--out-dir', str(tmp_path)])
    monkeypatch.setenv('PP_MODEL_DIR', str(tmp_path))
    monkeypatch.setenv('PP_DEMO_MOCK', '1')
    monkeypatch.delenv('PP_CVAE_WEIGHTS', raising=False)
    monkeypatch.delenv('PP_CONDITION_MARGINALS', raising=False)
    with TestClient(app) as c:
        r = c.get('/profiles')
        assert r.status_code == 200, r.text
        for p in r.json()['profiles']:
            assert p['n_train'] >= 500 and set(FULL) == set(p['condition'])
            assert c.post('/generate', json={'condition': p['condition'], 'n': 100}).status_code == 200


@pytest.mark.parametrize('glucose,expected', [(100., 0.), (250., 1.)])
def test_wilson_interval_handles_extreme_outcomes(glucose, expected):
    import pandas as pd
    from backend.outcome_stat_stub import outcome_stat
    r = outcome_stat(pd.DataFrame({'glucose_raw': [glucose]*100}), {'threshold_mg_dl': 200})
    assert r['rate'] == expected
    assert r['ci_low'] < r['ci_high'] and r['ci_low'] <= expected <= r['ci_high']
    assert r['monte_carlo_interval']['method'] == 'Wilson'


def test_small_counts_and_statistics_are_suppressed():
    from models.privacy import suppress_count, suppress_stat, safe_public_output, small_count_paths
    assert suppress_count(0) is None and suppress_count(29) is None and suppress_count(30) == 30
    assert suppress_stat(5., 29) is None
    report = safe_public_output({'n': 3, 'n_states': 2, 'n_train': 100, 'steps': [{'n_excluded': 1}]})
    assert report['n'] is None and report['n_states'] == 2
    assert not small_count_paths(report)


def test_dae_benchmark_rejects_failed_predictions_and_suppresses_small_support():
    from models.check_dae_benchmark import benchmark
    frame = make_mock_v2(1000, imputed=False)
    def broken(df):
        df = df.copy()
        df['waist_cm'] = np.nan
        return df
    with pytest.raises(ValueError, match='failed predictions'):
        benchmark(frame, frame, broken)
    res = benchmark(frame.iloc[:10], frame.iloc[:10], lambda x: x)
    assert all(r['status'] == 'insufficient coverage' and r['rmse_dae'] is None and r['n_masked'] is None for r in res.values())


def test_mock_training_public_outputs_have_no_small_counts(mock_model_dir):
    import json
    from models.privacy import small_count_paths
    for path in mock_model_dir.glob('*.json'):
        assert not small_count_paths(json.loads(path.read_text())), path.name
    log = json.loads((mock_model_dir/'cvae_train_log.json').read_text())
    assert log['scope_train']['steps'][0]['n_excluded_men'] is None or log['scope_train']['steps'][0]['n_excluded_men'] >= 30


def test_split_guards_preserve_membership_and_abort_on_mismatch():
    from split_and_aggregate_v2 import build_splits, row_hash
    df = make_mock_v2(1000)
    with pytest.raises(ValueError, match='init-split'):
        build_splits(df)
    parts = build_splits(df, init_split=True)
    rebuilt = build_splits(df, {k: v[['_row_id']] for k, v in parts.items()})
    assert {k: row_hash(v) for k, v in parts.items()} == {k: row_hash(v) for k, v in rebuilt.items()}
    with pytest.raises(ValueError, match='mismatch'):
        build_splits(df.iloc[:-1], {k: v[['_row_id']] for k, v in parts.items()})
    with pytest.raises(ValueError, match='Missing'):
        build_splits(df, {'train': parts['train']})


def test_nonlegacy_parquet_reads_are_centralized():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for path in list(root.glob('*.py')) + list((root/'models').glob('*.py')) + list((root/'backend').glob('*.py')):
        if path == root/'models/common.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'read_parquet'), f'Bypass in {path.name}:{node.lineno}'


def test_id_only_membership_cannot_read_outcomes(tmp_path):
    from models.common import read_parquet, LockedTestError
    with pytest.raises(LockedTestError, match='ID|row_id'):
        read_parquet(str(tmp_path/'test.parquet'), columns=['glucose_raw'], membership_only=True)


def test_manifest_errors_stop_before_reading_data(tmp_path):
    from models.step0_checks import main
    with pytest.raises(ValueError, match='Manifest integrity'):
        main(['--data-dir', str(tmp_path), '--out', str(tmp_path/'report.md')])
    assert not (tmp_path/'report.md').exists()


def test_final_run_requires_explicit_acknowledgement():
    from models.run_all import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--mock', '--final-test'])


def test_one_command_quick_mock_pipeline_and_public_scan(tmp_path, monkeypatch):
    import json
    from models.run_all import main
    from models.privacy import small_count_paths
    result = main(['--mock', '--quick', '--out-dir', str(tmp_path)])
    assert result['stages'] == ['step0', 'dae_benchmark', 'train_cvae', 'train_baselines', 'eval_dev', 'package']
    assert (tmp_path/'private_outputs/cvae_weights.pt').exists()
    assert not list(tmp_path.rglob('*.parquet')) and not list(tmp_path.rglob('*.csv'))
    for path in (tmp_path/'safe_outputs').glob('*.json'):
        assert not small_count_paths(json.loads(path.read_text())), path.name
    manifest = json.loads((tmp_path/'safe_outputs/manifest.json').read_text())
    assert manifest['mock'] is True
    assert manifest['run_type'] == 'mock'
    assert (tmp_path/'safe_outputs/timings.json').exists()
    monkeypatch.setenv('PP_MODEL_DIR', str(tmp_path/'private_outputs'))
    monkeypatch.setenv('PP_REPORT_DIR', str(tmp_path/'safe_outputs'))
    monkeypatch.setenv('PP_DEMO_MOCK', '1')
    monkeypatch.delenv('PP_CVAE_WEIGHTS', raising=False)
    monkeypatch.delenv('PP_CONDITION_MARGINALS', raising=False)
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        for endpoint in ('/validation', '/model-comparison'):
            payload = client.get(endpoint).json()
            assert payload['status'] == 'complete', payload
            assert payload['run_type'] == 'mock' and payload['status_banner'] == 'DEMO (mock data)'
            assert set(payload['models']) == {'CVAE', 'TVAE', 'CTGAN'}
            assert not small_count_paths(payload)
        evaluation = client.get('/validation').json()['metrics']
        assert 'marginals' in evaluation['models']['CVAE']
        assert 'wasserstein' in evaluation['models']['CVAE']['marginals']['glucose_raw']
        # Even checksum-consistent malformed/stale packages must not become complete reports.
        import copy
        for name, key, value in [('model_comparison_dev.json', 'models', []),
                                 ('model_card.json', 'model_fingerprint', 'different-fitted-model'),
                                 ('model_comparison_dev.json', 'run_type', 'quick')]:
            path = tmp_path/'safe_outputs'/name
            original = path.read_text()
            altered = json.loads(original)
            altered[key] = value
            path.write_text(json.dumps(altered))
            changed_manifest = copy.deepcopy(manifest)
            from models.common import sha256_file
            for entry in changed_manifest['files']:
                if entry['file'] == name:
                    entry['sha256'] = sha256_file(str(path))
            (tmp_path/'safe_outputs/manifest.json').write_text(json.dumps(changed_manifest))
            pending = client.get('/validation').json()
            assert pending['status'] == 'pending' and pending['models'] is None
            path.write_text(original)
            (tmp_path/'safe_outputs/manifest.json').write_text(json.dumps(manifest))
        monkeypatch.setenv('PP_REPORT_DIR', str(tmp_path/'missing'))
        payload = client.get('/validation').json()
        assert payload['status'] == 'pending' and payload['metrics'] is None

    expected = {p.name for p in (tmp_path/'safe_outputs').iterdir() if p.suffix in ('.json', '.md') and p.name != 'manifest.json'}
    assert {entry['file'] for entry in manifest['files']} == expected
    from models.common import sha256_file
    assert all(sha256_file(str(tmp_path/'safe_outputs'/entry['file'])) == entry['sha256'] for entry in manifest['files'])


def test_parser_negation_outcome_rejection_and_confirmation(real_generator_env):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        r = c.post('/parse', json={'text': 'urban women aged 25-34 do not smoke and drink no alcohol'})
        assert r.status_code == 200
        d = r.json()
        assert d['parsed_condition']['tobacco'] == 0 and d['parsed_condition']['alcohol'] == 0
        assert d['confidence'] is None and d['requires_confirmation']
        assert c.post('/parse', json={'text': 'elevated glucose'}).status_code == 422
        assert c.post('/parse', json={'text': 'improved BMI'}).json()['unresolved']
        r = c.post('/parse', json={'text': 'improved BMI', 'baseline': {'bmi_band': 'obese'}})
        assert r.json()['parsed_condition']['bmi_band'] == 'overweight'
        assert c.post('/parse', json={'text': 'unchanged'}).json()['parsed_condition']['sex'] is None


def test_api_version_and_real_parser_disabled(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    monkeypatch.delenv('PP_DEMO_MOCK', raising=False)
    with TestClient(app) as c:
        assert c.get('/openapi.json').json()['info']['version'] == '2.0.0'
        assert c.post('/parse', json={'text': 'urban women'}).status_code == 403
        assert c.get('/validation').json()['status'] == 'pending'


def test_normalization_rejects_insufficient_values_and_invalid_scales():
    from models.data import Preproc
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(1000), cfg)
    spec = build_spec(cfg, training_df=frame)
    generated = raw_generated(frame, spec)
    generated['waist_cm'] = np.nan
    with pytest.raises(ValueError, match='finite TRAIN'):
        fit_preproc(generated, spec)
    pre = fit_preproc(raw_generated(frame, spec), spec)
    pre.std['age'] = np.nan
    with pytest.raises(ValueError, match='scale'):
        make_arrays(frame, pre)


def test_invalid_numeric_domains_are_not_truncated():
    from models.data import condition_frame, state_index
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(1000), cfg)
    spec = build_spec(cfg, training_df=frame)
    for column in ('sex', 'wealth_quintile', 'state'):
        frame[column] = frame[column].astype(float)
    frame.loc[frame.index[:3], 'sex'] = .5
    frame.loc[frame.index[3:6], 'wealth_quintile'] = 1.5
    frame.loc[frame.index[6:9], 'any_tobacco'] = .5
    frame.loc[frame.index[9:12], 'age'] = 99
    frame.loc[frame.index[12:15], 'state'] = 1.5
    cf = condition_frame(frame, spec)
    assert cf.loc[frame.index[:3], 'sex'].isna().all()
    assert cf.loc[frame.index[3:6], 'wealth_quintile'].isna().all()
    assert cf.loc[frame.index[6:9], 'tobacco'].isna().all()
    assert cf.loc[frame.index[9:12], 'age_band'].isna().all()
    assert (state_index(frame, spec)[12:15] == -1).all()


def test_empty_samples_keep_typed_columns_and_invalid_state_stops(mock_model_dir):
    from models.sampling import CVAEBundle
    bundle = CVAEBundle.load(str(mock_model_dir/'cvae_weights.pt'))
    frame = bundle.sample(np.empty((0, len(bundle.spec.cond_names)), dtype=int), np.empty(0, dtype=int))
    assert len(frame) == 0 and 'glucose_raw' in frame and frame['age'].dtype.kind == 'i'
    with pytest.raises(ValueError, match='state'):
        bundle.sample(np.zeros((10, len(bundle.spec.cond_names)), dtype=int), np.full(10, -1))


def test_all_unfilled_baseline_keeps_schema():
    import pandas as pd
    from models.train_baselines import rejection_sample
    class NoMatch:
        def set_random_state(self, seed): pass
        def sample(self, n): return pd.DataFrame({'sex': ['1']*n, 'glucose_raw': [100.]*n})
    frame, stats = rejection_sample(NoMatch(), pd.DataFrame({'sex': ['0']*100}), ['sex'], 42, batch=100, max_factor=1)
    assert len(frame) == 100 and list(frame) == ['sex', 'glucose_raw'] and frame.isna().all().all()


def test_requested_retained_coverage_is_explicit_and_suppressed():
    import pandas as pd
    from models.eval_dev import coverage_report
    cf = pd.DataFrame({'sex': ['0']*100 + ['1']*5})
    report = coverage_report(cf, np.array([True]*50 + [False]*55))
    assert report['n_requested'] == 105 and report['n_retained'] == 50
    rare = next(x for x in report['condition_cells'] if x['condition_cell'] == '1')
    assert rare['n_requested'] is None and rare['n_retained'] is None and rare['retained_share'] is None


def test_dae_inference_batches_match_single_batch_and_preserve_glucose(tmp_path):
    import joblib
    import torch
    from dae_impute import DenoisingAutoencoder, impute_dae
    frame = make_mock_v2(100, imputed=False)
    columns = ['waist_cm', 'hip_cm']
    stats = {'cont_medians': {c: 80. for c in columns}, 'cont_stds': {c: 10. for c in columns},
             'cat_maps': {}, 'cat_modes': {}}
    torch.manual_seed(42)
    model = DenoisingAutoencoder(2, [8, 4])
    torch.save({'input_dim': 2, 'hidden_dims': [8, 4], 'model_state_dict': model.state_dict(),
                'cont_cols': columns, 'cat_cols': [], 'impute_cols': columns}, tmp_path/'weights.pt')
    joblib.dump(stats, tmp_path/'stats.pkl')
    a = impute_dae(frame, str(tmp_path/'weights.pt'), str(tmp_path/'stats.pkl'), batch_rows=7)
    b = impute_dae(frame, str(tmp_path/'weights.pt'), str(tmp_path/'stats.pkl'), batch_rows=1000)
    assert np.allclose(a[columns], b[columns], equal_nan=True)
    assert a['glucose_raw'].equals(frame['glucose_raw'])


def test_sampling_batches_respect_bounds_and_report_clipping(mock_model_dir):
    from models.sampling import CVAEBundle
    bundle = CVAEBundle.load(str(mock_model_dir/'cvae_weights.pt'))
    result = bundle.sample(np.zeros((100, len(bundle.spec.cond_names)), dtype=int), np.zeros(100, dtype=int), batch_rows=13)
    assert len(result) == 100 and np.isfinite(result['glucose_raw']).all()
    assert result['age'].between(15, 24).all()
    assert 0 <= result.attrs['sampling']['clipped_share'] <= 1


def test_config_precedence_and_condition_order(tmp_path):
    import json
    import yaml
    import torch
    from models.train_cvae import main
    cfg = load_config()
    cfg['model'].update(state_embedding=False, glucose_head='gaussian', seed=17, max_epochs=1)
    cfg['conditioning_variables'] = list(reversed(cfg['conditioning_variables']))
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    weights, log = main(['--mock', '--cpu', '--max-rows', '1500', '--config', str(config), '--out-dir', str(tmp_path/'configured')])
    checkpoint = torch.load(weights, weights_only=False)
    assert not checkpoint['preproc']['spec']['use_state']
    assert checkpoint['preproc']['spec']['cond_names'] == cfg['conditioning_variables']
    assert checkpoint['hparams']['glucose_components'] == 1 and len(log['epochs']) == 1
    card = json.loads((tmp_path/'configured/model_card.json').read_text())
    assert card['training']['seed'] == 17
    weights, log = main(['--mock', '--cpu', '--max-rows', '1500', '--config', str(config), '--out-dir', str(tmp_path/'overridden'),
                         '--epochs', '2', '--seed', '42', '--glucose-head', 'mixture'])
    checkpoint = torch.load(weights, weights_only=False)
    assert checkpoint['hparams']['glucose_components'] == 3 and len(log['epochs']) == 2
    cfg['conditioning_variables'].append('glucose_raw')
    with pytest.raises(ValueError, match='conditioning_variables'):
        build_spec(cfg)


def test_model_identity_changes_when_fitted_parameters_change(mock_model_dir):
    import torch
    from models.artifacts import validate_checkpoint, model_fingerprint
    checkpoint = torch.load(mock_model_dir/'cvae_weights.pt', weights_only=False)
    original = checkpoint['fingerprint']
    first = next(iter(checkpoint['state_dict']))
    checkpoint['state_dict'][first] = checkpoint['state_dict'][first] + .1
    assert model_fingerprint(checkpoint) != original
    with pytest.raises(RuntimeError, match='Model fingerprint'):
        validate_checkpoint(checkpoint)


def test_package_rejects_mock_as_real_and_preserves_old_outputs(tmp_path, mock_model_dir):
    import argparse
    import shutil
    from models.run_all import package
    (tmp_path/'working').mkdir()
    shutil.copytree(mock_model_dir, tmp_path/'working/model')
    old = tmp_path/'private_outputs'
    old.mkdir()
    (old/'user-file.md').write_text('preserve')
    args = argparse.Namespace(out_dir=str(tmp_path), mock=False, skipped_stages=[])
    with pytest.raises(ValueError, match='provenance'):
        package(args)
    assert (old/'user-file.md').read_text() == 'preserve'


def test_state_support_uses_rows_that_can_be_encoded():
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(4000), cfg)
    frame.loc[frame.state == 37, 'education'] = 99
    assert (frame.state == 37).sum() >= 30
    assert 37 not in build_spec(cfg, training_df=frame).state_codes


def test_derived_weight_respects_optional_measurement_support():
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(2000), cfg)
    frame.loc[frame.sex == 0, 'weight_kg'] = np.nan
    spec = build_spec(cfg, training_df=frame)
    assert 'height_cm' in spec.cont_cols and not spec.weight_derived


def test_mock_row_export_and_fractional_wealth_are_rejected(real_generator_env):
    from backend.generator import generate
    with pytest.raises(ValueError, match='export'):
        generate(FULL, 100, export_sample=True)
    with pytest.raises(ValueError, match='wealth_quintile'):
        generate({**FULL, 'wealth_quintile': 1.5}, 100)


def test_missing_artifacts_return_unavailable_for_full_requests(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    monkeypatch.setenv('PP_MODEL_DIR', str(tmp_path))
    for key in ('PP_DEMO_MOCK', 'PP_CVAE_WEIGHTS', 'PP_CONDITION_MARGINALS', 'PP_ALLOW_PARTIAL_PROFILE'):
        monkeypatch.delenv(key, raising=False)
    with TestClient(app) as client:
        assert client.get('/health').json()['mode'] == 'unavailable'
        assert client.post('/generate', json={'condition': FULL, 'n': 100}).status_code == 503
        assert client.post('/generate', json={'condition': {**FULL, 'state': 37}, 'n': 100}).status_code == 503


def test_final_marker_prevents_repeated_evaluation_or_retraining(tmp_path, mock_model_dir, monkeypatch):
    import json
    import shutil
    from models import run_all
    private = tmp_path/'private_outputs'
    private.mkdir()
    shutil.copy2(mock_model_dir/'cvae_weights.pt', private/'cvae_weights.pt')
    (tmp_path/'safe_outputs').mkdir()
    # Exercise orchestration/marker only; this unit test does not claim final metrics executed.
    def stage_only(command, **kwargs):
        assert '--final-test' in command
        (tmp_path/'stage_metrics').mkdir(exist_ok=True)
        (tmp_path/'stage_metrics/final_test.json').write_text(json.dumps({'stage': 'final_test', 'runtime_seconds': 0., 'peak_memory_mib': 0.}))
    monkeypatch.setattr(run_all.subprocess, 'run', stage_only)
    args = ['--mock', '--out-dir', str(tmp_path), '--final-test', '--i-understand-this-is-the-single-final-run']
    assert run_all.main(args)['stages'] == ['final_test']
    assert (tmp_path/'FINAL_TEST_STARTED.json').exists()
    with pytest.raises(RuntimeError, match='already entered'):
        run_all.main(args)
    with pytest.raises(RuntimeError, match='already entered'):
        run_all.main(['--mock', '--out-dir', str(tmp_path), '--full'])


def test_nonfinite_model_parameters_fail_artifact_validation(mock_model_dir):
    import torch
    from models.artifacts import model_fingerprint
    checkpoint = torch.load(mock_model_dir/'cvae_weights.pt', weights_only=False)
    first = next(iter(checkpoint['state_dict']))
    checkpoint['state_dict'][first].fill_(float('nan'))
    with pytest.raises(RuntimeError, match='not finite'):
        model_fingerprint(checkpoint)


def test_direction_statistics_suppress_both_sides_of_rare_cells(monkeypatch):
    from models import eval_dev
    from models.data import condition_frame
    cfg = load_config()
    frame, _ = apply_scope(make_mock_v2(1000), cfg)
    spec = build_spec(cfg, training_df=frame)
    real = condition_frame(frame, spec).reset_index(drop=True)
    real['glucose_raw'] = 100.
    real['residence'] = 'rural'
    real.loc[:9, 'residence'] = 'urban'
    monkeypatch.setattr(eval_dev, 'MIN_CELL_EVAL', 30)
    report = eval_dev.direction_checks(real, real.copy(), spec, 200)
    index = report['residence']['levels'].index('urban')
    assert report['residence']['real_rate'][index] is None and report['residence']['gen_rate'][index] is None
