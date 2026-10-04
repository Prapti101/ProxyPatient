"""One-command TRAIN/VAL workflow, run only on the private data holder's machine.

python -m models.run_all --data-dir DIR --out-dir OUT --quick
python -m models.run_all --data-dir DIR --out-dir OUT --full
python -m models.run_all --mock --out-dir OUT --quick

Final evaluation is separate, on frozen packaged artifacts, with explicit
--final-test --i-understand-this-is-the-single-final-run acknowledgement.
No respondent or mock rows are saved. safe_outputs contains aggregate JSON/MD;
private_outputs contains weights and fitted artifacts, never for public commit.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from models.common import load_config, sha256_file, write_json
from models.privacy import safe_public_output, small_count_paths

STAGES = ('step0', 'dae_benchmark', 'train_cvae', 'train_baselines', 'eval_dev', 'package')


def sanity(args):
    import numpy as np
    from models.data import apply_scope, build_spec, fit_preproc, make_arrays, raw_generated, require_sex_support
    from models.common import read_parquet, split_path
    if args.mock:
        from tests.mock_data import make_mock_v2
        raw = make_mock_v2(4000)
    else:
        from models.step0_checks import check_manifest
        checks = check_manifest(args.data_dir, False)
        if any(r['status'] != 'OK' and not r['status'].startswith('LOCKED') for r in checks):
            raise ValueError('Manifest integrity failed before training')
        raw = read_parquet(split_path(args.data_dir, 'train'))
    cfg = load_config()
    glucose = raw['glucose_raw'].to_numpy(float)
    finite = glucose[np.isfinite(glucose)]
    if len(finite) < 30 or not 60 <= np.median(finite) <= 200:
        raise ValueError('unit check failed: is this mmol/L? Verify the official codebook locally.')
    scoped, _ = apply_scope(raw, cfg)
    spec = build_spec(cfg, training_df=scoped)
    arr = make_arrays(scoped, fit_preproc(raw_generated(scoped, spec), spec))
    require_sex_support(arr, cfg, mock=args.mock)
    if 'log_glucose' not in spec.cont_cols:
        raise ValueError('Generated glucose is required')


def package(args):
    root = Path(args.out_dir)
    work, safe, private = root/'working', root/'safe_outputs', root/'private_outputs'
    safe.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    model = work/'model'
    for name in ('cvae_weights.pt', 'cvae_preproc.json', 'condition_marginals.json', 'supported_profiles.json'):
        shutil.copy2(model/name, private/name)
    if (work/'baselines').exists():
        for path in (work/'baselines').glob('*.pkl'):
            shutil.copy2(path, private/path.name)
    (private/'README.md').write_text('PRIVATE fitted artifacts and model weights. NOT to be committed to the public repository.\n'
                                    + ('MOCK DATA, not NFHS-5; development demonstration only.\n' if args.mock else 'Keep under the approved private data agreement.\n'))
    whitelist = [work/'step0.json', work/'dae_benchmark.json', model/'cvae_train_log.json',
                 model/'model_card.json', work/'baselines_train_log.json', work/'model_comparison_dev.json']
    manifest = []
    for path in whitelist:
        if not path.exists():
            continue
        payload = safe_public_output(json.loads(path.read_text()))
        if small_count_paths(payload):
            raise ValueError('Public output contains unsuppressed respondent counts')
        target = safe/path.name
        write_json(payload, str(target))
        manifest.append({'file': target.name, 'sha256': sha256_file(str(target))})
        markdown = path.with_suffix('.md')
        if markdown.exists():
            shutil.copy2(markdown, safe/markdown.name)
            manifest.append({'file': markdown.name, 'sha256': sha256_file(str(safe/markdown.name))})
    card = json.loads((safe/'model_card.json').read_text())
    card['evaluation'] = json.loads((safe/'model_comparison_dev.json').read_text())
    card['status'] = 'MOCK demonstration, not NFHS-5' if args.mock else 'NFHS-5 TRAIN/VAL run; no untouched-test claim'
    write_json(card, str(safe/'model_card.json'))
    for entry in manifest:
        entry['sha256'] = sha256_file(str(safe/entry['file']))
    write_json({'mock': args.mock, 'files': manifest, 'skipped_stages': args.skipped_stages}, str(safe/'manifest.json'))
    (safe/'README.md').write_text('Aggregate-only run reports. Counts below 30 are null. '
                                 'Review licensing and disclosure before public release. '
                                 + ('All numbers are MOCK smoke results, not NFHS-5 evidence.\n' if args.mock else 'These are computed private-run results, not a clinical assessment.\n'))


def stage(args):
    name = args._stage
    root = Path(args.out_dir)
    work = root/'working'
    work.mkdir(parents=True, exist_ok=True)
    data = ['--mock'] if args.mock else ['--data-dir', args.data_dir]
    quick = args.quick or args.mock
    started = time.monotonic()
    try:
        if name == 'step0':
            from models.step0_checks import main
            main(data + ['--out', str(work/'step0.md')])
            sanity(args)
        elif name == 'dae_benchmark':
            from models.check_dae_benchmark import main
            main(data + ['--out', str(work/'dae_benchmark.json')])
        elif name == 'train_cvae':
            from models.train_cvae import main
            main(data + ['--cpu', '--out-dir', str(work/'model')]
                 + (['--epochs', '2', '--max-rows', '4000' if args.mock else '50000'] if quick else []))
        elif name == 'train_baselines':
            from models.train_baselines import main
            main(data + ['--cpu', '--out-dir', str(work/'baselines'), '--log-out', str(work/'baselines_train_log.json')]
                 + (['--epochs', '1', '--subsample', '3000' if args.mock else '50000'] if quick else []))
        elif name == 'eval_dev':
            from models.eval_dev import main
            main(data + ['--cvae-weights', str(work/'model/cvae_weights.pt'), '--baselines-dir', str(work/'baselines'),
                         '--out-json', str(work/'model_comparison_dev.json')]
                 + (['--n-eval', '1500' if args.mock else '5000'] if quick else []))
        elif name == 'final_test':
            from models.eval_dev import main
            main(data + ['--final-test', '--cvae-weights', str(root/'private_outputs/cvae_weights.pt'),
                         '--baselines-dir', str(root/'private_outputs'),
                         '--out-json', str(root/'safe_outputs/model_comparison_final_test.json')])
        elif name == 'package':
            package(args)
        else:
            raise ValueError('Unknown stage')
    finally:
        # Fresh subprocess per stage: ru_maxrss is that stage's peak, in KiB on Linux.
        write_json({'stage': name, 'runtime_seconds': round(time.monotonic()-started, 3),
                    'peak_memory_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)},
                   str(root/'stage_metrics'/f'{name}.json'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=os.environ.get('PP_DATA_DIR'))
    parser.add_argument('--out-dir', default=os.environ.get('PP_RUN_DIR', 'outputs/run'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='small CPU smoke run; real per-sex guards still apply')
    mode.add_argument('--full', action='store_true', help='configuration defaults, CPU')
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--skip-dae-benchmark', action='store_true')
    parser.add_argument('--mock', action='store_true', help='in-memory MOCK data only; never persisted as rows')
    parser.add_argument('--final-test', action='store_true', help='separate frozen single final evaluation, no retraining')
    parser.add_argument('--i-understand-this-is-the-single-final-run', action='store_true', dest='acknowledge_final')
    parser.add_argument('--_stage', choices=STAGES+('final_test',), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.mock and not args.data_dir:
        parser.error('--data-dir or PP_DATA_DIR is required for private runs')
    if args.final_test and not args.acknowledge_final:
        parser.error('--final-test requires --i-understand-this-is-the-single-final-run')
    if args.final_test and (args.quick or args.full):
        parser.error('final evaluation uses frozen artifacts, not --quick/--full')
    args.skipped_stages = [s for s, skip in [('train_baselines', args.skip_baselines), ('dae_benchmark', args.skip_dae_benchmark)] if skip]
    return args


def main(argv=None):
    args = parse_args(argv)
    if args._stage:
        stage(args)
        return
    root = Path(args.out_dir).resolve()
    args.out_dir = str(root)
    marker = root/'FINAL_TEST_STARTED.json'
    if marker.exists():
        raise RuntimeError('Frozen output directory has already entered final-test stage; do not retrain or repeat')
    root.mkdir(parents=True, exist_ok=True)
    if args.final_test:
        from models.artifacts import validate_checkpoint
        from models.sampling import CVAEBundle
        weights = root/'private_outputs/cvae_weights.pt'
        bundle = CVAEBundle.load(str(weights))
        validate_checkpoint(bundle.ckpt)
        if bundle.ckpt['is_mock'] != args.mock:
            raise ValueError('Checkpoint MOCK mode does not match final run mode')
        with marker.open('x') as f:
            json.dump({'started': datetime.now(timezone.utc).isoformat(), 'weights_sha256': sha256_file(str(weights)),
                       'fingerprint': bundle.ckpt['fingerprint'], 'mock': args.mock}, f, indent=2)
        names = ['final_test']
    else:
        names = [s for s in STAGES if s not in args.skipped_stages]
        # Fresh working artifacts prevent accidentally evaluating stale baselines.
        work = root/'working'
        if work.exists():
            archive = root/('previous-working-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            work.rename(archive)
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    for name in names:
        command = [sys.executable, '-m', 'models.run_all', '--out-dir', str(root), '--_stage', name]
        if args.mock:
            command.append('--mock')
        else:
            command += ['--data-dir', args.data_dir]
        for flag, enabled in [('quick', args.quick), ('full', args.full), ('skip-baselines', args.skip_baselines),
                              ('skip-dae-benchmark', args.skip_dae_benchmark)]:
            if enabled:
                command.append('--'+flag)
        if args.final_test:
            command += ['--final-test', '--i-understand-this-is-the-single-final-run']
        print(f'Starting {name}', flush=True)
        subprocess.run(command, check=True, env=environment)
    metrics = [json.loads((root/'stage_metrics'/f'{name}.json').read_text()) for name in names]
    write_json({'mock': args.mock, 'stages': metrics, 'skipped_stages': args.skipped_stages}, str(root/'safe_outputs/run_metrics.json'))
    return {'stages': names, 'mock': args.mock, 'out_dir': str(root)}


if __name__ == '__main__':
    main()
