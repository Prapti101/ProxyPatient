"""Read only checksum-verified aggregate reports bound to the served model."""
import json
import os
from pathlib import Path
from backend.generator import _load, _paths, ModelUnavailable
from models.common import sha256_file
from models.privacy import safe_public_output, MIN_CELL_SIZE
from models.run_status import run_fields


def packaged_report(comparison=False):
    fields = run_fields('mock' if os.environ.get('PP_DEMO_MOCK') == '1' else None)
    pending = {'status': 'pending', 'metrics': None, 'models': None,
               'note': 'No compatible packaged TRAIN/VAL evaluation supplied', **fields}
    try:
        bundle, _ = _load()
        fields = run_fields(bundle.ckpt['run_type'])
        pending.update(fields)
        root = Path(os.environ.get('PP_REPORT_DIR', str(Path(_paths()[0]).parent/'safe_outputs')))
        manifest = json.loads((root/'manifest.json').read_text())
        hashes = {entry['file']: entry['sha256'] for entry in manifest['files']}
        def read(name):
            if name not in hashes or sha256_file(str(root/name)) != hashes[name]:
                raise ValueError('Report package checksum mismatch')
            return json.loads((root/name).read_text())
        card = read('model_card.json')
        report = read('model_comparison_dev.json')
        if (card['model_fingerprint'] != bundle.ckpt['fingerprint']
            or card['run_type'] != bundle.ckpt['run_type']
            or manifest['run_type'] != bundle.ckpt['run_type']
            or report['run_type'] != bundle.ckpt['run_type']
            or bool(report['_meta']['mock']) != bundle.ckpt['is_mock']):
            raise ValueError('Report provenance does not match served checkpoint')
        support = report['_meta'].get('n_eval_rows_all_models_filled')
        if not isinstance(support, int) or support < MIN_CELL_SIZE:
            pending['note'] = 'Insufficient retained support; evaluation statistics suppressed'
            return pending
        if report['_meta'].get('split') != 'val':
            raise ValueError('Development panel requires VAL evaluation')
        # Canonical metrics remain in eval_dev; there is no independent metric calculation here.
        payload = {'status': 'complete', **fields, 'model_fingerprint': bundle.ckpt['fingerprint'],
                   'evaluation_scope': report['_meta'], 'models': report['models'],
                   'real_reference': report.get('real'), 'metrics': None if comparison else report,
                   'note': 'Measured data only; no model ranking. Retained subset, unweighted sample.'}
        return safe_public_output(payload)
    except (OSError, ValueError, KeyError, TypeError, ModelUnavailable):
        return pending
