"""Canonical configuration identity shared by training and serving."""
import hashlib
import json
from models.common import load_config
from models.data import build_spec


def inference_cfg(cfg):
    return {
        'age_bands': cfg['age_bands'],
        'age_band_labels': {'women': cfg['whatif_options']['age_band']['women_options'],
                           'men': cfg['whatif_options']['age_band']['men_options']},
        'bmi_bands': cfg['bmi']['bands'],
        'threshold_mg_dl': cfg['outcome']['threshold_mg_dl'],
        'scope': cfg.get('model', {}).get('scope', 'complete_conditions'),
        'glucose_clip_mg_dl': cfg.get('model', {}).get('glucose_clip_mg_dl', [20, 600]),
    }


def fingerprint(cfg, spec):
    payload = {'inference': inference_cfg(cfg), 'spec': spec.to_dict()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_checkpoint(checkpoint, cfg=None):
    cfg = cfg or load_config()
    from models.data import Spec
    spec = Spec.from_dict(checkpoint['preproc']['spec'])
    expected = build_spec(cfg, use_state=spec.use_state, generate_bp=spec.generate_bp)
    optional = set(cfg.get('model', {}).get('optional_generated', ['height_cm', 'weight_kg']))
    if (spec.cond_names != expected.cond_names or spec.cond_levels != expected.cond_levels
        or spec.cat_cols != expected.cat_cols or spec.cat_levels != expected.cat_levels
        or set(spec.cont_cols) - set(expected.cont_cols)
        or set(expected.cont_cols) - set(spec.cont_cols) - optional):
        raise RuntimeError('Checkpoint variable lists do not match current configuration')
    if spec.use_state and (not spec.state_codes or len(spec.state_codes) != spec.n_states
                          or len(set(spec.state_codes)) != spec.n_states):
        raise RuntimeError('Invalid checkpoint state mapping')
    if checkpoint.get('scope') != cfg.get('model', {}).get('scope', 'complete_conditions'):
        raise RuntimeError('Checkpoint training scope does not match current configuration')
    if checkpoint.get('cfg') != inference_cfg(cfg) or checkpoint.get('config_fingerprint') != fingerprint(cfg, spec):
        raise RuntimeError('Checkpoint configuration fingerprint mismatch; retrain with current configuration')
    if checkpoint.get('fingerprint') != model_fingerprint(checkpoint):
        raise RuntimeError('Model fingerprint does not match fitted parameters and metadata')
    return checkpoint['fingerprint']


def model_fingerprint(checkpoint):
    """Identity includes fitted parameters, preprocessing, configuration and mode."""
    digest = hashlib.sha256()
    metadata = {key: checkpoint[key] for key in ('preproc', 'cfg', 'hparams', 'is_mock', 'scope')}
    metadata["run_type"] = checkpoint.get("run_type")
    digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name, tensor in sorted(checkpoint['state_dict'].items()):
        import numpy as np
        array = tensor.detach().cpu().contiguous().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError('Model parameters are not finite')
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
