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
    if checkpoint.get('cfg') != inference_cfg(cfg) or checkpoint.get('fingerprint') != fingerprint(cfg, spec):
        raise RuntimeError('Checkpoint configuration fingerprint mismatch; retrain with current configuration')
    return checkpoint['fingerprint']
