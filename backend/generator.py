"""
ProxyPatient — Generator (P2, real CVAE)
=========================================
    generate(condition: dict, n: int = 1000, seed: int = 42) -> pd.DataFrame

Decoder-only generator; serving validates artifact identity and mode.

condition: any subset of
    sex              "female"/"male" (API form) or 0/1
    age_band         "15-24", "25-34", "35-49" (women) / "35-54" (men); either
                     sex-specific label is accepted, the output echoes the
                     sex-appropriate label
    residence        "urban"/"rural"
    wealth_quintile  1-5
    bmi_band         "underweight"/"normal"/"overweight"/"obese"
    hypertension     0/1
    tobacco          0/1   (maps to any_tobacco)
    alcohol          0/1
    state            supported raw code from the checkpoint mapping
Unspecified keys are sampled from models/condition_marginals.json as
INDEPENDENT marginals, so the UI should send a FULL baseline profile plus the
what-if changes. Invalid values raise ValueError.

Returns n NEW rows sampled from the CVAE decoder, in ORIGINAL units: the
conditions, generated variables, glucose_raw (mg/dL), elevated_glucose_proxy
(derived from the generated glucose_raw and config threshold) and
is_synthetic=True. Deterministic per seed; CPU; writes nothing to disk unless
export_sample=True (to the git-ignored outputs/ folder).
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "models", "cvae_weights.pt")
DEFAULT_MARGINALS = os.path.join(BASE_DIR, "models", "condition_marginals.json")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

N_MIN, N_MAX = 100, 10_000

FORBIDDEN_INPUT_CONDITIONS = [
    "glucose", "glucose_raw", "elevated_glucose_proxy", "log_glucose",
    "sb74", "smb74", "hba1c",
]

_BUNDLES = {}


class ModelUnavailable(RuntimeError):
    pass


def _paths():
    root = os.environ.get("PP_MODEL_DIR", os.path.join(BASE_DIR, "models"))
    return (os.environ.get("PP_CVAE_WEIGHTS", os.path.join(root, "cvae_weights.pt")),
            os.environ.get("PP_CONDITION_MARGINALS", os.path.join(root, "condition_marginals.json")),
            os.path.join(root, "cvae_preproc.json"))


def _load():
    from models.common import load_config
    from models.artifacts import validate_checkpoint
    paths = _paths()
    demo = os.environ.get("PP_DEMO_MOCK") == "1"
    key = (paths, demo, tuple(os.stat(p).st_mtime_ns if os.path.isfile(p) else None for p in paths))
    try:
        if key not in _BUNDLES:
            if not all(os.path.isfile(p) for p in paths):
                raise ModelUnavailable("CVAE artifacts unavailable. Train the model first or explicitly create a MOCK demo checkpoint.")
            from models.sampling import CVAEBundle
            bundle = CVAEBundle.load(paths[0], "cpu")
            with open(paths[1], encoding="utf-8") as f:
                marg = json.load(f)
            with open(paths[2], encoding="utf-8") as f:
                pre = json.load(f)
            if pre != bundle.ckpt["preproc"]:
                raise ModelUnavailable("Preprocessor does not match checkpoint")
            is_mock = bundle.ckpt.get("is_mock", bundle.ckpt.get("mock"))
            if is_mock is not demo:
                raise ModelUnavailable("MOCK checkpoints require PP_DEMO_MOCK=1; demo mode requires a MOCK checkpoint")
            fp = validate_checkpoint(bundle.ckpt, load_config())
            if marg.get("_meta", {}).get("fingerprint") != fp:
                raise ModelUnavailable("Marginals fingerprint does not match checkpoint")
            _BUNDLES.clear()
            _BUNDLES[key] = (bundle, marg)
        bundle, marg = _BUNDLES[key]
        # Revalidate against live configuration even when weights are cached.
        validate_checkpoint(bundle.ckpt, load_config())
        return bundle, marg
    except ModelUnavailable:
        raise
    except Exception as exc:
        raise ModelUnavailable("CVAE artifacts invalid or incompatible: " + str(exc)) from exc


def _parse_binary(key, v):
    s = str(v).strip().lower()
    if s in ("0", "0.0", "false", "no"):
        return 0
    if s in ("1", "1.0", "true", "yes"):
        return 1
    raise ValueError(f"{key} must be 0 or 1; got {v!r}")


def _parse_condition(condition: dict, bundle) -> dict:
    """Validate and convert to level indices of the model's spec."""
    spec, cfg = bundle.spec, bundle.cfg
    if condition is None:
        condition = {}
    if not isinstance(condition, dict):
        raise ValueError("condition must be a dict")
    known = set(spec.cond_names) | {"state"}
    out = {}
    for key, v in condition.items():
        if key in FORBIDDEN_INPUT_CONDITIONS:
            raise ValueError(f"FORBIDDEN: '{key}' is the outcome and cannot be a conditioning input.")
        if v is None:
            continue
        if key not in known:
            raise ValueError(f"Unknown condition '{key}'. Allowed: {sorted(known)}")
        if key == "sex":
            s = str(v).strip().lower()
            sx = {"female": 0, "f": 0, "woman": 0, "women": 0, "male": 1, "m": 1, "man": 1, "men": 1}.get(s)
            out["sex"] = sx if sx is not None else _parse_binary("sex", v)
        elif key == "age_band":
            lab = str(v).strip()
            labels = cfg["age_band_labels"]
            if lab in labels["women"]:
                out["age_band"] = labels["women"].index(lab)
            elif lab in labels["men"]:
                out["age_band"] = labels["men"].index(lab)
            elif lab in spec.cond_levels["age_band"]:
                out["age_band"] = spec.cond_levels["age_band"].index(lab)
            else:
                raise ValueError(f"age_band must be one of {sorted(set(labels['women'] + labels['men']))}; got {v!r}")
            out["_age_label"] = lab
        elif key == "state":
            if not spec.use_state:
                raise ValueError("This model was trained without the state embedding; 'state' is not accepted.")
            try:
                st = int(str(v).strip())
            except ValueError:
                raise ValueError(f"state must be an integer supported DHS code; got {v!r}")
            if st not in spec.state_codes:
                raise ValueError(f"state must be one of {spec.state_codes}; got {v!r}")
            out["state"] = spec.state_codes.index(st)
        elif key in ("hypertension", "tobacco", "alcohol"):
            out[key] = _parse_binary(key, v)
        else:
            lab = str(v).strip().lower()
            if key == "wealth_quintile":
                try:
                    lab = str(int(float(lab)))
                except ValueError:
                    pass
            levels = spec.cond_levels[key]
            if lab not in levels:
                raise ValueError(f"{key} must be one of {levels}; got {v!r}")
            out[key] = levels.index(lab)
    # Either sex-specific label maps to the same harmonised band (e.g. '35-49'
    # for a man means 35+); the output echoes the sex-appropriate label.
    out.pop("_age_label", None)
    return out


def _marginal_probs(marg: dict, key: str, levels):
    counts = marg.get(key, {})
    w = np.array([float(counts.get(lab) or 0) for lab in levels])
    if w.sum() <= 0:
        raise ModelUnavailable(f"No supported marginal counts for {key}")
    return w / w.sum()


def generate(condition: dict, n: int = 1000, seed: int = 42,
             export_sample: bool = False) -> pd.DataFrame:
    """Generate a SYNTHETIC cohort of n rows. See module docstring."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if not N_MIN <= int(n) <= N_MAX:
        raise ValueError(f"n must be between {N_MIN} and {N_MAX}; got {n}")
    bundle, marg = _load()
    parsed = _parse_condition(condition, bundle)
    spec, cfg = bundle.spec, bundle.cfg
    n = int(n)
    rng = np.random.default_rng(int(seed))

    cond_idx = np.zeros((n, len(spec.cond_names)), dtype=np.int64)
    filled = []
    for j, key in enumerate(spec.cond_names):
        if key in parsed:
            cond_idx[:, j] = parsed[key]
        else:
            levels = spec.cond_levels[key]
            cond_idx[:, j] = rng.choice(len(levels), size=n, p=_marginal_probs(marg, key, levels))
            filled.append(key)
    state_idx = None
    if spec.use_state:
        if "state" in parsed:
            state_idx = np.full(n, parsed["state"], dtype=np.int64)
        else:
            codes = [str(s) for s in spec.state_codes]
            state_idx = rng.choice(spec.n_states, size=n, p=_marginal_probs(marg, "state", codes))
            filled.append("state")

    gen = bundle.sample(cond_idx, state_idx, seed=int(seed))

    ci = {c: j for j, c in enumerate(spec.cond_names)}
    sex = cond_idx[:, ci["sex"]]
    a3 = cond_idx[:, ci["age_band"]]
    labels = cfg["age_band_labels"]
    age_band = np.where(sex == 1, np.asarray(labels["men"], dtype=object)[a3],
                        np.asarray(labels["women"], dtype=object)[a3])
    df = pd.DataFrame({
        "sex": sex.astype(int),
        "age_band": age_band,
        "residence": np.asarray(spec.cond_levels["residence"], dtype=object)[cond_idx[:, ci["residence"]]],
        "wealth_quintile": np.asarray(spec.cond_levels["wealth_quintile"])[cond_idx[:, ci["wealth_quintile"]]].astype(int),
        "bmi_band": np.asarray(spec.cond_levels["bmi_band"], dtype=object)[cond_idx[:, ci["bmi_band"]]],
        "hypertension": cond_idx[:, ci["hypertension"]].astype(int),
        "any_tobacco": cond_idx[:, ci["tobacco"]].astype(int),
        "alcohol": cond_idx[:, ci["alcohol"]].astype(int),
    })
    if state_idx is not None:
        df["state"] = np.asarray(spec.state_codes, dtype=int)[state_idx]
    for c in ["age", "bmi", "weight_kg", "height_cm", "waist_cm", "hip_cm",
              "systolic_avg", "diastolic_avg", "education", "bp_ever_checked", "glucose_raw"]:
        if c in gen.columns:
            df[c] = gen[c].to_numpy()
    for c in ("education", "bp_ever_checked"):
        if c in df:
            df[c] = df[c].astype(int)
    df["elevated_glucose_proxy"] = (df["glucose_raw"] >= float(cfg["threshold_mg_dl"])).astype(int)
    df["is_synthetic"] = True
    df.attrs["fingerprint"] = bundle.ckpt["fingerprint"]
    df.attrs["demo"] = bool(bundle.ckpt["is_mock"])
    df.attrs["sampling"] = {**gen.attrs.get("sampling", {}), "filled_from_marginals": filled,
                            "seed": int(seed), "label": "SYNTHETIC"}

    if export_sample:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        df.to_csv(os.path.join(OUTPUTS_DIR, f"synthetic_sample_seed{seed}_{stamp}.csv"), index=False)
    return df
