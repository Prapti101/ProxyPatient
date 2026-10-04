"""
Data module for the CVAE: loads train/val, applies the training scope,
harmonises age bands, builds condition and generated-variable arrays, and fits
TRAIN-ONLY normalisation (saved as models/cvae_preproc.json).

All lists come from config.yaml (`model` section) so they can be changed
without touching code. Only aggregate information leaves this module.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.common import (AGE3_LABELS, MODELS_DIR, age3_index, as_num, as_str,
                           bmi_band_names, min_cell, write_json)

# Columns dropped before modelling (P1 handoff); kept here for the record.
DROP_COLUMNS = ["_row_id", "survey_weight", "glucose_ever_checked",
                "told_high_glucose", "on_glucose_medicine", "glucose_time",
                "self_reported_hypertension", "on_bp_medication"]


@dataclass
class Spec:
    """What the model conditions on and what it generates."""
    cond_names: List[str]                     # internal condition keys, in order
    cond_levels: Dict[str, List[str]]         # level labels per condition
    use_state: bool
    n_states: int
    cont_cols: List[str]                      # generated continuous (model space)
    cat_cols: List[str]                       # generated categorical
    cat_levels: Dict[str, List[float]]
    generate_bp: bool
    state_codes: List[int] = field(default_factory=list)
    weight_derived: bool = True               # weight_kg = bmi * (height/100)^2

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def build_spec(cfg: dict, use_state: Optional[bool] = None,
               generate_bp: Optional[bool] = None, training_df=None) -> Spec:
    m = cfg.get("model", {})
    use_state = m.get("state_embedding", True) if use_state is None else use_state
    generate_bp = m.get("generate_bp", False) if generate_bp is None else generate_bp
    w = cfg["whatif_options"]
    levels = {
        "sex": ["0", "1"],
        "age_band": list(AGE3_LABELS),
        "residence": list(w["residence"]["options"]),
        "wealth_quintile": [str(x) for x in w["wealth_quintile"]["options"]],
        "bmi_band": bmi_band_names(cfg),
        "hypertension": ["0", "1"],
        "tobacco": ["0", "1"],
        "alcohol": ["0", "1"],
    }
    names = cfg.get("conditioning_variables", list(levels))
    if len(names) != len(set(names)) or set(names) != set(levels):
        raise ValueError("conditioning_variables must contain each of the eight supported non-outcome conditions exactly once")
    levels = {name: levels[name] for name in names}
    cont = list(m.get("generated_continuous",
                      ["age", "bmi", "height_cm", "waist_cm", "hip_cm", "log_glucose"]))
    if not {"age", "bmi", "log_glucose"} <= set(cont):
        raise ValueError("Generated age, BMI and log_glucose are required")
    if generate_bp:
        cont += [c for c in ["systolic_avg", "diastolic_avg"] if c not in cont]
    cat = list(m.get("generated_categorical", ["education", "bp_ever_checked"]))
    cat_levels = {"education": [0.0, 1.0, 2.0, 3.0], "bp_ever_checked": [0.0, 1.0]}
    dropped = []
    if training_df is not None:
        minimum = float(m.get("optional_min_share", 0.8))
        for col in m.get("optional_generated", ["height_cm", "weight_kg"]):
            shares = [float(np.isfinite(as_num(training_df.loc[as_num(training_df["sex"]) == sex, col])).mean())
                      if col in training_df else 0.0 for sex in (0, 1)]
            if any(not np.isfinite(x) or x < minimum for x in shares):
                if col in cont:
                    cont.remove(col)
                dropped.append(col)
        if "height_cm" in dropped and "weight_kg" in cont:
            cont.remove("weight_kg")
        if dropped:
            print("Optional generated variables dropped for insufficient per-sex support: " + ", ".join(dropped))
    state_codes = []
    if training_df is not None and use_state:
        values = as_num(training_df["state"])
        if ((values.dropna() % 1) != 0).any():
            raise ValueError("State codes must be integers")
        candidate = Spec(cond_names=list(levels), cond_levels=levels, use_state=False, n_states=1,
                         cont_cols=cont, cat_cols=cat, cat_levels={c: cat_levels[c] for c in cat},
                         generate_bp=bool(generate_bp))
        eligible = (encode_conditions(condition_frame(training_df, candidate), candidate) >= 0).all(1)
        generated = raw_generated(training_df, candidate)
        eligible &= np.isfinite(generated[cont].to_numpy(float)).all(1)
        for col in cat:
            eligible &= generated[col].isin(cat_levels[col]).to_numpy()
        counts = values[eligible].value_counts()
        state_codes = sorted(int(code) for code, count in counts.items() if count >= min_cell(cfg))
        if not state_codes:
            raise ValueError("No state codes have privacy-safe training support")
    return Spec(cond_names=list(levels), cond_levels=levels, use_state=bool(use_state),
                n_states=max(len(state_codes), 1), state_codes=state_codes, cont_cols=cont, cat_cols=cat,
                cat_levels={c: cat_levels[c] for c in cat}, generate_bp=bool(generate_bp),
                weight_derived="height_cm" in cont and "weight_kg" not in dropped)


# ── Scope ─────────────────────────────────────────────────────────────────────

def apply_scope(df: pd.DataFrame, cfg: dict, scope: Optional[str] = None):
    """Return (in-scope frame, aggregate exclusion report)."""
    scope = scope or cfg.get("model", {}).get("scope", "complete_conditions")
    k = min_cell(cfg)
    g = as_num(df["glucose_raw"])
    rules = {"glucose_raw known": pd.Series(np.isfinite(g), index=df.index)}
    if scope == "complete_conditions":
        rules["bmi_measured == 1"] = as_num(df["bmi_measured"]) == 1
        rules["hypertension not null"] = as_num(df["hypertension"]).notna()
    elif scope != "glucose_known":
        raise ValueError(f"Unknown scope {scope!r}")
    keep = pd.Series(True, index=df.index)
    report = {"scope": scope, "n_input": int(len(df)), "steps": []}
    sex = as_num(df["sex"])
    for name, r in rules.items():
        drop = keep & ~r
        report["steps"].append({
            "rule": name,
            "n_excluded": int(drop.sum()),
            "n_excluded_women": int((drop & (sex == 0)).sum()),
            "n_excluded_men": int((drop & (sex == 1)).sum()),
        })
        keep &= r
    report["n_in_scope"] = int(keep.sum())
    report["in_scope_share"] = round(float(keep.mean()), 6) if len(df) else None
    # who is excluded, by sex / harmonised age band (aggregate, suppressed)
    by = {}
    a3 = age3_index(df["age"])
    for s in (0, 1):
        for i, lab in enumerate(AGE3_LABELS):
            cell = (sex == s) & (a3 == i)
            n = int(cell.sum())
            by[f"sex={s}, age={lab}"] = None if n < k else round(float((cell & ~keep).sum() / n), 6)
    report["excluded_share_by_sex_age"] = by
    return df.loc[keep].copy(), report


# ── Encoding ──────────────────────────────────────────────────────────────────

def condition_frame(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Condition labels (strings) per row from the v2 columns."""
    out = pd.DataFrame(index=df.index)
    sex = as_num(df["sex"])
    out["sex"] = sex.where(sex.isin([0, 1])).map(lambda v: None if pd.isna(v) else str(int(v)))
    age = as_num(df["age"])
    a3 = age3_index(age)
    a3 = a3.where(((sex == 0) & age.between(15, 49)) | ((sex == 1) & age.between(15, 54)))
    out["age_band"] = a3.map(lambda v: None if pd.isna(v) else AGE3_LABELS[int(v)])
    out["residence"] = as_str(df["residence"])
    out["wealth_quintile"] = as_num(df["wealth_quintile"]).where(as_num(df["wealth_quintile"]).isin([1, 2, 3, 4, 5])).map(lambda v: None if pd.isna(v) else str(int(v)))
    out["bmi_band"] = as_str(df["bmi_band"])
    for key, col in [("hypertension", "hypertension"), ("tobacco", "any_tobacco"), ("alcohol", "alcohol")]:
        out[key] = as_num(df[col]).where(as_num(df[col]).isin([0, 1])).map(lambda v: None if pd.isna(v) else str(int(v)))
    return out


def encode_conditions(cf: pd.DataFrame, spec: Spec) -> np.ndarray:
    """Integer level index per condition; -1 where missing/unknown."""
    cols = []
    for c in spec.cond_names:
        lut = {lab: i for i, lab in enumerate(spec.cond_levels[c])}
        cols.append(cf[c].map(lambda v: lut.get(v, -1) if v is not None and not pd.isna(v) else -1).to_numpy())
    return np.stack(cols, axis=1).astype(np.int64)


def state_index(df: pd.DataFrame, spec: Spec) -> np.ndarray:
    values = as_num(df["state"])
    mapping = {code: index for index, code in enumerate(spec.state_codes)}
    return values.where((values % 1) == 0).map(mapping).fillna(-1).to_numpy(dtype=np.int64)


def raw_generated(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Generated variables in model space (log glucose), unnormalised."""
    out = pd.DataFrame(index=df.index)
    for c in spec.cont_cols:
        if c == "log_glucose":
            g = as_num(df["glucose_raw"])
            out[c] = np.log(g.where(g > 0))
        else:
            out[c] = as_num(df[c])
    for c in spec.cat_cols:
        out[c] = as_num(df[c])
    return out


@dataclass
class Preproc:
    spec: Spec
    mean: Dict[str, float]
    std: Dict[str, float]
    lo: Dict[str, float]              # train min/max in model space (for clipping)
    hi: Dict[str, float]
    n_train_rows: int
    meta: dict = field(default_factory=dict)

    def to_json(self, path: str):
        write_json({"spec": self.spec.to_dict(), "mean": self.mean, "std": self.std,
                    "lo": self.lo, "hi": self.hi, "n_train_rows": self.n_train_rows,
                    "meta": self.meta}, path)

    @classmethod
    def from_dict(cls, d):
        return cls(spec=Spec.from_dict(d["spec"]), mean=d["mean"], std=d["std"],
                   lo=d["lo"], hi=d["hi"], n_train_rows=d["n_train_rows"], meta=d.get("meta", {}))

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def fit_preproc(train_gen: pd.DataFrame, spec: Spec) -> Preproc:
    mean, std, lo, hi = {}, {}, {}, {}
    for c in spec.cont_cols:
        v = train_gen[c]
        v = v[np.isfinite(v)]
        if len(v) < 30:
            raise ValueError(f"Insufficient finite TRAIN values for {c}; cannot fit normalization")
        mean[c] = float(v.mean())
        scale = float(v.std())
        std[c] = scale if np.isfinite(scale) and scale > 0 else 1.0
        # 0.1th / 99.9th percentiles rather than min/max so no single respondent's
        # value is stored; used only to clip extreme generated values.
        lo[c] = float(v.quantile(0.001))
        hi[c] = float(v.quantile(0.999))
    return Preproc(spec=spec, mean=mean, std=std, lo=lo, hi=hi, n_train_rows=int(len(train_gen)),
                   meta={"note": "Fit on TRAIN only (in-scope rows). Not P1's preprocess_v2.pkl."})


@dataclass
class Arrays:
    cond: np.ndarray        # (n, n_cond) int64
    state: np.ndarray       # (n,) int64, -1 unknown
    cont: np.ndarray        # (n, n_cont) float32, normalised
    cat: np.ndarray         # (n, n_cat) int64 level index
    n_dropped: int
    retained_mask: np.ndarray = None
    support: dict = field(default_factory=dict)
    exclusions: dict = field(default_factory=dict)


def make_arrays(df: pd.DataFrame, pre: Preproc) -> Arrays:
    spec = pre.spec
    for col in spec.cont_cols:
        if not np.isfinite(pre.std[col]) or pre.std[col] <= 0 or not np.isfinite(pre.mean[col]):
            raise ValueError("Invalid finite normalization scale")
    cf = condition_frame(df, spec)
    cond = encode_conditions(cf, spec)
    gen = raw_generated(df, spec)
    cont = np.stack([((gen[c] - pre.mean[c]) / pre.std[c]).to_numpy(dtype=float) for c in spec.cont_cols], 1)
    cat_cols = []
    for c in spec.cat_cols:
        lut = {float(v): i for i, v in enumerate(spec.cat_levels[c])}
        cat_cols.append(gen[c].map(lambda v: lut.get(float(v), -1) if not pd.isna(v) else -1).to_numpy())
    cat = np.stack(cat_cols, 1) if cat_cols else np.zeros((len(df), 0), dtype=np.int64)
    ok = (cond >= 0).all(1) & np.isfinite(cont).all(1) & (cat >= 0).all(1)
    st = state_index(df, spec)
    if spec.use_state:
        ok &= st >= 0
    return Arrays(cond=cond[ok], state=st[ok], cont=cont[ok].astype(np.float32),
                  cat=cat[ok].astype(np.int64), n_dropped=int((~ok).sum()), retained_mask=ok,
                  support={str(sex): int((ok & (as_num(df["sex"]).to_numpy() == sex)).sum()) for sex in (0, 1)},
                  exclusions={reason: {"n_excluded": int(mask.sum()),
                                       "excluded_per_sex": {str(sex): int((mask & (as_num(df["sex"]).to_numpy() == sex)).sum()) for sex in (0, 1)}}
                              for reason, mask in {"invalid_conditions": ~(cond >= 0).all(1),
                                                   "nonfinite_generated": ~np.isfinite(cont).all(1),
                                                   "invalid_categories": ~(cat >= 0).all(1),
                                                   "unsupported_state": (st < 0) if spec.use_state else np.zeros(len(df), dtype=bool)}.items()})


def condition_marginals(df: pd.DataFrame, spec: Spec, cfg: dict) -> dict:
    """Aggregate marginal shares of each condition (n < min_cell suppressed).
    Used by the generator to fill unspecified condition keys."""
    k = min_cell(cfg)
    cf = condition_frame(df, spec)
    out = {"_meta": {"source": "in-scope TRAIN rows", "n_rows": int(len(df)), "min_cell_size": k,
                     "note": "Independent marginals. The UI should send a FULL baseline profile."}}
    for c in spec.cond_names:
        vc = cf[c].value_counts()
        out[c] = {lab: (int(vc.get(lab, 0)) if vc.get(lab, 0) >= k else None) for lab in spec.cond_levels[c]}
    # age distribution within each harmonised band by sex is learned by the model,
    # state marginal is needed when state is not given
    sv = as_num(df["state"]).value_counts()
    out["state"] = {str(int(s)): (int(n) if n >= k else None) for s, n in sv.items() if not pd.isna(s) and (not spec.use_state or int(s) in spec.state_codes)}
    return out


def default_paths(models_dir: str = MODELS_DIR) -> dict:
    return {
        "weights": os.path.join(models_dir, "cvae_weights.pt"),
        "preproc": os.path.join(models_dir, "cvae_preproc.json"),
        "marginals": os.path.join(models_dir, "condition_marginals.json"),
        "train_log": os.path.join(models_dir, "cvae_train_log.json"),
        "model_card": os.path.join(models_dir, "model_card.json"),
    }


def require_sex_support(arr, cfg, mock=False):
    minimum = 30 if mock else int(cfg.get("model", {}).get("min_rows_per_sex", 5000))
    if minimum < 30:
        raise ValueError("min_rows_per_sex cannot be below privacy minimum 30")
    if any(arr.support.get(str(sex), 0) < minimum for sex in (0, 1)):
        raise ValueError(f"Insufficient complete rows per sex after encoding: require at least {minimum} for both sexes")
    return arr.support


def supported_profiles(df, spec, minimum=500, cfg=None):
    """Only observed FULL joint profiles from encoded TRAIN; no reconstructed individuals."""
    cf = condition_frame(df, spec)
    grouped = cf.groupby(spec.cond_names, dropna=True, observed=True).size().sort_values(ascending=False)
    profiles = []
    for labels, count in grouped.items():
        if count < max(500, minimum):
            continue
        profile = dict(zip(spec.cond_names, labels))
        for key in ("sex", "wealth_quintile", "hypertension", "tobacco", "alcohol"):
            profile[key] = int(profile[key])
        from models.common import load_config, age_band_label
        profile["age_band"] = age_band_label(cfg or load_config(), profile["sex"], AGE3_LABELS.index(profile["age_band"]))
        profiles.append({"label": f"Supported TRAIN profile {len(profiles)+1}", "condition": profile,
                         "n_train": int(count), "description": "Observed joint TRAIN cell; model scope, unweighted sample"})
        if len(profiles) == 3:
            break
    return {"profiles": profiles, "status": "supported" if len(profiles) >= 3 else "insufficient supported joint profiles",
            "minimum_training_cell": max(500, minimum)}
