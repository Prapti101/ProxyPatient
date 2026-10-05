"""
Shared helpers for the P2 model code: config, data paths, the locked test
split, small-cell suppression, age/BMI band logic.

Real data is read from the directory in env var PP_DATA_DIR (or --data-dir).
Nothing here prints or returns respondent-level rows.
"""

import hashlib
import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

MOCK_BANNER = "MOCK DATA, not NFHS-5"

# Harmonised age bands used inside the model (same meaning for both sexes).
AGE3_LABELS = ["15-24", "25-34", "35+"]


def load_config(path: Optional[str] = None) -> dict:
    with open(path or os.path.join(REPO_ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def min_cell(cfg: dict) -> int:
    value = int(cfg.get("privacy", {}).get("min_cell_size", 30))
    if value < 30:
        raise ValueError("privacy.min_cell_size cannot be below 30")
    return value


def threshold(cfg: dict) -> float:
    return float(cfg["outcome"]["threshold_mg_dl"])


# ── Data location and the locked test split ──────────────────────────────────

class LockedTestError(RuntimeError):
    pass


def data_dir(arg: Optional[str] = None) -> str:
    d = arg or os.environ.get("PP_DATA_DIR")
    if not d:
        raise RuntimeError(
            "No data directory. Set env var PP_DATA_DIR (or pass --data-dir) to the "
            "folder holding the v2 parquet files and MANIFEST.json.")
    return d


def split_path(ddir: str, split: str, imputed: bool = True,
               final_test: bool = False) -> str:
    """Path of a v2 split file. Refuses the test split unless final_test=True."""
    if split == "test" and not final_test:
        raise LockedTestError(
            "The test split is LOCKED. Model selection uses train/val only. "
            "Re-run with --final-test only for the single final evaluation.")
    name = f"dae_imputed_{split}_v2.parquet" if imputed else f"{split}_v2.parquet"
    return os.path.join(ddir, name)


def guard_filename(path: str, final_test: bool) -> None:
    if "test" in os.path.basename(path).lower() and not final_test:
        raise LockedTestError(f"Refusing to read {os.path.basename(path)} without --final-test.")


def read_parquet(path: str, final_test: bool = False, columns=None, membership_only=False) -> pd.DataFrame:
    if membership_only:
        if columns != ["_row_id"]:
            raise LockedTestError("Membership-only access permits _row_id only; never test outcomes")
    else:
        guard_filename(path, final_test)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {os.path.basename(path)} (looked in {os.path.dirname(path)})")
    return pd.read_parquet(path, columns=columns)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ── Value coercion (real dtypes are not known in advance) ────────────────────

def as_num(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    return pd.to_numeric(s, errors="coerce")


def as_str(s: pd.Series) -> pd.Series:
    """String labels with NaN kept as NaN (not the string 'nan')."""
    out = s.astype(object)
    mask = pd.isna(out)
    out = out.where(mask, out.astype(str).str.strip().str.lower())
    return out.where(~mask, np.nan)


def age3_index(age: pd.Series) -> pd.Series:
    """Harmonised age band index 0/1/2 (15-24, 25-34, 35+) from numeric age."""
    a = as_num(age)
    idx = pd.Series(np.where(a < 25, 0, np.where(a < 35, 1, 2)), index=age.index, dtype=float)
    return idx.where(a.between(15, 54))


def age_band_bounds(cfg: dict, sex: int, age3: int):
    """Inclusive (lo, hi) ages of a harmonised band for one sex, from config."""
    bands = cfg["age_bands"]["men" if int(sex) == 1 else "women"]
    lo, hi = bands[age3]
    return int(lo), int(hi)


def age_band_label(cfg: dict, sex: int, age3: int) -> str:
    key = "men_options" if int(sex) == 1 else "women_options"
    return cfg["whatif_options"]["age_band"][key][age3]


def parse_age_band(cfg: dict, label) -> int:
    """Accept either sex-specific label (or '35+') and return the harmonised index."""
    lab = str(label).strip()
    opts = cfg["whatif_options"]["age_band"]
    for key in ("women_options", "men_options"):
        if lab in opts[key]:
            return opts[key].index(lab)
    if lab in AGE3_LABELS:
        return AGE3_LABELS.index(lab)
    raise ValueError(
        f"age_band must be one of {sorted(set(opts['women_options'] + opts['men_options']))} "
        f"(or {AGE3_LABELS}); got {label!r}")


def bmi_band_names(cfg: dict):
    return list(cfg["bmi"]["bands"].keys())


def bmi_band_bounds(cfg: dict, band: str):
    lo, hi = cfg["bmi"]["bands"][band]
    return float(lo), float(hi)


def bmi_to_band_index(cfg: dict, bmi: np.ndarray) -> np.ndarray:
    """Left-closed bands [lo, hi) from config; NaN -> -1."""
    names = bmi_band_names(cfg)
    out = np.full(len(bmi), -1, dtype=int)
    for i, nm in enumerate(names):
        lo, hi = bmi_band_bounds(cfg, nm)
        out[(bmi >= lo) & (bmi < hi)] = i
    return out


# ── Aggregate output helpers ─────────────────────────────────────────────────

def suppress(n: int, value, k: int):
    """Return value only when the cell has at least k respondents."""
    return value if n >= k else None


def rate_cell(y: pd.Series, k: int) -> dict:
    y = y.dropna()
    n = int(len(y))
    if n < k:
        return {"n": None, "rate": None, "suppressed": True}
    return {"n": n, "rate": round(float(y.mean()), 6), "suppressed": False}


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        from models.privacy import safe_public_output
        json.dump(safe_public_output(to_jsonable(obj)), f, indent=2)


def md_table(rows, headers, none_text: str = "-") -> str:
    def fmt(v):
        if v is None:
            return none_text
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)
