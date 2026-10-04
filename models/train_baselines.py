"""
Baselines: TVAE and CTGAN (package `ctgan`), trained UNCONDITIONALLY on the
joint table (condition columns + generated variables incl. log_glucose) of a
stratified ~100K-row subsample of the in-scope TRAIN split. Conditioning is
done afterwards by rejection sampling on the condition columns.

Real data:  PP_DATA_DIR=/path python -m models.train_baselines
Mock smoke: python -m models.train_baselines --mock --epochs 1 --out-dir /tmp/x

Model pickles go to outputs/baselines/ (git-ignored). The training log
(times, sizes, rejection-sampling acceptance) is aggregate-only and goes to
docs/baselines_train_log.json.
"""

import argparse
import os
import pickle
import time
import resource
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from models.common import (DOCS_DIR, MOCK_BANNER, OUTPUTS_DIR, data_dir, load_config,
                           read_parquet, split_path, write_json)
from models.data import apply_scope, build_spec, condition_frame, raw_generated, fit_preproc, make_arrays, require_sex_support

STRATIFY_ON = ["sex", "age_band", "bmi_band", "hypertension"]


def joint_table(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Condition labels (str) + generated variables (model space). Complete rows only."""
    cf = condition_frame(df, spec)
    gen = raw_generated(df, spec)
    t = pd.concat([cf, gen], axis=1).dropna()
    for c in spec.cat_cols:
        t[c] = t[c].astype(int).astype(str)
    return t.reset_index(drop=True)


def discrete_columns(spec):
    return list(spec.cond_names) + list(spec.cat_cols)


def stratified_subsample(t: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(t) <= n:
        return t
    frac = n / len(t)
    parts = [g.sample(frac=frac, random_state=seed) for _, g in t.groupby(STRATIFY_ON, observed=True)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def to_units(t: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Baseline samples -> the same original-unit columns the CVAE path produces."""
    out = pd.DataFrame(index=t.index)
    lo, hi = cfg.get("model", {}).get("glucose_clip_mg_dl", [20, 600])
    for c in t.columns:
        if c == "log_glucose":
            out["glucose_raw"] = np.clip(np.round(np.exp(t[c].astype(float))), lo, hi)
        elif c in ("education", "bp_ever_checked"):
            out[c] = t[c].astype(float)
        else:
            out[c] = t[c]
    if "age" in out:
        out["age"] = np.round(out["age"].astype(float))
    if "bmi" in out and "height_cm" in out:
        out["weight_kg"] = np.round(out["bmi"] * (out["height_cm"] / 100.0) ** 2, 1)
    return out


def combo_key(cf: pd.DataFrame, cols) -> pd.Series:
    return cf[cols].astype(str).agg("|".join, axis=1)


def rejection_sample(model, wanted: pd.DataFrame, cond_cols, seed: int,
                     batch: int = 50_000, max_factor: float = 30.0, sample_columns=None):
    """Fill one generated row per row of `wanted` (condition labels) by drawing
    unconditional samples and keeping those whose condition columns match.
    Returns (samples aligned to wanted's index with NaN rows where unfilled, stats)."""
    model.set_random_state(seed)
    need = combo_key(wanted, cond_cols)
    slots = {k: list(ix) for k, ix in need.groupby(need).groups.items()}
    filled = {}
    columns = list(sample_columns or cond_cols)
    drawn = accepted = 0
    t0 = time.time()
    while slots and drawn < max_factor * len(wanted):
        s = model.sample(batch)
        columns = list(s.columns)
        drawn += len(s)
        keys = combo_key(s, cond_cols)
        for k, ix in keys.groupby(keys).groups.items():
            if k not in slots:
                continue
            free = slots[k]
            take = list(ix[:len(free)])
            for slot, row in zip(free[:len(take)], take):
                filled[slot] = s.loc[row]
            del free[:len(take)]
            accepted += len(take)
            if not free:
                del slots[k]
    res = pd.DataFrame([filled[i] for i in sorted(filled)], index=sorted(filled), columns=columns).reindex(wanted.index)
    stats = {"requested": int(len(wanted)), "filled": int(len(filled)),
             "fill_rate": round(len(filled) / max(len(wanted), 1), 6),
             "drawn": int(drawn), "acceptance_rate": round(accepted / max(drawn, 1), 6),
             "seconds": round(time.time() - t0, 2)}
    return res, stats


def fit_one(kind: str, t: pd.DataFrame, discrete, epochs: int, seed: int, gpu: bool):
    from ctgan import CTGAN, TVAE
    cls = CTGAN if kind == "ctgan" else TVAE
    m = cls(epochs=epochs, enable_gpu=gpu, batch_size=500)
    m.set_random_state(seed)
    t0 = time.time()
    m.fit(t, discrete_columns=discrete)
    return m, round(time.time() - t0, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir")
    p.add_argument("--out-dir", default=os.path.join(OUTPUTS_DIR, "baselines"))
    p.add_argument("--log-out", default=os.path.join(DOCS_DIR, "baselines_train_log.json"))
    p.add_argument("--models", default="tvae,ctgan")
    p.add_argument("--subsample", type=int, default=None, help="default: config baselines.subsample_rows")
    p.add_argument("--epochs", type=int, default=None, help="default: config baselines.epochs or 100")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mock", action="store_true", help="MOCK DATA smoke run (tests only)")
    args = p.parse_args(argv)
    cfg = load_config()
    args.seed = cfg.get("model", {}).get("seed", 42) if args.seed is None else args.seed
    if args.mock:
        from tests.mock_data import make_mock_v2
        print(f"*** {MOCK_BANNER} *** (smoke run)")
        raw = make_mock_v2(3000, seed=1)
    else:
        raw = read_parquet(split_path(data_dir(args.data_dir), "train"))
    tr, scope = apply_scope(raw, cfg)
    spec = build_spec(cfg, use_state=False, training_df=tr)
    arrays = make_arrays(tr, fit_preproc(raw_generated(tr, spec), spec))
    require_sex_support(arrays, cfg, mock=args.mock)
    args.subsample = args.subsample or int(cfg.get("baselines", {}).get("subsample_rows", 100_000))
    t = stratified_subsample(joint_table(tr, spec), args.subsample, args.seed)
    epochs = args.epochs or int(cfg.get("baselines", {}).get("epochs", 100))
    import torch
    gpu = torch.cuda.is_available() and not args.cpu
    os.makedirs(args.out_dir, exist_ok=True)
    log = {"created": datetime.now(timezone.utc).isoformat(), "mock": args.mock,
           "n_train_rows_used": int(len(t)), "subsample_target": args.subsample,
           "stratified_on": STRATIFY_ON, "epochs": epochs, "gpu": gpu,
           "columns": list(t.columns), "scope": scope["scope"], "models": {}}
    for kind in [k.strip() for k in args.models.split(",") if k.strip()]:
        print(f"training {kind} on {len(t):,} rows, {epochs} epochs ...")
        m, secs = fit_one(kind, t, discrete_columns(spec), epochs, args.seed, gpu)
        m.save(os.path.join(args.out_dir, f"{kind}.pkl"))
        # acceptance rate for a reference request: the train condition mix itself
        probe = t[spec.cond_names].sample(min(2000, len(t)), random_state=args.seed).reset_index(drop=True)
        _, st = rejection_sample(m, probe, spec.cond_names, args.seed, batch=20_000, max_factor=20)
        log["models"][kind] = {"train_seconds": secs, "probe_rejection_sampling": st}
        print(f"  {kind}: {secs}s, probe acceptance {st['acceptance_rate']}, fill {st['fill_rate']}")
    log["peak_memory_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)
    write_json(log, args.log_out)
    return log


if __name__ == "__main__":
    main()
