"""
Development evaluation of CVAE vs TVAE vs CTGAN on the VALIDATION split.

Real data:   PP_DATA_DIR=/path python -m models.eval_dev
Final test:  PP_DATA_DIR=/path python -m models.eval_dev --final-test   (ONCE, at the very end)
Mock smoke:  python -m models.eval_dev --mock --cvae-weights W --marginals M --baselines-dir B --out-json X

For every real evaluation row, each model generates one row with the same
condition values (CVAE: conditional decoder; baselines: rejection sampling), so
all models produce the same number of rows for the same condition mix. Rows a
baseline could not fill are dropped for ALL models (fill rates are reported).

Metrics (aggregate only): (1) per-column KS / Wasserstein (TVD for categorical)
(2) correlation-matrix difference (3) glucose tail quantiles and share >= threshold
(4) conditional outcome-rate fidelity over cells with >= 500 real rows
(5) direction checks (6) condition consistency (7) nearest-real-record distance
vs a real-to-real baseline (8) train-on-synthetic, test-on-real AUC.
"""

import argparse
import itertools
import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from models.common import (AGE3_LABELS, DOCS_DIR, MOCK_BANNER, MODELS_DIR, OUTPUTS_DIR,
                           bmi_band_names, data_dir, load_config, md_table, min_cell,
                           read_parquet, split_path, write_json)
from models.data import (apply_scope, build_spec, condition_frame, encode_conditions,
                         raw_generated, state_index)

CONT = ["age", "bmi", "height_cm", "weight_kg", "waist_cm", "hip_cm", "glucose_raw"]
CAT = ["education", "bp_ever_checked"]
MIN_CELL_EVAL = 500
DIRECTION_VARS = ["age_band", "bmi_band", "wealth_quintile", "residence", "tobacco",
                  "alcohol", "hypertension", "sex"]
ORDINAL = {"age_band", "bmi_band", "wealth_quintile"}


# ── real side ────────────────────────────────────────────────────────────────

def real_units(df: pd.DataFrame, spec) -> pd.DataFrame:
    cf = condition_frame(df, spec)
    gen = raw_generated(df, spec)
    out = cf.copy()
    for c in spec.cont_cols:
        if c == "log_glucose":
            out["glucose_raw"] = np.exp(gen[c])
        else:
            out[c] = gen[c]
    for c in spec.cat_cols:
        out[c] = gen[c]
    if spec.weight_derived:
        out["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    return out


def complete_rows(ru: pd.DataFrame, spec) -> pd.Series:
    cols = list(spec.cond_names) + [c for c in CONT + CAT if c in ru]
    return ru[cols].notna().all(1)


def outcome(df, thr):
    values = df["glucose_raw"].astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite glucose cannot be scored as outcome zero")
    return (values >= thr).astype(float)


# ── generation per model ─────────────────────────────────────────────────────

def gen_cvae(path, real_df, cf, spec_eval, seed):
    from models.sampling import CVAEBundle
    b = CVAEBundle.load(path, "cpu")
    cond_idx = encode_conditions(cf, b.spec)
    st = state_index(real_df, b.spec) if b.spec.use_state else None
    if st is not None:
        if (st < 0).any():
            raise ValueError("Evaluation contains unsupported state codes; no substitution is allowed")
    t0 = time.time()
    g = b.sample(cond_idx, st, seed=seed)
    secs = time.time() - t0
    out = cf.reset_index(drop=True).copy()
    for c in g.columns:
        out[c] = g[c].to_numpy()
    return out, {"sampling": g.attrs.get("sampling"), "generate_seconds": round(secs, 2)}


def gen_baseline(path, cf, spec, cfg, seed):
    from ctgan import CTGAN
    from models.train_baselines import rejection_sample, to_units
    m = CTGAN.load(path)          # works for TVAE pickles too (generic pickle load)
    wanted = cf.reset_index(drop=True)
    res, st = rejection_sample(m, wanted, spec.cond_names, seed, sample_columns=spec.cond_names + spec.cont_cols + spec.cat_cols)
    u = to_units(res.drop(columns=spec.cond_names), cfg)
    out = wanted.copy()
    for c in u.columns:
        out[c] = u[c].to_numpy()
    out.loc[res.isna().all(1).to_numpy(), [c for c in u.columns]] = np.nan
    return out, {"rejection_sampling": st}


# ── metrics ──────────────────────────────────────────────────────────────────

def marginal_metrics(real, gen):
    out = {}
    for c in CONT:
        if c not in real or c not in gen:
            continue
        r, g = real[c].dropna().to_numpy(float), gen[c].dropna().to_numpy(float)
        sd = r.std() or 1.0
        out[c] = {"ks": round(float(ks_2samp(r, g).statistic), 5),
                  "wasserstein": round(float(wasserstein_distance(r, g)), 5),
                  "wasserstein_over_sd": round(float(wasserstein_distance(r, g) / sd), 5)}
    for c in CAT:
        pr = real[c].value_counts(normalize=True); pg = gen[c].value_counts(normalize=True)
        idx = pr.index.union(pg.index)
        out[c] = {"tvd": round(float(0.5 * (pr.reindex(idx, fill_value=0) - pg.reindex(idx, fill_value=0)).abs().sum()), 5)}
    return out


def numeric_matrix(df, spec, cfg):
    m = pd.DataFrame(index=df.index)
    for c in spec.cond_names:
        lv = spec.cond_levels[c]
        m[c] = df[c].map({lab: i for i, lab in enumerate(lv)}).astype(float)
    for c in CONT + CAT:
        if c not in df:
            continue
        m[c] = df[c].astype(float)
    m["glucose_raw"] = np.log(m["glucose_raw"])
    return m


def corr_metrics(real, gen, spec, cfg):
    a = numeric_matrix(real, spec, cfg).corr().fillna(0).to_numpy()
    b = numeric_matrix(gen, spec, cfg).corr().fillna(0).to_numpy()
    d = a - b
    return {"frobenius": round(float(np.linalg.norm(d)), 5), "max_abs": round(float(np.abs(d).max()), 5)}


def tail_metrics(df, thr):
    g = df["glucose_raw"].dropna()
    q = g.quantile([0.5, 0.9, 0.95, 0.99])
    return {"p50": float(q[0.5]), "p90": float(q[0.9]), "p95": float(q[0.95]), "p99": float(q[0.99]),
            "share_ge_threshold": round(float((g >= thr).mean()), 6)}


def cell_errors(real, gen, thr, vars_, k=None):
    k = k or MIN_CELL_EVAL
    yr, yg = outcome(real, thr), outcome(gen, thr)
    rows = []
    if len(real) == 0:
        return rows
    for combo in itertools.chain(((v,) for v in vars_), itertools.combinations(vars_, 2)):
        key = real[list(combo)].astype(str).agg("|".join, axis=1)
        grp = pd.DataFrame({"k": key, "r": yr, "g": yg}).groupby("k")
        agg = grp.agg(n=("r", "size"), real=("r", "mean"), gen=("g", "mean"))
        for cell, r in agg[agg["n"] >= k].iterrows():
            rows.append({"vars": "+".join(combo), "cell": cell, "n_real": int(r["n"]),
                         "real_rate": round(float(r["real"]), 6), "gen_rate": round(float(r["gen"]), 6),
                         "abs_err_pp": round(abs(float(r["real"]) - float(r["gen"])) * 100, 4)})
    return rows


def summarise_errors(rows):
    if not rows:
        return {"n_cells": 0, "mean_abs_err_pp": None, "max_abs_err_pp": None}
    e = np.array([r["abs_err_pp"] for r in rows])
    s1 = [r["abs_err_pp"] for r in rows if "+" not in r["vars"]]
    s2 = [r["abs_err_pp"] for r in rows if "+" in r["vars"]]
    return {"n_cells": len(rows), "mean_abs_err_pp": round(float(e.mean()), 4),
            "max_abs_err_pp": round(float(e.max()), 4),
            "single_var_mean_pp": round(float(np.mean(s1)), 4) if s1 else None,
            "two_var_mean_pp": round(float(np.mean(s2)), 4) if s2 else None}


def conditional_fidelity(real, gen, spec, thr):
    rows = cell_errors(real, gen, thr, spec.cond_names)
    out = {"overall": summarise_errors(rows), "cells": rows, "by_sex": {}}
    others = [c for c in spec.cond_names if c != "sex"]
    for s in ("0", "1"):
        m = (real["sex"] == s).to_numpy()
        out["by_sex"][s] = summarise_errors(cell_errors(real[m], gen[m], thr, others))
    return out


def direction_checks(real, gen, spec, thr):
    yr, yg = outcome(real, thr), outcome(gen, thr)
    out = {}
    for v in DIRECTION_VARS:
        levels = spec.cond_levels[v]
        rr = yr.groupby(real[v]).mean().reindex(levels)
        rg = yg.groupby(gen[v]).mean().reindex(levels)
        n = real[v].value_counts().reindex(levels, fill_value=0)
        rr[n < MIN_CELL_EVAL] = np.nan     # too few real rows for a stable rate
        ok = rr.notna() & rg.notna()
        entry = {"levels": levels, "real_rate": [None if pd.isna(x) else round(float(x), 6) for x in rr],
                 "gen_rate": [None if pd.isna(x) else round(float(x), 6) for x in rg]}
        if ok.sum() >= 2:
            ra, ga = rr[ok].to_numpy(), rg[ok].to_numpy()
            entry["direction_match"] = bool(np.sign(ra[-1] - ra[0]) == np.sign(ga[-1] - ga[0]))
            if v in ORDINAL and ok.sum() >= 3:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho = spearmanr(ra, ga).statistic
                entry["spearman_real_vs_gen"] = None if np.isnan(rho) else round(float(rho), 4)
        else:
            entry["direction_match"] = None
        out[v] = entry
    vals = [e["direction_match"] for e in out.values() if e["direction_match"] is not None]
    out["_summary"] = {"matched": int(sum(vals)), "checked": len(vals)}
    return out


def consistency(gen, cfg):
    age_ok, bmi_ok = [], []
    names = bmi_band_names(cfg)
    for (sx, a3), g in gen.groupby(["sex", "age_band"]):
        bands = cfg["age_bands"]["men" if sx == "1" else "women"]
        lo, hi = bands[AGE3_LABELS.index(a3)]
        age_ok.append(((g["age"] >= lo) & (g["age"] <= hi)).to_numpy())
    for b, g in gen.groupby("bmi_band"):
        lo, hi = cfg["bmi"]["bands"][b]
        bmi_ok.append(((g["bmi"] >= lo) & (g["bmi"] < hi)).to_numpy())
    a = np.concatenate(age_ok) if age_ok else np.array([])
    b = np.concatenate(bmi_ok) if bmi_ok else np.array([])
    return {"age_in_band_share": round(float(a.mean()), 6) if len(a) else None,
            "bmi_in_band_share": round(float(b.mean()), 6) if len(b) else None}


def _feat(df, spec, mu=None, sd=None):
    m = numeric_matrix(df, spec, None)
    if mu is None:
        mu, sd = m.mean(), m.std().replace(0, 1)
    return ((m - mu) / sd).fillna(0).to_numpy(), mu, sd


def dcr(train_real, val_real, gen, spec, seed, n_ref=50_000, n_q=2_000):
    from sklearn.neighbors import NearestNeighbors
    ref = train_real.sample(min(n_ref, len(train_real)), random_state=seed)
    X, mu, sd = _feat(ref, spec)
    nn_ = NearestNeighbors(n_neighbors=1).fit(X)

    def stats(q):
        q = q.sample(min(n_q, len(q)), random_state=seed)
        d = nn_.kneighbors(_feat(q, spec, mu, sd)[0])[0][:, 0]
        return {"median": round(float(np.median(d)), 5), "p5": round(float(np.percentile(d, 5)), 5),
                "share_exact_copy": round(float((d < 1e-9).mean()), 6), "n_query": int(len(q))}
    return {"reference": f"{len(ref):,} in-scope TRAIN rows (standardised numeric space)",
            "real_val_to_train": stats(val_real), "generated_to_train": stats(gen)}


def _lr_X(df, spec):
    X = pd.get_dummies(df[spec.cond_names].astype(str), drop_first=False)
    for c in ["age", "bmi", "height_cm", "waist_cm", "hip_cm", "education", "bp_ever_checked"]:
        if c in df:
            X[c] = df[c].astype(float)
    return X


def tstr(train_real, test_real, gen, spec, thr, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xt = _lr_X(test_real, spec); yt = outcome(test_real, thr)

    def fit_auc(df):
        y = outcome(df, thr)
        if y.nunique() < 2 or yt.nunique() < 2:
            return None
        X = _lr_X(df, spec).reindex(columns=Xt.columns, fill_value=0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X, y)
        return round(float(roc_auc_score(yt, clf.predict_proba(Xt)[:, 1])), 5)
    real_sub = train_real.sample(min(len(gen), len(train_real)), random_state=seed)
    return {"auc_train_synthetic_test_real": fit_auc(gen),
            "auc_train_real_test_real": fit_auc(real_sub),
            "features": "conditions + age, bmi, height, waist, hip, education, bp_ever_checked (no glucose)"}


def coverage_report(conditions, retained):
    from models.privacy import suppress_count, suppress_stat
    retained = np.asarray(retained, dtype=bool)
    cells = conditions.astype(str).agg("|".join, axis=1)
    rows = []
    for key, indexes in cells.groupby(cells).groups.items():
        indexes = np.asarray(list(indexes), dtype=int)
        requested, kept = len(indexes), int(retained[indexes].sum())
        rows.append({"condition_cell": key, "n_requested": suppress_count(requested),
                     "n_retained": suppress_count(kept),
                     "retained_share": suppress_stat(kept/requested, requested)})
    return {"n_requested": suppress_count(len(conditions)), "n_retained": suppress_count(int(retained.sum())),
            "retained_share": suppress_stat(float(retained.mean()) if len(retained) else None, len(retained)),
            "condition_cells": rows}


# ── driver ───────────────────────────────────────────────────────────────────

def run(args, cfg):
    from models.privacy import safe_public_output, suppress_count
    thr = float(cfg["outcome"]["threshold_mg_dl"])
    spec = None
    eval_split = "test" if args.final_test else "val"
    if args.mock:
        from tests.mock_data import make_mock_v2
        print(f"*** {MOCK_BANNER} *** (smoke run; numbers are meaningless)")
        tr_raw, ev_raw = make_mock_v2(4000, seed=1), make_mock_v2(args.n_eval or 3000, seed=3, banner=False)
    else:
        d = data_dir(args.data_dir)
        tr_raw = read_parquet(split_path(d, "train"))
        ev_raw = read_parquet(split_path(d, eval_split, final_test=args.final_test), final_test=args.final_test)
    tr, _ = apply_scope(tr_raw, cfg)
    if args.cvae_weights and os.path.isfile(args.cvae_weights):
        from models.sampling import CVAEBundle
        from models.artifacts import validate_checkpoint
        checkpoint = CVAEBundle.load(args.cvae_weights)
        validate_checkpoint(checkpoint.ckpt, cfg)
        if checkpoint.ckpt['is_mock'] != bool(args.mock):
            raise ValueError("Evaluation mode does not match checkpoint MOCK provenance")
        spec = checkpoint.spec
    else:
        spec = build_spec(cfg, use_state=False, training_df=tr)
    ev, scope_rep = apply_scope(ev_raw, cfg)
    tr_u = real_units(tr, spec); tr_u = tr_u[complete_rows(tr_u, spec)]
    ev_u = real_units(ev, spec); keep = complete_rows(ev_u, spec)
    ev, ev_u = ev[keep.to_numpy()], ev_u[keep]
    if args.n_eval and len(ev_u) > args.n_eval:
        idx = ev_u.sample(args.n_eval, random_state=args.seed).index
        ev, ev_u = ev.loc[idx], ev_u.loc[idx]
    ev_u = ev_u.reset_index(drop=True); ev = ev.reset_index(drop=True)
    cf = ev_u[spec.cond_names]

    gens, info = {}, {}
    if args.cvae_weights and os.path.exists(args.cvae_weights):
        gens["CVAE"], info["CVAE"] = gen_cvae(args.cvae_weights, ev, cf, spec, args.seed)
    for extra in [w for w in (args.extra_cvae or []) if w]:
        name, path = extra.split("=", 1)
        if os.path.exists(path):
            gens[name], info[name] = gen_cvae(path, ev, cf, spec, args.seed)
    for kind, name in [("tvae", "TVAE"), ("ctgan", "CTGAN")]:
        p = os.path.join(args.baselines_dir, f"{kind}.pkl")
        if os.path.exists(p):
            gens[name], info[name] = gen_baseline(p, cf, spec, cfg, args.seed)
    if not gens:
        raise RuntimeError("No models found to evaluate (check --cvae-weights / --baselines-dir).")

    filled = np.ones(len(ev_u), dtype=bool)
    for g in gens.values():
        filled &= g["glucose_raw"].notna().to_numpy()
    coverage = {name: coverage_report(cf, np.isfinite(g["glucose_raw"].to_numpy(float))) for name, g in gens.items()}
    shared_coverage = coverage_report(cf, filled)
    real = ev_u[filled].reset_index(drop=True)
    if len(real) < min_cell(cfg) or len(tr_u) < min_cell(cfg):
        res = {"_meta": {"status": "insufficient coverage", "mock": bool(args.mock),
                         "n_eval_rows_requested": suppress_count(len(ev_u)),
                         "n_eval_rows_all_models_filled": suppress_count(len(real))},
               "models": {name: {"status": "insufficient coverage", "metrics": None} for name in gens}}
        write_json(res, args.out_json)
        with open(os.path.splitext(args.out_json)[0] + ".md", "w", encoding="utf-8") as f:
            f.write("# Evaluation: insufficient coverage\nStatistics suppressed (fewer than 30 respondents).\n")
        return res
    res = {"_meta": {"created": datetime.now(timezone.utc).isoformat(), "mock": bool(args.mock),
                     "split": eval_split, "n_eval_rows_requested": int(len(ev_u)),
                     "n_eval_rows_all_models_filled": int(filled.sum()),
                     "min_cell_for_conditional_metrics": MIN_CELL_EVAL, "threshold_mg_dl": thr,
                     "scope": scope_rep["scope"], "seed": args.seed,
                     "metrics_scope": "retained subset filled by every evaluated model, not full requested scope",
                     "shared_coverage": shared_coverage,
                     "comparison_note": "CVAE: full scoped TRAIN with optional state; TVAE/CTGAN: stratified TRAIN subsample without state. Different data/state use, not a controlled architecture-only comparison.",
                     "utility_note": "TSTR uses held-out matched conditions (transductive utility), not independent cohort TSTR"},
           "real": {"tail": tail_metrics(real, thr)}, "models": {}}
    for name, g in gens.items():
        g = g[filled].reset_index(drop=True)
        print(f"evaluating {name} on {len(g):,} rows ...")
        res["models"][name] = {
            "generation": info[name],
            "requested_vs_retained_coverage": coverage[name],
            "metrics_scope": "shared retained subset",
            "marginals": marginal_metrics(real, g),
            "correlation": corr_metrics(real, g, spec, cfg),
            "tail": tail_metrics(g, thr),
            "conditional_rate_fidelity": conditional_fidelity(real, g, spec, thr),
            "direction": direction_checks(real, g, spec, thr),
            "consistency": consistency(g, cfg),
            "nearest_record": dcr(tr_u, real, g, spec, args.seed),
            "tstr": tstr(tr_u, real, g, spec, thr, args.seed),
        }
    res = safe_public_output(res)
    write_json(res, args.out_json)
    with open(os.path.splitext(args.out_json)[0] + ".md", "w", encoding="utf-8") as f:
        f.write(summary_md(res))
    print(f"wrote {args.out_json}")
    return res


def summary_md(res) -> str:
    m = res["_meta"]
    head = [f"# Model comparison ({m['split'].upper()} split{', MOCK DATA, not NFHS-5' if m['mock'] else ''})", "",
            f"Created {m['created']}. Rows evaluated: {m['n_eval_rows_all_models_filled']:,} "
            f"(of {m['n_eval_rows_requested']:,} requested; rows a baseline could not fill by rejection "
            "sampling are dropped for all models). Outcome: elevated glucose (proxy), "
            f"glucose >= {m['threshold_mg_dl']:.0f} mg/dL. Aggregates only.",
            m["metrics_scope"], m["comparison_note"], m["utility_note"], ""]
    rows = []
    rt = res["real"]["tail"]
    rows.append(["REAL", None, None, rt["share_ge_threshold"] * 100, rt["p95"], rt["p99"], None, None, None, None, None, None])
    for name, r in res["models"].items():
        mk = r["marginals"]
        ks = np.mean([v["ks"] for k, v in mk.items() if "ks" in v])
        cf = r["conditional_rate_fidelity"]["overall"]
        rows.append([name, round(float(ks), 4), r["correlation"]["frobenius"],
                     r["tail"]["share_ge_threshold"] * 100, r["tail"]["p95"], r["tail"]["p99"],
                     cf["mean_abs_err_pp"], cf["max_abs_err_pp"],
                     f"{r['direction']['_summary']['matched']}/{r['direction']['_summary']['checked']}",
                     r["consistency"]["age_in_band_share"], r["consistency"]["bmi_in_band_share"],
                     r["tstr"]["auc_train_synthetic_test_real"]])
    tbl = md_table(rows, ["model", "mean KS", "corr diff (Frob.)", "% >= threshold", "P95", "P99",
                          "cond. MAE pp", "cond. max pp", "directions", "age in band", "BMI in band", "TSTR AUC"])
    extra = ["", "TRTR AUC (real train -> real eval): " + ", ".join(
        f"{n}: {r['tstr']['auc_train_real_test_real']}" for n, r in res["models"].items()), "",
        "Nearest-record distance, median (generated -> train vs real eval -> train): " + ", ".join(
        f"{n}: {r['nearest_record']['generated_to_train']['median']} vs "
        f"{r['nearest_record']['real_val_to_train']['median']}" for n, r in res["models"].items()), "",
        "Full per-column, per-cell and per-sex numbers are in the JSON file.", ""]
    return "\n".join(head + [tbl] + extra)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir")
    p.add_argument("--cvae-weights", default=os.path.join(MODELS_DIR, "cvae_weights.pt"))
    p.add_argument("--extra-cvae", action="append",
                   help="NAME=path of another CVAE checkpoint, e.g. CVAE-gru=outputs/cvae_gru.pt")
    p.add_argument("--baselines-dir", default=os.path.join(OUTPUTS_DIR, "baselines"))
    p.add_argument("--n-eval", type=int, default=50_000, help="eval rows (subsample of the split)")
    p.add_argument("--out-json", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-eval-cell", type=int, default=500,
                   help="min real rows per cell for conditional metrics (default 500)")
    p.add_argument("--final-test", action="store_true",
                   help="evaluate on the LOCKED test split. Run once, at the very end.")
    p.add_argument("--mock", action="store_true", help="MOCK DATA smoke run (tests only)")
    args = p.parse_args(argv)
    cfg = load_config()
    args.seed = cfg.get("model", {}).get("seed", 42) if args.seed is None else args.seed
    if args.out_json is None:
        args.out_json = os.path.join(DOCS_DIR, "model_comparison_final_test.json" if args.final_test
                                     else "model_comparison_dev.json")
    global MIN_CELL_EVAL
    MIN_CELL_EVAL = max(args.min_eval_cell, min_cell(load_config()))
    return run(args, load_config())


if __name__ == "__main__":
    main()
