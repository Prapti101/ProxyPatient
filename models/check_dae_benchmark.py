"""
Correct DAE-vs-median benchmark (T2b). Human runs it on real data:

    PP_DATA_DIR=/path/to/processed python -m models.check_dae_benchmark

Why a new benchmark: P1's `benchmark_dae_vs_median` reads the already
DAE-imputed validation file, so its "known" cells include earlier DAE guesses
(circular), and it relies on hard-coded Windows paths.

Here:
  * the UN-imputed val_v2.parquet is used, and only cells that are truly
    known in it are eligible;
  * 10% of known waist_cm (and, separately, of known hip_cm) are masked;
  * the masked frame is imputed by P1's `impute_dae` with EXPLICIT
    weights_path / fit_stats_path, and by the TRUE train median (from the
    UN-imputed train_v2.parquet, known cells only);
  * RMSE of both, the column SD and the number of masked cells are reported.
Aggregates only. Output: docs/p2_dae_benchmark.json and .md.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from models.common import (DOCS_DIR, MOCK_BANNER, REPO_ROOT, as_num, data_dir, load_config,
                           md_table, read_parquet, split_path, write_json)

BENCH_COLS = ["waist_cm", "hip_cm"]


def benchmark(val_raw: pd.DataFrame, train_raw: pd.DataFrame, impute_fn,
              cols=BENCH_COLS, mask_frac: float = 0.10, seed: int = 42) -> dict:
    from models.privacy import suppress_count
    rng = np.random.default_rng(seed)
    out = {}
    for col in cols:
        v = as_num(val_raw[col])
        known = np.flatnonzero(np.isfinite(v.to_numpy(float)))
        n_mask = int(round(len(known) * mask_frac))
        train_values = as_num(train_raw[col])
        train_values = train_values[np.isfinite(train_values)]
        if n_mask < 30 or len(train_values) < 30:
            out[col] = {"status": "insufficient coverage", "n_known_val": suppress_count(len(known)),
                        "n_masked": suppress_count(n_mask), "n_dae_returned_nan": None, "train_median": None,
                        "val_known_sd": None, "rmse_dae": None, "rmse_median": None,
                        "dae_beats_median": None, "improvement_pct": None}
            continue
        idx = rng.choice(known, n_mask, replace=False)
        truth = v.iloc[idx].to_numpy(float)
        masked = val_raw.copy()
        masked.iloc[idx, masked.columns.get_loc(col)] = np.nan
        imputed = impute_fn(masked)
        pred_dae = as_num(imputed[col]).iloc[idx].to_numpy(float)
        med = float(train_values.median())
        # Both comparators must be scored on identical finite known targets.
        if not np.isfinite(pred_dae).all():
            raise ValueError("DAE failed predictions on masked targets; unequal-coverage RMSE is forbidden")
        rmse_dae = float(np.sqrt(np.mean((truth - pred_dae) ** 2)))
        rmse_med = float(np.sqrt(np.mean((truth - med) ** 2)))
        out[col] = {
            "n_known_val": int(len(known)), "n_masked": n_mask,
            "n_dae_returned_nan": suppress_count(int((~np.isfinite(pred_dae)).sum())),
            "train_median": round(med, 3), "val_known_sd": round(float(v.dropna().std()), 4),
            "rmse_dae": round(rmse_dae, 4), "rmse_median": round(rmse_med, 4),
            "dae_beats_median": bool(rmse_dae < rmse_med),
            "improvement_pct": round((rmse_med - rmse_dae) / rmse_med * 100, 2) if rmse_med else None,
        }
    return out


def report_md(res, mock) -> str:
    rows = [[c, r["n_masked"], r["val_known_sd"], r["rmse_median"], r["rmse_dae"],
             "DAE" if r["dae_beats_median"] else "MEDIAN", r["improvement_pct"]] for c, r in res.items()]
    verdict = []
    for c, r in res.items():
        if r.get("status") == "insufficient coverage":
            verdict.append(f"- {c}: insufficient coverage; statistics suppressed.")
            continue
        if not r["dae_beats_median"]:
            verdict.append(f"- **{c}: the DAE does NOT beat the train median** (RMSE {r['rmse_dae']} vs {r['rmse_median']}).")
        else:
            verdict.append(f"- {c}: DAE RMSE {r['rmse_dae']} vs median {r['rmse_median']} "
                           f"({r['improvement_pct']}% lower); column SD {r['val_known_sd']}.")
    return "\n".join([
        "# DAE vs median benchmark (P2, corrected)" + (" - MOCK DATA, not NFHS-5" if mock else ""), "",
        f"Generated {datetime.now(timezone.utc).isoformat()} by `models/check_dae_benchmark.py`.",
        "Un-imputed val_v2, truly known cells only, 10% masked per column (one column at a time), "
        "median = true median of known train_v2 cells.", "",
        md_table(rows, ["column", "n masked", "SD (val known)", "RMSE median", "RMSE DAE", "better", "improvement %"]),
        "", *verdict, "",
        "An RMSE close to the SD means the imputer is not much better than a constant.", ""])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir")
    p.add_argument("--weights", help="dae_weights_v2.pt (default <data-dir>/dae_weights_v2.pt)")
    p.add_argument("--fit-stats", help="dae_fit_stats_v2.pkl (default <data-dir>/dae_fit_stats_v2.pkl)")
    p.add_argument("--mask-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=os.path.join(DOCS_DIR, "p2_dae_benchmark.json"))
    p.add_argument("--mock", action="store_true", help="MOCK DATA smoke run with a toy imputer (tests only)")
    args = p.parse_args(argv)
    args.seed = load_config().get("model", {}).get("seed", 42) if args.seed is None else args.seed
    if args.mock:
        from tests.mock_data import make_mock_v2
        print(f"*** {MOCK_BANNER} *** (toy linear imputer stands in for the DAE)")
        val, train = make_mock_v2(3000, 5, imputed=False), make_mock_v2(3000, 6, imputed=False, banner=False)

        def impute_fn(df):
            out = df.copy()
            for col, other in [("waist_cm", "hip_cm"), ("hip_cm", "waist_cm")]:
                ok = out[[col, other]].notna().all(1)
                a, b = np.polyfit(out.loc[ok, other], out.loc[ok, col], 1)
                m = out[col].isna()
                out.loc[m, col] = (a * out.loc[m, other] + b).fillna(out[col].median())
            return out
    else:
        d = data_dir(args.data_dir)
        val = read_parquet(split_path(d, "val", imputed=False))
        train = read_parquet(split_path(d, "train", imputed=False))
        sys.path.insert(0, REPO_ROOT)
        from dae_impute import impute_dae   # P1's inference function
        w = args.weights or os.path.join(d, "dae_weights_v2.pt")
        fs = args.fit_stats or os.path.join(d, "dae_fit_stats_v2.pkl")

        def impute_fn(df):
            return impute_dae(df, weights_path=w, fit_stats_path=fs)
    res = benchmark(val, train, impute_fn, mask_frac=args.mask_frac, seed=args.seed)
    write_json({"mock": args.mock, "results": res}, args.out)
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write(report_md(res, args.mock))
    print(report_md(res, args.mock))
    return res


if __name__ == "__main__":
    main()
