"""
Step 0: real-data checks (run on the human's machine, never in the cloud).

    PP_DATA_DIR=/path/to/processed python -m models.step0_checks
    python -m models.step0_checks --mock --out /tmp/x.md      (MOCK DATA smoke run)

Writes docs/p2_step0_checks.md (+ .json). Aggregates only: shapes, dtypes,
counts, rates, quantiles. Any cell backed by fewer than privacy.min_cell_size
respondents is suppressed. No row-level output. Test files are neither read
nor hashed unless --final-test is given.
"""

import argparse
import itertools
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from models.common import (AGE3_LABELS, DOCS_DIR, MOCK_BANNER, age3_index, as_num, as_str,
                           data_dir, load_config, md_table, min_cell, read_parquet,
                           sha256_file, split_path, write_json)
from models.data import apply_scope, build_spec, condition_frame

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def manifest_hashes(obj, out=None, key=None):
    """Collect {filename: sha256} from a MANIFEST.json of unknown layout."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        name = obj.get("file") or obj.get("filename") or obj.get("name") or obj.get("path")
        sha = obj.get("sha256") or obj.get("sha") or obj.get("hash")
        if isinstance(name, str) and isinstance(sha, str) and HEX64.match(sha):
            out[os.path.basename(name)] = sha.lower()
        for k, v in obj.items():
            if isinstance(v, str) and HEX64.match(v) and "." in str(k):
                out[os.path.basename(k)] = v.lower()
            else:
                manifest_hashes(v, out, k)
    elif isinstance(obj, list):
        for v in obj:
            manifest_hashes(v, out, key)
    return out


def check_manifest(ddir, final_test):
    path = os.path.join(ddir, "MANIFEST.json")
    if not os.path.exists(path):
        return [{"file": "MANIFEST.json", "status": "MISSING"}]
    with open(path, encoding="utf-8") as f:
        expected = manifest_hashes(json.load(f))
    rows = []
    names = sorted(set(expected) | {f for f in os.listdir(ddir) if f.endswith((".parquet", ".pkl", ".pt"))})
    for name in names:
        fp = os.path.join(ddir, name)
        if "test" in name.lower() and not final_test:
            rows.append({"file": name, "status": "LOCKED (not hashed without --final-test)"})
            continue
        if not os.path.exists(fp):
            rows.append({"file": name, "status": "listed in manifest, file missing"})
            continue
        got = sha256_file(fp)
        exp = expected.get(name)
        rows.append({"file": name, "sha256_prefix": got[:16],
                     "status": "OK" if exp == got else ("NOT IN MANIFEST" if exp is None else "MISMATCH")})
    return rows


def cnt(x, k):
    """A count is itself a cell: show 0, the count if >= k, else '<k'."""
    from models.privacy import suppress_count
    return None if x is None else suppress_count(x, k)


def rate_table(df, cols, y, k):
    """n and outcome rate for every level of each single variable and each pair."""
    rows = []
    for combo in itertools.chain(((c,) for c in cols), itertools.combinations(cols, 2)):
        key = df[list(combo)].astype(str).agg("|".join, axis=1)
        g = pd.DataFrame({"k": key, "y": y}).dropna().groupby("k")["y"].agg(["size", "mean"])
        for cell, r in g.iterrows():
            n = int(r["size"])
            rows.append(["+".join(combo), cell, n if n >= k else None,
                         round(float(r["mean"]) * 100, 3) if n >= k else None])
    return rows


def split_section(name, raw, imp, cfg, k):
    out, md = {}, [f"## Split: {name}", ""]
    md.append(f"Rows: raw {cnt(len(raw), k) if raw is not None else None}, DAE-imputed {cnt(len(imp), k)}; columns: {len(imp.columns)}.")
    md.append("")
    cols = list(imp.columns)
    rows = []
    for c in cols:
        rn = cnt(raw[c].isna().sum(), k) if raw is not None and c in raw else None
        rows.append([c, str(imp[c].dtype), rn, cnt(imp[c].isna().sum(), k)])
    md += ["### Columns, dtypes, NaN counts (raw v2 vs DAE-imputed v2)", "",
           md_table(rows, ["column", "dtype (imputed)", "NaN raw", "NaN imputed"]), ""]
    if raw is not None:
        extra = sorted(set(raw.columns) ^ set(imp.columns))
        md.append(f"Columns present in only one of raw/imputed: {extra or 'none'}")
        md.append("")
    out["nan"] = {r[0]: {"raw": r[2], "imputed": r[3]} for r in rows}

    df = imp
    sex = as_num(df["sex"])
    g = as_num(df["glucose_raw"])
    thr = float(cfg["outcome"]["threshold_mg_dl"])
    cav = []
    for s, lab in [(0, "women"), (1, "men")]:
        m = sex == s
        n = int(m.sum())
        gs = g[m].dropna()
        cav.append([f"{lab}: rows", n if n >= k else None])
        cav.append([f"{lab}: glucose known", int(len(gs)) if len(gs) >= k else None])
        if len(gs) >= k:
            q = gs.quantile([0.05, 0.5, 0.95, 0.99])
            cav.append([f"{lab}: glucose P5/P50/P95/P99", " / ".join(f"{v:.1f}" for v in q)])
            cav.append([f"{lab}: share glucose >= {thr:.0f}", round(float((gs >= thr).mean()), 5)])
        for col in ["any_tobacco", "alcohol", "hypertension", "bmi_measured", "bp_measured"]:
            v = as_num(df.loc[m, col]).dropna()
            cav.append([f"{lab}: mean {col} (non-missing n={len(v) if len(v) >= k else 'suppressed'})",
                        round(float(v.mean()), 5) if len(v) >= k else None])
        b = as_num(df.loc[m, "bmi"]).dropna()
        if len(b) >= k:
            cav.append([f"{lab}: BMI P1/P5/P50/P95/P99", " / ".join(f"{v:.2f}" for v in b.quantile([.01, .05, .5, .95, .99]))])
        for col in ["height_cm", "weight_kg", "waist_cm", "hip_cm"]:
            v = as_num(df.loc[m, col]).dropna()
            if len(v) >= k:
                cav.append([f"{lab}: {col} P1/P50/P99", " / ".join(f"{x:.1f}" for x in v.quantile([.01, .5, .99]))])
    med = g.median() if g.notna().sum() >= k else np.nan
    unit = "mg/dL" if med > 40 else ("mmol/L?" if med < 15 else "UNCLEAR")
    cav.append(["glucose median (all) -> unit verdict", f"{med:.1f} -> {unit}" if np.isfinite(med) else None])
    sw = as_num(df["sex"]).value_counts().to_dict()
    cav.append(["glucose known: women + men == total known?",
                f"{cnt(int((g.notna() & (sex == 0)).sum()), k)} + {cnt(int((g.notna() & (sex == 1)).sum()), k)} vs {cnt(int(g.notna().sum()), k)}"])
    cav.append(["sex values present", sorted(str(int(x)) for x in sw)])
    ep = as_num(df["elevated_glucose_proxy"])
    both = g.notna() & ep.notna()
    cav.append(["rows where elevated_glucose_proxy != (glucose_raw >= threshold)",
                cnt(int(((g[both] >= thr).astype(int) != ep[both].astype(int)).sum()), k)])
    cav.append(["rows with glucose known but proxy missing (or reverse)", cnt(int((g.notna() ^ ep.notna()).sum()), k)])
    a3 = age3_index(df["age"])
    lab_ok = 0
    for s, key in [(0, "women"), (1, "men")]:
        labels = cfg["whatif_options"]["age_band"][f"{key}_options"]
        m = (sex == s) & a3.notna()
        exp = a3[m].astype(int).map(lambda i: labels[i])
        lab_ok += int((as_str(df.loc[m, "age_band"]) != exp.str.lower()).sum())
    cav.append(["age_band label disagrees with numeric age (sex-specific bands)", cnt(lab_ok, k)])
    for s, key in [(0, "women"), (1, "men")]:
        lo, hi = cfg["age_bands"][key][0][0], cfg["age_bands"][key][-1][1]
        a = as_num(df.loc[sex == s, "age"])
        cav.append([f"{key}: ages outside [{lo}, {hi}]", cnt(int(((a < lo) | (a > hi)).sum()), k)])
    bm = as_num(df["bmi"]); bmf = as_num(df["bmi_measured"])
    cav.append(["bmi missing but bmi_measured==1", cnt(int((bm.isna() & (bmf == 1)).sum()), k)])
    cav.append(["bmi present but bmi_measured==0", cnt(int((bm.notna() & (bmf == 0)).sum()), k)])
    cav.append(["bmi_band labels", sorted(as_str(df["bmi_band"]).dropna().unique().tolist())])
    cav.append(["residence labels", sorted(as_str(df["residence"]).dropna().unique().tolist())])
    st = as_num(df["state"])
    cav.append(["state codes: n distinct / range", f"{st.nunique()} / {st.min():.0f}-{st.max():.0f}"])
    md += ["### Caveat checks", "", md_table(cav, ["check", "value"], "suppressed (n<30)"), ""]
    out["caveats"] = {r[0]: r[1] for r in cav}
    return out, md


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir")
    p.add_argument("--out", default=os.path.join(DOCS_DIR, "p2_step0_checks.md"))
    p.add_argument("--final-test", action="store_true", help="also hash and check the LOCKED test split")
    p.add_argument("--mock", action="store_true", help="MOCK DATA smoke run (tests only)")
    args = p.parse_args(argv)
    cfg = load_config()
    k = min_cell(cfg)
    md = ["# P2 Step 0: real-data checks" + (" (MOCK DATA, not NFHS-5)" if args.mock else ""), "",
          f"Generated {datetime.now(timezone.utc).isoformat()} by `models/step0_checks.py`. "
          f"Aggregates only; cells with n < {k} are suppressed.", ""]
    res = {"mock": args.mock}
    splits = ["train", "val"] + (["test"] if args.final_test else [])
    frames = {}
    if args.mock:
        from tests.mock_data import make_mock_v2
        print(f"*** {MOCK_BANNER} ***")
        for i, s in enumerate(splits):
            frames[s] = (make_mock_v2(3000, seed=10 + i, imputed=False, banner=False),
                         make_mock_v2(3000, seed=10 + i, imputed=True, banner=False))
        md += ["## File integrity", "", "MOCK run: MANIFEST check skipped.", ""]
    else:
        d = data_dir(args.data_dir)
        man = check_manifest(d, args.final_test)
        res["manifest"] = man
        md += ["## File integrity (SHA-256 vs MANIFEST.json)", "",
               md_table([[r["file"], r.get("sha256_prefix"), r["status"]] for r in man],
                        ["file", "sha256 (first 16)", "status"]), ""]
        for s in splits:
            ft = s == "test"
            raw = read_parquet(split_path(d, s, imputed=False, final_test=ft), final_test=ft)
            imp = read_parquet(split_path(d, s, imputed=True, final_test=ft), final_test=ft)
            frames[s] = (raw, imp)
    res["splits"] = {}
    for s, (raw, imp) in frames.items():
        o, m = split_section(s, raw, imp, cfg, k)
        res["splits"][s] = o
        md += m

    # scope, cells, rates: TRAIN only (val/test would be the same picture)
    tr = frames["train"][1]
    scoped, rep = apply_scope(tr, cfg)
    res["scope"] = {**rep, "steps": [{kk: (cnt(v, k) if kk.startswith("n_") else v) for kk, v in st_.items()}
                                     for st_ in rep["steps"]]}
    md += ["## Training scope (train split)", "",
           f"Scope `{rep['scope']}`: {rep['n_in_scope']:,} of {rep['n_input']:,} rows in scope "
           f"({rep['in_scope_share']:.2%}).", "",
           md_table([[s["rule"], cnt(s["n_excluded"], k), cnt(s["n_excluded_women"], k), cnt(s["n_excluded_men"], k)]
                     for s in rep["steps"]],
                    ["rule (applied in order)", "excluded", "women", "men"]), "",
           "Share excluded by sex and harmonised age band:", "",
           md_table([[kk, v] for kk, v in rep["excluded_share_by_sex_age"].items()], ["cell", "share excluded"],
                    "suppressed (n<30)"), ""]
    spec = build_spec(cfg)
    cf = condition_frame(scoped, spec)
    y = as_num(scoped["elevated_glucose_proxy"])
    full = cf.dropna()
    sizes = full.astype(str).agg("|".join, axis=1).value_counts()
    n_possible = int(np.prod([len(spec.cond_levels[c]) for c in spec.cond_names]))
    cells = {"possible_full_condition_cells": n_possible, "observed": int(len(sizes)),
             "n<30": int((sizes < 30).sum()), "30<=n<500": int(((sizes >= 30) & (sizes < 500)).sum()),
             "n>=500": int((sizes >= 500).sum()), "never_observed": n_possible - int(len(sizes))}
    res["full_condition_cells"] = cells
    md += ["## Condition cell sizes (in-scope train, all 8 conditions jointly)", "",
           md_table([[a, b] for a, b in cells.items()], ["", "count"]), "",
           "Rows with any missing condition after scope: "
           f"{cnt(len(cf) - len(full), k)} (by condition: "
           + ", ".join(f"{c}={cnt(cf[c].isna().sum(), k)}" for c in spec.cond_names) + ")", ""]
    rows = rate_table(cf, spec.cond_names, y, k)
    res["rates"] = rows
    md += ["## Outcome: elevated glucose (proxy) rate, single variables and pairs (in-scope train, unweighted)", "",
           f"Overall: n={cnt(y.notna().sum(), k)}, rate={round(float(y.mean()) * 100, 3) if y.notna().sum() >= k else None}%.", "",
           md_table(rows, ["variables", "cell", "n", "rate %"], "suppressed (n<30)"), ""]
    st = as_num(tr["state"]).value_counts().sort_index()
    small = [[int(s), int(n) if n >= k else None] for s, n in st.items() if n < 500]
    md += ["## States with fewer than 500 rows (train, before scope)", "",
           md_table(small, ["state (DHS code)", "n"], "suppressed (n<30)") if small else "None.", ""]
    res["small_states"] = small
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    from models.privacy import safe_public_output
    res = safe_public_output(res, k)
    write_json(res, os.path.splitext(args.out)[0] + ".json")
    print(f"wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
