"""
Train the ProxyPatient CVAE.

Real data (human runs this; data never leaves their machine):
    PP_DATA_DIR=/path/to/processed python -m models.train_cvae
Mock smoke run (MOCK DATA, not NFHS-5; writes into a temp dir):
    python -m models.train_cvae --mock --epochs 2 --out-dir /tmp/x

Reads dae_imputed_{train,val}_v2.parquet only (never test). Saves
cvae_weights.pt (git-ignored), cvae_preproc.json, condition_marginals.json,
cvae_train_log.json and model_card.json. Logs are aggregate-only.
"""

import argparse
import copy
import json
import os
import random
import resource
import time
from datetime import datetime, timezone

import numpy as np
import torch

from models.common import (MOCK_BANNER, MODELS_DIR, data_dir, load_config, read_parquet,
                           split_path, write_json)
from models.cvae import build_model
from models.data import (DROP_COLUMNS, apply_scope, build_spec, condition_marginals,
                         default_paths, fit_preproc, make_arrays, raw_generated, require_sex_support, supported_profiles)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


from models.artifacts import inference_cfg, fingerprint, model_fingerprint


def load_frames(args, cfg):
    if args.mock:
        from tests.mock_data import make_mock_v2
        n = args.max_rows or 6000
        train = make_mock_v2(n, seed=1)
        if args.demo_profiles:
            # Construct MOCK joint cells, never copy survey rows or save samples.
            for group, (sex, bmi, band) in enumerate([(0, 17., "underweight"), (0, 22., "normal"), (1, 32., "obese")]):
                mask = np.arange(n) % 3 == group
                train.loc[mask, ["sex", "age", "age_band", "residence", "wealth_quintile", "bmi", "bmi_band",
                                 "bmi_measured", "hypertension", "any_tobacco", "alcohol"]] = [sex, 30, "25-34", "rural", 3, bmi, band, 1, 0, 0, 0]
                train.loc[mask, "weight_kg"] = bmi * (train.loc[mask, "height_cm"] / 100)**2
        return train, make_mock_v2(max(n // 4, 500), seed=2, banner=False)
    d = data_dir(args.data_dir)
    tr = read_parquet(split_path(d, "train"))
    va = read_parquet(split_path(d, "val"))
    return tr, va


def batches(arr, idx, bs, device):
    for b in range(0, len(idx), bs):
        sl = idx[b:b + bs]
        yield (torch.as_tensor(arr.cont[sl], device=device),
               torch.as_tensor(arr.cat[sl], device=device),
               torch.as_tensor(arr.cond[sl], device=device),
               torch.as_tensor(arr.state[sl], device=device))


def evaluate(model, arr, bs, device, use_state):
    model.eval()
    tot = {"recon": 0.0, "kl": 0.0}
    n = len(arr.cont)
    g = torch.random.get_rng_state()
    torch.manual_seed(0)                       # fixed noise -> comparable val ELBO
    with torch.no_grad():
        for cont, cat, cond, st in batches(arr, np.arange(n), bs, device):
            out = model.loss(cont, cat, cond, st if use_state else None, beta=1.0)
            w = len(cont) / n
            tot["recon"] += float(out["recon"]) * w
            tot["kl"] += float(out["kl"]) * w
    torch.random.set_rng_state(g)
    tot["elbo"] = -(tot["recon"] + tot["kl"])
    return tot


def train(args, cfg):
    args.seed = cfg.get("model", {}).get("seed", 42) if args.seed is None else args.seed
    args.variant = cfg.get("model", {}).get("variant", "mlp") if args.variant is None else args.variant
    args.glucose_head = cfg.get("model", {}).get("glucose_head", "mixture") if args.glucose_head is None else args.glucose_head
    for value in (args.epochs, args.patience, args.batch_size, args.max_rows, args.latent_dim):
        if value is not None and value <= 0:
            raise ValueError("Explicit training sizes/epochs must be positive")
    if args.demo_profiles and not args.mock:
        raise ValueError("--demo-profiles is MOCK-only")
    set_seed(args.seed)
    mcfg = dict(cfg.get("model", {}))
    for k in ("latent_dim",):
        if getattr(args, k) is not None:
            mcfg[k] = getattr(args, k)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    paths = default_paths(args.out_dir)
    if args.mock:
        print(f"*** {MOCK_BANNER} *** (smoke run; outputs are NOT a shipped model)")

    tr_raw, va_raw = load_frames(args, cfg)
    if args.max_rows and not args.mock and len(tr_raw) > args.max_rows:
        tr_raw = tr_raw.sample(args.max_rows, random_state=args.seed)
    tr, scope_tr = apply_scope(tr_raw, cfg, args.scope)
    va, scope_va = apply_scope(va_raw, cfg, args.scope)
    spec = build_spec(cfg, use_state=False if args.no_state else None, generate_bp=args.generate_bp, training_df=tr)
    pre = fit_preproc(raw_generated(tr, spec), spec)
    a_tr, a_va = make_arrays(tr, pre), make_arrays(va, pre)
    require_sex_support(a_tr, cfg, mock=args.mock)
    if len(a_va.cont) < 30:
        raise ValueError("Insufficient complete validation rows")
    from models.privacy import suppress_count, safe_public_output
    print(f"Encoded train rows: {suppress_count(len(a_tr.cont))}; excluded incomplete: {suppress_count(a_tr.n_dropped)}; "
          f"validation rows: {suppress_count(len(a_va.cont))}; excluded: {suppress_count(a_va.n_dropped)}")

    model = build_model(spec, mcfg, variant=args.variant, glucose_head=args.glucose_head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(mcfg.get("lr", 1e-3)))
    beta_max = float(mcfg.get("beta", 1.0))
    anneal = max(int(mcfg.get("kl_annealing_epochs", 10)), 1)
    bs = int(args.batch_size or mcfg.get("batch_size", 1024))
    epochs = int(args.epochs or mcfg.get("max_epochs", 100))
    patience = int(args.patience or mcfg.get("early_stopping_patience", 8))

    log = {"started": datetime.now(timezone.utc).isoformat(), "mock": bool(args.mock),
           "device": device, "variant": args.variant, "glucose_head": args.glucose_head,
           "n_train": len(a_tr.cont), "n_val": len(a_va.cont),
           "dropped_incomplete": {"train": a_tr.n_dropped, "val": a_va.n_dropped},
           "scope_train": scope_tr, "scope_val": scope_va,
           "encoded_per_sex": a_tr.support, "encoding_exclusions": a_tr.exclusions, "epochs": []}
    best, best_state, bad_epochs = -np.inf, None, 0
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        beta = beta_max * min(1.0, ep / anneal)
        perm = rng.permutation(len(a_tr.cont))
        s = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        nb = 0
        for cont, cat, cond, st in batches(a_tr, perm, bs, device):
            out = model.loss(cont, cat, cond, st if spec.use_state else None, beta=beta)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k in s:
                s[k] += out[k].item(); nb += (k == "loss")
        v = evaluate(model, a_va, 8192, device, spec.use_state)
        row = {"epoch": ep, "beta": round(beta, 4), **{f"train_{k}": round(x / nb, 5) for k, x in s.items()},
               **{f"val_{k}": round(x, 5) for k, x in v.items()}, "seconds": round(time.time() - t0, 1)}
        log["epochs"].append(row)
        print(json.dumps(row))
        # early stopping only once annealing is complete (ELBO comparable at beta=1)
        if ep >= anneal or epochs < anneal:
            if v["elbo"] > best + 1e-4:
                best, best_state, bad_epochs = v["elbo"], copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop at epoch {ep}")
                    break
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    log["best_val_elbo"] = round(best, 5) if np.isfinite(best) else None
    log["peak_memory_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)
    log["train_seconds"] = round(time.time() - t0, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "hparams": model.hparams, "preproc": json.loads(json.dumps(_pre_dict(pre))),
            "cfg": inference_cfg(cfg), "variant": args.variant, "glucose_head": args.glucose_head,
            "mock": bool(args.mock), "is_mock": bool(args.mock),
            "config_fingerprint": fingerprint(cfg, spec), "created": datetime.now(timezone.utc).isoformat()}
    ckpt["fingerprint"] = model_fingerprint(ckpt)
    weights = args.weights_out or paths["weights"]
    torch.save(ckpt, weights)
    pre.to_json(paths["preproc"])
    marginals = condition_marginals(tr.loc[a_tr.retained_mask], spec, cfg)
    marginals["_meta"]["fingerprint"] = ckpt["fingerprint"]
    write_json(marginals, paths["marginals"])
    profiles = supported_profiles(tr.loc[a_tr.retained_mask], spec, cfg=cfg)
    profiles["fingerprint"] = ckpt["fingerprint"]
    write_json(profiles, os.path.join(args.out_dir, "supported_profiles.json"))
    log = safe_public_output(log)
    write_json(log, paths["train_log"])
    write_json(model_card(cfg, model, pre, log, args), paths["model_card"])
    print(f"saved weights -> {weights} (git-ignored; do not commit)")
    return weights, log


def _pre_dict(pre):
    return {"spec": pre.spec.to_dict(), "mean": pre.mean, "std": pre.std, "lo": pre.lo,
            "hi": pre.hi, "n_train_rows": pre.n_train_rows, "meta": pre.meta}


def model_card(cfg, model, pre, log, args) -> dict:
    return {
        "model": "ProxyPatient CVAE" + ("" if args.variant == "mlp" else f" ({args.variant} variant)"),
        "status": "MOCK SMOKE RUN - not a real model" if args.mock else "trained on NFHS-5 v2 train split",
        "intended_use": "Generate SYNTHETIC cohorts for population-level health awareness. "
                        "Outcome: elevated glucose (proxy), random capillary glucose >= "
                        f"{cfg['outcome']['threshold_mg_dl']} mg/dL. Not a diagnosis, not an individual risk.",
        "not_for": ["diagnosis", "treatment decisions", "individual prediction", "causal claims"],
        "data": {"source": "NFHS-5 India 2019-21 (DHS)", "splits_used": ["train", "val"],
                 "test_split": "locked; not used for training or model selection",
                 "scope": log["scope_train"]["scope"], "n_train": log["n_train"], "n_val": log["n_val"]},
        "conditions": pre.spec.cond_names + (["state"] if pre.spec.use_state else []),
        "generated": pre.spec.cont_cols + pre.spec.cat_cols + (["weight_kg (derived)"] if pre.spec.weight_derived else []),
        "architecture": model.hparams,
        "training": {"best_val_elbo": log.get("best_val_elbo"), "epochs_run": len(log["epochs"]),
                     "seconds": log.get("train_seconds"), "device": log["device"], "seed": args.seed},
        "survey_weights": "not used in training (unweighted model)",
        "limitations": [
            "Trained only on respondents with glucose, measured BMI and known hypertension status.",
            "Full profiles required in serving; independent marginal fill is explicit demo/development only.",
            "What-if = how the synthetic cohort shifts, not a causal intervention.",
            "No external validation dataset (NMB-2017 has no data file).",
        ],
        "evaluation": "see docs/model_comparison_dev.json (validation only)",
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir")
    p.add_argument("--out-dir", default=MODELS_DIR)
    p.add_argument("--weights-out", help="override weights path (default <out-dir>/cvae_weights.pt)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--latent-dim", dest="latent_dim", type=int)
    p.add_argument("--max-rows", type=int, help="subsample train rows (debugging)")
    p.add_argument("--scope", choices=["complete_conditions", "glucose_known"])
    p.add_argument("--variant", choices=["mlp", "gru", "cnn"], default=None)
    p.add_argument("--glucose-head", choices=["mixture", "gaussian"], default=None)
    p.add_argument("--no-state", action="store_true", default=None, help="disable the state embedding")
    p.add_argument("--generate-bp", action="store_true", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mock", action="store_true", help="MOCK DATA smoke run (tests only)")
    p.add_argument("--demo-profiles", action="store_true", help="construct supported MOCK demo profile cells")
    p.add_argument("--config")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    return train(args, cfg)


if __name__ == "__main__":
    main()
