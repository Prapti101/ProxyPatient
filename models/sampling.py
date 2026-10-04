"""
Turns CVAE decoder draws into rows in ORIGINAL units and enforces condition
consistency (generated age inside the requested age band for that sex,
generated BMI inside the requested BMI band; with BP generation,
hypertension=0 rows must have systolic < 140 and diastolic < 90).
Inconsistent rows are re-drawn from the decoder (never copied from data);
after max_rounds the remainder is clipped into range and counted.

Shared by backend/generator.py and models/eval_dev.py.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from models.cvae import CVAE
from models.data import Preproc


class CVAEBundle:
    def __init__(self, ckpt: dict, device: str = "cpu"):
        self.ckpt = ckpt
        self.pre = Preproc.from_dict(ckpt["preproc"])
        self.spec = self.pre.spec
        self.cfg = ckpt["cfg"]                   # the config subset needed at inference
        self.model = CVAE(**ckpt["hparams"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.device = device

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        return cls(torch.load(path, map_location=device, weights_only=False), device)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _bounds(self, cond_idx: np.ndarray):
        s = self.spec
        ci = {c: i for i, c in enumerate(s.cond_names)}
        sex, a3, bb = cond_idx[:, ci["sex"]], cond_idx[:, ci["age_band"]], cond_idx[:, ci["bmi_band"]]
        age_lo = np.empty(len(sex)); age_hi = np.empty(len(sex))
        for sx in (0, 1):
            bands = self.cfg["age_bands"]["men" if sx == 1 else "women"]
            for j, (lo, hi) in enumerate(bands):
                m = (sex == sx) & (a3 == j)
                age_lo[m], age_hi[m] = lo, hi
        names = s.cond_levels["bmi_band"]
        bmi_lo = np.array([self.cfg["bmi_bands"][names[k]][0] for k in bb], dtype=float)
        bmi_hi = np.array([self.cfg["bmi_bands"][names[k]][1] for k in bb], dtype=float)
        ht = cond_idx[:, ci["hypertension"]]
        return age_lo, age_hi, bmi_lo, bmi_hi, ht

    def _to_units(self, cont: np.ndarray, cat: np.ndarray) -> dict:
        s, p = self.spec, self.pre
        out = {}
        for j, c in enumerate(s.cont_cols):
            v = cont[:, j].astype(np.float64) * p.std[c] + p.mean[c]
            if c == "log_glucose":
                lo, hi = self.cfg["glucose_clip_mg_dl"]
                out["glucose_raw"] = np.clip(np.round(np.exp(v)), lo, hi)
            else:
                out[c] = v
        for j, c in enumerate(s.cat_cols):
            out[c] = np.asarray(s.cat_levels[c], dtype=float)[cat[:, j]]
        return out

    def _consistent(self, u: dict, bounds) -> np.ndarray:
        age_lo, age_hi, bmi_lo, bmi_hi, ht = bounds
        age = np.round(u["age"])
        ok = (age >= age_lo) & (age <= age_hi) & (u["bmi"] >= bmi_lo) & (u["bmi"] < bmi_hi)
        if "systolic_avg" in u:
            ok &= ~((ht == 0) & ((u["systolic_avg"] >= 140) | (u["diastolic_avg"] >= 90)))
        return ok

    # ── main entry ───────────────────────────────────────────────────────────
    def sample(self, cond_idx: np.ndarray, state_idx: Optional[np.ndarray] = None,
               seed: int = 42, max_rounds: int = 30, batch_rows: int = 200_000) -> pd.DataFrame:
        t0 = time.time()
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        n = len(cond_idx)
        bounds = self._bounds(cond_idx)
        cond_t = torch.as_tensor(cond_idx, dtype=torch.long, device=self.device)
        st_t = torch.as_tensor(state_idx, dtype=torch.long, device=self.device) if self.spec.use_state else None

        def draw(idx):
            conts, cats = [], []
            for b in range(0, len(idx), batch_rows):
                sl = idx[b:b + batch_rows]
                ct, ca = self.model.sample(cond_t[sl], st_t[sl] if st_t is not None else None, generator=g)
                conts.append(ct.cpu().numpy()); cats.append(ca.cpu().numpy())
            return np.concatenate(conts), np.concatenate(cats)

        cont, cat = draw(np.arange(n))
        units = self._to_units(cont, cat)
        bad = np.where(~self._consistent(units, bounds))[0]
        first_pass_reject = len(bad) / max(n, 1)
        n_redraws = 0
        for _ in range(max_rounds):
            if len(bad) == 0:
                break
            n_redraws += len(bad)
            c2, k2 = draw(bad)
            u2 = self._to_units(c2, k2)
            sub = tuple(b[bad] for b in bounds)
            ok2 = self._consistent(u2, sub)
            for key in units:
                units[key][bad[ok2]] = u2[key][ok2]
            bad = bad[~ok2]
        n_clipped = len(bad)
        age_lo, age_hi, bmi_lo, bmi_hi, ht = bounds
        units["age"] = np.clip(np.round(units["age"]), age_lo, age_hi)
        units["bmi"] = np.clip(units["bmi"], bmi_lo, np.nextafter(bmi_hi, -np.inf))
        if "systolic_avg" in units:
            m = ht == 0
            units["systolic_avg"][m] = np.minimum(units["systolic_avg"][m], 139)
            units["diastolic_avg"][m] = np.minimum(units["diastolic_avg"][m], 89)
        for c in ("height_cm", "waist_cm", "hip_cm", "systolic_avg", "diastolic_avg"):
            if c in units:
                units[c] = np.clip(units[c], self.pre.lo[c], self.pre.hi[c])
        units["bmi"] = np.floor(units["bmi"] * 100) / 100   # floor keeps it inside [lo, hi)
        if self.spec.weight_derived:
            units["height_cm"] = np.round(units["height_cm"], 1)
            units["weight_kg"] = np.round(units["bmi"] * (units["height_cm"] / 100.0) ** 2, 1)
        for c in ("waist_cm", "hip_cm"):
            units[c] = np.round(units[c], 1)
        for c in ("systolic_avg", "diastolic_avg"):
            if c in units:
                units[c] = np.round(units[c])
        df = pd.DataFrame(units)
        df["age"] = df["age"].astype(int)
        df.attrs["sampling"] = {
            "n": n, "first_pass_inconsistent_share": round(first_pass_reject, 6),
            "redrawn_rows": int(n_redraws), "clipped_after_max_rounds": int(n_clipped),
            "consistent_without_clipping_share": round(1 - n_clipped / max(n, 1), 6),
            "seconds": round(time.time() - t0, 4),
        }
        return df
