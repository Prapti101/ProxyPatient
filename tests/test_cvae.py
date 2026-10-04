"""MOCK DATA, not NFHS-5: CVAE training/sampling smoke tests."""
import json

import numpy as np
import pytest

from models.sampling import CVAEBundle


def test_two_epoch_run_completes(mock_model_dir):
    log = json.loads((mock_model_dir / "cvae_train_log.json").read_text())
    assert log["mock"] is True and len(log["epochs"]) == 2
    assert all(np.isfinite(e["val_elbo"]) for e in log["epochs"])
    for f in ["cvae_weights.pt", "cvae_preproc.json", "condition_marginals.json", "model_card.json"]:
        assert (mock_model_dir / f).exists()
    card = json.loads((mock_model_dir / "model_card.json").read_text())
    assert "MOCK" in card["status"]


def test_sampling_deterministic_and_complete(mock_model_dir):
    b = CVAEBundle.load(str(mock_model_dir / "cvae_weights.pt"))
    rng = np.random.default_rng(0)
    cond = np.stack([rng.integers(0, len(b.spec.cond_levels[c]), 500) for c in b.spec.cond_names], 1)
    st = rng.integers(0, 36, 500)
    a, c = b.sample(cond, st, seed=7), b.sample(cond, st, seed=7)
    d = b.sample(cond, st, seed=8)
    assert a.equals(c) and not a.equals(d)
    assert len(a) == 500 and not a.isna().any().any()
    assert {"age", "bmi", "weight_kg", "height_cm", "waist_cm", "hip_cm", "glucose_raw",
            "education", "bp_ever_checked"} <= set(a.columns)


@pytest.mark.parametrize("variant", ["gru", "cnn"])
def test_variants_train(tmp_path, variant):
    from models.train_cvae import main
    w, log = main(["--mock", "--epochs", "1", "--max-rows", "1500", "--variant", variant,
                   "--out-dir", str(tmp_path), "--cpu"])
    b = CVAEBundle.load(w)
    df = b.sample(np.zeros((50, len(b.spec.cond_names)), dtype=int), np.zeros(50, dtype=int), seed=1)
    assert len(df) == 50 and not df.isna().any().any()


def test_gaussian_head_and_bp(tmp_path):
    from models.train_cvae import main
    w, _ = main(["--mock", "--epochs", "1", "--max-rows", "1500", "--glucose-head", "gaussian",
                 "--generate-bp", "--no-state", "--out-dir", str(tmp_path), "--cpu"])
    b = CVAEBundle.load(w)
    cond = np.zeros((200, len(b.spec.cond_names)), dtype=int)    # hypertension index 0 -> "0"
    df = b.sample(cond, None, seed=3)
    assert ((df["systolic_avg"] < 140) & (df["diastolic_avg"] < 90)).all()
