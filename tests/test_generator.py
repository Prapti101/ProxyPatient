"""MOCK DATA, not NFHS-5: backend/generator.py contract tests (weights trained
2 epochs on mock data in a temp dir)."""
import importlib
import inspect
import os
import time

import numpy as np
import pytest

from models.common import REPO_ROOT, load_config

CFG = load_config()
FULL = {"sex": "female", "age_band": "25-34", "residence": "urban", "wealth_quintile": "3",
        "bmi_band": "overweight", "hypertension": 0, "tobacco": "0", "alcohol": "0"}


@pytest.fixture()
def gen(real_generator_env):
    import backend.generator as g
    g._BUNDLES.clear()
    return g


def test_signature_matches_stub():
    import backend.generator as g
    import backend.generator_stub as s
    real = list(inspect.signature(g.generate).parameters)[:3]
    assert real == list(inspect.signature(s.generate).parameters)[:3] == ["condition", "n", "seed"]
    assert inspect.signature(g.generate).parameters["n"].default == 1000
    assert inspect.signature(g.generate).parameters["seed"].default == 42


def test_deterministic_per_seed(gen):
    a, b = gen.generate(FULL, n=300, seed=5), gen.generate(FULL, n=300, seed=5)
    c = gen.generate(FULL, n=300, seed=6)
    assert a.equals(b) and not a.equals(c)


def test_columns_units_and_labels(gen):
    df = gen.generate(FULL, n=500, seed=1)
    need = {"sex", "age_band", "residence", "wealth_quintile", "bmi_band", "hypertension",
            "any_tobacco", "alcohol", "age", "bmi", "weight_kg", "height_cm", "waist_cm", "hip_cm",
            "education", "bp_ever_checked", "glucose_raw", "elevated_glucose_proxy", "is_synthetic"}
    assert need <= set(df.columns) and len(df) == 500
    assert df["is_synthetic"].all() and not df.isna().any().any()
    assert (df["elevated_glucose_proxy"] == (df["glucose_raw"] >= CFG["outcome"]["threshold_mg_dl"])).all()
    assert 60 < df["glucose_raw"].median() < 160          # mg/dL, original units
    assert (df["sex"] == 0).all() and (df["age_band"] == "25-34").all() and (df["residence"] == "urban").all()


def test_condition_consistency(gen):
    for cond in [FULL, {**FULL, "sex": "male", "age_band": "35-54", "bmi_band": "obese"}]:
        df = gen.generate(cond, n=1000, seed=2)
        lo, hi = CFG["bmi"]["bands"][cond["bmi_band"]]
        assert ((df["bmi"] >= lo) & (df["bmi"] < hi)).all()
        sex = 1 if cond["sex"] == "male" else 0
        bands = CFG["age_bands"]["men" if sex else "women"]
        idx = ["15-24", "25-34"].index(cond["age_band"]) if cond["age_band"] in ("15-24", "25-34") else 2
        assert df["age"].between(*bands[idx]).all()
        bmi_calc = df["weight_kg"] / (df["height_cm"] / 100) ** 2
        assert (abs(bmi_calc - df["bmi"]) < 0.3).all()
        assert "first_pass_inconsistent_share" in df.attrs["sampling"]


def test_sex_specific_age_label_echo(gen):
    df = gen.generate({"sex": "male", "age_band": "35-49"}, n=200, seed=1)
    assert (df["age_band"] == "35-54").all() and df["age"].between(35, 54).all()
    df = gen.generate({"sex": 0, "age_band": "35-54"}, n=200, seed=1)
    assert (df["age_band"] == "35-49").all() and df["age"].between(35, 49).all()


def test_unspecified_keys_filled_from_marginals(gen):
    df = gen.generate({}, n=2000, seed=3)
    assert df["sex"].nunique() == 2 and df["bmi_band"].nunique() >= 3
    assert "sex" in df.attrs["sampling"]["filled_from_marginals"]


@pytest.mark.parametrize("cond", [{"sex": "x"}, {"age_band": "50-60"}, {"wealth_quintile": 9},
                                  {"bmi_band": "huge"}, {"tobacco": 2}, {"glucose_raw": 250},
                                  {"elevated_glucose_proxy": 1}, {"colour": "red"}, {"state": 99}])
def test_invalid_conditions_raise(gen, cond):
    with pytest.raises(ValueError):
        gen.generate(cond, n=100)


@pytest.mark.parametrize("n", [99, 10_001, 0, 1.5])
def test_n_range(gen, n):
    with pytest.raises(ValueError):
        gen.generate(FULL, n=n)


def test_no_file_writes(gen, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def snapshot():
        out = set()
        for base in (REPO_ROOT, str(tmp_path)):
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
                out |= {os.path.join(root, f) for f in files}
        return out
    before = snapshot()
    gen.generate(FULL, n=1000, seed=1)
    assert snapshot() == before


def test_speed_10k(gen):
    gen.generate(FULL, n=100, seed=0)          # warm-up / load
    t = time.time()
    gen.generate(FULL, n=10_000, seed=0)
    assert time.time() - t < 2.0


def test_missing_weights_message(monkeypatch, tmp_path):
    import backend.generator as g
    g._BUNDLES.clear()
    monkeypatch.setenv("PP_CVAE_WEIGHTS", str(tmp_path / "nope.pt"))
    with pytest.raises(RuntimeError, match="Train the model first"):
        g.generate(FULL, n=100)


def test_main_switch(real_generator_env, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.generator as g
    g._BUNDLES.clear()
    monkeypatch.setenv("PP_GENERATOR", "real")
    import backend.main as m
    m = importlib.reload(m)
    try:
        assert m.MODEL_USED == "CVAE"
        with TestClient(m.app) as client:
            r = client.post("/generate", json={"condition": {"sex": "female", "age_band": "25-34"}, "n": 200, "seed": 1})
            assert r.status_code == 200, r.text
            assert r.json()["model_used"] == "CVAE"
            r = client.post("/compare", json={"scenarios": [
                {"label": "base", "condition": {"sex": "male", "bmi_band": "normal"}, "n": 200},
                {"label": "obese", "condition": {"sex": "male", "bmi_band": "obese"}, "n": 200}]})
            assert r.status_code == 200, r.text
    finally:
        monkeypatch.delenv("PP_GENERATOR")
        importlib.reload(m)
        assert m.MODEL_USED.startswith("CVAE-stub")
