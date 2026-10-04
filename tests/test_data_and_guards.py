"""MOCK DATA, not NFHS-5: mock harness, data module and locked-test guards."""
import numpy as np
import pandas as pd
import pytest

from models.common import LockedTestError, load_config, read_parquet, split_path
from models.data import apply_scope, build_spec, fit_preproc, make_arrays, raw_generated
from tests.mock_data import V2_COLUMNS, make_mock_v2

CFG = load_config()


def test_mock_has_exact_v2_columns_and_labels():
    df = make_mock_v2(2000, seed=0)
    assert list(df.columns) == V2_COLUMNS and len(V2_COLUMNS) == 31
    assert df.attrs["MOCK"] == "MOCK DATA, not NFHS-5"
    assert set(df["residence"]) <= set(CFG["whatif_options"]["residence"]["options"])
    assert set(df["bmi_band"].dropna().astype(str)) <= set(CFG["bmi"]["bands"])
    allowed_age = set(CFG["whatif_options"]["age_band"]["women_options"] + CFG["whatif_options"]["age_band"]["men_options"])
    assert set(df["age_band"]) <= allowed_age
    assert df.loc[df.sex == 0, "age"].max() <= 49 and df.loc[df.sex == 1, "age"].max() <= 54
    assert 0.005 < df["elevated_glucose_proxy"].mean() < 0.08
    assert df["bmi"].isna().any() and df["glucose_raw"].isna().any()


def test_mock_unimputed_has_sporadic_gaps():
    df = make_mock_v2(2000, seed=0, imputed=False)
    assert df["waist_cm"].isna().any() and df["any_tobacco"].isna().any()


@pytest.mark.parametrize("final", [False])
def test_test_split_is_locked(tmp_path, final):
    with pytest.raises(LockedTestError):
        split_path(str(tmp_path), "test", final_test=final)
    with pytest.raises(LockedTestError):
        read_parquet(str(tmp_path / "dae_imputed_test_v2.parquet"))
    assert split_path(str(tmp_path), "test", final_test=True).endswith("dae_imputed_test_v2.parquet")


def test_scope_and_train_only_normalisation():
    df = make_mock_v2(3000, seed=1)
    sc, rep = apply_scope(df, CFG)
    assert rep["n_in_scope"] == len(sc) < len(df)
    assert sc["glucose_raw"].notna().all() and (sc["bmi_measured"] == 1).all() and sc["hypertension"].notna().all()
    spec = build_spec(CFG)
    pre = fit_preproc(raw_generated(sc, spec), spec)
    arr = make_arrays(sc, pre)
    assert arr.cont.shape[1] == len(spec.cont_cols) and np.isfinite(arr.cont).all()
    assert abs(arr.cont.mean(0)).max() < 0.05          # train-normalised
    assert "log_glucose" in spec.cont_cols and abs(pre.mean["log_glucose"] - np.log(100)) < 0.5


def test_age_bands_harmonised():
    from models.data import condition_frame
    df = make_mock_v2(2000, seed=2)
    cf = condition_frame(df, build_spec(CFG))
    assert set(cf["age_band"]) == {"15-24", "25-34", "35+"}
    assert (cf.loc[df["age"] >= 35, "age_band"] == "35+").all()
