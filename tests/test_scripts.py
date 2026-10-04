"""MOCK DATA, not NFHS-5: end-to-end smoke runs of the real-data scripts."""
import json
import os

from models import check_dae_benchmark, eval_dev, step0_checks


def test_step0_checks_mock(tmp_path):
    out = tmp_path / "s0.md"
    res = step0_checks.main(["--mock", "--out", str(out)])
    text = out.read_text()
    assert "MOCK DATA, not NFHS-5" in text and "Training scope" in text
    assert res["scope"]["n_in_scope"] > 0
    # every reported rate cell has n >= 30 or is suppressed
    for var, cell, n, rate in res["rates"]:
        assert (n is None and rate is None) or n >= 30


def test_dae_benchmark_mock(tmp_path):
    res = check_dae_benchmark.main(["--mock", "--out", str(tmp_path / "d.json")])
    for col in ("waist_cm", "hip_cm"):
        r = res[col]
        assert r["n_masked"] > 0 and r["rmse_dae"] > 0 and r["rmse_median"] > 0 and "val_known_sd" in r


def test_baselines_and_eval_mock(tmp_path, mock_model_dir, mock_baselines_dir):
    log = json.loads((mock_baselines_dir / "log.json").read_text())
    assert set(log["models"]) == {"tvae", "ctgan"}
    for m in log["models"].values():
        assert m["train_seconds"] >= 0 and 0 <= m["probe_rejection_sampling"]["acceptance_rate"] <= 1
    out = tmp_path / "cmp.json"
    res = eval_dev.main(["--mock", "--min-eval-cell", "50", "--n-eval", "1500",
                         "--cvae-weights", str(mock_model_dir / "cvae_weights.pt"),
                         "--baselines-dir", str(mock_baselines_dir), "--out-json", str(out)])
    assert set(res["models"]) == {"CVAE", "TVAE", "CTGAN"}
    for name, r in res["models"].items():
        for key in ["marginals", "correlation", "tail", "conditional_rate_fidelity", "direction",
                    "consistency", "nearest_record", "tstr"]:
            assert key in r, (name, key)
    assert res["models"]["CVAE"]["consistency"]["age_in_band_share"] == 1.0
    assert res["models"]["CVAE"]["consistency"]["bmi_in_band_share"] == 1.0
    assert (tmp_path / "cmp.md").exists() and "MOCK" in (tmp_path / "cmp.md").read_text()
    assert os.path.getsize(out) < 2_000_000          # aggregate-only output
