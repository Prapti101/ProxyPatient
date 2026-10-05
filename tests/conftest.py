"""Shared fixtures. Everything here uses MOCK DATA, not NFHS-5."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def mock_model_dir(tmp_path_factory):
    """CVAE trained for 2 epochs on MOCK data, saved in a temp dir."""
    from models.train_cvae import main
    out = tmp_path_factory.mktemp("cvae_mock")
    main(["--mock", "--epochs", "2", "--max-rows", "4000", "--out-dir", str(out), "--cpu"])
    return out


@pytest.fixture(scope="session")
def mock_baselines_dir(tmp_path_factory):
    from models.train_baselines import main
    out = tmp_path_factory.mktemp("baselines_mock")
    main(["--mock", "--epochs", "1", "--out-dir", str(out), "--log-out", str(out / "log.json"), "--cpu"])
    return out


@pytest.fixture()
def real_generator_env(mock_model_dir, monkeypatch):
    monkeypatch.setenv("PP_MODEL_DIR", str(mock_model_dir))
    monkeypatch.setenv("PP_DEMO_MOCK", "1")
    monkeypatch.setenv("PP_ALLOW_PARTIAL_PROFILE", "1")
    monkeypatch.setenv("PP_CVAE_WEIGHTS", str(mock_model_dir / "cvae_weights.pt"))
    monkeypatch.setenv("PP_CONDITION_MARGINALS", str(mock_model_dir / "condition_marginals.json"))
    yield mock_model_dir
