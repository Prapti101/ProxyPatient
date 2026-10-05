"""
ProxyPatient — FastAPI Backend
================================
Author  : P1 (Data & Backend Lead)
Strictly generative AI — NO agents, NO LangChain, NO autonomous workflows.
Plain request/response API only.

Endpoints:
    GET  /health       -> system health check
    GET  /schema       -> variable schema (from schema.json)
    GET  /profiles     -> baseline reference profiles (from aggregates.json)
    POST /generate     -> generate synthetic cohort + compute outcome stat
    POST /compare      -> compare multiple what-if scenarios
    GET  /validation   -> validation report (from P3's validation_report.json)
    POST /parse        -> (optional) parse natural language condition (P3's parser)

RULES ENFORCED:
  - Glucose is never a conditioning input (blocked in schemas.py).
  - Raw data rows are never returned.
  - Outcome always labelled "elevated glucose (proxy)".
  - Disclaimer always included in every response that shows a rate.
  - All numbers come from code — nothing hardcoded.
  - Switch from stub to real model: change 1 import line below.
"""

import os
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.generator import generate, ModelUnavailable, _load, FORBIDDEN_INPUT_CONDITIONS
from backend.outcome_stat_stub import outcome_stat
from models.run_status import run_fields
MODEL_USED = "CVAE"

from backend.schemas import (
    GenerateRequest, GenerateResponse, OutcomeStat,
    CompareRequest,  CompareResponse,  ScenarioResult,
    HealthResponse, ProfilesResponse, ProfileEntry,
    ParseRequest, ParseResponse, Condition,
    DISCLAIMER
)

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxypatient")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(application):
    load_assets()
    yield


app = FastAPI(
    lifespan=lifespan,
    title       = "ProxyPatient API",
    description = (
        "Synthetic Scenario Exploration for Elevated Glucose (Proxy). "
        "STRICTLY generative AI. NOT a diagnostic tool. "
        "Source: NFHS-5 India (2019-21)."
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
)

# Allow frontend (P4's React app) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load static assets at startup ────────────────────────────────────────────
_schema     = None
_aggregates = None
_validation = None

from models.common import load_config
_cfg = load_config()
RULE = {"threshold_mg_dl": _cfg["outcome"]["threshold_mg_dl"]}


def load_assets():
    global _schema, _aggregates, _validation
    try:
        with open(os.path.join(DOCS_DIR, "schema.json"),        encoding="utf-8") as f:
            _schema = json.load(f)
        with open(os.path.join(DOCS_DIR, "aggregates_v2.json"), encoding="utf-8") as f:
            _aggregates = json.load(f)
        logger.info("Schema and aggregates_v2 loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load assets: {e}")

    val_path = os.path.join(DOCS_DIR, "validation_report.json")
    if os.path.exists(val_path):
        with open(val_path, encoding="utf-8") as f:
            _validation = json.load(f)
        logger.info("Validation report loaded.")
    else:
        logger.warning("validation_report.json not found — P3 has not delivered it yet.")
        _validation = {"status": "pending", "note": "Awaiting P3 (Validation Lead)."}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """System health check."""
    try:
        bundle, _ = _load()
        return HealthResponse(**run_fields(bundle.ckpt["run_type"]), mode="demo" if bundle.ckpt["is_mock"] else "real",
                              model_fingerprint=bundle.ckpt["fingerprint"])
    except ModelUnavailable as exc:
        return HealthResponse(status="unavailable", mode="unavailable", detail=str(exc))


@app.get("/schema", tags=["Data"])
def get_schema():
    """
    Return the full variable schema.
    Shows all confirmed DHS variable codes and their roles.
    Does NOT return any raw data rows.
    """
    if _schema is None:
        raise HTTPException(503, "Schema not loaded. Check server startup logs.")
    return {**_schema, "supported_state_codes": _state_options()}


@app.get("/profiles", response_model=ProfilesResponse, tags=["Data"])
def get_profiles():
    """
    Return baseline reference profiles for the UI's Explore page.
    Full profiles come from encoded TRAIN joint cells with at least 500 rows.
    """
    bundle, _ = _load_or_503()
    root = os.path.dirname(__import__("backend.generator", fromlist=["_paths"])._paths()[0])
    path = os.path.join(root, "supported_profiles.json")
    try:
        with open(path, encoding="utf-8") as f:
            assets = json.load(f)
        if assets.get("fingerprint") != bundle.ckpt["fingerprint"]:
            raise ValueError("Supported profiles fingerprint mismatch")
        if len(assets["profiles"]) < 3:
            raise ValueError("Fewer than three supported TRAIN profiles; each requires 500 encoded rows")
        from backend.schemas import FULL_KEYS
        for profile in assets["profiles"]:
            parsed = Condition(**profile["condition"])
            if profile["n_train"] < 500 or any(getattr(parsed, key) is None for key in FULL_KEYS):
                raise ValueError("Invalid or under-supported full TRAIN profile")
        return ProfilesResponse(profiles=[ProfileEntry(**p) for p in assets["profiles"]])
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, str(exc)) from exc


def _load_or_503():
    try:
        return _load()
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


def _response_fields(condition, df, stat):
    from models.common import age_band_label, parse_age_band, load_config
    effective = condition.model_dump(exclude_none=True)
    if "sex" in effective and "age_band" in effective:
        effective["age_band"] = age_band_label(load_config(), effective["sex"], parse_age_band(load_config(), effective["age_band"]))
    from models.privacy import safe_public_output
    diagnostics = safe_public_output(df.attrs["sampling"])
    return dict(condition=condition, effective_conditions=effective, outcome_stat=OutcomeStat(**stat),
                model_fingerprint=df.attrs["fingerprint"], **run_fields(df.attrs["run_type"]),
                banner="MOCK DEMO — not NFHS-5" if df.attrs["demo"] else None,
                sampling_diagnostics={**diagnostics, "rejection_rate": diagnostics["first_pass_inconsistent_share"],
                                      "clipped_share": diagnostics["clipped_share"]})


@app.post("/generate", response_model=GenerateResponse, tags=["Scenarios"])
def generate_cohort(req: GenerateRequest):
    """
    Generate a synthetic cohort under the given what-if conditions
    and compute the elevated glucose (proxy) outcome statistic.

    RULES:
      - Glucose is NEVER a conditioning input.
      - Rate is computed from generated data, never hardcoded.
      - Disclaimer always included.
    """
    condition_dict = req.condition.model_dump(exclude_none=True)

    # Extra safety: block glucose conditioning even if schema validation missed it
    for key in condition_dict:
        if key in FORBIDDEN_INPUT_CONDITIONS:
            raise HTTPException(
                400,
                f"'{key}' is the outcome variable and cannot be used as "
                "a what-if conditioning input."
            )

    try:
        logger.info(f"Generating cohort: n={req.n}, condition={condition_dict}")
        df = generate(condition=condition_dict, n=req.n, seed=req.seed)
    except ModelUnavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")

    try:
        stat = outcome_stat(df, rule=RULE, seed=req.seed)
    except Exception as e:
        logger.error(f"Outcome stat error: {e}")
        raise HTTPException(500, f"Outcome computation failed: {str(e)}")

    return GenerateResponse(**_response_fields(req.condition, df, stat))


@app.post("/compare", response_model=CompareResponse, tags=["Scenarios"])
def compare_scenarios(req: CompareRequest):
    """
    Compare multiple what-if scenarios side-by-side.
    Returns outcome rate for each scenario and delta vs baseline (first scenario).
    """
    results = []
    baseline_rate = None

    for scenario in req.scenarios:
        condition_dict = scenario.condition.model_dump(exclude_none=True)

        for key in condition_dict:
            if key in FORBIDDEN_INPUT_CONDITIONS:
                raise HTTPException(
                    400,
                    f"'{key}' is forbidden as a conditioning input."
                )
        try:
            df   = generate(condition=condition_dict, n=scenario.n, seed=scenario.seed)
            stat = outcome_stat(df, rule=RULE, seed=scenario.seed)
        except ModelUnavailable as e:
            raise HTTPException(503, str(e))
        except ValueError as e:
            raise HTTPException(400, f"Scenario '{scenario.label}': {e}")
        except Exception as e:
            raise HTTPException(500, f"Scenario '{scenario.label}' failed: {e}")

        if baseline_rate is None:
            baseline_rate = stat["rate"]

        delta_pp = round((stat["rate"] - baseline_rate) * 100, 4) \
                   if baseline_rate is not None else None

        results.append(ScenarioResult(label=scenario.label, delta_pp=delta_pp,
                                      **_response_fields(scenario.condition, df, stat)))

    return CompareResponse(**run_fields(df.attrs["run_type"]), scenarios=results, model_fingerprint=df.attrs["fingerprint"])


@app.get("/validation", tags=["Validation"])
def get_validation():
    """
    Return the validation report produced by P3 (Validation Lead).
    Includes fidelity scores, KS/Wasserstein stats, model comparison table.
    Returns 'pending' if P3 has not yet delivered the report.
    """
    root = os.environ.get("PP_MODEL_DIR", os.path.join(BASE_DIR, "models"))
    path = os.path.join(root, "validation_report.json")
    if not os.path.isfile(path):
        return {"status": "pending", "note": "No real-run validation report supplied"}
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("status") not in ("complete", "pending"):
        return {"status": "pending", "note": "Validation adapter requires explicit complete/pending status"}
    return report


@app.post("/parse", response_model=ParseResponse, tags=["NLP"])
def parse_condition(req: ParseRequest):
    """Demo-only rule-based parser. Returns proposals for confirmation; runs nothing."""
    import re
    if os.environ.get("PP_DEMO_MOCK") != "1":
        raise HTTPException(403, "Rule-based parser is demo-only; select explicit full conditions in real mode")
    text = req.text.lower()
    if re.search(r"glucose|hba1c|sb74|smb74", text):
        raise HTTPException(422, "glucose is the outcome, not an input")
    condition, unresolved = {}, []
    if re.search(r"\bunchanged\b|same as baseline", text):
        return ParseResponse(parsed_condition=Condition(), raw_text=req.text)
    if re.search(r"\b(female|women|woman)\b", text):
        condition["sex"] = 0
    elif re.search(r"\b(male|men|man)\b", text):
        condition["sex"] = 1
    for residence in ("urban", "rural"):
        if re.search(r"\b"+residence+r"\b", text):
            condition["residence"] = residence
    for band in sorted(set(_cfg["whatif_options"]["age_band"]["women_options"] + _cfg["whatif_options"]["age_band"]["men_options"])):
        if band in text:
            condition["age_band"] = band
    for key, words in [("tobacco", r"tobacco|smok(?:e|ing|er)"), ("alcohol", r"alcohol|drink(?:ing)?")]:
        negative = re.search(r"\b(?:no|without|not|never|do not|does not|don't)\s+(?:use\s+|consume\s+|drink\s+)?(?:"+words+r")\b", text)
        negative = negative or (key == "alcohol" and re.search(r"drink\s+no\s+alcohol", text))
        if negative:
            condition[key] = 0
        elif re.search(r"\b(?:"+words+r")\b", text):
            condition[key] = 1
    if "improved bmi" in text:
        levels = list(_cfg["bmi"]["bands"])
        baseline = req.baseline.bmi_band if req.baseline else None
        if baseline in levels and levels.index(baseline) > levels.index("normal"):
            condition["bmi_band"] = levels[levels.index(baseline)-1]
        else:
            unresolved.append("improved BMI: choose a concrete BMI band")
    else:
        for band in _cfg["bmi"]["bands"]:
            if re.search(r"\b"+band+r"\b", text):
                condition["bmi_band"] = band
    return ParseResponse(parsed_condition=Condition(**condition), raw_text=req.text, unresolved=unresolved)


def _state_options():
    from backend.generator import _load
    try:
        return _load()[0].spec.state_codes
    except RuntimeError:
        return []


@app.get("/options", tags=["Data"])
def options():
    from backend.schemas import FULL_KEYS
    return {"state": _state_options(), "sex": [0, 1], "age_band": _cfg["whatif_options"]["age_band"],
            "residence": _cfg["whatif_options"]["residence"]["options"],
            "wealth_quintile": [1, 2, 3, 4, 5], "bmi_band": list(_cfg["bmi"]["bands"]),
            "hypertension": [0, 1], "tobacco": [0, 1], "alcohol": [0, 1],
            "required_profile_keys": list(FULL_KEYS)}


class DemoBannerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        try:
            fields = run_fields(_load()[0].ckpt["run_type"])
        except ModelUnavailable:
            fields = run_fields("mock" if os.environ.get("PP_DEMO_MOCK") == "1" else None)
        banner = "MOCK DEMO — not NFHS-5; synthetic rates are not research results"
        response.headers["X-ProxyPatient-Status"] = fields["status_banner"]
        if fields["demo"]:
            response.headers["X-ProxyPatient-Demo"] = banner.encode("ascii", "replace").decode()
        if "application/json" in response.headers.get("content-type", ""):
            body = b"".join([chunk async for chunk in response.body_iterator])
            payload = json.loads(body)
            if isinstance(payload, dict):
                payload.update(fields)
                if fields["demo"]:
                    payload.update(banner=banner)
            headers = dict(response.headers)
            headers.pop("content-length", None)
            return Response(json.dumps(payload), status_code=response.status_code,
                            headers=headers, media_type="application/json")
        return response


app.add_middleware(DemoBannerMiddleware)
