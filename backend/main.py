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

# ── Generator: stub by default; P2's CVAE when env var PP_GENERATOR=real ─────
if os.environ.get("PP_GENERATOR", "").lower() == "real":
    from backend.generator      import generate      # REAL: P2's CVAE
    MODEL_USED = "CVAE"
else:
    from backend.generator_stub import generate      # STUB
    MODEL_USED = "CVAE-stub (set PP_GENERATOR=real for P2 model)"

# ── Swap these two lines when P3 delivers the real outcome_stat ──────────────
from backend.outcome_stat_stub import outcome_stat   # STUB: replace with real
# from backend.outcome_stat    import outcome_stat    # REAL: P3's implementation

from backend.schemas import (
    GenerateRequest, GenerateResponse, OutcomeStat,
    CompareRequest,  CompareResponse,  ScenarioResult,
    HealthResponse, ProfilesResponse, ProfileEntry,
    ParseRequest, ParseResponse, Condition,
    DISCLAIMER
)
from backend.generator_stub import FORBIDDEN_INPUT_CONDITIONS

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxypatient")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

app = FastAPI(
    title       = "ProxyPatient API",
    description = (
        "Synthetic Patient Scenario Generation for Diabetes Risk Awareness. "
        "STRICTLY generative AI. NOT a diagnostic tool. "
        "Source: NFHS-5 India (2019-21)."
    ),
    version     = "1.0.0",
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

RULE = {"threshold_mg_dl": 200}   # from config.yaml — never hardcode elsewhere


@app.on_event("startup")
def load_assets():
    global _schema, _aggregates, _validation
    try:
        with open(os.path.join(DOCS_DIR, "schema.json"),     encoding="utf-8") as f:
            _schema = json.load(f)
        with open(os.path.join(DOCS_DIR, "aggregates.json"), encoding="utf-8") as f:
            _aggregates = json.load(f)
        logger.info("Schema and aggregates loaded successfully.")
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
    return HealthResponse()


@app.get("/schema", tags=["Data"])
def get_schema():
    """
    Return the full variable schema.
    Shows all confirmed DHS variable codes and their roles.
    Does NOT return any raw data rows.
    """
    if _schema is None:
        raise HTTPException(503, "Schema not loaded. Check server startup logs.")
    return _schema


@app.get("/profiles", response_model=ProfilesResponse, tags=["Data"])
def get_profiles():
    """
    Return baseline reference profiles for the UI's Explore page.
    All rates come from the precomputed aggregates.json (no raw rows).
    Only cells with n >= 30 are shown.
    """
    if _aggregates is None:
        raise HTTPException(503, "Aggregates not loaded.")

    profiles = []

    # Overall baseline
    overall = _aggregates.get("overall", {})
    if not overall.get("suppressed"):
        profiles.append(ProfileEntry(
            label       = "Overall Population",
            condition   = {},
            description = (
                f"All NFHS-5 respondents with glucose readings. "
                f"n={overall.get('n', 'N/A'):,}. "
                f"Elevated glucose (proxy) rate: "
                f"{round(overall.get('rate', 0) * 100, 2)}%"
            )
        ))

    # By sex
    for sex_key, sex_label in [("female", "Women"), ("male", "Men")]:
        entry = _aggregates.get("by_sex", {}).get(sex_key, {})
        if not entry.get("suppressed") and entry.get("rate") is not None:
            profiles.append(ProfileEntry(
                label       = f"{sex_label} (All Ages)",
                condition   = {"sex": sex_key},
                description = (
                    f"n={entry.get('n', 0):,}. "
                    f"Rate: {round(entry.get('rate', 0) * 100, 2)}%"
                )
            ))

    # By residence
    for res in ["urban", "rural"]:
        entry = _aggregates.get("by_residence", {}).get(res, {})
        if not entry.get("suppressed") and entry.get("rate") is not None:
            profiles.append(ProfileEntry(
                label       = f"{res.capitalize()} Residents",
                condition   = {"residence": res},
                description = (
                    f"n={entry.get('n', 0):,}. "
                    f"Rate: {round(entry.get('rate', 0) * 100, 2)}%"
                )
            ))

    return ProfilesResponse(profiles=profiles)


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
    condition_dict = req.condition.dict(exclude_none=True)

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
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")

    try:
        stat = outcome_stat(df, rule=RULE, seed=req.seed)
    except Exception as e:
        logger.error(f"Outcome stat error: {e}")
        raise HTTPException(500, f"Outcome computation failed: {str(e)}")

    return GenerateResponse(
        condition    = req.condition,
        outcome_stat = OutcomeStat(**stat),
        model_used   = MODEL_USED
    )


@app.post("/compare", response_model=CompareResponse, tags=["Scenarios"])
def compare_scenarios(req: CompareRequest):
    """
    Compare multiple what-if scenarios side-by-side.
    Returns outcome rate for each scenario and delta vs baseline (first scenario).
    """
    results = []
    baseline_rate = None

    for scenario in req.scenarios:
        condition_dict = scenario.condition.dict(exclude_none=True)

        for key in condition_dict:
            if key in FORBIDDEN_INPUT_CONDITIONS:
                raise HTTPException(
                    400,
                    f"'{key}' is forbidden as a conditioning input."
                )
        try:
            df   = generate(condition=condition_dict, n=scenario.n, seed=scenario.seed)
            stat = outcome_stat(df, rule=RULE, seed=scenario.seed)
        except ValueError as e:
            raise HTTPException(400, f"Scenario '{scenario.label}': {e}")
        except Exception as e:
            raise HTTPException(500, f"Scenario '{scenario.label}' failed: {e}")

        if baseline_rate is None:
            baseline_rate = stat["rate"]

        delta_pp = round((stat["rate"] - baseline_rate) * 100, 4) \
                   if baseline_rate is not None else None

        results.append(ScenarioResult(
            label        = scenario.label,
            condition    = scenario.condition,
            outcome_stat = OutcomeStat(**stat),
            delta_pp     = delta_pp
        ))

    return CompareResponse(scenarios=results)


@app.get("/validation", tags=["Validation"])
def get_validation():
    """
    Return the validation report produced by P3 (Validation Lead).
    Includes fidelity scores, KS/Wasserstein stats, model comparison table.
    Returns 'pending' if P3 has not yet delivered the report.
    """
    return _validation


@app.post("/parse", tags=["NLP"])
def parse_condition(req: ParseRequest):
    """
    (Optional) Parse a natural language condition string into a Condition object.
    Implemented by P3 using BERT. Returns a basic rule-based parse as stub.
    """
    text  = req.text.lower()
    cond  = {}

    # Rule-based stub — P3 replaces with BERT parser
    if "female" in text or "women" in text or "woman" in text:
        cond["sex"] = "female"
    elif "male" in text or "men" in text or "man" in text:
        cond["sex"] = "male"

    if "urban" in text:
        cond["residence"] = "urban"
    elif "rural" in text:
        cond["residence"] = "rural"

    for band in ["15-24", "25-34", "35-49", "35-54"]:
        if band in text:
            cond["age_band"] = band
            break

    if "tobacco" in text or "smok" in text:
        cond["tobacco"] = "1"
    if "alcohol" in text or "drink" in text:
        cond["alcohol"] = "1"

    if "obese" in text:
        cond["bmi_band"] = "obese"
    elif "overweight" in text:
        cond["bmi_band"] = "overweight"

    return ParseResponse(
        parsed_condition = Condition(**cond),
        raw_text         = req.text,
        confidence       = 0.6 if cond else 0.1
    )
