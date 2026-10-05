"""Strict version 2 request/response contracts; options come from configuration."""
import os
import re
from typing import Any, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from models.common import load_config, parse_age_band

DISCLAIMER = ('SYNTHETIC scenario exploration for elevated glucose (proxy). '
              'Not a medical diagnosis, individual prediction, treatment recommendation, '
              'or causal effect. Consult a qualified healthcare professional.')
UNCERTAINTY_NOTE = ('Monte Carlo variation of the synthetic cohort conditional on the fitted model; '
                    'does not measure model or survey uncertainty.')
SCOPE = 'Measured BMI, known blood-pressure status and finite glucose; both sexes; complete encoded TRAIN rows.'
FULL_KEYS = ('sex', 'age_band', 'residence', 'wealth_quintile', 'bmi_band', 'hypertension', 'tobacco', 'alcohol')


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Condition(StrictModel):
    sex: Optional[int] = None
    age_band: Optional[str] = None
    residence: Optional[str] = None
    wealth_quintile: Optional[int] = None
    bmi_band: Optional[str] = None
    hypertension: Optional[int] = None
    tobacco: Optional[int] = None
    alcohol: Optional[int] = None
    state: Optional[int] = None

    @model_validator(mode='before')
    @classmethod
    def reject_outcome(cls, value):
        if isinstance(value, dict):
            for key in value:
                if any(t in key.lower() for t in ('glucose', 'hba1c', 'sb74', 'smb74')):
                    raise ValueError('glucose is the outcome, not an input')
        return value

    @field_validator('sex', mode='before')
    @classmethod
    def sex_value(cls, value):
        return {'female': 0, 'male': 1}.get(str(value).lower(), value)

    @field_validator('sex', 'hypertension', 'tobacco', 'alcohol', 'wealth_quintile', 'state', mode='before')
    @classmethod
    def integer_value(cls, value, info):
        if value is None:
            return value
        if info.field_name == 'sex':
            value = {'female': 0, 'male': 1}.get(str(value).lower(), value)
        if isinstance(value, bool) or not re.fullmatch(r'-?\d+', str(value).strip()):
            raise ValueError('must be an integer code')
        return int(value)

    @field_validator('sex', 'hypertension', 'tobacco', 'alcohol')
    @classmethod
    def binary(cls, value):
        if value is not None and value not in (0, 1):
            raise ValueError('must be 0 or 1')
        return value

    @field_validator('age_band')
    @classmethod
    def age(cls, value):
        if value is not None:
            cfg = load_config()
            options = cfg['whatif_options']['age_band']
            if value not in options['women_options'] + options['men_options']:
                raise ValueError('age_band must be a configured sex-specific label')
        return value

    @field_validator('residence', 'bmi_band')
    @classmethod
    def labels(cls, value, info):
        cfg = load_config()
        options = cfg['whatif_options']['residence']['options'] if info.field_name == 'residence' else cfg['bmi']['bands']
        if value is not None and value not in options:
            raise ValueError(f'{info.field_name} must be one of {list(options)}')
        return value

    @field_validator('wealth_quintile')
    @classmethod
    def wealth(cls, value):
        if value is not None and value not in (1, 2, 3, 4, 5):
            raise ValueError('wealth_quintile must be 1-5')
        return value

    @field_validator('state')
    @classmethod
    def state_supported(cls, value):
        if value is not None:
            from backend.generator import _load, ModelUnavailable
            try:
                codes = _load()[0].spec.state_codes
            except ModelUnavailable as exc:
                from fastapi import HTTPException
                raise HTTPException(503, 'state options unavailable until compatible artifacts are loaded') from exc
            if value not in codes:
                raise ValueError(f'state must be one of {codes}')
        return value


class GenerateRequest(StrictModel):
    condition: Condition
    n: int = Field(default_factory=lambda: int(load_config()['outcome_stat']['default_cohort_size']), ge=100, le=10000, strict=True)
    n_examples: int = Field(default=3, ge=1, le=5, strict=True)
    seed: int = Field(default_factory=lambda: int(load_config()['model']['seed']), strict=True)

    @model_validator(mode='after')
    def full_profile(self):
        missing = [k for k in FULL_KEYS if getattr(self.condition, k) is None]
        if missing and os.environ.get('PP_ALLOW_PARTIAL_PROFILE') != '1':
            raise ValueError('Missing full profile conditions: ' + ', '.join(missing))
        if missing and os.environ.get('PP_DEMO_MOCK') != '1':
            raise ValueError('Independent marginal fill is an explicit demo/development option only')
        return self


class OutcomeStat(StrictModel):
    rate: float
    ci_low: float
    ci_high: float
    n: int
    rate_pct: float
    dropped_nonfinite: Optional[int] = None
    monte_carlo_interval: Optional[dict[str, Any]] = None


class RunStatus(StrictModel):
    run_type: Optional[Literal["mock", "quick", "full"]] = None
    preliminary: bool = False
    status_banner: str = "UNAVAILABLE"
    run_note: Optional[str] = None
    demo: bool = False


class SyntheticExample(StrictModel):
    example_id: str
    synthetic: Literal[True] = True
    label: str
    effective_conditions: dict[str, Any]
    generated_features: dict[str, Any]
    glucose_raw: int
    elevated_glucose_proxy: bool
    illustrative_elevated: bool = False


class GenerateResponse(RunStatus):
    examples: Optional[list[SyntheticExample]] = None
    examples_status: str
    examples_note: str = 'Nearest-record screening is a heuristic, not a privacy guarantee. Examples do not determine the cohort statistic.' 
    condition: Condition
    effective_conditions: dict[str, Any]
    outcome_stat: OutcomeStat
    model_fingerprint: str
    sampling_diagnostics: dict[str, Any]
    synthetic: bool = True
    demo: bool = False
    banner: Optional[str] = None
    model_used: str = 'CVAE'
    scope: str = SCOPE
    weighting: str = 'unweighted sample'
    uncertainty_note: str = UNCERTAINTY_NOTE
    disclaimer: str = DISCLAIMER
    note: str = 'Descriptive synthetic scenario comparison; what-if is not causal.'


class CompareScenario(GenerateRequest):
    label: str = Field(min_length=1, max_length=200)


class CompareRequest(StrictModel):
    scenarios: list[CompareScenario] = Field(min_length=2, max_length=5)


class ScenarioResult(GenerateResponse):
    label: str
    delta_pp: float


class CompareResponse(RunStatus):
    scenarios: list[ScenarioResult]
    model_fingerprint: str
    disclaimer: str = DISCLAIMER
    weighting: str = 'unweighted sample'
    uncertainty_note: str = UNCERTAINTY_NOTE
    note: str = 'Differences describe synthetic scenarios, not causal effects.'


class HealthResponse(RunStatus):
    status: str = 'ok'
    mode: str = 'unavailable'
    model_fingerprint: Optional[str] = None
    detail: Optional[str] = None
    version: str = '2.0.0'
    model: str = 'CVAE'
    disclaimer_present: bool = True


class ProfileEntry(StrictModel):
    label: str
    condition: dict[str, Any]
    description: str
    n_train: int


class ProfilesResponse(StrictModel):
    profiles: list[ProfileEntry]
    source: str = 'Encoded TRAIN model scope; unweighted sample'
    reference_scope: str = 'Historical reference aggregates use all known-glucose respondents; not the model scope.'
    disclaimer: str = DISCLAIMER


class ParseRequest(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    baseline: Optional[Condition] = None


class ParseResponse(StrictModel):
    parsed_condition: Condition
    raw_text: str
    confidence: Optional[float] = None
    parser: str = 'rule-based demo parser'
    requires_confirmation: bool = True
    unresolved: list[str] = Field(default_factory=list)
    optional_token_hook: Optional[dict[str, Any]] = None


class ReportResponse(RunStatus):
    status: str
    metrics: Optional[dict[str, Any]] = None
    models: Optional[dict[str, Any]] = None
    evaluation_scope: Optional[dict[str, Any]] = None
    real_reference: Optional[dict[str, Any]] = None
    model_fingerprint: Optional[str] = None
    note: str
