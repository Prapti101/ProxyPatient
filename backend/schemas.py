"""
ProxyPatient — Backend Pydantic Schemas
Request and response models for all FastAPI endpoints.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


DISCLAIMER = (
    "This tool generates SYNTHETIC patient scenarios for health-awareness "
    "and research purposes only. Results are based on population-level "
    "statistical patterns, not individual predictions. This is NOT a medical "
    "diagnosis, clinical assessment, or treatment recommendation. "
    "'Elevated glucose (proxy)' refers to a random capillary glucose reading "
    ">= 200 mg/dL and cannot distinguish Type 1 from Type 2 diabetes. "
    "Always consult a qualified healthcare professional."
)

# Valid values for each conditioning variable (from schema.json / config.yaml)
VALID_SEX         = ["female", "male"]
VALID_AGE_BAND    = ["15-24", "25-34", "35-49", "35-54"]
VALID_RESIDENCE   = ["urban", "rural"]
VALID_WEALTH      = ["1", "2", "3", "4", "5"]   # 1=poorest, 5=richest
VALID_BMI_BAND    = ["underweight", "normal", "overweight", "obese"]
VALID_TOBACCO     = ["0", "1"]    # 0=no, 1=yes
VALID_ALCOHOL     = ["0", "1"]    # 0=no, 1=yes

# Glucose is NEVER a valid conditioning variable
FORBIDDEN_CONDITIONS = ["glucose", "glucose_raw", "elevated_glucose_proxy",
                         "sb74", "smb74", "hba1c"]


class Condition(BaseModel):
    """What-if conditions for scenario generation."""
    sex:            Optional[str] = Field(None, description="female or male")
    age_band:       Optional[str] = Field(None, description="e.g. 25-34")
    residence:      Optional[str] = Field(None, description="urban or rural")
    wealth_quintile:Optional[str] = Field(None, description="1-5 (1=poorest)")
    bmi_band:       Optional[str] = Field(None, description="underweight/normal/overweight/obese")
    tobacco:        Optional[str] = Field(None, description="0=no, 1=yes")
    alcohol:        Optional[str] = Field(None, description="0=no, 1=yes")

    @validator("sex")
    def validate_sex(cls, v):
        if v is not None and v not in VALID_SEX:
            raise ValueError(f"sex must be one of {VALID_SEX}")
        return v

    @validator("residence")
    def validate_residence(cls, v):
        if v is not None and v not in VALID_RESIDENCE:
            raise ValueError(f"residence must be one of {VALID_RESIDENCE}")
        return v

    @validator("wealth_quintile")
    def validate_wealth(cls, v):
        if v is not None and str(v) not in VALID_WEALTH:
            raise ValueError(f"wealth_quintile must be one of {VALID_WEALTH}")
        return v

    @validator("bmi_band")
    def validate_bmi(cls, v):
        if v is not None and v not in VALID_BMI_BAND:
            raise ValueError(f"bmi_band must be one of {VALID_BMI_BAND}")
        return v


class GenerateRequest(BaseModel):
    condition: Condition = Field(..., description="What-if conditioning variables")
    n:         int       = Field(1000, ge=100, le=10000,
                                  description="Cohort size (100-10000)")
    seed:      int       = Field(42, description="Random seed for reproducibility")


class OutcomeStat(BaseModel):
    rate:     float = Field(..., description="Proportion with elevated glucose (proxy)")
    ci_low:   float = Field(..., description="Bootstrap 95% CI lower bound")
    ci_high:  float = Field(..., description="Bootstrap 95% CI upper bound")
    n:        int   = Field(..., description="Number of generated rows with glucose reading")
    rate_pct: float = Field(..., description="rate as percentage (rate * 100)")


class GenerateResponse(BaseModel):
    condition:    Condition
    outcome_stat: OutcomeStat
    disclaimer:   str = DISCLAIMER
    model_used:   str = Field("CVAE", description="Which generative model produced this")
    note:         str = ("What-if means 'how the synthetic cohort shifts when the "
                         "population profile changes', NOT a causal intervention.")


class CompareScenario(BaseModel):
    label:     str       = Field(..., description="Human-readable scenario name")
    condition: Condition
    n:         int       = Field(1000, ge=100, le=10000)
    seed:      int       = Field(42)


class CompareRequest(BaseModel):
    scenarios: List[CompareScenario] = Field(..., min_items=2, max_items=5)


class ScenarioResult(BaseModel):
    label:        str
    condition:    Condition
    outcome_stat: OutcomeStat
    delta_pp:     Optional[float] = Field(
        None, description="Difference in percentage points vs first scenario (baseline)"
    )


class CompareResponse(BaseModel):
    scenarios:  List[ScenarioResult]
    disclaimer: str = DISCLAIMER
    note:       str = ("Differences are descriptive shifts in synthetic cohorts, "
                       "not causal effects.")


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    model:   str = "CVAE"
    disclaimer_present: bool = True


class ProfileEntry(BaseModel):
    label:       str
    condition:   Dict[str, Any]
    description: str


class ProfilesResponse(BaseModel):
    profiles:   List[ProfileEntry]
    source:     str = "NFHS-5 India (2019-21)"
    disclaimer: str = DISCLAIMER


class ParseRequest(BaseModel):
    text: str = Field(..., description="Natural language condition e.g. 'rural women aged 25-34'")


class ParseResponse(BaseModel):
    parsed_condition: Condition
    raw_text:         str
    confidence:       float
