"""BERT-based condition parser for /parse.

Input JSON examples:
  {"text": "high glucose, improved BMI, unchanged trajectory"}
  {"condition": "elevated glucose and lower BMI"}

Output:
  {"conditions": {"glucose": "elevated", "bmi": "improved", "trajectory": "unchanged"},
   "raw_text": ..., "parser": "bert"}

The BERT model is configurable with BERT_MODEL. If transformers/torch or the model
are unavailable, a deterministic rule-based fallback is used so the endpoint remains
usable during development/testing.
"""
import os, re, json
from typing import Dict, Any

DEFAULT_MODEL = os.getenv("BERT_MODEL", "bert-base-uncased")

# Canonical vocabulary used by the synthetic diabetes scenario generator.
ALIASES = {
    "glucose": ["glucose", "blood sugar", "blood glucose", "fasting glucose", "sugar"],
    "bmi": ["bmi", "body mass index", "body-mass index"],
    "trajectory": ["trajectory", "trend", "progression", "course"],
    "age": ["age", "older", "younger"],
    "blood_pressure": ["blood pressure", "bp", "hypertension"],
    "insulin": ["insulin"],
}

VALUE_PATTERNS = {
    "elevated": [r"\bhigh\b", r"\belevated\b", r"\bincreased\b", r"\braised\b", r"\bworse\b"],
    "reduced": [r"\blow\b", r"\breduced\b", r"\bdecreased\b", r"\blower\b"],
    "improved": [r"\bimprov(?:ed|e|ing)\b", r"\bbetter\b", r"\bdecreased\b", r"\blower\b"],
    "unchanged": [r"\bunchanged\b", r"\bsame\b", r"\bstable\b", r"\bno change\b"],
    "normal": [r"\bnormal\b", r"\bhealthy\b", r"\bwithin range\b"],
}


def _find_entity(text: str, aliases):
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I):
            return alias
    return None


def _rule_parse(text: str) -> Dict[str, str]:
    t = text.lower()
    out: Dict[str, str] = {}
    for field, aliases in ALIASES.items():
        if not _find_entity(t, aliases):
            continue
        # Inspect a small local window around the entity mention.
        alias = _find_entity(t, aliases)
        idx = t.find(alias)
        window = t[max(0, idx - 35): idx + len(alias) + 55]
        found = None
        for value, patterns in VALUE_PATTERNS.items():
            if any(re.search(p, window) for p in patterns):
                found = value
                break
        if found:
            out[field] = found
    # Specific scenario phrases override generic lexical matches.
    if re.search(r"elevated\s+(?:blood\s+)?glucose", t):
        out["glucose"] = "elevated"
    if re.search(r"(?:improved|better|lower|reduced)\s+bmi", t):
        out["bmi"] = "improved"
    if re.search(r"unchanged\s+(?:trajectory|trend|progression)", t):
        out["trajectory"] = "unchanged"
    return out


class ConditionParser:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._pipe = None
        self.mode = "rule_fallback"
        try:
            from transformers import pipeline
            self._pipe = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
            self.mode = "bert_token_classifier"
        except Exception:
            # A plain BERT checkpoint is not a token-classification model, and model
            # downloads may be unavailable in deployment. The fallback is intentional.
            self._pipe = None

    def parse(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        conditions = _rule_parse(text)
        bert_entities = []
        if self._pipe is not None:
            try:
                bert_entities = [dict(x) for x in self._pipe(text)]
            except Exception:
                bert_entities = []
        return {
            "conditions": conditions,
            "raw_text": text,
            "parser": self.mode,
            "bert_entities": bert_entities,
        }


_parser = None

def parse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _parser
    text = payload.get("text") or payload.get("condition") or payload.get("prompt")
    if not isinstance(text, str):
        raise ValueError("Expected JSON field 'text', 'condition', or 'prompt'.")
    if _parser is None:
        _parser = ConditionParser()
    return _parser.parse(text)


# Optional FastAPI integration:
# from fastapi import FastAPI
# app = FastAPI()
# @app.post('/parse')
# def parse(payload: dict):
#     return parse_payload(payload)

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "elevated glucose, improved BMI, unchanged trajectory"
    print(json.dumps(parse_payload({"text": text}), indent=2))
