"""One guarded rule-based proposal parser; optional offline annotations never generate conditions.

P3's original is archived in legacy and is never imported. Its useful aliases and
relative-BMI phrases are reconciled here. No temporal modelling is implied.
"""
import os
import re
from pathlib import Path
from functools import lru_cache
from fastapi import HTTPException
from backend.schemas import Condition, ParseResponse
from models.common import load_config


@lru_cache(maxsize=1)
def _offline_pipeline(directory):
    # Imports are lazy and occur only for an explicitly configured existing local model.
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(directory, local_files_only=True)
    return pipeline('token-classification', model=model, tokenizer=tokenizer, aggregation_strategy='simple', device=-1)


def offline_token_hook(text):
    directory = os.environ.get('PP_BERT_MODEL_DIR')
    if not directory:
        return {'status': 'disabled', 'used_for_conditions': False}
    if not Path(directory).is_dir():
        return {'status': 'unavailable', 'used_for_conditions': False, 'note': 'Offline model directory missing'}
    try:
        entities = _offline_pipeline(str(Path(directory).resolve()))(text)
        return {'status': 'available', 'used_for_conditions': False,
                'annotations': [{'token': str(e['word']), 'label': str(e['entity_group'])} for e in entities]}
    except (ImportError, OSError, ValueError, RuntimeError, KeyError):
        return {'status': 'unavailable', 'used_for_conditions': False, 'note': 'Offline token model could not be loaded'}


def propose(req):
    _cfg = load_config()
    text = req.text.lower()
    if re.search(r"blood[ -]?sugar|\binsulin\b", text):
        raise HTTPException(422, "glucose is the outcome, not an input; medication is not a supported condition")
    text = re.sub(r"body[ -]mass[ -]index", "bmi", text)
    hook = offline_token_hook(req.text)
    if re.search(r"glucose|hba1c|sb74|smb74", text):
        raise HTTPException(422, "glucose is the outcome, not an input")
    condition, unresolved = {}, []
    if re.search(r"\bunchanged\b|same as baseline|no change|\bstable\b", text):
        return ParseResponse(parsed_condition=Condition(), raw_text=req.text, optional_token_hook=hook)
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
    if re.search(r"(?:improved|better|lower|reduced)\s+bmi", text):
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
    if re.search(r"\b(?:no|without|not)\s+(?:hypertension|high blood pressure)\b", text):
        condition["hypertension"] = 0
    elif re.search(r"\bhypertension\b|high blood pressure", text):
        condition["hypertension"] = 1
    wealth = re.search(r"wealth(?: quintile)?\s*([1-5])\b", text)
    if wealth:
        condition["wealth_quintile"] = int(wealth.group(1))
    state = re.search(r"\bstate\s*(\d+)\b", text)
    if state:
        condition["state"] = int(state.group(1))
    try:
        proposed = Condition(**condition)
    except ValueError as exc:
        raise HTTPException(422, "Unsupported parsed condition; choose values from /options") from exc
    return ParseResponse(parsed_condition=proposed, raw_text=req.text, unresolved=unresolved,
                         optional_token_hook=hook)

