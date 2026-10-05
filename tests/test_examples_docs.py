"""Validate captured mock documentation against the current public contract."""
import json
from pathlib import Path
import re
from backend.schemas import GenerateRequest, CompareRequest, SyntheticExample


def test_captured_scenario_contracts():
    text = Path('docs/API_CONTRACT.md').read_text()
    for endpoint, model in [('generate', GenerateRequest), ('compare', CompareRequest)]:
        heading = f'### POST /{endpoint}'
        def block(kind):
            match = re.search(re.escape(heading + ' ' + kind) + r'[^\n]*\n\n```json\n(.*?)\n```', text, re.S)
            assert match, heading
            return json.loads(match.group(1))
        request = model.model_validate(block('request'))
        response = block('response')
        scenarios = response['scenarios'] if endpoint == 'compare' else [response]
        requests = request.scenarios if endpoint == 'compare' else [request]
        for result, req in zip(scenarios, requests, strict=True):
            assert result['demo'] and result['examples_status'] == 'available: MOCK'
            assert len(result['examples']) <= req.n_examples
            assert result['outcome_stat']['n'] == req.n
            for example in result['examples']:
                SyntheticExample.model_validate(example)
                assert 'MOCK' in example['label'] and example['synthetic']
