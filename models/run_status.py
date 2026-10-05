"""Run provenance shared by packaging and HTTP contracts."""
RUN_TYPES = ('mock', 'quick', 'full')


def run_fields(run_type):
    if run_type is not None and run_type not in RUN_TYPES:
        raise ValueError('run_type must be mock, quick or full')
    return {'run_type': run_type, 'preliminary': run_type == 'quick',
            'demo': run_type == 'mock',
            'status_banner': {'mock': 'DEMO (mock data)', 'quick': 'PRELIMINARY (quick run)',
                              'full': 'FULL RUN'}.get(run_type, 'UNAVAILABLE'),
            'run_note': 'reduced, untuned run; results are indicative only' if run_type == 'quick' else None}


def resolve_run_type(mock=False, quick=False):
    return 'mock' if mock else 'quick' if quick else 'full'
