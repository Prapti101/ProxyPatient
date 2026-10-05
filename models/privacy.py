"""Shared disclosure threshold for respondent-backed counts and statistics."""
MIN_CELL_SIZE = 30


def suppress_count(n, minimum=MIN_CELL_SIZE):
    if n is None:
        return None
    return int(n) if int(n) >= max(MIN_CELL_SIZE, minimum) else None


def suppress_stat(value, n, minimum=MIN_CELL_SIZE):
    return value if int(n) >= max(MIN_CELL_SIZE, minimum) else None

# Counts of algorithmic settings (epochs, layers, states, metric cells) are not
# respondent-backed counts; they must not be confused with disclosure support.
STRUCTURAL_COUNTS = {'n_states', 'n_cond', 'n_cont', 'n_cat', 'n_cells', 'n_components'}
COUNT_KEYS = {'n', 'requested', 'filled', 'drawn', 'accepted', 'redrawn_rows',
              'clipped_after_max_rounds', 'dropped_nonfinite', 'count', 'support'}


def is_respondent_count(key):
    return (key.startswith('n_') and key not in STRUCTURAL_COUNTS) or key in COUNT_KEYS


def safe_public_output(obj, minimum=MIN_CELL_SIZE):
    """Sanitize declared respondent counts, retaining structural model settings."""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if is_respondent_count(str(key)) and isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = suppress_count(value, minimum)
            elif key in ('dropped_incomplete', 'encoded_per_sex', 'excluded_per_sex') and isinstance(value, dict):
                result[key] = {k: suppress_count(v, minimum) for k, v in value.items()}
            else:
                result[key] = safe_public_output(value, minimum)
        return result
    if isinstance(obj, (list, tuple)):
        return [safe_public_output(value, minimum) for value in obj]
    return obj


def small_count_paths(obj, path=''):
    """Audit declared respondent-backed counts, including zero support."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f'{path}.{key}'
            if is_respondent_count(str(key)) and isinstance(value, (int, float)) and not isinstance(value, bool) and value < MIN_CELL_SIZE:
                found.append(here)
            found += small_count_paths(value, here)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found += small_count_paths(value, f'{path}[{i}]')
    return found
