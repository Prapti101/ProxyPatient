"""Shared disclosure threshold for respondent-backed counts and statistics."""
MIN_CELL_SIZE = 30


def suppress_count(n, minimum=MIN_CELL_SIZE):
    return int(n) if int(n) >= max(MIN_CELL_SIZE, minimum) else None


def suppress_stat(value, n, minimum=MIN_CELL_SIZE):
    return value if int(n) >= max(MIN_CELL_SIZE, minimum) else None
