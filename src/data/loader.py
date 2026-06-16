"""Data loading utilities for the StreamFlix synthetic A/B test dataset."""
from pathlib import Path
import pandas as pd


EXPECTED_COLS = {
    "user_id", "timestamp", "group", "landing_page",
    "country", "device", "source", "is_returning",
    "prior_watch_hours", "converted",
    "trial_watch_hours", "distinct_titles", "day7_active", "page_load_ms",
}


def load_experiment(path: str | Path = "data/experiment.csv") -> pd.DataFrame:
    """Load the synthetic StreamFlix A/B test dataset.

    Generate it first with `python src/data/simulate.py` if `experiment.csv`
    is missing.

    Parameters
    ----------
    path : str or Path
        Path to experiment.csv.

    Returns
    -------
    pd.DataFrame with experiment-ready columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with: python src/data/simulate.py"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df
