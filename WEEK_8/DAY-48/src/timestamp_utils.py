"""
Timestamp parsing utilities for chat_logs.csv.

The assignment warns: "chat_logs.csv timestamps are not in a consistent
format — resolve this before any feature engineering". This module
diagnoses the format mix and parses each row with the correct strategy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


MIN_PLAUSIBLE_YEAR = 2015
MAX_PLAUSIBLE_YEAR = 2030
UNIX_EPOCH_LOWER = pd.Timestamp(f"{MIN_PLAUSIBLE_YEAR}-01-01").timestamp()
UNIX_EPOCH_UPPER = pd.Timestamp(f"{MAX_PLAUSIBLE_YEAR}-01-01").timestamp()


def diagnose_timestamp_formats(series: pd.Series, sample_size: int = 20) -> Dict[str, int]:
    """Sniff the timestamp column and report how many look like each format."""
    sample = series.dropna().astype(str).head(sample_size).tolist()
    counts = {"iso_like": 0, "eu_slash": 0, "unix_epoch_str": 0, "unknown": 0}
    for s in sample:
        if s.isdigit() and UNIX_EPOCH_LOWER <= int(s) <= UNIX_EPOCH_UPPER:
            counts["unix_epoch_str"] += 1
        elif "T" in s and "-" in s:
            counts["iso_like"] += 1
        elif "/" in s:
            counts["eu_slash"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _parse_one(value):
    """Parse a single timestamp string using format detection."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    s = str(value).strip()
    if not s:
        return pd.NaT

    if s.isdigit():
        try:
            epoch = int(s)
            if UNIX_EPOCH_LOWER <= epoch <= UNIX_EPOCH_UPPER:
                return pd.to_datetime(epoch, unit="s")
        except (ValueError, OverflowError):
            pass

    if "T" in s:
        try:
            return pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S", errors="raise")
        except (ValueError, TypeError):
            pass

    if "/" in s:
        try:
            return pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="raise")
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(s, dayfirst=True, errors="raise")
            except (ValueError, TypeError):
                return pd.NaT

    try:
        return pd.to_datetime(s, errors="raise")
    except (ValueError, TypeError):
        return pd.NaT


def parse_mixed_timestamps(series: pd.Series) -> pd.Series:
    """Parse a column containing mixed timestamp formats."""
    return series.map(_parse_one)
