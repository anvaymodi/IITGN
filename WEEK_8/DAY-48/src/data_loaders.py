"""
Data loading for stock_prices.csv and chat_logs.csv.

Tries real LMS files first at LMS_UPLOAD_DIR. If unavailable, synthesizes
datasets that mirror the Kaggle reference schemas:
  - Nifty50 stock market data (rohanrao/nifty50-stock-market-data)
  - Bitext customer-support chatbot dataset (bitext/bitext-customer-support-...)

All synthetic data is deterministic (seeded) so reruns are reproducible.
The chat_logs.csv we synthesize deliberately has the timestamp-parsing
issue the assignment warns about — Sub-step 2 resolves it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Constants (no magic numbers)
LMS_UPLOAD_DIR = Path("/mnt/user-data/uploads")
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_TRADING_DAYS = 1500          # ~6 years of daily bars per stock
N_STOCKS = 5                   # five Indian equities (matches scenario)
STOCK_TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
CHAT_N_CUSTOMERS = 3000        # customer base for chat_logs
CHAT_MIN_INTERACTIONS = 1
CHAT_MAX_INTERACTIONS = 25
CHURN_RATE_TARGET = 0.18       # realistic fintech churn base rate
RANDOM_SEED = 42


def _try_read_csv(filename: str) -> pd.DataFrame | None:
    """Return DataFrame if CSV exists in LMS upload or local data dir, else None."""
    for folder in (LMS_UPLOAD_DIR, LOCAL_DATA_DIR):
        candidate = folder / filename
        if candidate.exists():
            try:
                return pd.read_csv(candidate)
            except Exception as exc:   # noqa: BLE001 - log and fall through
                print(f"[data_loaders] Failed to read {candidate}: {exc}")
    return None


def _synthesize_stock_prices(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate stock_prices.csv that mirrors the Nifty50 Kaggle schema.

    Columns: Date, Symbol, Open, High, Low, Close, Volume, VWAP
    Uses geometric Brownian motion with a mild trend + volatility clusters
    so the LSTM has something non-trivial to learn.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start_date = pd.Timestamp("2019-01-02")
    dates = pd.bdate_range(start_date, periods=N_TRADING_DAYS)

    for i, ticker in enumerate(STOCK_TICKERS):
        mu = 0.0003 + 0.0002 * i
        sigma = 0.012 + 0.003 * (i % 3)
        start_price = 500.0 + 250.0 * i

        returns = rng.normal(mu, sigma, size=N_TRADING_DAYS)
        for regime_start in range(0, N_TRADING_DAYS, 200):
            regime_end = min(regime_start + 200, N_TRADING_DAYS)
            vol_mult = rng.uniform(0.7, 1.8)
            returns[regime_start:regime_end] *= vol_mult

        close = start_price * np.exp(np.cumsum(returns))
        daily_range = np.abs(rng.normal(0, 0.008, N_TRADING_DAYS)) * close
        open_ = close + rng.normal(0, 0.004, N_TRADING_DAYS) * close
        high = np.maximum(open_, close) + daily_range / 2
        low = np.minimum(open_, close) - daily_range / 2
        volume = rng.integers(500_000, 5_000_000, N_TRADING_DAYS)
        vwap = (open_ + high + low + close) / 4

        for d, o, h, l, c, v, w in zip(dates, open_, high, low, close, volume, vwap):
            rows.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Symbol": ticker,
                "Open": round(o, 2),
                "High": round(h, 2),
                "Low": round(l, 2),
                "Close": round(c, 2),
                "Volume": int(v),
                "VWAP": round(w, 2),
            })
    return pd.DataFrame(rows)


def _synthesize_chat_logs(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate chat_logs.csv with deliberate data-quality issues.

    Columns: customer_id, timestamp, channel, intent, message_length,
             sentiment_score, resolved, churned_within_30d

    Schema mirrors the Bitext customer-support training dataset
    (intents like "cancel_account", "payment_issue") but enriched with
    customer and churn signal.

    KEY WART (Sub-step 2 must find this): timestamps are written in a mix
    of formats — some as ISO, some as "DD/MM/YYYY HH:MM", some as unix
    epoch integers rendered as strings. A naive pd.to_datetime() call
    will fail or silently mis-parse a chunk.
    """
    rng = np.random.default_rng(seed)
    intents_churn_heavy = [
        "cancel_account", "payment_issue", "complaint",
        "contact_human_agent", "delivery_problem",
    ]
    intents_neutral = [
        "check_balance", "get_invoice", "track_order",
        "edit_account", "change_password", "newsletter_subscription",
    ]
    intents_happy = ["place_order", "recover_password", "create_account"]
    channels = ["web_chat", "whatsapp", "mobile_app", "email"]

    rows = []
    for customer_id in range(1, CHAT_N_CUSTOMERS + 1):
        n_interactions = rng.integers(CHAT_MIN_INTERACTIONS, CHAT_MAX_INTERACTIONS + 1)
        will_churn = rng.random() < CHURN_RATE_TARGET
        # Label noise: ~15% of customers behave "like the other class".
        # This keeps the task realistic — otherwise AUC tops out at 1.0
        # and we can't meaningfully compare models.
        behavioural_flip = rng.random() < 0.15
        effective_churn_pattern = will_churn ^ behavioural_flip

        if effective_churn_pattern:
            # Churners: some complaint signal, but not overwhelmingly so.
            # Real-world data has weaker separation than a schoolbook example.
            intent_pool = (
                intents_churn_heavy * 2 + intents_neutral * 3 + intents_happy * 1
            )
            sentiment_center = -0.15
            resolve_rate = 0.55
        else:
            intent_pool = (
                intents_neutral * 3 + intents_happy * 2 + intents_churn_heavy * 1
            )
            sentiment_center = 0.10
            resolve_rate = 0.75

        base_time = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
        for k in range(n_interactions):
            ts_raw = base_time + pd.Timedelta(hours=int(rng.integers(1, 200)))
            fmt_choice = rng.integers(0, 3)
            if fmt_choice == 0:
                ts_str = ts_raw.strftime("%Y-%m-%dT%H:%M:%S")
            elif fmt_choice == 1:
                ts_str = ts_raw.strftime("%d/%m/%Y %H:%M")
            else:
                ts_str = str(int(ts_raw.timestamp()))

            intent = rng.choice(intent_pool)
            msg_len = max(5, int(rng.normal(80 if not effective_churn_pattern else 55, 30)))
            sentiment = float(np.clip(rng.normal(sentiment_center, 0.25), -1.0, 1.0))
            resolved = bool(rng.random() < resolve_rate)
            rows.append({
                "customer_id": customer_id,
                "timestamp": ts_str,
                "channel": rng.choice(channels),
                "intent": intent,
                "message_length": msg_len,
                "sentiment_score": round(sentiment, 3),
                "resolved": resolved,
                "churned_within_30d": will_churn,
            })
            base_time = ts_raw
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_stock_prices() -> Tuple[pd.DataFrame, str]:
    """Load stock prices; returns (df, source) where source is 'lms' or 'synthetic'."""
    df = _try_read_csv("stock_prices.csv")
    if df is not None:
        return df, "lms"
    print("[data_loaders] stock_prices.csv not found on LMS — synthesizing.")
    return _synthesize_stock_prices(), "synthetic"


def load_chat_logs() -> Tuple[pd.DataFrame, str]:
    """Load chat logs; returns (df, source) where source is 'lms' or 'synthetic'."""
    df = _try_read_csv("chat_logs.csv")
    if df is not None:
        return df, "lms"
    print("[data_loaders] chat_logs.csv not found on LMS — synthesizing.")
    return _synthesize_chat_logs(), "synthetic"


def save_to_local_data(df: pd.DataFrame, filename: str) -> Path:
    """Persist a synthesized dataset for reproducibility."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOCAL_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path
