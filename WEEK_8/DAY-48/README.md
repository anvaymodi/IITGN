# Week 08 · Thursday — RNNs & Sequential Data

PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar

## What this is

Two fintech sequential-data problems for Vikram Anand's architecture review:

1. **Stock forecasting** — next-day close price for five Indian equities (regression, time-series LSTM).
2. **Churn prediction** — customer-churn risk from chat logs (classification, sequence vs. tabular comparison).

All seven sub-steps are covered:

| Sub-step | Difficulty | Status |
|---|---|---|
| 1 — Stock sequence construction + split defence | 🟢 Easy | ✓ |
| 2 — Chat log timestamp repair + EDA | 🟢 Easy | ✓ |
| 3 — LSTM stock model + directional-accuracy bar | 🟡 Medium | ✓ |
| 4 — Churn: sequence vs. tabular comparison | 🟡 Medium | ✓ |
| 5 — Cost-aware outreach list + threshold selection | 🟡 Medium | ✓ |
| 6 — AR baseline audit (Hard) | 🔴 Hard | ✓ |
| 7 — Manual BPTT + vanishing gradient demo (Hard) | 🔴 Hard | ✓ |

Both Hard sub-steps are attempted (only one required for Band 4 — this gives a safety margin).

## How to run

```bash
# Python 3.11+ required (tested on 3.12)
pip install -r requirements.txt
jupyter notebook week08_day48_rnn.ipynb
# Then: Kernel -> Restart & Run All
```

The notebook runs end-to-end in ~2–3 minutes on CPU.

## Data

The notebook looks for the LMS datasets in this order:

1. `/mnt/user-data/uploads/stock_prices.csv` and `chat_logs.csv` (the grading environment)
2. `./data/stock_prices.csv` and `./data/chat_logs.csv` (local run)
3. If neither found, **deterministic synthetic data is generated** that mirrors the Kaggle reference schemas (Nifty50 stock market data; Bitext customer-support chatbot dataset). The synthetic `chat_logs.csv` deliberately contains the mixed-format timestamp wart that Sub-step 2 is about.

The TA running from a clean environment will get consistent results via the seeded synthesis fallback.

## Project layout

```
week-08/DAY-48/
├── README.md                         ← you are here
├── requirements.txt
├── week08_day48_rnn.ipynb            ← the submission
├── prompts.md                        ← AI prompts + critique per sub-step
├── churn_risk_list.csv               ← Sub-step 5 output
└── src/
    ├── __init__.py
    ├── data_loaders.py               ← hybrid real/synthetic loaders
    ├── timestamp_utils.py            ← mixed-format timestamp parser (Sub-step 2)
    ├── sequence_builders.py          ← windowing + chronological split
    ├── models.py                     ← StockLSTM, ChurnLSTM, trainer
    ├── evaluation.py                 ← metrics + cost-aware threshold
    └── manual_bptt.py                ← hand-rolled BPTT for Sub-step 7
```

## Engineering-quality indicators (Dimension B)

- **Readable naming.** `compute_adf_test()`-style naming throughout; no `x`, `temp2`, or magic one-letter variables.
- **Modular structure.** Every sub-step's logic lives in a `src/` module and the notebook is thin glue. 7+ functions across 6 modules.
- **No magic numbers.** Window size, cost parameters, hidden dims, spectral-radius scales — all named constants (`DEFAULT_WINDOW_SIZE`, `DEFAULT_COST_FALSE_NEGATIVE`, `STOCK_LSTM_HIDDEN`, etc.) with inline justification.
- **Defensive handling.** `try/except` around CSV I/O in `data_loaders._try_read_csv`; NaT handling in `timestamp_utils`; empty-split and edge-case checks in `sequence_builders.chronological_split`; shape validation in `build_sequences`.

## Key methodological commitments

These show up in the notebook narrative but worth stating here too:

1. **Chronological splits on time-series.** Random splits leak future data into training — documented in Sub-step 1.
2. **Scaler fitted on training data only.** Using full-series statistics before splitting is a subtler leak.
3. **Metric chosen by application.** Directional accuracy for trading (not RMSE); PR-AUC for imbalanced churn (not plain accuracy or ROC-AUC).
4. **Costs quantified.** Outreach threshold comes from a stated cost model (FP ₹5, FN ₹80), not a default 0.5.
5. **Baselines run.** Every complex model is audited against a simpler one (naive t-1, AR(30), tabular RF). Complexity must earn its place.

## AI Usage Policy (Dimension B)

AI-assisted sub-steps are documented in `prompts.md` with the prompts
used and a critique of each output — which parts of the AI's defaults
I kept, which I modified, and why. Read that file for the methodology
decisions behind every sub-step.
