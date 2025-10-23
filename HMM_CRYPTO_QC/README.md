# HMM Crypto QuantConnect Strategy

This repository scaffolds a QuantConnect (LEAN) project that trains hourly Hidden Markov Models (HMM) for BTC, ETH, SOL, and BNB spot trading. The layout mirrors the structure required to collect raw market data, engineer features, train models offline, and deploy them inside QuantConnect cloud backtests or live trading.

## Getting Started

1. **Clone and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Copy credentials template**
   ```bash
   cp config/credentials.example.json secrets/credentials.json
   ```
   Fill the file with the API keys for CryptoQuant and CoinStats. Binance order book snapshots do not require an API key.
3. **Environment variables**
   Copy `.env.example` to `.env` when running locally so scripts can load secrets via `python-dotenv`.

## QuantConnect Notes

- Upload the `qcsrc/` folder and all required data exports in `qc_data/` to the QuantConnect project.
- QuantConnect already bundles `numpy`, `pandas`, `requests`, `scipy`, `statsmodels`, and `scikit-learn`. Packages such as `hmmlearn`, `pyarrow`, and `pydantic` must be added via the QC **Project Dependencies** panel using wheels that match Python 3.10.
- The `lean.json` file is optional for local Lean CLI usage, referencing the main algorithm at `qcsrc/HMMCryptoAlgorithm.py`.

## Repository Layout (abridged)

```
HMM_CRYPTO_QC/
├─ config/            # YAML settings for assets, features, schedules
├─ data/              # Raw, interim, processed datasets and models
├─ qc_data/           # Custom data exports formatted for QC
├─ qcsrc/             # Algorithm, feature builders, pipelines, utilities
├─ notebooks/         # EDA and diagnostics notebooks
└─ tests/             # Unit tests covering pipeline and algorithm helpers
```

## Backtesting & Evaluation

1. **Prepare data** – run the ingestion + processing pipeline:
   ```bash
   python -m qcsrc.pipeline.run_all --start 2022-01-01 --end 2023-01-01
   ```
   This populates `data/interim/`, `data/processed/`, and the trained models in `data/models/hmm/`.
2. **Upload resources to QuantConnect** – copy the contents of `qc_data/` and `data/models/` to the QC Object Store so the algorithm can warm start in cloud backtests.
3. **Launch a 1-year backtest** – inside QuantConnect set the start/end dates (e.g., 2022-01-01 to 2023-01-01) and select hourly resolution. Ensure the brokerage model is spot (`BinanceUS` or default crypto brokerage).
4. **Collect QC statistics** – after the run finishes, export the backtest results JSON. Key metrics such as Sharpe, Sortino, drawdown, win and loss rates are automatically captured by QuantConnect and mirrored in the algorithm’s metrics reports.
5. **Retrieve metrics JSON** – each asset writes a diagnostics file to `data/models/diagnostics/<SYMBOL>_metrics.json` containing:
   - Latest MAE, RMSE, and MAPE of the return forecasts.
   - Win/loss distribution derived from `TradeBuilder.ClosedTrades`.
   - Sharpe/Sortino/Drawdown snapshots from `self.Statistics`.
   These files are created during `OnEndOfAlgorithm` and can be downloaded from the cloud filesystem post-backtest.
6. **Analyse notebooks** – open the notebooks under `notebooks/` locally to visualise state sequences, transition matrices, and asset-specific EDA. They expect the processed parquet files generated earlier.

## Parameter Tuning Workflow

Use `config/settings.yaml` and `config/features.yaml` to iterate on thresholds, HMM state counts, and feature windows:

1. Adjust `threshold` (e.g., 0.6–0.8) and volatility/return windows.
2. Rerun the offline training pipeline (`python -m qcsrc.pipeline.train_hmm`).
3. Refresh probabilities with `python -m qcsrc.pipeline.infer_proba` and re-upload exports.
4. Trigger a new backtest and compare the generated metrics JSON plus QC dashboard statistics.

Document experiment outcomes (parameter combinations, metric deltas) directly in the notebooks or by appending to the `Backtesting & Evaluation` section.

## Deployment Runbook

1. **Update datasets** – execute `python -m qcsrc.pipeline.run_all --live` to pull the latest data snapshots.
2. **Retrain models** – run `python -m qcsrc.pipeline.train_hmm --retrain --symbols BTC,ETH,SOL,BNB` before deploying changes.
3. **Refresh custom data exports** – rerun `python -m qcsrc.pipeline.infer_proba` to regenerate the CSV feeds used by the QuantConnect algorithm.
4. **Upload to QuantConnect** – sync `qcsrc/`, `qc_data/`, and `data/models/` through the QC CLI or GUI, ensuring secrets remain excluded.
5. **Smoke backtest** – launch a short (e.g., 7-day) backtest to verify that metrics reports are produced and thresholds behave as expected.
6. **Promote to live** – once satisfied, schedule the same files in a live QuantConnect project with spot brokerage. Monitor the generated metrics JSON after trading sessions to confirm forecasts remain stable.

## Next Steps

- Phase 2: Implement the API fetching layer (`qcsrc/pipeline/fetch_*.py`).
- Phase 3: Normalize and align datasets into hourly panels.
- Phase 4: Build feature matrices used by the HMM.
- Phase 5: Train and evaluate Gaussian HMMs for each asset.
- Phase 6-8: Export probabilities and integrate into the QuantConnect algorithm.
- Phase 9: Backtest, evaluate Sortino, Drawdown, Sharpe, Win/Loss statistics, and refine parameters.

