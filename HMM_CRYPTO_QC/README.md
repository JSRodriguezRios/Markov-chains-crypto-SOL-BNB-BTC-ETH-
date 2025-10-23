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

## Bootstrapping & Evaluation

Follow these steps to move from raw data to a QuantConnect backtest that mirrors the nine-phase roadmap:

1. **Wrapper chain** – execute the ingestion and processing pipeline to refresh every staged dataset.
   ```bash
   python -m qcsrc.pipeline.run_all --start 2022-01-01 --end 2023-01-01
   ```
   The command hydrates `data/interim/`, builds feature matrices in `data/processed/`, and stores retrained HMMs beneath `data/models/hmm/`.
2. **QC resources** – upload the generated artifacts (`qc_data/` exports and `data/models/`) to the QuantConnect Object Store or project data folder so the algorithm can warm start online.
3. **Backtest window** – configure an hourly backtest of at least one year inside QuantConnect (for example, 2022-01-01 through 2023-01-01) using a spot brokerage model such as `BinanceUS`.
4. **Metrics capture** – once the run completes, download the backtest JSON to archive the standard QC statistics (Sharpe, Sortino, drawdown, win/loss rate, trade counts). The algorithm also logs Sortino and drawdown snapshots through its diagnostics channel.
5. **Local diagnostics** – every asset persists a JSON report to `data/models/diagnostics/<SYMBOL>_metrics.json` summarising MAE, RMSE, MAPE, and trade outcomes derived from `TradeBuilder.ClosedTrades`. Fetch these files from the QC filesystem to compare runs.
6. **Notebook review** – inspect the notebooks stored under `notebooks/` to visualise regime sequences, transition matrices, and asset-specific feature behaviour. They expect the parquet outputs produced in step 1.

## Parameter Tuning Workflow

Use `config/settings.yaml` and `config/features.yaml` to iterate on thresholds, HMM state counts, and feature windows:

1. Adjust `threshold` (e.g., 0.6–0.8) and volatility/return windows.
2. Rerun the offline training pipeline (`python -m qcsrc.pipeline.train_hmm`).
3. Refresh probabilities with `python -m qcsrc.pipeline.infer_proba` and re-upload exports.
4. Trigger a new backtest and compare the generated metrics JSON plus QC dashboard statistics.
5. Document experiment outcomes (parameter combinations, metric deltas) directly in the notebooks or by appending to the `Bootstrapping & Evaluation` section.

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

