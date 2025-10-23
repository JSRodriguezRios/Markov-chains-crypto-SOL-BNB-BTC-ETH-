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

## Next Steps

- Phase 2: Implement the API fetching layer (`qcsrc/pipeline/fetch_*.py`).
- Phase 3: Normalize and align datasets into hourly panels.
- Phase 4: Build feature matrices used by the HMM.
- Phase 5: Train and evaluate Gaussian HMMs for each asset.
- Phase 6-8: Export probabilities and integrate into the QuantConnect algorithm.
- Phase 9: Backtest, evaluate Sortino, Drawdown, Sharpe, Win/Loss statistics, and refine parameters.

