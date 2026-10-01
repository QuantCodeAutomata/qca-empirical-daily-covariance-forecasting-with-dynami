# Empirical Daily Covariance Forecasting with Dynamic Attenuation

Replication and extension of the HARQ framework for multivariate realized
covariance matrices. The project implements HAR, HARQ, HAR-DRD, HARQ-DRD,
EWMA, and HEAVY model families and evaluates them on 10 DJIA stocks using
5-minute intraday data.

---

## Project Description

The central idea is that high-frequency realized covariance estimators contain
measurement noise whose magnitude varies day-to-day. The **HARQ** model
(Bollerslev et al.) dynamically attenuates the daily autoregressive lag by a
noise proxy π_t — the ratio of realised quarticity to squared realised
covariance. When noise is high, HARQ shrinks the daily signal; when noise is
low it is passed through unchanged.

The **DRD** (Diagonal–Rotation–Diagonal) decomposition separates each
covariance matrix into its volatility dynamics (D) and correlation dynamics
(R), allowing HAR and HARQ to be applied separately and then recombined.

---

## Repository Structure

```
project/
├── data/                          # raw and processed data (not committed)
│   ├── raw/                       # downloaded tick / 5-min OHLCV files
│   ├── realized_covariances.parquet
│   ├── pi_matrices.parquet
│   └── daily_returns.parquet
│
├── src/
│   ├── utils.py                   # vech, unvech, nearest_spd, is_spd, …
│   ├── losses.py                  # frobenius_loss, qlike_loss, DM test
│   ├── realized_measures.py       # Parzen kernel, MK estimator, quarticity
│   ├── portfolio.py               # gmv_weights, CEQ, turnover, …
│   └── models/
│       ├── har.py                 # HAR and HARQ (vectorized OLS)
│       ├── har_drd.py             # HAR-DRD and HARQ-DRD
│       ├── ewma.py                # EWMA and EWMAQ
│       └── heavy.py               # HEAVY and HEAVYQ
│
├── exp/
│   ├── exp1_forecasting.py        # OOS covariance loss comparison
│   ├── exp2_portfolio.py          # unconstrained GMV portfolio exercise
│   ├── exp3_robustness.py         # reliability split, long-only, multi-horizon, shrinkage
│   └── exp4_simulation.py         # Monte Carlo simulation study
│
├── tests/
│   ├── test_utils.py
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_portfolio.py
│   └── test_realized_measures.py
│
├── results/
│   ├── RESULTS.md                 # human-readable summary (committed)
│   └── exp{1..4}_results.csv      # numeric results tables (committed)
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Set the MASSIVE_TOKEN environment variable

Data is fetched via the `massive` / `quantStates` API. Export your token
before running any experiment or data-loading step:

```bash
export MASSIVE_TOKEN="<your-token-here>"
```

### 3 — Prepare data

Run the data pipeline to download 5-minute DJIA returns and compute
realized covariances, quarticity proxies, and daily returns:

```bash
python -m src.data_pipeline          # downloads and caches to data/
```

This writes three parquet files consumed by all four experiments:
- `data/realized_covariances.parquet` — (T, K) vech of N×N realized cov
- `data/pi_matrices.parquet`          — (T, K) vech of noise proxy
- `data/daily_returns.parquet`        — (T, N) close-to-close log-returns

---

## Running Experiments

Each script is self-contained. Results and plots are written to `results/`.

```bash
# Experiment 1: OOS forecast losses (Frobenius, QLIKE) for all models
python exp/exp1_forecasting.py

# Experiment 2: GMV portfolio performance (vol, Sharpe, turnover, CEQ)
python exp/exp2_portfolio.py

# Experiment 3: Robustness — reliability split, long-only GMV,
#               multi-horizon (weekly/monthly), ex-post shrinkage
python exp/exp3_robustness.py

# Experiment 4: Monte Carlo simulation — HAR vs HARQ vs HAR_∞
#               across sampling frequencies (1-min, 5-min, 15-min)
python exp/exp4_simulation.py --seed 42
```

---

## Testing

Run the full test suite with:

```bash
pytest tests/ -v
```

Tests cover:
- `test_utils.py`             — vech/unvech roundtrip, nearest_spd, is_spd
- `test_losses.py`            — Frobenius, QLIKE, Diebold–Mariano
- `test_models.py`            — HAR/HARQ shapes, DRD roundtrip, EWMA/HEAVY
- `test_portfolio.py`         — GMV optimality, turnover, CEQ, insanity filter
- `test_realized_measures.py` — Parzen kernel, kernel RCov, quarticity, pi

---

## Models

| Model       | Description                                          |
|-------------|------------------------------------------------------|
| HAR         | Heterogeneous AutoRegression on vech(S_t)            |
| HARQ        | HAR + daily-lag attenuation by noise proxy π_t       |
| HAR-DRD     | HAR applied to DRD-decomposed D_t and R_t separately |
| HARQ-DRD    | HARQ applied to DRD components                       |
| EWMA        | Exponentially Weighted Moving Average (scalar α)     |
| EWMAQ       | EWMA with noise-proxy reweighting                    |
| HEAVY       | High-frEquency-bAsed VolatilitY model                |
| HEAVYQ      | HEAVY with noise attenuation                         |

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for populated tables after
running all four experiments.

---

## Reference

Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016).
*Exploiting the errors: A simple approach for improved volatility forecasting.*
Journal of Econometrics, 192(1), 1–18.
