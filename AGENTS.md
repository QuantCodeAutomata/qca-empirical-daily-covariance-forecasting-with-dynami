# Project: Empirical Daily Covariance Forecasting with Dynamic Attenuation

## Overview
Quantitative-finance research project. Compares 8 covariance-forecasting models
(HAR, HARQ, HAR-DRD, HARQ-DRD, EWMA, EWMAQ, HEAVY, HEAVYQ) on DJIA stocks
using rolling-window OOS evaluation and GMV portfolio construction.

## Key Architecture

### Data pipeline  (`data/`)
| File | Purpose |
|------|---------|
| `data/loader.py` | Massive API → 5-min OHLCV bars; daily returns; cross-section alignment |
| `data/preprocess.py` | Realized covariance S_t, quarticity Pi_t, vech panels |

### Experiments (`exp/`)
| File | Purpose |
|------|---------|
| `exp/exp1_forecasting.py` | Rolling-window OOS forecast losses, DM tests, 90% MCS, split-sample analysis |
| `exp/exp2_portfolio.py` | GMV portfolio performance, CEQ gains, White Reality Check |

### Source modules (`src/`) — to be implemented separately
```
src/utils.py           – vech, unvech, is_spd, nearest_spd
src/realized_measures.py – compute_multivariate_kernel_cov, compute_pi_matrix/vector
src/losses.py          – frobenius_loss, qlike_loss, compute_losses_series, diebold_mariano_test
src/mcs.py             – model_confidence_set
src/models/har.py      – fit_har_ols, forecast_har, fit_harq_ols, forecast_harq
src/models/har_drd.py  – fit_har_drd, forecast_har_drd, fit_harq_drd, forecast_harq_drd
src/models/ewma.py     – EWMAModel, EWMAQModel
src/models/heavy.py    – HEAVYModel, HEAVYQModel
src/portfolio.py       – gmv_weights, compute_turnover, compute_concentration, insanity_filter
```

## Massive API (massive==2.8.0)
- `RESTClient(api_key=token)` — `pagination=True` by default.
- `client.list_aggs(ticker, multiplier, timespan, from_=start, to=end, sort='asc', limit=50000)`
  returns an auto-paginating **iterator** of `Agg` objects.
- `Agg` fields: `open, high, low, close, volume, vwap, timestamp (ms UTC), transactions, otc`.
- Auth: read `MASSIVE_TOKEN` env var; raise `ValueError` if unset (do not use default key).
- Use `list_aggs(multiplier=1, timespan='day')` for daily bars (no separate daily endpoint).

## Data Structures
- **S_panel / Pi_panel**: `pd.DataFrame(T, K)`, columns = `pd.MultiIndex[(asset_i, asset_j)]`
  for `i ≤ j` (upper-triangle / vech ordering).  K = N(N+1)/2.
- **S_matrices**: `np.ndarray(T, N, N)` — raw realized covariance matrices.
- **S_vech**: `np.ndarray(T, K)` — flattened upper triangle.
- **pi_vech**: `np.ndarray(T, K)` — realized quarticity vech.

## Performance Notes (exp1)
- **Critical bottleneck**: DRD decomposition of the full IS window at every OOS step.
- **Fix**: `_drd_decompose_panel(S_vech, n)` pre-computes the entire T-row DRD panel
  once; the rolling loop then takes O(1) slices.
- **Rolling HAR features**: use `_rolling_mean_cumsum(Y, w)` (cumsum trick) to
  pre-compute 5-day and 22-day means for the full series; slice IS portions in the loop.
- Together these reduce runtime from ~200 s → ~8 s (T=1500, N=5, T_oos=500).

## src Import Strategy
Both `exp/exp1_forecasting.py` and `exp/exp2_portfolio.py` wrap all `src.*` imports
in `try/except ImportError`. When src modules are absent (`_SRC_AVAILABLE = False`),
fully self-contained inline fallback implementations are used automatically so the
experiments can run and be validated before src is built.

## Synthetic Data Path
- Triggered when `data/S_matrices.npy` / `data/pi_vech.npy` do not exist.
- `_generate_synthetic_data(T=1500, n=5, M=78)`: AR(1) Cholesky latent covariance
  → Wishart(V_t, M)/M realized covariance; quarticity from (M/3) R_sq' R_sq.
- Exp2 synthetic path uses Gaussian returns drawn from the synthetic actuals.
- All synthetic output is labelled clearly in print statements and plot titles.

## Experiment Outputs (results/)
```
exp1_results.csv         – mean Frobenius/QLIKE per model + MCS membership
exp1_losses.png          – bar chart of mean losses (* = in 90% MCS)
exp1_cum_loss_diff.png   – cumulative loss differential curves (Q vs benchmark)
exp1_split_sample.png    – split-sample heatmap by pi_t tercile

exp2_results.csv         – ann. return/vol/Sharpe/turnover/concentration/net returns
exp2_ceq_gains.csv       – CEQ gains × {Q-pair, γ, cost}
exp2_wrc.csv             – White Reality Check p-values
exp2_volatility.png      – volatility bar chart
exp2_turnover.png        – turnover comparison
exp2_ceq_heatmap.png     – CEQ gain heatmap (γ × cost)
exp2_cumulative_returns.png – cumulative GMV return curves
```

## Model Reference
| Model | Formula |
|-------|---------|
| HAR | vech(H_t) = β₀ + β_d s_{t-1} + β_w s̄_w + β_m s̄_m |
| HARQ | HAR + (β_dQ π̃_{t-1}) · s_{t-1} interaction |
| HAR-DRD | HAR on log-vols + Fisher-corr from S=DPD decomposition |
| HARQ-DRD | HARQ on DRD components |
| EWMA | H_t = λ H_{t-1} + (1-λ) S_{t-1}, λ=0.94 |
| EWMAQ | EWMA with λ_{ij,t} = clip(0.94 + 0.02 π̃_{ij,t-1}, 0.5, 0.99) |
| HEAVY | H_t = ω + c S_{t-1} (OLS approximation of GARCH-type recursion) |
| HEAVYQ | HEAVY with c_eff = c + c_Q π̃_{t-1} |
