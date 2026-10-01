"""
Data loader for 5-minute OHLCV bar data via the Massive REST API.

Study universe: 10 DJIA stocks (AXP, BA, CVX, DD, GE, IBM, JPM, KO, MSFT, XOM).
Full study period: 2010-01-01 to 2013-12-31.
  - ~1,006 trading days across 4 years.
  - A 1,000-day in-sample window yields a minimal but workable OOS period; a
    shorter feasibility window (e.g., 2010-2012) can be used when API rate-limits
    or storage are a concern.
"""

import os
import numpy as np
import pandas as pd
from massive import RESTClient
from typing import List, Dict, Tuple, Optional

# ── constants ────────────────────────────────────────────────────────────────

ASSETS: List[str] = ['AXP', 'BA', 'CVX', 'DD', 'GE', 'IBM', 'JPM', 'KO', 'MSFT', 'XOM']

# Regular trading hours in Eastern Time
_ET_OPEN  = pd.Timestamp('09:30').time()
_ET_CLOSE = pd.Timestamp('16:00').time()

# Maximum bars per Massive API request
_API_LIMIT = 50_000


# ── client ───────────────────────────────────────────────────────────────────

def get_massive_client() -> RESTClient:
    """
    Create a Massive RESTClient authenticated from the MASSIVE_TOKEN env var.

    The client is initialised with ``pagination=True`` (the library default),
    so ``list_aggs`` returns a lazy iterator that automatically fetches
    successive pages until the API is exhausted.
    """
    token = os.environ.get('MASSIVE_TOKEN')
    if not token:
        raise ValueError(
            "Environment variable MASSIVE_TOKEN is not set. "
            "Export your Massive API key before running the loader."
        )
    return RESTClient(api_key=token)


# ── intraday bars ─────────────────────────────────────────────────────────────

def load_intraday_bars(
    ticker: str,
    start: str,
    end: str,
    multiplier: int = 5,
    timespan: str = 'minute',
    client: Optional[RESTClient] = None,
) -> pd.DataFrame:
    """
    Load intraday aggregate bars for *ticker* from the Massive API.

    Parameters
    ----------
    ticker     : Exchange ticker symbol.
    start, end : Date strings 'YYYY-MM-DD' (inclusive).
    multiplier : Bar size multiplier (default 5 → 5-minute bars).
    timespan   : Timespan unit accepted by Massive ('minute', 'hour', …).
    client     : Pre-constructed RESTClient; one is created when *None*.

    Returns
    -------
    pd.DataFrame
        Columns: [open, high, low, close, volume, vwap, timestamp].
        Index   : UTC-aware DatetimeIndex.
        Rows are filtered to regular trading hours (09:30–16:00 ET) and
        cleaned of NaN/zero prices and price-change outliers (>10 σ).
    """
    if client is None:
        client = get_massive_client()

    # list_aggs handles pagination automatically (RESTClient default: pagination=True).
    records = []
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start,
        to=end,
        sort='asc',
        limit=_API_LIMIT,
    ):
        records.append({
            'open':      agg.open,
            'high':      agg.high,
            'low':       agg.low,
            'close':     agg.close,
            'volume':    agg.volume,
            'vwap':      agg.vwap,
            'timestamp': agg.timestamp,
        })

    if not records:
        return pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp']
        )

    df = pd.DataFrame(records)

    # Millisecond UTC epoch → timezone-aware DatetimeIndex
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()

    # ── filter to regular trading hours ──────────────────────────────────────
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert('America/New_York')
    rth_mask = (df_et.index.time >= _ET_OPEN) & (df_et.index.time < _ET_CLOSE)
    df_et = df_et.loc[rth_mask]

    # ── basic cleaning ────────────────────────────────────────────────────────
    price_cols = ['open', 'high', 'low', 'close']

    # Drop rows with any NaN price
    df_et = df_et.dropna(subset=price_cols)

    # Drop rows where any price is non-positive
    df_et = df_et.loc[(df_et[price_cols] > 0).all(axis=1)]

    # Remove outliers: bars whose close-to-close log-return exceeds 10 σ.
    # Cross-day differences are set to NaN so only within-day changes are tested.
    if len(df_et) > 1:
        log_ret    = np.log(df_et['close']).diff()
        date_arr   = df_et.index.normalize().date
        day_starts = pd.Series(
            np.concatenate([[True], date_arr[1:] != date_arr[:-1]]),
            index=df_et.index, dtype=bool,
        )
        log_ret[day_starts] = np.nan
        mu, sigma = log_ret.mean(), log_ret.std()
        if sigma > 0:
            outlier_ok = (log_ret - mu).abs() <= 10.0 * sigma
            outlier_ok.iloc[0] = True          # always keep the first bar
            outlier_ok = outlier_ok.fillna(True)  # NaN day-starts pass through
            df_et = df_et.loc[outlier_ok]

    # Return with UTC index
    df_et.index = df_et.index.tz_convert('UTC')
    return df_et


# ── daily returns ─────────────────────────────────────────────────────────────

def load_daily_returns(
    ticker: str,
    start: str,
    end: str,
    client: Optional[RESTClient] = None,
) -> pd.Series:
    """
    Load daily bars and compute open-to-close returns.

    Returns
    -------
    pd.Series
        Index : date-normalised UTC DatetimeIndex (one entry per trading day).
        Values: (close / open) − 1  (simple open-to-close return).
        Name  : *ticker*.
    """
    if client is None:
        client = get_massive_client()

    records = []
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan='day',
        from_=start,
        to=end,
        sort='asc',
        limit=_API_LIMIT,
    ):
        records.append({
            'open':      agg.open,
            'close':     agg.close,
            'timestamp': agg.timestamp,
        })

    if not records:
        return pd.Series(dtype=float, name=ticker)

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()

    # Normalise to date-level (midnight UTC) so indices align across assets
    df.index = df.index.normalize()

    # Drop any rows where open or close are missing / zero
    df = df.dropna(subset=['open', 'close'])
    df = df.loc[(df['open'] > 0) & (df['close'] > 0)]

    returns = df['close'] / df['open'] - 1.0
    returns.name = ticker
    return returns


# ── bulk asset loading ────────────────────────────────────────────────────────

def load_all_assets(
    assets: List[str],
    start: str,
    end: str,
    data_dir: str = 'data/raw/',
) -> Dict[str, pd.DataFrame]:
    """
    Load 5-minute bars for every ticker in *assets*, caching to *data_dir*.

    Files are written as Apache Parquet (``{ticker}_5min.parquet``). On
    subsequent calls the cache is used without contacting the API.

    Returns
    -------
    dict mapping ticker → cleaned bar DataFrame (see :func:`load_intraday_bars`).
    """
    os.makedirs(data_dir, exist_ok=True)
    client = get_massive_client()

    result: Dict[str, pd.DataFrame] = {}
    for ticker in assets:
        cache_path = os.path.join(data_dir, f'{ticker}_5min.parquet')
        if os.path.exists(cache_path):
            print(f'[loader] {ticker}: reading from cache ({cache_path})')
            df = pd.read_parquet(cache_path)
        else:
            print(f'[loader] {ticker}: fetching from Massive API ({start} → {end}) …')
            df = load_intraday_bars(ticker, start, end, client=client)
            df.to_parquet(cache_path)
            print(f'[loader] {ticker}: saved {len(df):,} bars → {cache_path}')
        result[ticker] = df

    return result


# ── return series construction ────────────────────────────────────────────────

def compute_5min_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5-minute log-returns from close prices.

    The first bar of each trading day is dropped (no prior close to diff from).

    Parameters
    ----------
    bars : DataFrame returned by :func:`load_intraday_bars` (UTC index).

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(date, interval)`` where *date* is a ``datetime.date``
        and *interval* is the 0-based intraday bar index within that day.
        Single column: ``log_return``.
    """
    if bars.empty:
        idx = pd.MultiIndex.from_tuples([], names=['date', 'interval'])
        return pd.DataFrame({'log_return': pd.Series(dtype=float, index=idx)})

    bars_et = bars.copy()
    bars_et.index = bars_et.index.tz_convert('America/New_York')

    log_close  = np.log(bars_et['close'])
    log_ret    = log_close.diff()

    dates      = bars_et.index.normalize()
    date_arr   = dates.date                      # array of datetime.date objects
    day_starts = pd.Series(
        np.concatenate([[True], date_arr[1:] != date_arr[:-1]]),
        index=bars_et.index, dtype=bool,
    )
    log_ret[day_starts] = np.nan          # no cross-day return

    df = pd.DataFrame({
        'log_return': log_ret,
        'date':       date_arr,
        'interval':   bars_et.groupby(dates).cumcount(),
    })
    df = df.dropna(subset=['log_return'])
    df = df.set_index(['date', 'interval'])
    return df


def align_returns_crosssectional(
    returns_dict: Dict[str, pd.Series],
    min_assets: int = 8,
) -> pd.DataFrame:
    """
    Align daily return series across assets and drop incomplete days.

    Parameters
    ----------
    returns_dict : Mapping of ticker → daily return Series (date index).
    min_assets   : Days with fewer non-NaN assets than this are dropped.

    Returns
    -------
    pd.DataFrame
        Shape (T, N), columns are ticker symbols, index is date.
    """
    df = pd.DataFrame(returns_dict)
    df = df.dropna(thresh=min_assets)
    return df


def load_spy_returns(
    start: str,
    end: str,
    client: Optional[RESTClient] = None,
) -> pd.Series:
    """Load SPY daily open-to-close returns (used as market benchmark)."""
    return load_daily_returns('SPY', start, end, client=client)
