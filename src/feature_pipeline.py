"""
feature_pipeline.py — Etapa 4

Transforms raw TimescaleDB data into a clean (X, y) tabular dataset
ready for ML model training. Exports to data/ml_dataset.parquet.

The bank does the heavy lifting (aggregations, window functions via
Continuous Aggregates). Python only post-processes and validates.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from db import get_db_connection

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
PARQUET_PATH = DATA_DIR / "ml_dataset.parquet"
META_PATH = DATA_DIR / "ml_dataset_meta.json"

# ---------------------------------------------------------------------------
# Imputation defaults
# ---------------------------------------------------------------------------

SPREAD_DEFAULT: float = 0.002   # typical liquid market spread (0.2%)
PRICE_STD_DEFAULT: float = 0.0  # no observed volatility — conservative
VOLUME_DEFAULT: float = 0.0     # no volume data — conservative

# ---------------------------------------------------------------------------
# Feature extraction SQL
# ---------------------------------------------------------------------------

# One snapshot per (token, day). The TimescaleDB Continuous Aggregate
# price_volatility_1d is used for volatility features. The bank does all
# GROUP BY / window logic; Python only post-processes and validates.

SQL_FEATURE_QUERY = """
WITH daily_snapshots AS (
  -- One snapshot per (token, day): uses the daily median price
  SELECT
    token_id,
    time_bucket('1 day', time)          AS snapshot_date,
    percentile_cont(0.5)
      WITHIN GROUP (ORDER BY price)     AS implied_prob,
    MAX(COALESCE(ask - bid, NULL))      AS spread_width_direct
  FROM price_history
  WHERE side IN ('MID', 'SNAPSHOT')
  GROUP BY token_id, snapshot_date
),

spread_fallback AS (
  -- Fallback spread: 'spread' field saved in market raw_data JSONB
  SELECT
    mo.token_id,
    (m.raw_data->>'spread')::numeric    AS market_spread
  FROM market_outcomes mo
  JOIN markets m ON mo.market_id = m.id
),

volatility_7d AS (
  -- 7-day rolling average of daily price stddev via Continuous Aggregate
  SELECT
    token_id,
    bucket,
    price_std_1d,
    AVG(price_std_1d) OVER (
      PARTITION BY token_id
      ORDER BY bucket
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                   AS price_std_7d_sql
  FROM price_volatility_1d
)

SELECT
  ds.token_id,
  mo.market_id,
  ds.snapshot_date,
  ds.implied_prob,
  COALESCE(ds.spread_width_direct,
           sf.market_spread)            AS spread_width,
  EXTRACT(EPOCH FROM
    (m.resolved_at - ds.snapshot_date::timestamptz)
  ) / 86400.0                           AS time_to_expiry_days,
  v.price_std_1d,
  v.price_std_7d_sql,
  (m.raw_data->>'volumeNum')::numeric   AS volume_total,
  (m.raw_data->>'volume1wk')::numeric   AS volume_1wk,
  (m.raw_data->>'category')             AS category,
  mo.outcome_index,
  CASE WHEN mo.token_id = m.winner_token_id
       THEN 1 ELSE 0 END               AS y,
  m.resolved_at

FROM daily_snapshots ds
JOIN market_outcomes mo  ON mo.token_id    = ds.token_id
JOIN markets m           ON m.id           = mo.market_id
LEFT JOIN spread_fallback sf ON sf.token_id = ds.token_id
LEFT JOIN volatility_7d v    ON v.token_id  = ds.token_id
                             AND v.bucket   = ds.snapshot_date

WHERE m.uma_resolution_status = 'resolved'
  AND m.winner_token_id IS NOT NULL
  AND ds.snapshot_date < m.resolved_at::date
  AND EXTRACT(EPOCH FROM
        (m.resolved_at - ds.snapshot_date::timestamptz)) > 3600

ORDER BY ds.snapshot_date, ds.token_id;
"""


# ---------------------------------------------------------------------------
# Transformation steps
# ---------------------------------------------------------------------------


def load_raw_data(conn) -> pd.DataFrame:
    """Execute the feature query and return a DataFrame with all raw columns."""
    logging.info("Executing feature query against TimescaleDB...")
    df = pd.read_sql(SQL_FEATURE_QUERY, conn)
    if df.empty:
        logging.info("Raw query returned 0 rows.")
        return df
    logging.info(
        f"Raw query returned {len(df):,} rows from "
        f"{df['market_id'].nunique():,} unique markets."
    )
    return df


def compute_price_std_7d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7-day rolling average of daily price stddev per token.

    Overrides the SQL window-function value with an equivalent Python
    rolling computation. This is more robust when the Continuous Aggregate
    is only partially populated (e.g., newly added tokens).

    The rolling mean uses min_periods=1 so tokens with fewer than 7 days
    of history still get a valid (though less stable) estimate.

    Args:
        df: DataFrame sorted by snapshot_date, containing 'price_std_1d'.

    Returns:
        DataFrame with 'price_std_7d' column added/overwritten.
    """
    df = df.sort_values(["token_id", "snapshot_date"]).copy()
    df["price_std_7d"] = df.groupby("token_id")["price_std_1d"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'category' field into stable boolean feature columns.

    Uses the top-level 'category' from Gamma API raw_data JSONB.
    This is more robust than parsing individual tags, which are heavily
    biased toward 2024 US election candidates.

    Produced columns: is_sports, is_crypto, is_politics, is_other.
    is_other = 1 when none of the three named categories match.

    Args:
        df: DataFrame with a 'category' string column.

    Returns:
        DataFrame with four new integer (0/1) category columns.
    """
    cat = df["category"].fillna("").str.lower()
    df = df.copy()
    df["is_sports"] = cat.str.contains("sport", na=False).astype(int)
    df["is_crypto"] = cat.str.contains("crypto", na=False).astype(int)
    df["is_politics"] = (
        cat.str.contains("affair", na=False)
        | cat.str.contains("election", na=False)
        | cat.str.contains("politic", na=False)
    ).astype(int)
    df["is_other"] = (
        (df["is_sports"] == 0)
        & (df["is_crypto"] == 0)
        & (df["is_politics"] == 0)
    ).astype(int)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values with domain-appropriate defaults.

    Imputation strategy:
      - spread_width: 0.002 — typical liquid market spread (order of magnitude
        correct even as a rough estimate for the liquidity feature)
      - price_std_1d/7d: 0.0 — conservative; no observed volatility
      - volume_total/1wk: 0.0 — no volume data available

    Args:
        df: DataFrame with potentially missing feature values.

    Returns:
        DataFrame with NaN values filled.
    """
    df = df.copy()
    df["spread_width"] = df["spread_width"].fillna(SPREAD_DEFAULT)
    df["price_std_1d"] = df["price_std_1d"].fillna(PRICE_STD_DEFAULT)
    df["price_std_7d"] = df["price_std_7d"].fillna(PRICE_STD_DEFAULT)
    df["volume_total"] = df["volume_total"].fillna(VOLUME_DEFAULT)
    df["volume_1wk"] = df["volume_1wk"].fillna(VOLUME_DEFAULT)
    return df


def validate_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assert data integrity invariants and absence of look-ahead bias.

    Checks performed:
      1. snapshot_date < resolved_at — no future information used
      2. time_to_expiry_days > 0 — no snapshot at resolution time
      3. implied_prob in [0.001, 0.999] — clips and warns if violated

    Args:
        df: Fully transformed feature DataFrame (before dropping helper columns).

    Returns:
        DataFrame with implied_prob clipped if needed.

    Raises:
        AssertionError: if invariants 1 or 2 are violated.
    """
    df = df.copy()

    # 1. No look-ahead: snapshot strictly before resolution date
    snapshot_dt = pd.to_datetime(df["snapshot_date"]).dt.normalize()
    resolved_dt = pd.to_datetime(df["resolved_at"]).dt.normalize()

    # Ensure both are tz-aware (UTC) before comparing — pd.read_sql may return
    # time_bucket results as tz-naive even when resolved_at is tz-aware.
    if snapshot_dt.dt.tz is None:
        snapshot_dt = snapshot_dt.dt.tz_localize("UTC")
    if resolved_dt.dt.tz is None:
        resolved_dt = resolved_dt.dt.tz_localize("UTC")

    bad_dates = snapshot_dt >= resolved_dt
    assert bad_dates.sum() == 0, (
        f"Look-ahead leakage: {bad_dates.sum()} rows have "
        f"snapshot_date >= resolved_at."
    )

    # 2. Positive time-to-expiry
    non_positive = df["time_to_expiry_days"] <= 0
    assert non_positive.sum() == 0, (
        f"Look-ahead leakage: {non_positive.sum()} rows have "
        f"time_to_expiry_days <= 0."
    )

    # 3. Probability bounds — clip instead of assert (API data can be noisy)
    out_of_range = (df["implied_prob"] < 0.001) | (df["implied_prob"] > 0.999)
    n_clipped = int(out_of_range.sum())
    if n_clipped > 0:
        logging.warning(
            f"Clipping {n_clipped} implied_prob values outside [0.001, 0.999]."
        )
        df["implied_prob"] = df["implied_prob"].clip(0.001, 0.999)

    y_rate = df["y"].mean()
    logging.info(f"Target distribution: {y_rate:.3%} positive (expected ~50%).")
    if not (0.35 <= y_rate <= 0.65):
        logging.warning(
            f"y distribution {y_rate:.3%} deviates significantly from 50%."
        )

    return df


def build_quality_report(df: pd.DataFrame) -> dict:
    """
    Build a quality summary dict for export as metadata JSON.

    Args:
        df: Transformed DataFrame, still containing 'category' and 'resolved_at'.

    Returns:
        Dict with row counts, date range, y distribution, and spread coverage.
    """
    spread_real_count = int(
        (df["spread_width"] != SPREAD_DEFAULT).sum()
    )
    return {
        "total_rows": len(df),
        "unique_markets": int(df["market_id"].nunique()),
        "unique_tokens": int(df["token_id"].nunique()),
        "date_range_start": str(df["snapshot_date"].min()),
        "date_range_end": str(df["snapshot_date"].max()),
        "y_positive_rate": float(round(df["y"].mean(), 4)),
        "pct_spread_real": float(
            round(spread_real_count / max(len(df), 1), 4)
        ),
        "y_by_category": (
            df.groupby("category")["y"]
            .mean()
            .round(4)
            .to_dict()
        ),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Final feature columns written to the parquet file
FEATURE_COLS = [
    "token_id",
    "market_id",
    "snapshot_date",
    "implied_prob",
    "time_to_expiry_days",
    "price_std_1d",
    "price_std_7d",
    "spread_width",
    "volume_total",
    "volume_1wk",
    "is_sports",
    "is_crypto",
    "is_politics",
    "is_other",
    "outcome_index",
    "y",
]


def run(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Execute the full feature pipeline end-to-end.

    Steps:
      1. Query TimescaleDB → raw DataFrame
      2. Compute price_std_7d rolling window
      3. One-hot encode categories
      4. Impute missing values
      5. Validate anti-leakage invariants
      6. Build quality report
      7. Export to Parquet + JSON metadata

    Args:
        output_dir: Override the default data/ directory (useful for testing).

    Returns:
        The final feature DataFrame (FEATURE_COLS only).
    """
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "ml_dataset.parquet"
    meta_out = out_dir / "ml_dataset_meta.json"

    conn = get_db_connection()
    try:
        df = load_raw_data(conn)
    finally:
        conn.close()

    if df.empty:
        logging.warning("Feature query returned no rows. Aborting pipeline.")
        return df

    logging.info("Step 2/5 — Computing price_std_7d rolling window.")
    df = compute_price_std_7d(df)

    logging.info("Step 3/5 — Encoding category features.")
    df = encode_categories(df)

    logging.info("Step 4/5 — Imputing missing values.")
    df = impute_missing_values(df)

    logging.info("Step 5/5 — Validating anti-leakage invariants.")
    df = validate_no_leakage(df)

    # Build report before dropping helper columns
    report = build_quality_report(df)

    # Keep only final feature columns
    df = df[FEATURE_COLS].copy()

    df.to_parquet(parquet_out, index=False, compression="snappy")
    logging.info(f"Dataset saved → {parquet_out}  ({len(df):,} rows)")

    meta_out.write_text(json.dumps(report, indent=2, default=str))
    logging.info(f"Metadata saved → {meta_out}")

    logging.info("=== Quality Report ===")
    for k, v in report.items():
        logging.info(f"  {k}: {v}")

    return df


if __name__ == "__main__":
    run()
