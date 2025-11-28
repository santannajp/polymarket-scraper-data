-- 4. PRICE HISTORY (OHLCV or Ticks)
-- Stores the price movement of specific outcome tokens (Yes/No)
CREATE TABLE price_history (
    time TIMESTAMPTZ NOT NULL,
    token_id TEXT NOT NULL REFERENCES market_outcomes(token_id), -- Links to "Yes" or "No"
    price NUMERIC,
    amount NUMERIC,    -- Volume of this specific trade/candle
    side TEXT,         -- 'BUY' or 'SELL' (if capturing ticks)
    PRIMARY KEY (time, token_id)
);

-- Convert to Hypertable (TimescaleDB magic)
SELECT create_hypertable('price_history', 'time');


-- 5. MARKET STATS HISTORY
-- Tracks the liquidity and volume growth of the market as a whole over time
CREATE TABLE market_stats_history (
    time TIMESTAMPTZ NOT NULL,
    market_id TEXT NOT NULL REFERENCES markets(id),
    liquidity_clob NUMERIC,
    volume_24hr NUMERIC,
    volume_total NUMERIC,
    open_interest NUMERIC
);

SELECT create_hypertable('market_stats_history', 'time');

ALTER TABLE price_history SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'token_id'
);
SELECT add_compression_policy('price_history', INTERVAL '7 days');