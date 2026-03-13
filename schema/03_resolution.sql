-- Etapa 1: Schema adaptation for resolved markets and bid/ask spread
-- Run after 01_core.sql and 02_time_series.sql

-- 1a. Resolution fields on the markets table
ALTER TABLE markets
  ADD COLUMN IF NOT EXISTS resolved_at           TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS winner_token_id       TEXT,
  ADD COLUMN IF NOT EXISTS uma_resolution_status TEXT;

-- winner_token_id has no FK intentionally:
-- upsert order is (1) INSERT markets → (2) INSERT market_outcomes → (3) UPDATE markets SET winner_token_id
-- A FK would violate constraint at step 1.

-- 1b. Bid and Ask columns on price_history (nullable — compatible with existing MID-only data)
ALTER TABLE price_history
  ADD COLUMN IF NOT EXISTS bid NUMERIC,
  ADD COLUMN IF NOT EXISTS ask NUMERIC;

-- 1c. Continuous Aggregate for daily price volatility (TimescaleDB)
CREATE MATERIALIZED VIEW IF NOT EXISTS price_volatility_1d
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 day', time)  AS bucket,
  token_id,
  stddev(price)               AS price_std_1d,
  avg(price)                  AS price_avg_1d,
  count(*)                    AS sample_count
FROM price_history
GROUP BY bucket, token_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
  'price_volatility_1d',
  start_offset      => INTERVAL '7 days',
  end_offset        => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour'
);

-- Index to speed up resolution-based filtering in the feature pipeline
CREATE INDEX IF NOT EXISTS idx_markets_resolved_at ON markets(resolved_at);
CREATE INDEX IF NOT EXISTS idx_markets_winner      ON markets(winner_token_id);
