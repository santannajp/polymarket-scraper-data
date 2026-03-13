-- Migration 04: Add category, slug (markets), liquidity, and volume columns
-- Run after 01_core.sql, 02_time_series.sql, and 03_resolution.sql

-- Events: category is a plain string at the root of the event object.
-- Liquidity and volume are numeric at the root of the event object.
ALTER TABLE events
  ADD COLUMN IF NOT EXISTS category  TEXT,
  ADD COLUMN IF NOT EXISTS liquidity NUMERIC,
  ADD COLUMN IF NOT EXISTS volume    NUMERIC;

-- Markets: slug and category are plain strings.
-- Liquidity/volume come from the numeric fields (liquidityNum / volumeNum),
-- because the top-level "liquidity" and "volume" fields are returned as strings
-- by the Gamma API.
ALTER TABLE markets
  ADD COLUMN IF NOT EXISTS slug      TEXT,
  ADD COLUMN IF NOT EXISTS category  TEXT,
  ADD COLUMN IF NOT EXISTS liquidity NUMERIC,
  ADD COLUMN IF NOT EXISTS volume    NUMERIC;

-- Optional indexes for common filter patterns
CREATE INDEX IF NOT EXISTS idx_events_category  ON events(category);
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS idx_markets_slug     ON markets(slug);
