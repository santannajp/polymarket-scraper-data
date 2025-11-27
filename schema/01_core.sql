-- 1. EVENTS: The high-level grouping (e.g., "Fed Rates")
CREATE TABLE events (
    id TEXT PRIMARY KEY,               -- Root 'id' (e.g., "16084")
    ticker TEXT,                       -- "fed-rate-hike-in-2025"
    slug TEXT,
    title TEXT,
    description TEXT,
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    creation_date TIMESTAMPTZ,
    image_url TEXT,
    icon_url TEXT,
    active BOOLEAN,
    is_restricted BOOLEAN,             -- "restricted": true
    cyom BOOLEAN,                      -- "Create Your Own Market"
    raw_data JSONB,                    -- Store full root JSON here for safety
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. MARKETS: The specific tradeable questions
CREATE TABLE markets (
    id TEXT PRIMARY KEY,               -- 'markets[].id' (e.g., "516706")
    event_id TEXT REFERENCES events(id),
    condition_id TEXT UNIQUE,          -- '0x431...' (Vital for blockchain lookups)
    question_id TEXT,                  -- '0x842...'
    question TEXT,
    resolution_source TEXT,            -- Often a URL or text
    market_maker_address TEXT,

    -- Status flags
    active BOOLEAN,
    closed BOOLEAN,
    archived BOOLEAN,
    accepting_orders BOOLEAN,

    -- Configurations
    uma_bond NUMERIC,
    order_min_size NUMERIC,
    order_price_min_tick_size NUMERIC,

    raw_data JSONB,                    -- Store the specific object from the markets array
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. MARKET OUTCOMES (Linking Tokens to Labels)
-- This "unzips" the 'clobTokenIds' and 'outcomes' arrays.
CREATE TABLE market_outcomes (
    token_id TEXT PRIMARY KEY,         -- From 'clobTokenIds' (The long numeric string)
    market_id TEXT REFERENCES markets(id),
    outcome_label TEXT,                -- "Yes", "No", "Trump", "Biden"
    outcome_index INT,                 -- 0 for Yes, 1 for No (helps maintain order)

    -- Current snapshot stats (updated frequently, queried often)
    current_price NUMERIC,             -- From 'lastTradePrice' or 'outcomePrices'
    best_bid NUMERIC,
    best_ask NUMERIC
);

CREATE TABLE tags (
    id TEXT PRIMARY KEY, -- "100196"
    slug TEXT UNIQUE,    -- "fed-rates"
    label TEXT           -- "Fed Rates"
);

CREATE TABLE event_tags (
    event_id TEXT REFERENCES events(id),
    tag_id TEXT REFERENCES tags(id),
    PRIMARY KEY (event_id, tag_id)
);

-- Indexes for frequent filtering
CREATE INDEX idx_events_active ON events(active);
CREATE INDEX idx_markets_condition ON markets(condition_id);
CREATE INDEX idx_outcomes_market ON market_outcomes(market_id);