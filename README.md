Building a scraper for a prediction market like Polymarket requires handling two distinct types of data: **Relational Metadata** (Market details) and **Time-Series Data** (Price history).
Here is the detailed architecture breakdown.

---

### 1. The Data Classification
Before choosing tools, we must understand how the data behaves:

1.  **Market Metadata (Low Velocity, High Complexity):**
    *   *Data:* Title, description, categories, resolution dates, icon URLs.
    *   *Behavior:* Updates rarely. Requires complex filtering (e.g., "Find all Sports markets ending in 2025").
    *   *Storage:* Standard Relational Tables (PostgreSQL).
2.  **Price & Volume History (High Velocity, Append-Only):**
    *   *Data:* Price of "Yes" at 10:00 AM, Volume at 10:05 AM.
    *   *Behavior:* Massive volume of writes. Queries are usually aggregations (e.g., "OHLC candles").
    *   *Storage:* Time-Series Hypertable (TimescaleDB).

### 2. The Database Architecture (PostgreSQL + TimescaleDB)

Why this stack?
*   **Unified SQL:** You can join your price history with market metadata in a single query.
*   **Compression:** TimescaleDB compresses historical price data by 90%+, saving massive disk space.
*   **JSON Support:** Postgres has native JSONB support for storing raw API responses that might change structure.

#### The Schema Design

#### 1. Core Metadata Tables (Relational)

These tables store the "Facts" about the markets.

```sql
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

-- Indexes for frequent filtering
CREATE INDEX idx_events_active ON events(active);
CREATE INDEX idx_markets_condition ON markets(condition_id);
CREATE INDEX idx_outcomes_market ON market_outcomes(market_id);
```

#### 2. Tagging System (Normalized)

The JSON contains a list of tags. Normalizing this allows you to query "All Business markets".

```sql
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
```

#### 3. Time-Series Data (TimescaleDB)

This is where the massive data volume lives. We separate "Trade History" from "Market Stats History".

```sql
-- 4. PRICE HISTORY (OHLCV or Ticks)
-- Stores the price movement of specific outcome tokens (Yes/No)
CREATE TABLE price_history (
    time TIMESTAMPTZ NOT NULL,
    token_id TEXT NOT NULL REFERENCES market_outcomes(token_id), -- Links to "Yes" or "No"
    price NUMERIC,
    amount NUMERIC,    -- Volume of this specific trade/candle
    side TEXT          -- 'BUY' or 'SELL' (if capturing ticks)
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
```


### 3. The Scraper Architecture (Python-based)

You cannot simply loop through all markets sequentially; it will be too slow. You need an asynchronous pipeline.

**Tech Stack:** Python, Aiohttp (Async requests), Celery (Task Queue), Redis (Broker).

#### Component 1: The Discovery Worker (The "Manager")
*   **Frequency:** Runs every 1-5 minutes.
*   **Job:** Hits the Polymarket Gamma API (`/events` or `/markets`).
*   **Logic:**
    1.  Fetches list of active markets.
    2.  Upserts metadata into the `markets` table.
    3.  Detects new markets.
    4.  **Crucial:** Pushes a job to the Queue for price updates.

#### Component 2: The Price Worker (The "Grunt")
*   **Trigger:** Triggered by the Discovery Worker.
*   **Job:** Fetches price history (candles) or Order Book data.
*   **Logic:**
    1.  Receives `market_id` from Queue.
    2.  Hits the CLOB (Central Limit Order Book) API for the latest tick/candle.
    3.  Batches data.
    4.  Performs a bulk insert (`COPY`) into the `price_history` table.

#### Component 3: The Snapshot Worker (Top Holders)
*   **Frequency:** Runs less often (e.g., every 6 hours) as holder data changes slower.
*   **Job:** Queries the leaderboard/holders endpoint.
*   **Logic:** Replaces or updates entries in the `positions` table.

---

### 4. Handling API Limitations & Optimization

Polymarket (and the underlying blockchain RPCs) have rate limits.

1.  **Proxy Rotation:** If you scrape aggressively, you will need a proxy service.
2.  **CLOB vs. Gamma API:**
    *   Use **Gamma API** for metadata (Descriptions, Images).
    *   Use **CLOB API** for Prices. It is faster and geared towards trading bots.
3.  **ETags / Caching:** In your `markets` table, store a `last_updated` timestamp. Only scrape deep data for markets that have activity.

---

### 5. Example Query Capabilities

With this architecture, you can run powerful analytical queries:

**Query: Get the price history of "Yes" for all "Sports" markets:**
```sql
SELECT time_bucket('1 hour', p.time) as bucket,
       m.question,
       avg(p.price)
FROM price_history p
JOIN tokens t ON p.token_id = t.token_id
JOIN markets m ON t.market_id = m.id
WHERE m.category = 'Sports'
GROUP BY bucket, m.question;
```

**Query: Find "Whales" (Top holders across multiple markets):**
```sql
SELECT user_address, sum(value_usdc) as total_exposure
FROM positions
GROUP BY user_address
ORDER BY total_exposure DESC
LIMIT 10;
```

### 6. Summary of Recommended Stack

1.  **Language:** Python 3.11+ (Type hinting is helpful for complex data).
2.  **Scraping Libs:** `httpx` (Async HTTP), `tenacity` (Retrying failed requests).
3.  **Database:** PostgreSQL 16 + TimescaleDB extension.
4.  **Queue:** Redis (Keep it simple, no need for RabbitMQ yet).
5.  **Containerization:** Docker Compose (to spin up Python, Postgres, and Redis together).