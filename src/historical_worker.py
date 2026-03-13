"""
historical_worker.py — Etapa 2

Collects all resolved markets from the Polymarket Gamma API, extracts the
winner of each market, persists everything to the DB, and enqueues market_ids
for price_worker.py to populate the price history.

Usage:
  python historical_worker.py --mode full
  python historical_worker.py --mode incremental [--lookback-days 7]

Modes:
  full         Scan all ~546K closed markets (one-time job, checkpoint-aware).
  incremental  Only process events closed within --lookback-days days.
               Suitable for periodic cron/Docker scheduling.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from db import (
    get_db_connection,
    link_tags_to_events,
    update_market_resolution,
    upsert_events,
    upsert_market_outcomes,
    upsert_markets,
    upsert_tags,
)
from gamma_api import PolymarketGammaClient
from redis_client import get_redis_connection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_KEY = "historical_worker:last_offset"
PRICE_QUEUE = "price_worker_queue"
PAGE_SIZE = 100
REQUEST_DELAY = 0.5  # seconds between API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def extract_winner_token_id(market: dict) -> Optional[str]:
    """
    Determines the winning token_id of a resolved market.

    Returns None if resolution is ambiguous (e.g. all prices == 0.5, N/A) or
    if the market data is missing / malformed.

    The Gamma API represents the winner as the outcome where
    ``outcomePrices[i] == "1"``. All values are strings ("0" or "1") in normal
    resolved markets.

    Args:
        market: Raw market dict as returned by the Gamma API.

    Returns:
        The winning token_id string, or None.
    """
    try:
        prices = json.loads(market.get("outcomePrices", "[]"))
        token_ids = json.loads(market.get("clobTokenIds", "[]"))

        for i, price_str in enumerate(prices):
            if float(price_str) == 1.0 and i < len(token_ids):
                return token_ids[i]

    except (json.JSONDecodeError, ValueError, IndexError):
        return None

    return None  # N/A or ambiguous resolution — exclude from ML


def _is_within_lookback(event: dict, cutoff: datetime) -> bool:
    """
    Returns True if at least one market in the event closed after *cutoff*.

    Used by incremental mode to decide whether to keep processing a page.

    Args:
        event: Raw event dict from the Gamma API.
        cutoff: Timezone-aware datetime; events older than this are skipped.
    """
    for market in event.get("markets", []):
        closed_time_str = market.get("closedTime")
        if not closed_time_str:
            continue
        try:
            closed_time = datetime.fromisoformat(
                closed_time_str.replace("Z", "+00:00")
            )
            if closed_time >= cutoff:
                return True
        except ValueError:
            continue
    return False


def _process_event(conn, redis_conn, event: dict) -> int:
    """
    Persists a single closed event and all its markets/outcomes/tags to the DB,
    updates resolution metadata, and enqueues each market_id for price history
    collection.

    Args:
        conn: Active psycopg2 connection.
        redis_conn: Active Redis connection.
        event: Raw event dict from the Gamma API.

    Returns:
        Number of market_ids enqueued to ``PRICE_QUEUE``.
    """
    event_id = event.get("id")
    if not event_id:
        logging.warning("Skipping event with no ID.")
        return 0

    # --- Persist event, tags, and the event↔tag relationship ---
    upsert_events(conn, [event])
    upsert_tags(conn, [event])
    link_tags_to_events(conn, [event])

    markets = event.get("markets", [])
    if markets:
        upsert_markets(conn, markets, event_id)
        upsert_market_outcomes(conn, markets)

    # --- Per-market: set resolution fields and enqueue for price scraping ---
    enqueued = 0
    for market in markets:
        market_id = market.get("id")
        if not market_id:
            continue

        winner = extract_winner_token_id(market)
        update_market_resolution(
            conn,
            market_id=market_id,
            winner_token_id=winner,
            resolved_at=market.get("closedTime"),
            uma_resolution_status=market.get("umaResolutionStatus"),
        )

        redis_conn.lpush(PRICE_QUEUE, market_id)
        enqueued += 1

    return enqueued


# ---------------------------------------------------------------------------
# Operational modes
# ---------------------------------------------------------------------------


def run_full(
    gamma_client: PolymarketGammaClient,
    conn,
    redis_conn,
) -> None:
    """
    Full-scan mode: iterate all closed events from the last saved checkpoint.

    Resumes automatically after a crash by reading the offset stored in Redis.
    The checkpoint is updated *after* each successful page so no data is lost.

    Args:
        gamma_client: Configured Gamma API client.
        conn: Active psycopg2 connection.
        redis_conn: Active Redis connection (used for checkpoint + queue).
    """
    offset = int(redis_conn.get(CHECKPOINT_KEY) or 0)
    logging.info(f"[full] Starting from offset={offset}")

    total_events = 0
    total_enqueued = 0

    while True:
        events = gamma_client.get_events(
            limit=PAGE_SIZE,
            offset=offset,
            active=False,
            closed=True,
        )

        if not events:
            logging.info(f"[full] Reached end of data at offset={offset}. Done.")
            break

        for event in events:
            enqueued = _process_event(conn, redis_conn, event)
            total_enqueued += enqueued

        total_events += len(events)
        offset += PAGE_SIZE
        redis_conn.set(CHECKPOINT_KEY, offset)

        logging.info(
            f"[full] offset={offset}, events_so_far={total_events}, "
            f"enqueued_so_far={total_enqueued}"
        )
        time.sleep(REQUEST_DELAY)

    logging.info(
        f"[full] Complete. total_events={total_events}, "
        f"markets_enqueued={total_enqueued}"
    )


def run_incremental(
    gamma_client: PolymarketGammaClient,
    conn,
    redis_conn,
    lookback_days: int = 7,
) -> None:
    """
    Incremental mode: collect only events closed within the last *lookback_days*.

    Always starts from offset=0 (no checkpoint) and stops as soon as an entire
    page contains no events within the lookback window.

    Args:
        gamma_client: Configured Gamma API client.
        conn: Active psycopg2 connection.
        redis_conn: Active Redis connection (used for queue only).
        lookback_days: How far back to look for recently-closed events.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    logging.info(
        f"[incremental] lookback_days={lookback_days}, cutoff={cutoff.isoformat()}"
    )

    offset = 0
    total_events = 0
    total_enqueued = 0

    while True:
        events = gamma_client.get_events(
            limit=PAGE_SIZE,
            offset=offset,
            active=False,
            closed=True,
        )

        if not events:
            logging.info(f"[incremental] No more events at offset={offset}. Done.")
            break

        any_within_window = False
        for event in events:
            if _is_within_lookback(event, cutoff):
                any_within_window = True
                enqueued = _process_event(conn, redis_conn, event)
                total_enqueued += enqueued
                total_events += 1

        if not any_within_window:
            logging.info(
                "[incremental] All events in this page are outside the lookback "
                "window. Stopping."
            )
            break

        offset += PAGE_SIZE
        logging.info(
            f"[incremental] offset={offset}, events_so_far={total_events}, "
            f"enqueued_so_far={total_enqueued}"
        )
        time.sleep(REQUEST_DELAY)

    logging.info(
        f"[incremental] Complete. total_events={total_events}, "
        f"markets_enqueued={total_enqueued}"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket Historical Data Worker (Etapa 2)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="full: scan all closed markets; incremental: only recent closes",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="(incremental only) How many days back to look for closed events",
    )
    args = parser.parse_args()

    conn = get_db_connection()
    redis_conn = get_redis_connection()
    gamma_client = PolymarketGammaClient()

    try:
        if args.mode == "full":
            run_full(gamma_client, conn, redis_conn)
        else:
            run_incremental(gamma_client, conn, redis_conn, args.lookback_days)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
