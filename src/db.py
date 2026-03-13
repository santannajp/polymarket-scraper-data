import os
import json
import psycopg2
from psycopg2.extras import Json, execute_values
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "db"),
            dbname=os.environ.get("DB_NAME", "polymarket"),
            user=os.environ.get("DB_USER", "admin"),
            password=os.environ.get("DB_PASSWORD", "password"),
            port=os.environ.get("DB_PORT", "5432"),
        )
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Could not connect to PostgreSQL database: {e}")
        raise

def _flatten_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens the nested JSON structure of an event for database insertion."""
    return {
        "id": event.get("id"),
        "ticker": event.get("ticker"),
        "slug": event.get("slug"),
        "title": event.get("title"),
        "description": event.get("description"),
        "start_date": event.get("start_date"),
        "end_date": event.get("end_date"),
        "creation_date": event.get("creation_date"),
        "image_url": event.get("image"),
        "icon_url": event.get("icon"),
        "active": event.get("active"),
        "is_restricted": event.get("restricted"),
        "raw_data": Json(event),
    }

def upsert_events(conn, events: List[Dict[str, Any]]):
    """
    Upserts a list of events into the 'events' table.
    An upsert is an "update" or "insert" operation.
    """
    if not events:
        return

    flat_events = [_flatten_event(e) for e in events]
    
    # Filter out events with no ID
    flat_events = [e for e in flat_events if e['id'] is not None]
    if not flat_events:
        logging.warning("Skipping event upsert, no valid events with IDs found.")
        return

    columns = flat_events[0].keys()
    
    # Create the string for the ON CONFLICT clause
    update_cols = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])

    # Use execute_values for efficient batch insertion
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            f"""
            INSERT INTO events ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                {update_cols},
                updated_at = NOW()
            """,
            [list(event.values()) for event in flat_events],
            template=None,
            page_size=100
        )
    conn.commit()
    logging.info(f"Upserted {len(flat_events)} events.")

def _flatten_market(market: Dict[str, Any], event_id: str) -> Dict[str, Any]:
    """Flattens the nested JSON structure of a market for database insertion."""
    return {
        "id": market.get("id"),
        "event_id": event_id,
        "condition_id": market.get("conditionId"),
        "question_id": market.get("questionID"),
        "question": market.get("question"),
        "description": market.get("description"),
        "created_at_date": market.get("createdAt"),
        "start_date": market.get("startDate"),
        "end_date": market.get("endDate"),
        "accepting_orders_date": market.get("acceptingOrdersTimestamp"),
        "competitive": market.get("competitive"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "archived": market.get("archived"),
        "accepting_orders": market.get("acceptingOrders"),
        "uma_bond": market.get("umaBond"),
        "order_min_size": market.get("orderMinSize"),
        "order_price_min_tick_size": market.get("orderPriceMinTickSize"),
        # Resolution fields (Etapa 1)
        "resolved_at": market.get("closedTime"),
        "winner_token_id": None,           # set later via update_market_resolution()
        "uma_resolution_status": market.get("umaResolutionStatus"),
        "raw_data": Json(market),
    }

def upsert_markets(conn, markets: List[Dict[str, Any]], event_id: str):
    """
    Upserts a list of markets into the 'markets' table.
    """
    if not markets:
        return

    flat_markets = [_flatten_market(m, event_id) for m in markets]
    
    # Filter out markets with no ID
    flat_markets = [m for m in flat_markets if m['id'] is not None and m['condition_id']]
    if not flat_markets:
        logging.warning("Skipping market upsert, no valid markets with IDs found.")
        return

    columns = flat_markets[0].keys()
    
    # Create the string for the ON CONFLICT clause
    update_cols = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            f"""
            INSERT INTO markets ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                {update_cols},
                updated_at = NOW()
            """,
            [list(market.values()) for market in flat_markets],
            template=None,
            page_size=100
        )
    conn.commit()
    logging.info(f"Upserted {len(flat_markets)} markets for event {event_id}.")

def upsert_market_outcomes(conn, markets: List[Dict[str, Any]]):
    """
    Upserts market outcomes into the 'market_outcomes' table.
    Outcomes are extracted from the market objects, handling JSON strings.
    """
    if not markets:
        return

    all_outcomes = []
    for market in markets:
        market_id = market.get("id")
        
        try:
            # These fields are JSON-encoded strings in the API response
            outcomes_str = market.get("outcomes", "[]")
            clob_token_ids_str = market.get("clobTokenIds", "[]")

            outcome_labels = json.loads(outcomes_str)
            token_ids = json.loads(clob_token_ids_str)

        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Skipping outcomes for market {market_id} due to JSON parsing error: {e}")
            continue

        if not market_id or len(outcome_labels) != len(token_ids):
            logging.warning(f"Skipping outcomes for market {market_id} due to mismatched outcomes/token IDs.")
            continue

        for i, label in enumerate(outcome_labels):
            all_outcomes.append({
                "token_id": token_ids[i],
                "market_id": market_id,
                "outcome_label": label,
                "outcome_index": i,
            })

    if not all_outcomes:
        logging.warning("No valid market outcomes to upsert.")
        return

    columns = all_outcomes[0].keys()
    # Only update label and index on conflict, as price is not stored.
    update_cols = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['token_id', 'market_id']])

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            f"""
            INSERT INTO market_outcomes ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (token_id) DO UPDATE SET
                {update_cols}
            """,
            [list(outcome.values()) for outcome in all_outcomes],
            template=None,
            page_size=100
        )
    conn.commit()
    logging.info(f"Upserted {len(all_outcomes)} market outcomes.")

def upsert_tags(conn, events: List[Dict[str, Any]]):
    """
    Upserts tags from a list of events into the 'tags' table.
    """
    if not events:
        return

    all_tags = []
    for event in events:
        tags = event.get("tags", [])
        for tag in tags:
            all_tags.append({
                "id": tag.get("id"),
                "slug": tag.get("slug"),
                "label": tag.get("label"),
            })

    if not all_tags:
        return
        
    # Remove duplicates
    unique_tags = list({t['id']: t for t in all_tags if t['id']}.values())
    if not unique_tags:
        return

    columns = unique_tags[0].keys()
    update_cols = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            f"""
            INSERT INTO tags ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                {update_cols}
            """,
            [list(tag.values()) for tag in unique_tags],
            template=None,
            page_size=100
        )
    conn.commit()
    logging.info(f"Upserted {len(unique_tags)} tags.")

def link_tags_to_events(conn, events: List[Dict[str, Any]]):
    """
    Links tags to events in the 'event_tags' join table.
    This should be called after upserting both events and tags.
    """
    if not events:
        return

    event_tag_pairs = []
    for event in events:
        event_id = event.get("id")
        tags = event.get("tags", [])
        if not event_id:
            continue
        for tag in tags:
            tag_id = tag.get("id")
            if tag_id:
                event_tag_pairs.append((event_id, tag_id))
    
    if not event_tag_pairs:
        return

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO event_tags (event_id, tag_id)
            VALUES %s
            ON CONFLICT (event_id, tag_id) DO NOTHING
            """,
            event_tag_pairs,
            template=None,
            page_size=100
        )
    conn.commit()
    logging.info(f"Linked {len(event_tag_pairs)} event-tag relationships.")

def get_token_ids_for_market(conn, market_id: str) -> List[str]:
    """Fetches all token_ids for a given market_id."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT token_id FROM market_outcomes WHERE market_id = %s", (market_id,))
        return [row[0] for row in cursor.fetchall()]

def get_all_token_ids(conn) -> List[str]:
    """Fetches all token_ids from the market_outcomes table."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT token_id FROM market_outcomes")
        return [row[0] for row in cursor.fetchall()]

def get_active_token_ids(conn) -> List[str]:
    """
    Fetches token_ids for all outcomes belonging to currently active, non-closed markets.

    Used by the orderbook_snapshot_worker to determine which tokens need bid/ask snapshots.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT mo.token_id
            FROM market_outcomes mo
            JOIN markets m ON mo.market_id = m.id
            WHERE m.active = true AND m.closed = false
            """
        )
        return [row[0] for row in cursor.fetchall()]

def insert_price_history(
    conn,
    token_id: str,
    price_data: List[Dict[str, Any]],
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    side: str = "MID",
):
    """
    Inserts a list of price data points into the 'price_history' table.

    Args:
        conn: Active psycopg2 connection.
        token_id: The CLOB token identifier.
        price_data: List of dicts with keys 't' (unix timestamp) and 'p' (price).
        bid: Optional best-bid price to store alongside each record (Etapa 1).
        ask: Optional best-ask price to store alongside each record (Etapa 1).
        side: Record type — 'MID' for historical series, 'SNAPSHOT' for live bid/ask captures.
    """
    if not price_data:
        return

    records = []
    for point in price_data:
        ts = datetime.fromtimestamp(point['t'], tz=timezone.utc)
        price = point['p']
        records.append((ts, token_id, price, None, side, bid, ask))

    if not records:
        return

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO price_history (time, token_id, price, amount, side, bid, ask)
            VALUES %s
            ON CONFLICT (token_id, time) DO NOTHING
            """,
            records,
            template=None,
            page_size=500
        )
    conn.commit()
    logging.info(f"Inserted {len(records)} price points for token {token_id}.")


def update_market_resolution(
    conn,
    market_id: str,
    winner_token_id: Optional[str],
    resolved_at: Optional[str],
    uma_resolution_status: Optional[str],
) -> None:
    """
    Updates the resolution fields of a market after its outcomes have been upserted.

    This is a separate UPDATE (not part of upsert_markets) because winner_token_id
    is derived from market_outcomes data that is inserted after the market row itself.

    Args:
        conn: Active psycopg2 connection.
        market_id: The market primary key.
        winner_token_id: token_id of the winning outcome, or None if ambiguous/unresolved.
        resolved_at: ISO timestamp string from the API's 'closedTime' field.
        uma_resolution_status: Value from the API's 'umaResolutionStatus' field.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE markets
            SET
                winner_token_id       = %s,
                resolved_at           = %s,
                uma_resolution_status = %s,
                updated_at            = NOW()
            WHERE id = %s
            """,
            (winner_token_id, resolved_at, uma_resolution_status, market_id),
        )
    conn.commit()
    logging.info(
        f"Updated resolution for market {market_id}: winner={winner_token_id}, "
        f"status={uma_resolution_status}"
    )
