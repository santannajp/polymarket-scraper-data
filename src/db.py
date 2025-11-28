import os
import json
import psycopg2
from psycopg2.extras import Json, execute_values
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            port=os.environ.get("DB_PORT"),
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
