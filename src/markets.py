import time
from gamma_api import PolymarketGammaClient
from db import (
    get_db_connection,
    upsert_events,
    upsert_markets,
    upsert_market_outcomes,
    upsert_tags,
    link_tags_to_events,
)
import logging

def scrape_and_process_events():
    """
    Scrapes all active events, and their associated markets, from the 
    Polymarket Gamma API and stores them in the database.
    """
    client = PolymarketGammaClient()
    conn = None
    try:
        conn = get_db_connection()
        offset = 0
        limit = 100

        while True:
            logging.info(f"Fetching events page at offset {offset}...")
            events_response = client.get_events(limit=limit, offset=offset, active=True, closed=False)

            if not events_response:
                logging.info("No more events to fetch.")
                break

            logging.info(f"Processing {len(events_response)} events...")

            # Upsert the batch of events
            upsert_events(conn, events_response)
            
            # Process each event individually for its nested data
            for event in events_response:
                event_id = event.get('id')
                if not event_id:
                    logging.warning("Skipping event with no ID.")
                    continue

                # Extract and upsert markets associated with the event
                markets = event.get('markets', [])
                if markets:
                    upsert_markets(conn, markets, event_id=event_id)
                    # Outcomes are nested in markets, so we process them here
                    upsert_market_outcomes(conn, markets)

                # Tags are also nested within the event object
                tags = event.get('tags', [])
                if tags:
                    # We can pass the single event to the tag functions
                    upsert_tags(conn, [event])
                    link_tags_to_events(conn, [event])

            # TODO: Send market_ids to Redis Queue for Price Worker
            # all_market_ids = [m['id'] for e in events_response for m in e.get('markets', [])]
            # if all_market_ids:
            #     send_to_redis_queue(all_market_ids)

            offset += limit
            time.sleep(0.5)  # Be nice to the API

    except Exception as e:
        logging.error(f"An error occurred during the scraping process: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    scrape_and_process_events()
