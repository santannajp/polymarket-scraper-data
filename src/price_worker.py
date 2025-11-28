import time
import logging
import redis
from db import get_db_connection, get_token_ids_for_market, insert_price_history
from redis_client import get_redis_connection
from clob_api import get_price_history
from py_clob_client.exceptions import PolyApiException


def price_worker():
    """
    Continuously fetches market IDs from a Redis queue, scrapes their price history,
    and stores it in the database.
    """
    logging.info("Starting price worker...")
    redis_conn = get_redis_connection()
    db_conn = get_db_connection()
    
    queue_name = "price_worker_queue"

    while True:
        try:
            # Blocking pop from the Redis queue
            market_id_tuple = redis_conn.brpop(queue_name)
            if market_id_tuple:
                market_id = market_id_tuple[1]
                logging.info(f"Processing market_id: {market_id}")

                # 1. Get token_ids for the given market_id
                token_ids = get_token_ids_for_market(db_conn, market_id)
                if not token_ids:
                    logging.warning(f"No token_ids found for market_id: {market_id}")
                    continue

                # 2. For each token_id, fetch and store price history
                for token_id in token_ids:
                    try:
                        price_history = get_price_history(token_id)
                        
                        if price_history:
                            insert_price_history(db_conn, token_id, price_history)
                        else:
                            logging.info(f"No price history found for token_id: {token_id}")

                    except PolyApiException as e:
                        logging.error(f"CLOB API Error for token_id {token_id}: {e}")
                    except Exception as e:
                        logging.error(f"An unexpected error occurred for token_id {token_id}: {e}", exc_info=True)
                        # If a database error occurs, the transaction will be aborted.
                        # We need to roll back to start a new one.
                        db_conn.rollback()
                
                logging.info(f"Finished processing market_id: {market_id}")

        except redis.exceptions.ConnectionError as e:
            logging.error(f"Redis connection error: {e}. Reconnecting...")
            time.sleep(5)
            redis_conn = get_redis_connection()
        except Exception as e:
            logging.error(f"An unexpected error occurred in the worker loop: {e}", exc_info=True)
            # In case of a non-connection error, wait a bit before continuing
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    price_worker()
