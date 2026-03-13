"""
orderbook_snapshot_worker.py — Etapa 3

Captura snapshots periódicos de bid/ask para todos os tokens de mercados ativos.
Separa claramente as responsabilidades:
  - price_worker.py          → histórico MID via /prices-history
  - orderbook_snapshot_worker → snapshots live de bid/ask via /price
"""

import os
import logging
import time
from datetime import datetime, timezone
from typing import List

from db import get_db_connection, get_active_token_ids, insert_price_history
from clob_api import get_bid_ask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Configuração via variáveis de ambiente com defaults razoáveis
SNAPSHOT_INTERVAL_SECONDS: int = int(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "900"))  # 15 min
BATCH_SIZE: int = int(os.getenv("SNAPSHOT_BATCH_SIZE", "50"))
INTER_BATCH_SLEEP: float = float(os.getenv("SNAPSHOT_INTER_BATCH_SLEEP", "0.2"))


def _process_batch(conn, token_ids: List[str], now: datetime) -> int:
    """
    Fetches bid/ask for a batch of tokens and inserts SNAPSHOT records into price_history.

    Args:
        conn: Active psycopg2 connection.
        token_ids: Batch of token identifiers to snapshot.
        now: Timestamp to use for all records in this batch (ensures consistency).

    Returns:
        Number of tokens successfully snapshotted.
    """
    inserted = 0
    for token_id in token_ids:
        try:
            bid, ask = get_bid_ask(token_id)
            if bid is None or ask is None:
                logging.debug(f"No bid/ask available for token {token_id}, skipping.")
                continue

            mid = (bid + ask) / 2.0
            insert_price_history(
                conn,
                token_id=token_id,
                price_data=[{"t": now.timestamp(), "p": mid}],
                bid=bid,
                ask=ask,
                side="SNAPSHOT",
            )
            inserted += 1

        except Exception as e:
            logging.warning(f"Snapshot failed for token {token_id}: {e}")

    return inserted


def run() -> None:
    """Main loop: snapshot all active tokens every SNAPSHOT_INTERVAL_SECONDS."""
    logging.info(
        f"orderbook_snapshot_worker started | "
        f"interval={SNAPSHOT_INTERVAL_SECONDS}s | "
        f"batch_size={BATCH_SIZE}"
    )

    while True:
        cycle_start = time.monotonic()
        conn = None
        try:
            conn = get_db_connection()
            token_ids = get_active_token_ids(conn)
            logging.info(f"Active tokens to snapshot: {len(token_ids)}")

            if not token_ids:
                logging.info("No active tokens found. Sleeping until next cycle.")
            else:
                now = datetime.now(tz=timezone.utc)
                total_inserted = 0

                for i in range(0, len(token_ids), BATCH_SIZE):
                    batch = token_ids[i : i + BATCH_SIZE]
                    inserted = _process_batch(conn, batch, now)
                    total_inserted += inserted

                    # Rate-limit between batches (not needed after the last one)
                    if i + BATCH_SIZE < len(token_ids):
                        time.sleep(INTER_BATCH_SLEEP)

                logging.info(
                    f"Snapshot cycle complete: {total_inserted}/{len(token_ids)} tokens inserted "
                    f"in {time.monotonic() - cycle_start:.1f}s"
                )

        except Exception as e:
            logging.error(f"Snapshot cycle failed: {e}", exc_info=True)

        finally:
            if conn is not None:
                conn.close()

        elapsed = time.monotonic() - cycle_start
        sleep_duration = max(0.0, SNAPSHOT_INTERVAL_SECONDS - elapsed)
        logging.info(f"Next snapshot in {sleep_duration:.0f}s")
        time.sleep(sleep_duration)


if __name__ == "__main__":
    run()
