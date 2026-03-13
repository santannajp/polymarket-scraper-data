import logging
from typing import Optional, Tuple

from py_clob_client.client import ClobClient
from py_clob_client.http_helpers.helpers import get, post

from constants import CLOB_BASE_URL, CLOB_PRICE_HISTORY_ENDPOINT, CLOB_PRICE_ENDPOINT

client = ClobClient(CLOB_BASE_URL)  # Level 0 (no auth)

def get_price_history(token_id, start_time: Optional[int] = None, end_time: Optional[int] = None, interval: str = "max", fidelity: int = 1):
    parameters = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    parameters = "&".join(f"{k}={v}" for k, v in parameters.items())
    return get("{}{}?{}".format(client.host, CLOB_PRICE_HISTORY_ENDPOINT, parameters))["history"]


def get_bid_ask(token_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetches the current best bid and ask prices for a token from the CLOB API.

    Uses two lightweight GET /price requests (side=SELL for bid, side=BUY for ask)
    instead of the full /book endpoint, as we only need best prices.

    All values from the API are strings ("0.491"), so explicit float conversion is required.

    Args:
        token_id: The CLOB token identifier.

    Returns:
        Tuple (bid, ask) as floats, or (None, None) if the data is unavailable.
    """
    try:
        bid_response = get(
            "{}{}?token_id={}&side=SELL".format(client.host, CLOB_PRICE_ENDPOINT, token_id)
        )
        ask_response = get(
            "{}{}?token_id={}&side=BUY".format(client.host, CLOB_PRICE_ENDPOINT, token_id)
        )

        bid_str = bid_response.get("price") if isinstance(bid_response, dict) else None
        ask_str = ask_response.get("price") if isinstance(ask_response, dict) else None

        if bid_str is None or ask_str is None:
            return None, None

        return float(bid_str), float(ask_str)

    except Exception as e:
        logging.warning(f"get_bid_ask failed for token {token_id}: {e}")
        return None, None


if __name__ == "__main__":
    ok = client.get_ok()
    time = client.get_server_time()
    print(ok, time)

