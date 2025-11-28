from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.http_helpers.helpers import get, post

from constants import CLOB_BASE_URL, CLOB_PRICE_HISTORY_ENDPOINT

client = ClobClient(CLOB_BASE_URL)  # Level 0 (no auth)

def get_price_history(token_id, start_time: Optional[int] = None, end_time: Optional[int] = None, interval: str = "max", fidelity: int = 1):
    parameters = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    parameters = "&".join(f"{k}={v}" for k, v in parameters.items())
    return get("{}{}?{}".format(client.host, CLOB_PRICE_HISTORY_ENDPOINT, parameters))["history"]


if __name__ == "__main__":
    ok = client.get_ok()
    time = client.get_server_time()
    print(ok, time)

