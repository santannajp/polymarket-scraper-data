import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import time
import json


class PolymarketGammaClient:
    """
    A client for the Polymarket Gamma API to fetch Markets and Events.
    Documentation: https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, retries: int = 3, backoff_factor: float = 0.5):
        self.session = requests.Session()
        # Basic retry logic setup
        adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=retries,
                backoff_factor=backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Internal helper to make GET requests with error handling."""
        url = urljoin(self.BASE_URL, endpoint)
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response Body: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Failed: {e}")
            raise

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch a list of markets with filtering options.

        Common kwargs:
            order (str): Field to sort by (e.g., 'volume').
            ascending (bool): Sort direction.
            id (str): Specific market ID.
            slug (str): Specific market slug.
            tag_id (int): Filter by category tag.
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            **kwargs
        }
        # Remove None values to keep query clean
        params = {k: v for k, v in params.items() if v is not None}

        return self._get("/markets", params=params)

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch a list of events (groups of markets).
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            **kwargs
        }
        params = {k: v for k, v in params.items() if v is not None}

        return self._get("/events", params=params)

    def get_event_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Helper to get a specific event by its URL slug."""
        events = self.get_events(limit=1, slug=slug)
        return events[0] if events else None

    def get_market_by_id(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Helper to get a specific market by its ID."""
        markets = self.get_markets(limit=1, id=market_id)
        return markets[0] if markets else None


# --- Usage Example ---
if __name__ == "__main__":
    client = PolymarketGammaClient()

    print("--- Fetching Top 5 Active Markets ---")
    # We use 'volume' to find the most popular ones
    markets = client.get_markets(limit=5, active=True, order="volume", ascending=False)

    for m in markets:
        print(f"ID: {m.get('id')}")
        print(f"Question: {m.get('question')}")
        # Identify the outcome prices (usually in 'outcomes' or 'clobTokenIds')
        # Note: Gamma returns raw metadata; outcome prices might be in nested fields or require CLOB lookup
        print(f"Slug: {m.get('slug')}")
        print("-" * 30)

    print("\n--- Fetching Events (aggregated markets) ---")
    events = client.get_events(limit=1, active=True)
    for e in events:
        print(f"Event: {e.get('title')}")
        print(f"Markets Count: {len(e.get('markets', []))}")
        print(json.dumps(e, indent=2))