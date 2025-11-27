from gamma_api import PolymarketGammaClient

def scrape_all_markets():
    client = PolymarketGammaClient()
    offset = 0
    limit = 100

    while True:
        print(f"Fetching page at offset {offset}...")
        markets = client.get_markets(limit=limit, offset=offset)

        if not markets:
            break

        # TODO: Insert 'markets' data into PostgreSQL here
        # TODO: Send market_ids to Redis Queue for Price Worker

        offset += limit
        time.sleep(0.5)  # Be nice to the API