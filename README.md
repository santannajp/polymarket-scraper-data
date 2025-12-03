# Polymarket Scraper

This project is a Python-based scraper for the Polymarket prediction market platform. It fetches data from the Polymarket APIs, processes it, and stores it in a PostgreSQL database with the TimescaleDB extension for efficient time-series data handling.

## Features

- Scrapes market and event data from the Polymarket Gamma API.
- Scrapes price history data from the Polymarket CLOB API.
- Stores data in a PostgreSQL + TimescaleDB database.
- Uses Redis as a message queue to decouple market discovery and price scraping.
- Containerized with Docker and Docker Compose for easy setup and deployment.

## Technology Stack

- **Python 3.11**
- **PostgreSQL 16** with **TimescaleDB**
- **Redis**
- **Docker** & **Docker Compose**
- `psycopg2` for PostgreSQL connection.
- `redis-py` for Redis connection.
- `py-clob-client` for Polymarket CLOB API.
- `requests` for Polymarket Gamma API.

## Project Structure

```
/
├── data/                     # Data volumes for Postgres and Redis
├── schema/                   # Database schema files
│   ├── 01_core.sql           # Core relational tables
│   └── 02_time_series.sql    # Time-series hypertables
├── scripts/                  # Utility scripts
│   ├── check_setup.py        # Checks DB and Redis connections
│   └── init_db.py            # Initializes the database schema
├── src/                      # Main application source code
│   ├── clob_api.py           # CLOB API client
│   ├── constants.py          # Project constants
│   ├── db.py                 # Database interaction logic
│   ├── gamma_api.py          # Gamma API client
│   ├── markets.py            # Market discovery worker
│   ├── price_worker.py       # Price history worker
│   └── redis_client.py       # Redis client
├── .env.example              # Example environment variables
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Dockerfile for Python workers
└── README.md                 # This file
```

## How It Works

The application consists of two main services that run as workers:

1.  **Market Worker (`market-worker`)**:
    - Runs periodically (every 1 hour by default).
    - Fetches active events and their associated markets from the Polymarket Gamma API.
    - Upserts event, market, tag, and outcome data into the PostgreSQL database.
    - Pushes the ID of each discovered market into a Redis queue (`price_worker_queue`).

2.  **Price Worker (`price-worker`)**:
    - Runs continuously.
    - Listens for market IDs on the Redis queue.
    - For each market ID, it fetches the corresponding token IDs from the database.
    - For each token ID, it scrapes the full price history from the CLOB API.
    - Inserts the price history data into the `price_history` hypertable in the database.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/polymarket-scraper.git
    cd polymarket-scraper
    ```

2.  **Create an environment file:**
    - Copy the example environment file:
      ```bash
      cp .env.example .env
      ```
    - (Optional) Modify the variables in `.env` if you want to change the default database credentials or ports.

3.  **Build and run the services:**
    ```bash
    docker-compose up --build
    ```
    This command will:
    - Build the Docker image for the Python workers.
    - Start the PostgreSQL database (`db`).
    - Start the Redis server (`redis`).
    - Start the market discovery worker (`market-worker`).
    - Start the price history worker (`price-worker`).

4.  **Initialize the database schema:**
    - In a separate terminal, wait for the `db` service to be healthy. You can check the logs with `docker-compose logs -f db`.
    - Once the database is ready, run the initialization script:
      ```bash
      docker-compose exec market-worker uv run python scripts/init_db.py
      ```
    - This will create all the necessary tables and hypertables defined in the `schema/` directory.

The scrapers will now be running. You can monitor their activity by viewing the logs:
```bash
# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f market-worker
docker-compose logs -f price-worker
```

### Accessing the Data

You can connect to the PostgreSQL database to query the scraped data:
- **Host**: `localhost` (or your Docker host IP)
- **Port**: `5432` (or as configured in `.env`)
- **Database**: `polymarket`
- **User**: `admin`
- **Password**: `password`

Use any standard SQL client (like `psql`, DBeaver, or DataGrip) to connect.

### Example Queries

**Find the most recently updated active markets:**
```sql
SELECT question, updated_at
FROM markets
WHERE active = TRUE
ORDER BY updated_at DESC
LIMIT 10;
```

**Get the latest price for a specific outcome token:**
```sql
SELECT price
FROM price_history
WHERE token_id = '<your_token_id>'
ORDER BY time DESC
LIMIT 1;
```
