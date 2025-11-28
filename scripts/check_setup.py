import os
import time
import psycopg2
import redis
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_postgres():
    """Checks the PostgreSQL connection."""
    retries = 10
    for i in range(retries):
        try:
            conn = psycopg2.connect(
                host=os.environ.get("DB_HOST", "db"),
                dbname=os.environ.get("DB_NAME", "polymarket"),
                user=os.environ.get("DB_USER", "admin"),
                password=os.environ.get("DB_PASSWORD", "password"),
                port=os.environ.get("DB_PORT", "5432"),
                connect_timeout=3,
            )
            conn.close()
            logging.info("PostgreSQL is ready.")
            return True
        except psycopg2.OperationalError as e:
            logging.warning(f"PostgreSQL not ready yet (attempt {i+1}/{retries}): {e}")
            time.sleep(5)
    logging.error("Could not connect to PostgreSQL after several retries.")
    return False

def check_redis():
    """Checks the Redis connection."""
    retries = 10
    for i in range(retries):
        try:
            r = redis.Redis(
                host=os.environ.get("REDIS_HOST"),
                port=int(os.environ.get("REDIS_PORT")),
                db=int(os.environ.get("REDIS_DB")),
                socket_connect_timeout=3,
            )
            r.ping()
            logging.info("Redis is ready.")
            return True
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logging.warning(f"Redis not ready yet (attempt {i+1}/{retries}): {e}")
            time.sleep(5)
    logging.error("Could not connect to Redis after several retries.")
    return False

if __name__ == "__main__":
    if check_postgres() and check_redis():
        logging.info("All services are ready. Exiting with success.")
        exit(0)
    else:
        logging.error("One or more services are not ready. Exiting with failure.")
        exit(1)
