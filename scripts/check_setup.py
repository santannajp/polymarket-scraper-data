import asyncio
import os
import asyncpg
from redis import asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()


async def check_db():
    dsn = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    try:
        conn = await asyncpg.connect(dsn)
        version = await conn.fetchval("SELECT version()")
        print(f"✔ Database Connected: {version}")

        # Check if Timescale extension is loaded
        timescale = await conn.fetchval(
            "SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb'")
        print(f"✔ TimescaleDB Extension Available: v{timescale}")

        await conn.close()
    except Exception as e:
        print(f"❌ Database Error: {e}")


async def check_redis():
    try:
        r = aioredis.from_url(f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}")
        await r.set("test_key", "Hello Polymarket")
        value = await r.get("test_key")
        print(f"✔ Redis Connected: {value.decode('utf-8')}")
        await r.close()
    except Exception as e:
        print(f"❌ Redis Error: {e}")


async def main():
    print("--- Checking Infrastructure ---")
    await check_db()
    await check_redis()


if __name__ == "__main__":
    asyncio.run(main())