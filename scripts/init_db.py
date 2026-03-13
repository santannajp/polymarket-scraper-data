import asyncio
import os
import sys
import asyncpg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Define the order of your schema files here
SCHEMA_FILES = [
    "01_core.sql",
    "02_time_series.sql",
    "03_resolution.sql",
]


async def init_db():
    dsn = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

    print(f"Connecting to {dsn}...")

    conn = await asyncpg.connect(dsn)

    schema_dir = Path("schema")

    for filename in SCHEMA_FILES:
        file_path = schema_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found in schema/ folder.")

        print(f"Running {filename}...")
        sql = file_path.read_text()

        # Execute the SQL
        await conn.execute(sql)
        print(f"  {filename} applied successfully.")

    await conn.close()
    print("Database initialization complete.")


if __name__ == "__main__":
    try:
        asyncio.run(init_db())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
