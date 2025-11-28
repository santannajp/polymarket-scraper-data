# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv
RUN uv sync --no-cache

# Copy the rest of the application's code
COPY . .

# Make scripts executable
RUN chmod +x /app/scripts/check_setup.py

# This command is overridden in docker-compose.yml, but it's good practice to have a default.
CMD ["/app/.venv/bin/uv", "run", "src/price_worker.py"]
