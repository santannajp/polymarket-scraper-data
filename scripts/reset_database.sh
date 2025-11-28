sudo docker-compose down
sudo rm -rf data/postgres
sudo docker-compose up -d
uv run scripts/init_db.py