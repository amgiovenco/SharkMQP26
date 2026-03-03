#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
    sleep 2
done
echo "PostgreSQL is ready."

echo "Running database seed..."
python seed.py || true

echo "Starting API server..."
exec uvicorn app.main:sio_asgi_app --host 0.0.0.0 --port 8000 --workers 1
