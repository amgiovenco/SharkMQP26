#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
    sleep 2
done
echo "PostgreSQL is ready."

echo "Starting ML worker..."
exec python -m worker.worker
