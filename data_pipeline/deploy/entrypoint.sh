#!/bin/sh

echo "Starting Dagster webserver..."
poetry run dagster-webserver -h 0.0.0.0 -p 3000 &

echo "Starting Dagster daemon..."
poetry run dagster-daemon run &

wait
