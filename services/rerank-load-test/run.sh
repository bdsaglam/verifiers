#!/bin/bash

# Run locust in headless mode
# -H: Host to load test
# -u: Number of users to simulate
# -r: Spawn rate (users per second)
# -t: Run time
# --headless: Run without web UI
# --only-summary: Only show the summary stats
locust -f locustfile.py \
    -H http://localhost:8003 \
    -u 10 \
    -r 1 \
    -t 1m \
    --headless \
    --only-summary 