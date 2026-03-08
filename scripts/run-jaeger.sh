#!/usr/bin/env bash
# Start Jaeger all-in-one and generate sample OTel trace data.
#
# Jaeger UI:   http://localhost:16686
# OTLP gRPC:   localhost:4317
# OTLP HTTP:   localhost:4318
#
# Usage:
#   bash scripts/run-jaeger.sh          # start + generate sample data
#   bash scripts/run-jaeger.sh --stop   # tear down

set -euo pipefail

CONTAINER_NAME="constat-jaeger"

if [[ "${1:-}" == "--stop" ]]; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null && echo "Removed $CONTAINER_NAME" || echo "Not running"
    exit 0
fi

# Start Jaeger all-in-one (v2 with OTLP native support)
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Jaeger already running"
else
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run -d --name "$CONTAINER_NAME" \
        -p 16686:16686 \
        -p 4317:4317 \
        -p 4318:4318 \
        jaegertracing/all-in-one:latest
    echo "Waiting for Jaeger to start..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:16686/api/services >/dev/null 2>&1; then
            echo "Jaeger ready at http://localhost:16686"
            break
        fi
        sleep 1
    done
fi

# Generate sample trace data via OTLP HTTP
echo "Generating sample traces..."
python3 "$(dirname "$0")/generate-jaeger-data.py"
echo "Done. Open http://localhost:16686 to browse traces."
