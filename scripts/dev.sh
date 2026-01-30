#!/bin/bash
# Development launcher - starts constat server and UI dev server
# Usage: ./scripts/dev.sh [config_file]
#
# Examples:
#   ./scripts/dev.sh                    # uses demo/config.yaml
#   ./scripts/dev.sh my/config.yaml     # uses custom config

set -e

# Suppress multiprocessing resource_tracker warnings at shutdown
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-demo/config.yaml}"

cd "$PROJECT_ROOT"

echo "Constat Development Environment"
echo "================================"
echo "Config: $CONFIG_FILE"
echo ""

# Create log directory
mkdir -p .logs

# Track PIDs for cleanup
SERVER_PID=""
UI_PID=""
SERVER_LOG=".logs/server.log"
UI_LOG=".logs/ui.log"

cleanup() {
    echo ""
    echo "Shutting down..."

    if [ -n "$UI_PID" ] && kill -0 "$UI_PID" 2>/dev/null; then
        echo "Stopping UI dev server (PID $UI_PID)..."
        kill "$UI_PID" 2>/dev/null || true
    fi

    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping constat server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true

        # Wait up to 3 seconds for graceful shutdown
        for i in 1 2 3; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Force killing server..."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
    fi

    # Wait for processes to exit
    wait 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start constat server with debug (output goes to log file)
# Force unbuffered output with PYTHONUNBUFFERED environment variable
echo "Starting constat server..."
echo "Server log: $SERVER_LOG"
PYTHONUNBUFFERED=1 constat serve -c "$CONFIG_FILE" --debug > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Give server a moment to start and check if it's still running
sleep 2
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "================================"
    echo "ERROR: Server failed to start"
    echo "================================"
    echo ""
    echo "Last 50 lines from server log:"
    echo "---"
    tail -50 "$SERVER_LOG"
    echo "---"
    echo ""
    echo "Full log available at: $SERVER_LOG"
    exit 1
fi

# Start UI dev server (output goes to log file)
# Use --strictPort to fail if port 5173 is in use rather than silently using another port
echo "Starting UI dev server..."
echo "UI log: $UI_LOG"
(cd constat-ui && npm run dev -- --port 5173 --strictPort) > "$UI_LOG" 2>&1 &
UI_PID=$!
echo "UI PID: $UI_PID"

# Give UI a moment to start
sleep 2
if ! kill -0 "$UI_PID" 2>/dev/null; then
    echo ""
    echo "================================"
    echo "ERROR: UI dev server failed to start"
    echo "================================"
    echo ""
    echo "Last 50 lines from UI log:"
    echo "---"
    tail -50 "$UI_LOG"
    echo "---"
    echo ""
    echo "Full log available at: $UI_LOG"
    exit 1
fi

echo ""
echo "Both services running:"
echo "  - Server: http://localhost:8000 (API)"
echo "  - UI:     http://localhost:5173 (Vite)"
echo ""
echo "Logs:"
echo "  - Server: $SERVER_LOG"
echo "  - UI:     $UI_LOG"
echo ""
echo "To monitor logs in real-time:"
echo "  tail -f $SERVER_LOG"
echo "  tail -f $UI_LOG"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Monitor both processes
while true; do
    # Check if server is still running
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "================================"
        echo "Server exited unexpectedly"
        echo "================================"
        echo ""
        echo "Last 100 lines from server log:"
        echo "---"
        tail -100 "$SERVER_LOG"
        echo "---"
        echo ""
        echo "Full log available at: $SERVER_LOG"
        break
    fi

    # Check if UI is still running
    if ! kill -0 "$UI_PID" 2>/dev/null; then
        echo ""
        echo "================================"
        echo "UI dev server exited unexpectedly"
        echo "================================"
        echo ""
        echo "Last 100 lines from UI log:"
        echo "---"
        tail -100 "$UI_LOG"
        echo "---"
        echo ""
        echo "Full log available at: $UI_LOG"
        break
    fi

    sleep 1
done