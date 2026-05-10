#!/bin/bash
# Auto-restart constat server on crash
# Usage: ./scripts/serve_with_restart.sh [config_file] [extra_args...]
#
# Examples:
#   ./scripts/serve_with_restart.sh demo/config.yaml --debug
#   ./scripts/serve_with_restart.sh  # uses default config

set -e

CONFIG_FILE="${1:-demo/config.yaml}"
shift 2>/dev/null || true  # Remove first arg if present
EXTRA_ARGS="$@"

MAX_RESTARTS=10
RESTART_DELAY=2
restart_count=0
SERVER_PID=""

echo "Constat Server Auto-Restart Wrapper"
echo "===================================="
echo "Config: $CONFIG_FILE"
echo "Extra args: $EXTRA_ARGS"
echo ""

kill_server_tree() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Killing server process tree (PID $SERVER_PID)..."
        # Kill the entire process group to catch uvicorn reloader children
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        # Wait briefly for processes to release file locks
        sleep 1
    fi
    SERVER_PID=""
}

cleanup() {
    echo ""
    echo "Received shutdown signal, stopping server..."
    kill_server_tree
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting constat server (attempt $((restart_count + 1)))..."

    # Run the server in a new process group so we can kill it cleanly
    set +e
    set -m  # Enable job control for process groups
    constat serve -c "$CONFIG_FILE" $EXTRA_ARGS &
    SERVER_PID=$!
    wait "$SERVER_PID"
    EXIT_CODE=$?
    set +m
    set -e

    # Check exit code
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server exited normally"
        break
    fi

    # Handle crash — kill any lingering child processes
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server crashed with exit code $EXIT_CODE"
    kill_server_tree

    restart_count=$((restart_count + 1))

    # Check if we've exceeded max restarts
    if [ $restart_count -ge $MAX_RESTARTS ]; then
        echo "ERROR: Max restarts ($MAX_RESTARTS) exceeded. Giving up."
        exit 1
    fi

    echo "Restarting in $RESTART_DELAY seconds... (restart $restart_count/$MAX_RESTARTS)"
    sleep $RESTART_DELAY

    # Exponential backoff (cap at 30 seconds)
    RESTART_DELAY=$((RESTART_DELAY * 2))
    if [ $RESTART_DELAY -gt 30 ]; then
        RESTART_DELAY=30
    fi
done
