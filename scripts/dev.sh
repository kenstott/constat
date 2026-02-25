#!/bin/bash
# Development launcher - starts constat server and UI dev server
# Usage: ./scripts/dev.sh [config_file]
#
# Examples:
#   ./scripts/dev.sh                    # uses demo/config.yaml
#   ./scripts/dev.sh my/config.yaml     # uses custom config
#
# Controls:
#   Ctrl+C - Stop both services and exit
#   Ctrl+R - Restart both services

set -e

# Suppress multiprocessing resource_tracker warnings at shutdown
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-demo/config.yaml}"

cd "$PROJECT_ROOT"

# Create log directory
mkdir -p .logs

# Track PIDs for cleanup
SERVER_PID=""
UI_PID=""
SERVER_LOG=".logs/server.log"
UI_LOG=".logs/ui.log"

stop_servers() {
    # Stop UI dev server and all child processes
    if [ -n "$UI_PID" ] && kill -0 "$UI_PID" 2>/dev/null; then
        echo "Stopping UI dev server (PID $UI_PID)..."
        # Kill the process group to get all child processes
        kill -- -"$UI_PID" 2>/dev/null || kill "$UI_PID" 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if kill -0 "$UI_PID" 2>/dev/null; then
            kill -9 -- -"$UI_PID" 2>/dev/null || kill -9 "$UI_PID" 2>/dev/null || true
        fi
    fi

    # Kill anything still using port 5173
    lsof -ti :5173 2>/dev/null | xargs kill -9 2>/dev/null || true

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

    # Kill anything still using port 8000
    lsof -ti :8000 2>/dev/null | xargs kill -9 2>/dev/null || true

    # Wait for processes to exit
    wait 2>/dev/null || true
    SERVER_PID=""
    UI_PID=""

    # Wait for ports to be released
    sleep 1
}

start_server() {
    echo "Starting constat server..."
    echo "Server log: $SERVER_LOG"
    PYTHONUNBUFFERED=1 constat serve -c "$CONFIG_FILE" --debug --reload > "$SERVER_LOG" 2>&1 &
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
        return 1
    fi
    return 0
}

start_ui() {
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
        return 1
    fi
    return 0
}

start_all() {
    if ! start_server; then
        return 1
    fi
    if ! start_ui; then
        return 1
    fi
    return 0
}

restart_servers() {
    echo ""
    echo "================================"
    echo "Restarting servers..."
    echo "================================"
    echo ""
    stop_servers
    sleep 1
    if start_all; then
        show_status
    fi
}

show_status() {
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
    echo "Press Ctrl+R to restart, Ctrl+C to stop"
    echo ""
}

cleanup() {
    echo ""
    echo "Shutting down..."
    stop_servers
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "Constat Development Environment"
echo "================================"
echo "Config: $CONFIG_FILE"
echo ""

# Initial startup
if ! start_all; then
    exit 1
fi

show_status

# Monitor both processes and handle key input
while true; do
    # Check for key input (with 1 second timeout)
    if read -rsn1 -t 1 key 2>/dev/null; then
        # Check for Ctrl+R (ASCII 18)
        if [[ "$key" == $'\x12' ]]; then
            restart_servers
            continue
        fi
    fi

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
done
