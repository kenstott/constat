#!/usr/bin/env bash
# Start Jupyter Lab with constat-jupyter available.
# Assumes the constat server is already running.
#
# Usage:
#   ./scripts/start_jupyter.sh                    # defaults
#   ./scripts/start_jupyter.sh --port 8889        # custom Jupyter port
#   CONSTAT_PORT=9000 ./scripts/start_jupyter.sh  # custom server port
#
# Prerequisites:
#   pip install jupyterlab ipywidgets
#   pip install -e constat-jupyter/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SERVER_PORT="${CONSTAT_PORT:-8000}"
SERVER_HOST="${CONSTAT_HOST:-127.0.0.1}"

cd "$PROJECT_ROOT"

# Check dependencies
if ! command -v jupyter &>/dev/null; then
    echo "Error: jupyter not found. Install with: pip install jupyterlab"
    exit 1
fi

if ! python -c "import constat_jupyter" 2>/dev/null; then
    echo "Error: constat_jupyter not importable. Install with: pip install -e constat-jupyter/"
    exit 1
fi

# Check server is running
if ! curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" >/dev/null 2>&1; then
    echo "Error: constat server not reachable at http://$SERVER_HOST:$SERVER_PORT"
    echo "Start it first with: constat serve -c demo/config.yaml"
    exit 1
fi

# Export connection info for notebooks
export CONSTAT_SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"

echo ""
echo "=== Constat Jupyter ==="
echo "Server: $CONSTAT_SERVER_URL"
echo ""
echo "In notebooks:"
echo "  from constat_jupyter import ConstatClient"
echo "  client = ConstatClient('$CONSTAT_SERVER_URL')"
echo ""

jupyter lab "$@"
