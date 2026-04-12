#!/usr/bin/env bash
# Launch the FastAPI server.
# Usage:
#   ./run.sh              # auto-pick a free port starting at 8000
#   ./run.sh 8090         # use port 8090
#   ./run.sh --reload     # auto-pick port + enable hot reload
#   ./run.sh 8090 --reload
set -euo pipefail

cd "$(dirname "$0")"

PORT=""
RELOAD=""
for arg in "$@"; do
    case "$arg" in
        --reload) RELOAD="--reload" ;;
        ''|*[!0-9]*) echo "Unknown arg: $arg" >&2; exit 1 ;;
        *) PORT="$arg" ;;
    esac
done

if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "No python found (.venv/bin/python or python3)" >&2
    exit 1
fi

port_in_use() {
    lsof -i ":$1" -sTCP:LISTEN >/dev/null 2>&1
}

if [[ -z "$PORT" ]]; then
    for p in 8000 8001 8080 8090 8888 9000; do
        if ! port_in_use "$p"; then
            PORT="$p"
            break
        fi
    done
    if [[ -z "$PORT" ]]; then
        echo "No free port found in the default list" >&2
        exit 1
    fi
elif port_in_use "$PORT"; then
    echo "Port $PORT is already in use" >&2
    exit 1
fi

echo "Starting server on http://localhost:${PORT}"
echo "UI: http://localhost:${PORT}/graph/static/index.html"
exec "$PY" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" $RELOAD
