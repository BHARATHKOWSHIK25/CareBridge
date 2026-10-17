#!/bin/sh
set -e

# CareBridge Container Entrypoint
echo "=========================================="
echo " Starting CareBridge Deployment Container"
echo "=========================================="

# Create runtime data directory for SQLite if it does not exist
mkdir -p /app/runtime-data

PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-1}"
RUN_MODE="${RUN_MODE:-demo}"

echo "Configuration:"
echo " - Mode:     ${RUN_MODE}"
echo " - Port:     ${PORT}"
echo " - Workers:  ${WORKERS}"
echo " - Database: ${DATABASE_URL:-sqlite:////app/runtime-data/carebridge.db}"

if [ "${RUN_MODE}" = "live" ]; then
    echo "Running in LIVE production mode."
    if [ -z "${LLM_BASE_URL}" ]; then
        echo "ERROR: RUN_MODE=live requires LLM_BASE_URL to be set." >&2
        echo "Please provide your OpenAI-compatible endpoint URL in your environment." >&2
        exit 1
    fi
    if [ -z "${NURSE_PASSWORD}" ]; then
        echo "ERROR: RUN_MODE=live requires NURSE_PASSWORD to protect nurse endpoints." >&2
        echo "Please provide a strong NURSE_PASSWORD in your environment." >&2
        exit 1
    fi
else
    echo "Running in DEMO mode (deterministic extractive fallback, synthetic records)."
fi

echo "Launching Uvicorn server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}" "$@"
