#!/usr/bin/env bash
set -euo pipefail

# CareBridge Automated Deployment Script
# Usage:
#   ./scripts/deploy.sh [--prod | --demo]

DEPLOY_MODE="auto"
COMPOSE_FILE="docker-compose.yml"

for arg in "$@"; do
    case "$arg" in
        --prod)
            DEPLOY_MODE="prod"
            COMPOSE_FILE="docker-compose.prod.yml"
            ;;
        --demo)
            DEPLOY_MODE="demo"
            COMPOSE_FILE="docker-compose.yml"
            ;;
        *)
            ;;
    esac
done

echo "=========================================================="
echo "          CareBridge Deployment Orchestrator              "
echo "=========================================================="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: 'docker' is not installed or not in PATH." >&2
    echo "Please install Docker: https://docs.docker.com/get-docker/" >&2
    exit 1
fi

# Check Docker daemon is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running." >&2
    echo "Please start the Docker service and try again." >&2
    exit 1
fi

# Check for docker compose support
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "ERROR: Neither 'docker compose' nor 'docker-compose' found." >&2
    exit 1
fi

# Check .env file
if [ ! -f .env ]; then
    echo "[!] .env file not found."
    if [ "$DEPLOY_MODE" = "prod" ]; then
        echo "    Creating .env from .env.production.example..."
        cp .env.production.example .env
        echo "    CRITICAL: Please edit .env to set your LLM_BASE_URL and NURSE_PASSWORD before running in production!"
    else
        echo "    Creating .env from .env.example..."
        cp .env.example .env
    fi
fi

# Validate live mode requirements
RUN_MODE_VAL=$(grep -E '^RUN_MODE=' .env | cut -d '=' -f2- | tr -d ' "' || echo "demo")
if [ "$RUN_MODE_VAL" = "live" ] || [ "$DEPLOY_MODE" = "prod" ]; then
    echo "[*] Validating live mode configuration..."
    LLM_BASE_URL_VAL=$(grep -E '^LLM_BASE_URL=' .env | cut -d '=' -f2- | tr -d ' "' || echo "")
    NURSE_PASSWORD_VAL=$(grep -E '^NURSE_PASSWORD=' .env | cut -d '=' -f2- | tr -d ' "' || echo "")
    
    if [ -z "$LLM_BASE_URL_VAL" ] || [ "$LLM_BASE_URL_VAL" = "https://your-llm-endpoint.example.com/v1" ]; then
        echo "ERROR: RUN_MODE=live requires a valid LLM_BASE_URL in .env." >&2
        echo "Please set your remote OpenAI-compatible endpoint URL in .env." >&2
        exit 1
    fi
    if [ -z "$NURSE_PASSWORD_VAL" ] || [ "$NURSE_PASSWORD_VAL" = "generate_a_strong_password_here" ]; then
        echo "ERROR: RUN_MODE=live requires a strong NURSE_PASSWORD in .env." >&2
        echo "Generate one with: openssl rand -base64 24" >&2
        exit 1
    fi
fi

echo "[*] Using compose file: $COMPOSE_FILE"
echo "[*] Building and starting CareBridge containers..."

$DOCKER_COMPOSE_CMD -f "$COMPOSE_FILE" build
$DOCKER_COMPOSE_CMD -f "$COMPOSE_FILE" up -d

echo "[*] Waiting for CareBridge health check..."
PORT_VAL=$(grep -E '^PORT=' .env | cut -d '=' -f2- | tr -d ' "' || echo "8000")
if [ "$COMPOSE_FILE" = "docker-compose.prod.yml" ]; then
    PORT_VAL=$(grep -E '^HTTP_PORT=' .env | cut -d '=' -f2- | tr -d ' "' || echo "80")
fi

HEALTH_URL="http://127.0.0.1:${PORT_VAL}/health"
MAX_ATTEMPTS=30
ATTEMPT=1
HEALTHY=0

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s -f "$HEALTH_URL" &> /dev/null; then
        HEALTHY=1
        break
    fi
    printf "."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done
echo ""

if [ $HEALTHY -eq 1 ]; then
    echo "=========================================================="
    echo " [OK] CareBridge is successfully deployed and healthy!"
    echo "=========================================================="
    echo "  - Patient Chat:       http://localhost:${PORT_VAL}/"
    echo "  - Nurse Handoff Queue: http://localhost:${PORT_VAL}/nurse"
    echo "  - Health Endpoint:     http://localhost:${PORT_VAL}/health"
    echo "  - API Documentation:   http://localhost:${PORT_VAL}/docs"
    echo "=========================================================="
else
    echo "=========================================================="
    echo " [!] Deployment started, but healthcheck did not respond in time."
    echo " Check container logs with: $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs -f"
    echo "=========================================================="
fi
