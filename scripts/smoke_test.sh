#!/usr/bin/env bash
set -euo pipefail

created_env=0
if [[ ! -f .env ]]; then
  cp .env.example .env
  created_env=1
fi
cleanup() {
  docker compose down
  if [[ "$created_env" == "1" ]]; then rm -f .env; fi
}
trap cleanup EXIT

docker compose up -d --build
ready=0
for attempt in $(seq 1 60); do
  if curl --fail --silent http://localhost:8000/health >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  docker compose logs --no-color
  echo "FAIL: CareBridge did not become healthy." >&2
  exit 1
fi

python3 - <<'PY'
import json
import urllib.error
import urllib.request

base = "http://localhost:8000"

def post(message, patient="demo-appendectomy"):
    payload = json.dumps({
        "patient_id": patient,
        "message": message,
        "session_id": "docker-smoke",
    }).encode()
    request = urllib.request.Request(
        f"{base}/api/chat", payload, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

checks = [
    ("answerable", post("Can I shower?"), "answered"),
    ("unanswerable", post("When can I travel?"), "escalated"),
    ("red-flag English", post("I have severe chest pain"), "red_flag"),
    ("red-flag Hindi", post("सीने में दर्द हो रहा है"), "red_flag"),
    ("red-flag Telugu", post("నాకు ఛాతి నొప్పి ఉంది"), "red_flag"),
]
failed = []
for name, result, expected in checks:
    valid = result.get("status") == expected
    if name == "answerable":
        valid = valid and bool(result.get("answer")) and bool(result.get("citations"))
    if name.startswith("red-flag"):
        valid = valid and bool(result.get("ticket_id"))
    print(f"{'PASS' if valid else 'FAIL'}: {name} ({result.get('status')})")
    if not valid:
        failed.append(name)
if failed:
    raise SystemExit(f"Smoke checks failed: {', '.join(failed)}")
print("PASS: all CareBridge smoke checks")
PY
