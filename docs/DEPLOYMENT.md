# CareBridge Production Deployment Guide

This guide covers everything required to take CareBridge from local development to a secure, resilient, and production-ready deployment.

---

## 1. Architecture & Operating Modes

CareBridge is a safety-first discharge instructions explainer featuring:
- **FastAPI backend** running Python 3.11 with asynchronous endpoints.
- **Embedded Sentence-Transformers model** (`all-MiniLM-L6-v2`) on CPU for semantic retrieval.
- **SQLite persistent database** for ticket handoffs, audit trails, and metrics.
- **Remote OpenAI-compatible LLM** for synthesis, translation, and red-flag classification.
- **Static single-page web UI** for patient chat (`/`) and nurse triage queue (`/nurse`).

### Run Modes

CareBridge operates in one of two modes controlled by `RUN_MODE`:

| Setting | Behavior | Requirements |
|---|---|---|
| `RUN_MODE=demo` | Offline deterministic extractive demo. Ideal for testing, CI/CD, and staging. | No remote LLM required. Uses synthetic discharge sheets. |
| `RUN_MODE=live` | **Production clinical mode.** Enforces confidence gating, model grounding, and real-time remote LLM synthesis. | **Requires** `LLM_BASE_URL` and `NURSE_PASSWORD`. Server fails fast on startup if either is missing. |

---

## 2. Pre-Deployment Security Checklist

Before deploying CareBridge to a public server:

- [ ] **Generate a Strong Nurse Password**:
  ```bash
  openssl rand -base64 24
  ```
  Set this as `NURSE_PASSWORD` in your production environment.
- [ ] **Configure Restricted CORS Origins**:
  Set `CORS_ORIGINS` to your exact frontend domain (e.g., `https://carebridge.yourhospital.org`), never `*`.
- [ ] **Configure Your Remote LLM Endpoint**:
  Point `LLM_BASE_URL` to your OpenAI-compatible inference endpoint (e.g., AMD ROCm vLLM instance, self-hosted Ollama, or cloud endpoint).
- [ ] **Enforce TLS/HTTPS**:
  Never expose patient communications over unencrypted HTTP. Use a reverse proxy (Nginx, Caddy, Cloudflare, or AWS ALB) with SSL/TLS certificates.
- [ ] **Persistent Volume Mount**:
  Ensure `/app/runtime-data` is mounted to persistent storage so `carebridge.db` survives container restarts.
- [ ] **Hardware Sizing**:
  - **CPU**: 2 vCPU minimum (for CPU embedding inference).
  - **RAM**: 2 GB minimum (4 GB recommended for production).
  - **Disk**: 5 GB minimum (to store base image, cached MiniLM weights, and SQLite db).

---

## 3. Deployment Methods

### Option A: Production Docker Compose with Nginx (Recommended for VPS / Dedicated Servers)

CareBridge includes a complete production compose configuration with Nginx reverse proxy, rate limiting, and security headers.

1. **Clone the repository onto your server**:
   ```bash
   git clone https://github.com/your-org/CareBridge.git /opt/carebridge
   cd /opt/carebridge
   ```

2. **Create your production `.env`**:
   ```bash
   cp .env.production.example .env
   ```
   Edit `.env` and set:
   ```ini
   RUN_MODE=live
   ENVIRONMENT=production
   LLM_BASE_URL=https://your-llm-service.internal/v1
   LLM_API_KEY=your-api-key
   LLM_MODEL=Qwen2.5-7B-Instruct
   NURSE_PASSWORD=your-secure-nurse-password
   CORS_ORIGINS=https://carebridge.yourdomain.com
   HTTP_PORT=80
   WEB_CONCURRENCY=2
   ```

3. **Deploy using the automated script or docker compose**:
   ```bash
   # Using the deployment script:
   ./scripts/deploy.sh --prod

   # Or directly with Docker Compose:
   docker compose -f docker-compose.prod.yml up -d --build
   ```

4. **Add SSL/TLS via Certbot (Let's Encrypt)**:
   ```bash
   sudo apt-get install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d carebridge.yourdomain.com
   ```

---

### Option B: Cloud PaaS (1-Click & Managed Services)

#### 1. Render (`render.yaml`)
CareBridge includes a pre-configured `render.yaml` blueprint with persistent disk:
1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Blueprint Instance** pointing to `render.yaml`.
3. Fill in secret environment variables in the Render dashboard:
   - `LLM_BASE_URL`
   - `LLM_API_KEY`
   - `NURSE_PASSWORD` (or allow Render to auto-generate)
   - `CORS_ORIGINS` (your `https://<service-name>.onrender.com` URL)
4. Deploy!

#### 2. Fly.io (`fly.toml`)
1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Create and allocate persistent volume for SQLite:
   ```bash
   fly volumes create carebridge_data --size 1 --region iad
   ```
3. Set your production secrets:
   ```bash
   fly secrets set \
     LLM_BASE_URL="https://your-llm.example.com/v1" \
     LLM_API_KEY="your-key" \
     NURSE_PASSWORD="your-strong-password" \
     CORS_ORIGINS="https://carebridge.fly.dev"
   ```
4. Deploy:
   ```bash
   fly deploy
   ```

#### 3. Hugging Face Spaces (Free CPU Docker Space)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Docker** as the SDK.
3. Push this repository to your Space git remote.
4. Under **Settings -> Variables and secrets**:
   - Add Secret: `LLM_BASE_URL`
   - Add Secret: `LLM_API_KEY`
   - Add Secret: `NURSE_PASSWORD`
   - Add Variable: `RUN_MODE=live`
   - Add Variable: `CORS_ORIGINS=https://<your-space-name>.hf.space`
5. The container entrypoint automatically maps Hugging Face's default port (`7860`) and serves the UI.

#### 4. AWS ECS / GCP Cloud Run / Azure Container Apps
Build and push the Docker image to your container registry:
```bash
docker build -t your-registry/carebridge:latest .
docker push your-registry/carebridge:latest
```
Deploy the container with:
- `PORT`: Handled automatically (or set to `8000`).
- Mount a persistent volume at `/app/runtime-data` (e.g. AWS EFS or GCP Cloud Storage FUSE).
- Inject environment variables from AWS Secrets Manager or GCP Secret Manager.

---

### Option C: Bare-Metal Linux Service (systemd)

To run directly on a Linux server without Docker:

1. **Create system user and directories**:
   ```bash
   sudo useradd -r -s /bin/false carebridge
   sudo mkdir -p /opt/carebridge /var/lib/carebridge
   sudo chown -R carebridge:carebridge /opt/carebridge /var/lib/carebridge
   ```

2. **Clone and setup virtual environment**:
   ```bash
   cd /opt/carebridge
   git clone https://github.com/your-org/CareBridge.git .
   python3.11 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
   .venv/bin/pip install -r requirements.txt
   cp .env.production.example .env
   # Edit .env with your settings and set DATABASE_URL=sqlite:////var/lib/carebridge/carebridge.db
   ```

3. **Install systemd unit file (`/etc/systemd/system/carebridge.service`)**:
   ```ini
   [Unit]
   Description=CareBridge Clinical Explainer Service
   After=network.target

   [Service]
   Type=simple
   User=carebridge
   Group=carebridge
   WorkingDirectory=/opt/carebridge
   EnvironmentFile=/opt/carebridge/.env
   ExecStart=/opt/carebridge/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2
   Restart=always
   RestartSec=5
   LimitNOFILE=65536

   [Install]
   WantedBy=multi-user.target
   ```

4. **Enable and start the service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now carebridge
   sudo systemctl status carebridge
   ```

---

## 4. Operational Monitoring & Healthchecks

### Monitoring Endpoints

- **`GET /health`**:
  Liveness and readiness probe. Returns HTTP 200 `{"status": "ok", "service": "CareBridge"}`.
- **`GET /api/about`**:
  Returns current run mode (`live` vs `demo`), LLM connectivity status (`llm_probe_ok`), circuit breaker status, and whether AMD acceleration label is active.
- **`GET /api/metrics`**:
  Live throughput statistics: requests handled, answered vs escalated counts, red-flag triggers, and circuit breaker trip counts.

### Healthcheck Command
For container orchestration or external monitoring:
```bash
curl -f http://localhost:8000/health || exit 1
```

---

## 5. Maintenance & Database Backups

CareBridge persists nurse handoff tickets and resolution notes in SQLite.

### Automated SQLite Backup Script
Create `/usr/local/bin/backup-carebridge.sh`:
```bash
#!/usr/bin/env bash
set -e
BACKUP_DIR="/var/backups/carebridge"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$BACKUP_DIR"

# Safe SQLite snapshot without locking
sqlite3 /app/runtime-data/carebridge.db ".backup '${BACKUP_DIR}/carebridge_${TIMESTAMP}.db'"
gzip "${BACKUP_DIR}/carebridge_${TIMESTAMP}.db"

# Retain backups for 30 days
find "$BACKUP_DIR" -type f -name "*.db.gz" -mtime +30 -delete
```
Schedule via cron (`crontab -e`):
```cron
0 2 * * * /usr/local/bin/backup-carebridge.sh
```

---

## 6. Troubleshooting Common Production Issues

| Issue | Cause | Resolution |
|---|---|---|
| Server fails to start with `RuntimeError: RUN_MODE=live requires LLM_BASE_URL` | Container is in live mode without endpoint | Provide valid `LLM_BASE_URL` in `.env` or set `RUN_MODE=demo` for testing. |
| HTTP 503 on `/nurse` or `/api/tickets` | `NURSE_PASSWORD` is not configured in live mode | Set `NURSE_PASSWORD` in your environment. |
| HTTP 401 on `/nurse` | Incorrect Basic Auth credentials | Provide username `nurse` and the password set in `NURSE_PASSWORD`. |
| Chat responses always escalate to ticket | LLM endpoint unreachable or circuit breaker tripped | Check `/api/about` for `circuit_breaker` state; verify network connectivity to `LLM_BASE_URL`. |
| Port binding error on PaaS | Hardcoded port conflict | Docker container dynamically respects `$PORT`. Ensure your host maps the expected port. |
