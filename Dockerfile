FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    RUN_MODE=demo \
    HF_HOME=/opt/huggingface \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    EMBEDDINGS_ENABLED=true \
    DATABASE_URL=sqlite:////app/runtime-data/carebridge.db

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system carebridge \
    && useradd --system --gid carebridge --home-dir /app carebridge \
    && mkdir -p /app/runtime-data /opt/huggingface \
    && chown -R carebridge:carebridge /app /opt/huggingface

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.6.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=carebridge:carebridge app ./app
COPY --chown=carebridge:carebridge data ./data
COPY --chown=carebridge:carebridge ui ./ui
COPY --chown=carebridge:carebridge eval ./eval
COPY --chown=carebridge:carebridge scripts ./scripts
RUN chmod +x ./scripts/docker-entrypoint.sh

USER carebridge
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)"

ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
