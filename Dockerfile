# UniBot backend - production image
#
# Design goals for a "no trade-offs" deploy:
#  - Knowledge base (chroma_db_fastembed/) is baked into the image at
#    build time, not loaded from a runtime-attached disk. The app never
#    writes to it while serving chat, so it doesn't need to be "state"
#    at all - it's a build artifact, like the code itself. Works
#    identically on hosts with or without persistent volumes.
#  - FastEmbed's model is pre-downloaded at build time into
#    ./fastembed_cache (see retrieval/rag_system.py, which pins
#    FASTEMBED_CACHE_DIR to this path) so the container needs zero
#    network access to HuggingFace at runtime / on cold start.
#  - No Playwright browser binaries installed: the live-scraping tool is
#    intentionally excluded from the chat agent's tool list (see
#    agents/unibot.py), so a full Chromium install here would only add
#    image weight for a code path that's never reachable in production.
#  - GOOGLE_API_KEY (and any other secret) is read from the environment
#    at runtime, never baked into the image. Do not COPY .env - it's in
#    .dockerignore. Set it via your platform's env var / secrets UI.

FROM python:3.12-slim

WORKDIR /app

# System deps: gcc/build-essential for any packages without prebuilt
# wheels on slim images, libgomp1 for onnxruntime (used by FastEmbed).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer caches across code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the FastEmbed model at build time (needs network access
# during `docker build`, not at runtime). Must match FASTEMBED_CACHE_DIR
# in retrieval/rag_system.py.
ENV FASTEMBED_CACHE_DIR=/app/fastembed_cache
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; \
    FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/fastembed_cache').embed_query('warm cache')"

# App code + the pre-built knowledge base (see .dockerignore for what's
# excluded - notably venv/, node_modules/, .env, the old chroma_db/
# backup, and frontend/ since that's deployed separately as static files)
COPY . .

ENV PYTHONUNBUFFERED=1
# Most PaaS hosts (Render, Railway, etc.) inject $PORT at runtime and
# expect the app to bind to it; 8001 is the local-dev fallback.
ENV PORT=8001
EXPOSE 8001

CMD ["sh", "-c", "uvicorn backend.api_server:app --host 0.0.0.0 --port ${PORT}"]