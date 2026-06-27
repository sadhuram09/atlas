# ============================================================
# ATLAS Dockerfile — multi-stage build
# ============================================================
#
# Stage 1 (builder): install ALL dependencies including dev tools
# Stage 2 (runtime): copy only what's needed — smaller final image
#
# Why multi-stage?
#   The builder installs gcc, poetry, and build tools (~800MB).
#   The runtime copies the venv + code only (~200MB).
#   Railway charges by memory — smaller = faster + cheaper.
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed to build Python packages
# (Some packages like asyncpg compile C extensions)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry — we use it for deterministic dependency resolution
RUN pip install --no-cache-dir poetry==1.8.4

# Copy dependency files first (Docker layer caching)
# If pyproject.toml hasn't changed, this layer is cached — fast rebuilds
COPY pyproject.toml poetry.lock* ./

# Install dependencies into a virtualenv at /app/.venv
# --no-dev: skip pytest, ruff, mypy in production image
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi --without dev

# --- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime system deps only (libpq for asyncpg)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from builder (has all pip packages)
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY atlas/ ./atlas/

# Add .venv to PATH so `python` uses the venv Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Don't buffer Python output — important for Railway log streaming
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Railway injects PORT — default to 8000
ENV PORT=8000

# Non-root user for security
RUN useradd --create-home --shell /bin/bash atlas \
    && chown -R atlas:atlas /app
USER atlas

# Health check — Railway probes this
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health').raise_for_status()"

EXPOSE ${PORT}

# Start the server — shell form so $PORT is expanded from environment
CMD ["sh", "-c", "uvicorn atlas.api.app:app --host 0.0.0.0 --port ${PORT}"]
