# syntax=docker/dockerfile:1
# Build from repository root:  docker build -f docker/Dockerfile -t fdi-lighthouse:py314 .
# ACR:  az acr build -r <registry> -f docker/Dockerfile -t fdi-lighthouse:py314 ..

FROM python:3.14-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lh-v2

# Install the library (Lighthouse) only — do not copy tests or local .venv
COPY pyproject.toml README.md ./
COPY lh_v2 ./lh_v2
COPY uv.lock ./uv.lock

# uv gives reproducible installs; curl already present
RUN pip install --no-cache-dir "uv>=0.5" \
    && UV_PROJECT_ENVIRONMENT=/opt/lh-v2/.venv uv sync --no-dev --frozen

ENV PATH="/opt/lh-v2/.venv/bin:${PATH}"

# Verify imports used by the training path
RUN python -c "import lh_v2; print('lh_v2', lh_v2.__file__ if hasattr(lh_v2, '__file__') else 'ok')"

# Default: smoke import (override in `az ml job` command)
CMD ["python", "-c", "import lh_v2, sys; print('python', sys.version); import lh_v2; print('lh_v2 ok')"]
