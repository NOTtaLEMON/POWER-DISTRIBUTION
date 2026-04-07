ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "numpy>=1.19.0" && \
    pip install --no-cache-dir -e .

# Set PYTHONPATH so package imports resolve correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "energy_grid_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
