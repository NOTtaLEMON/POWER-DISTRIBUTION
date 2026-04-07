FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY energy_grid_env/ ./energy_grid_env/

RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "numpy>=1.19.0"

RUN pip install --no-cache-dir -e ./energy_grid_env/

ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

CMD ["uvicorn", "energy_grid_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
