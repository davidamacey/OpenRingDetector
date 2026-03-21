FROM python:3.12-slim

WORKDIR /app

RUN adduser --disabled-password --uid 1000 appuser && \
    apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc curl coreutils \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install backend requirements
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install ring_detector package from repo root
COPY pyproject.toml /app/pyproject.toml
COPY ring_detector /app/ring_detector
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic
RUN pip install --no-cache-dir -e .

# Copy backend app
COPY backend/app /app/app

# Watcher needs write access to tokens dir (Ring token refresh)
RUN mkdir -p /app/tokens && chown -R appuser:appuser /app/tokens

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
