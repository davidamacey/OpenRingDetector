FROM python:3.12-slim

WORKDIR /app

RUN adduser --disabled-password --uid 1000 appuser && \
    apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Install backend requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ring_detector package from repo root
COPY ../pyproject.toml /repo/pyproject.toml
COPY ../ring_detector /repo/ring_detector
RUN pip install --no-cache-dir -e /repo

# Copy backend app
COPY . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
