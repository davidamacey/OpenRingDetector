FROM python:3.12-slim

# System libraries required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps in two stages so the heavy torch layer is cached
# separately from the application code.
COPY pyproject.toml README.md ./

# Install torch with CUDA 12.1 wheels first (pinned to facenet-pytorch compat window)
RUN pip install --no-cache-dir \
        torch==2.2.2+cu121 \
        torchvision==0.17.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps with stub so code changes don't bust this layer
RUN mkdir -p ring_detector && touch ring_detector/__init__.py && \
    pip install --no-cache-dir -e .

# Copy actual source after deps are cached
COPY ring_detector/ ./ring_detector/

# Create runtime directories
RUN mkdir -p /app/tokens /app/models /app/logs \
    && chown -R appuser:appuser /app

USER appuser

# Health check: attempt a Ring status probe (requires DB to be up)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD ring-status 2>&1 | grep -q "\[+" || exit 1

CMD ["ring-watch"]
