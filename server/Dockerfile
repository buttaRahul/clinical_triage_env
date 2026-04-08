FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy environment package
COPY . /app/env/

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    pydantic==2.7.4 \
    httpx==0.27.0 \
    python-multipart==0.0.9 \
    openai>=1.0.0

# Expose port
EXPOSE 8000

# Environment defaults
ENV TRIAGE_TASK=easy
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app/env

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
