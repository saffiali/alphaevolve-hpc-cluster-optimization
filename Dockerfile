FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt google-auth requests

COPY pillars/ /app/pillars/
COPY app/ /app/app/
COPY scripts/ /app/scripts/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
