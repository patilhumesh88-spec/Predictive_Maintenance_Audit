FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install .[api,dashboard]

COPY maint_audit/ maint_audit/
COPY models/ models/
COPY data/ data/
COPY reports/ reports/
COPY dashboard/ dashboard/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "maint_audit.api:app", "--host", "0.0.0.0", "--port", "8000"]
