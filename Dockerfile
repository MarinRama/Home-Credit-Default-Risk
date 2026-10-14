FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MONITORING_DB_PATH=/app/data/production/predictions.sqlite3

WORKDIR /app

# Bibliothèque OpenMP utilisée par LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN python -m pip install -r requirements-api.txt \
    && python -m pip check

# Utilisateur dédié pour exécuter l'API.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home appuser \
    && mkdir -p /app/data/production \
    && chown -R appuser:appuser /app/data

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]