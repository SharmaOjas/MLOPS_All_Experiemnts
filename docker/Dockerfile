# Aircraft RUL prediction API. Build from the project root:
#   docker build -f docker/Dockerfile -t rul-api:1.0.0 .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY docker/requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY config/ config/
COPY src/ src/
COPY api/ api/
# Trained bundle produced by `python -m src.train` (model + feature spec + drift reference)
COPY models/rul_model.joblib models/metadata.json models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
