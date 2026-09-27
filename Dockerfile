# DefectLens serving image: FastAPI + Gradio + OpenVINO INT8, CPU only, no torch (~0.6 GB).
# Build context must contain deploy/model/ (scripts/export_bundle.py).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    DEFECTLENS_MODEL_DIR=/app/deploy/model \
    DEFECTLENS_DEVICE=CPU

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user   # Hugging Face Spaces run containers as uid 1000

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY --chown=user src/__init__.py src/__init__.py
COPY --chown=user src/edge/__init__.py src/edge/preprocess.py src/edge/runtime.py src/edge/
COPY --chown=user app/ app/
COPY --chown=user deploy/model/ deploy/model/

USER user
ENV HOME=/home/user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/v1/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
