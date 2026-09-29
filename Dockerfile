FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements-gpu.txt .
COPY wheels/ /wheels/

RUN pip install \
    --no-index \
    --find-links=/wheels \
    --prefix=/install \
    -r requirements-gpu.txt


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEEPFACE_HOME=/opt/deepface \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512'); from retinaface import RetinaFace; RetinaFace.build_model()"

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "main.py"]
