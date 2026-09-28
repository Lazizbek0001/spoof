FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPFACE_HOME=/opt/deepface \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies layer (~9 GB) is rebuilt only when requirements-gpu.txt changes.
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt

# Bake DeepFace weights into the image so pods never download at startup.
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512'); \
from retinaface import RetinaFace; RetinaFace.build_model()"

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One process: concurrency comes from thread pools + GPU batching inside it.
CMD ["python", "main.py"]