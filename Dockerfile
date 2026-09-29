# syntax=docker/dockerfile:1
#
# Build (host with unreliable internet: download wheels once, build offline):
#   pip download -r requirements-gpu.txt -d wheels/ --timeout 120 --retries 20 \
#       --python-version 3.11 --only-binary=:all: \
#       --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64
#   DOCKER_BUILDKIT=1 docker compose build
#
# Build online (CI or a host with a stable connection): same command without
# the wheels/ folder present; pip falls back to the package indexes.
#
# Host needs: NVIDIA driver >= 570, nvidia-container-toolkit, and a CPU (or VM
# CPU type "host") that exposes AVX2 -- TensorFlow wheels crash without it.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEEPFACE_HOME=/opt/deepface \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (~9 GB layer, rebuilt only when requirements change)
COPY requirements-gpu.txt .
# wheels/ is optional. A glob that matches nothing makes COPY fail, so the
# requirements file is listed too (it always matches).
COPY requirements-gpu.txt wheels* /wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    if ls /wheels/*.whl >/dev/null 2>&1; then \
        pip install --no-index --find-links /wheels -r requirements-gpu.txt; \
    else \
        pip install --timeout 120 --retries 10 -r requirements-gpu.txt; \
    fi \
 && rm -rf /wheels

# ---- Model weights baked in, so containers never download at startup
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512'); \
from retinaface import RetinaFace; RetinaFace.build_model()"

# ---- Application
COPY . .
RUN rm -rf /app/wheels

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One process: concurrency comes from thread pools + GPU batching inside it.
CMD ["python", "main.py"]
