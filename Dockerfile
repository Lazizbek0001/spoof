# syntax=docker/dockerfile:1

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
      libegl1 \
      libgles2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gpu.txt .

RUN --mount=type=bind,from=wheels,target=/wheels,readonly \
    pip install \
      --no-cache-dir \
      --no-index \
      --find-links=/wheels \
      -r requirements-gpu.txt

COPY . .

RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512'); from retinaface import RetinaFace; RetinaFace.build_model()"

EXPOSE 8000

CMD ["python", "main.py"]
