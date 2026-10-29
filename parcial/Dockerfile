# Entorno reproducible del examen parcial (CPU).
# Construir:  docker build -t cc0f4-parcial .
# Ejecutar:   docker run --rm -p 8888:8888 -v "$PWD":/workspace/parcial cc0f4-parcial

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace/parcial

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt \
    && python -m pip check

COPY . /workspace/parcial

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.root_dir=/workspace/parcial"]
