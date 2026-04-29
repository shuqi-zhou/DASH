# Base image: CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget curl ca-certificates unzip \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3       /usr/bin/pip

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["demo"]
