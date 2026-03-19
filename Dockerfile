FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY setup.py README.md ./
COPY molfun/ molfun/

RUN pip install --upgrade pip && \
    pip install -e ".[openfold,streaming]"

COPY recipes/ recipes/

EXPOSE 8000

ENTRYPOINT ["molfun"]
CMD ["info"]
