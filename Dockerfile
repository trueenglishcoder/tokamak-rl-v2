FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/tokamak-rl-v2

COPY pyproject.toml README.md ./
COPY tokamak_rl_v2 ./tokamak_rl_v2
COPY scripts ./scripts
COPY configs ./configs

RUN python -m pip install --upgrade pip && python -m pip install .

CMD ["python", "scripts/train.py", "--help"]
