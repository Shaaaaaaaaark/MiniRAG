# MiniRAG 检索服务镜像
# 使用 uv 管理依赖，Python 3.11 slim 基础镜像
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 安装 uv（快速依赖解析/安装）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先拷贝依赖描述，利用层缓存
COPY pyproject.toml ./
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install -r pyproject.toml

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# 拷贝应用代码与配置
COPY minirag ./minirag
COPY config.example.yaml ./config.example.yaml

EXPOSE 8090

# uvicorn 启动 FastAPI；主机端口在 compose 中映射
CMD ["uvicorn", "minirag.api.server:app", "--host", "0.0.0.0", "--port", "8090"]
