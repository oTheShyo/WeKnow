# syntax=docker/dockerfile:1
# WeKnow — 单容器生产部署
# 构建：docker build -t weknow .
# 运行：docker compose up -d   （compose 会挂载 .env 与数据卷）

FROM python:3.11-slim

# 安装 uv（依赖管理），配合 uv.lock 保证可复现安装。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先复制依赖清单再复制源码，利用 Docker 层缓存：依赖不变时跳过安装。
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# server extra 包含 Web 前端所需的 fastapi / uvicorn / python-multipart。
RUN uv sync --frozen --no-dev --extra server

# HF_ENDPOINT 指向镜像站：容器内首次启动要下载 SaT 分句模型（约 400 MB）。
# LITELLM_LOCAL_MODEL_COST_MAP 禁止 litellm 联网拉取模型价格表。
ENV HF_ENDPOINT=https://hf-mirror.com \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 配置（.env）与数据库（data/）由 compose 挂载进来；容器内不落任何状态。
CMD [".venv/bin/weknow", "serve", "--host", "0.0.0.0", "--port", "8000"]
