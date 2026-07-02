FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.routes.process_session:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
