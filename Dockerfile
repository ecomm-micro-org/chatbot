FROM ghcr.io/astral-sh/uv:0.11-python3.14-trixie-slim
WORKDIR /app
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --no-dev
COPY . .
EXPOSE 8086
CMD [ "uv","run","-m","src.server" ]
