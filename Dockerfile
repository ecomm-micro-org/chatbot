FROM python:3.14-slim
WORKDIR /app
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --no-dev
COPY . .
EXPOSE 8086
CMD [ "uv","run","-m","src.server" ]
