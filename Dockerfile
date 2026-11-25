# Runs the guarded UR5e from MuJoCo Menagerie over stdio. Used by MCP directories to start and inspect the server.
FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home guard
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-compile .
USER guard
# Download the Menagerie model at build time, so the server starts in about a second.
RUN mjguard check ur5e > /dev/null
ENTRYPOINT ["mjguard", "serve", "ur5e"]
