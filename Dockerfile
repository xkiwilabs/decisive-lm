# DecisiveLM server (decisivelm/serve.py). Configured only by environment variables; see that module's docstring.
#
#   docker build -t decisivelm-serve .
#   docker run --gpus '"device=0"' -p 127.0.0.1:8003:8003 \
#       -v ~/models/hub:/models/hub:ro -v ~/bundles:/bundles:ro \
#       -e DECISIVELM_BUNDLES=/bundles -e DECISIVELM_MEMORY_GB=20 -e DECISIVELM_ALLOW_UNAUTHENTICATED=1 decisivelm-serve
#
# The backbone's weights are read from the mounted Hugging Face cache (offline by default); bundles from /bundles.
FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve --extra audio --no-install-project
COPY decisivelm decisivelm
COPY configs/model.yaml configs/model.yaml
RUN uv sync --frozen --no-dev --extra serve --extra audio
RUN useradd --system --uid 10001 decisivelm
USER decisivelm
# The container listens on all its interfaces, so it starts only with DECISIVELM_AUTH set (keys or litellm), or with
# DECISIVELM_ALLOW_UNAUTHENTICATED=1 when its port is published to loopback alone (docker run -p 127.0.0.1:8003:8003).
ENV HF_HUB_CACHE=/models/hub HF_HUB_OFFLINE=1 DECISIVELM_HOST=0.0.0.0 DECISIVELM_PORT=8003
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['DECISIVELM_PORT'], timeout=4)"]
CMD ["/app/.venv/bin/python", "-m", "decisivelm.serve"]
