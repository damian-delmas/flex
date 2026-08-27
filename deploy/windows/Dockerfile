FROM python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17

ARG FLEX_VERSION
ARG FLEX_WHEEL_SHA256
ARG FLEX_SOURCE_REVISION=unknown
ARG FLEX_IMAGE_REVISION=unknown

LABEL org.opencontainers.image.title="Flex" \
      org.opencontainers.image.description="Provider-general Flex runtime for Docker Desktop/WSL2" \
      org.opencontainers.image.source="https://github.com/damiandelmas/flex" \
      org.opencontainers.image.version="${FLEX_VERSION}" \
      org.opencontainers.image.revision="${FLEX_SOURCE_REVISION}" \
      io.getflex.wheel.sha256="${FLEX_WHEEL_SHA256}" \
      io.getflex.image.revision="${FLEX_IMAGE_REVISION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY dist/getflex-*.whl /tmp/flex-wheel/

RUN set -eux; \
    count="$(find /tmp/flex-wheel -maxdepth 1 -type f -name 'getflex-*.whl' | wc -l)"; \
    test "$count" -eq 1; \
    wheel="$(find /tmp/flex-wheel -maxdepth 1 -type f -name 'getflex-*.whl' -print)"; \
    actual="$(sha256sum "$wheel" | awk '{print $1}')"; \
    if [ -n "${FLEX_WHEEL_SHA256}" ]; then test "$actual" = "${FLEX_WHEEL_SHA256}"; fi; \
    python -m pip install --no-cache-dir "$wheel"; \
    installed="$(python -c 'import importlib.metadata; print(importlib.metadata.version("getflex"))')"; \
    if [ -n "${FLEX_VERSION}" ]; then test "$installed" = "${FLEX_VERSION}"; fi; \
    rm -rf /tmp/flex-wheel

ENV FLEX_HOME=/root/.flex \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /root/.flex /sources /module-packages

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD python -c "import os; os.kill(1, 0)"

CMD ["python", "-m", "flex.daemon", "--no-refresh", "--no-background"]
