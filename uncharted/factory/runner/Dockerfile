# Sandbox for running AI written code: no network at test time, pytest preinstalled.
# The extra packages are the ones v3lib specs may use (build.LIB_IMPORTS, spec.SANDBOX_LIBS).
# Bump build.IMAGE's tag when this file changes.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.* \
        numpy cryptography aioquic networkx lark pillow msgpack zstandard pyyaml tomli-w \
    && useradd -m runner
USER runner
WORKDIR /w
