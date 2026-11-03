# Sandbox for running AI written code: no network at test time, pytest preinstalled.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.* && useradd -m runner
USER runner
WORKDIR /w
