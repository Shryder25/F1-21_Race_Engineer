# Cloud engineer: HUD + analysis + AI brain, fed by tools/cloud_relay.py on the game PC.
# Railway and Fly.io build this file as-is. See docs/cloud-deployment.md.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLOUD_MODE=1 \
    DATA_DIR=/data \
    PORT=8080

WORKDIR /app
# The server needs numpy only: voice, microphone and controller packages stay on the PC.
RUN pip install --no-cache-dir "numpy>=1.26"

COPY config.toml cloud_server.py ./
COPY ai_engineer/ ai_engineer/

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "cloud_server.py"]
