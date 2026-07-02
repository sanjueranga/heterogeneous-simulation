FROM python:3.11-slim

# ffmpeg for cutting and audio decode, libgomp for ctranslate2
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git libgomp1 build-essential postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# whisperx pulls in torch, so this layer is big and slow the first time
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1 DJANGO_SETTINGS_MODULE=config.settings
