# real whisperx `base` int8 + wav2vec2 align, loaded per prefork worker inside a
# memory-capped container
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # model caches live in the image so the sweep runs offline
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    OMP_NUM_THREADS=2

WORKDIR /work

# whisperx.load_audio shells out to ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# cpu-only torch, then whisperx and ctranslate2
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchaudio==2.8.0 && \
    pip install --no-cache-dir "whisperx>=3.7.6" "ctranslate2>=4.5.0"

# 30s of 16kHz mono noise, enough to exercise decode/transcribe/align
RUN ffmpeg -y -f lavfi -i "anoisesrc=d=30:c=pink:r=16000:a=0.3" \
        -ar 16000 -ac 1 /work/sample.wav -loglevel error

COPY whisper_wall.py /work/whisper_wall.py

# download both models at build time
RUN python -c "import whisperx; \
    m=whisperx.load_model('base', device='cpu', compute_type='int8'); \
    a,meta=whisperx.load_align_model(language_code='en', device='cpu'); \
    print('[build] cached base int8 + en align model')"

ENTRYPOINT ["python", "/work/whisper_wall.py"]
