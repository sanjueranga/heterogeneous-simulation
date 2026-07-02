# harness

The real pipeline, used to collect the measurements. Django + Celery + Redis +
Postgres in docker compose, WhisperX for transcription, ffmpeg for cutting and
the Claude API for classification (with a synthetic taxonomy).

    pipeline/   models, stages, classify, v0 and v1 tasks, management commands
    config/     django settings + celery app
    probes/     memory probe and the scripts that aggregate raw logs into data/
    sweep.sh    runs everything

## Setup

1. `cp .env.example .env` and put your `ANTHROPIC_API_KEY` in it
2. drop a few `.mp4` files into `videos_in/` (they're gitignored)
3. `docker compose up -d --build`, the first build is slow because of torch

## One video end to end

    docker compose exec web python manage.py submit_batch --k 1 --mode v1
    docker compose exec web python manage.py cost_sweep --scopes 1 --n 5 --headline

Look at `_data/results/stage_timings.csv` and `classify_calls.csv`. From the second
classify call on, `cache_read_input_tokens` should be non-zero.

## Full run

    ./sweep.sh

That does the latency runs (A), the cache/cost runs (C) and the memory runs (B)
and writes the CSVs to `../data`. The memory probe restarts the heavy worker
many times so it takes a while.

## Settings

All in `.env`: `PIPELINE_MODE` (v0/v1), `HEAVY_C`, `LIGHT_C`, `CUT_THREADS`,
`CLASSIFY_THREADS`, `CLASSIFY_MODE` (injection/agentic), `CACHE_SCOPES`,
`N_SCENES`, `WHISPER_MODEL`, `MEM_CAP_GB`, `CLASSIFY_MODEL`.
