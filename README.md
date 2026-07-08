# media-pipeline-scaling

Code and data from a small study of what goes wrong when you scale out a video
processing pipeline (transcribe, cut into scenes, classify each scene with an
LLM). Two things bit us:

- **memory**: every celery worker loads its own copy of the whisper models, so
  total memory grows with worker count until the container gets OOM-killed.
- **cost**: classification calls share a big prompt prefix (the taxonomy). If the
  calls get spread over several cache scopes, each scope goes cold between calls,
  the prompt cache stops hitting and the bill goes up even though the number of
  calls is the same.

Everything here is a clean-room rebuild with a synthetic taxonomy and generic
prompts. None of it is the original system.

## What's in here

- `harness/` Django + Celery pipeline that runs the real thing (WhisperX, ffmpeg,
  Claude API) and logs timings, memory and token usage. Has a serial version (v0)
  and a fork-join version (v1). See `harness/README.md`.
- `sims/` small simulations of the same effects (these are simulations, not
  measurements). `MODEL.md` has the closed-form model they check.
- `container/` standalone memory-wall probes that run in a docker container with a
  memory cap, with and without the real whisper models.
- `data/` the CSVs. `data/raw` is the harness' raw logs and the rest of `data/` are
  the aggregated measurements from it. `data/sim` is simulation output only, it
  is not measured data.

## Running

Simulations (a few minutes at most, the memory one allocates real RAM):

    cd sims
    pip install -r requirements.txt
    ./run_all.sh

The real harness needs docker, some of your own `.mp4` files and an Anthropic API
key, see `harness/README.md`. The cache sweep makes real API calls and costs
money, the number of calls is capped by `--n`.

## Notes

The measured CSVs in `data/` are from short runs on a laptop and are not a full
sweep (for example only K up to 3 for the latency crossover, and one point for the
cache curve). Re-run `harness/sweep.sh` for the full matrix.

Prices in `harness/pipeline/classify.py` and the sims are hardcoded, check them
before reading anything into the dollar numbers.
