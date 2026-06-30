#!/usr/bin/env python3
"""
whisper_wall.py -- memory wall with the real models.

Loads the real whisperx `base` int8 model plus the wav2vec2 (en) alignment model
in each of `-c` prefork workers inside a memory-capped container, runs an actual
transcribe + align so every worker reaches its real peak, then reads the cgroup
`memory.peak`. If the cap is exceeded the kernel OOM-kills the container.

Workers are started with mp `spawn`, so each one imports torch and loads its own
copy of the weights, same as a celery prefork child that imports whisperx lazily
on its first task (no copy-on-write sharing from the parent).

    per-worker peak = transcription weights + align model
                      + (1 + prefetch) audio working sets + torch/python runtime
    total peak     ~= base + concurrency * per-worker

The parent samples memory.peak once every worker has reached its peak. If a worker
is OOM-killed first the container exits 137 before RESULT prints, and the host
script picks that up from docker's OOMKilled flag.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

GiB = 1024 ** 3
CG = "/sys/fs/cgroup"  # cgroup v2 unified hierarchy

# Baked into the image by the Dockerfile (a real 16kHz mono wav of noise).
AUDIO_WAV = os.environ.get("AUDIO_WAV", "/work/sample.wav")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
ALIGN_LANG = os.environ.get("ALIGN_LANG", "en")


def _cg_read(name):
    """Read an integer cgroup-v2 memory file; None if unavailable ('max')."""
    try:
        with open(os.path.join(CG, name)) as fh:
            val = fh.read().strip()
        return None if val == "max" else int(val)
    except (FileNotFoundError, ValueError):
        return None


def _worker(prefetch, ready_q, release_evt):
    """Load the models and a working set, say we're at peak, then hold it until released."""
    import whisperx  # imported inside the worker so nothing is shared with the parent

    held = []  # keep references so nothing gets garbage collected before sampling

    # 1. transcription model, held for the worker's whole life
    model = whisperx.load_model(MODEL_SIZE, device="cpu", compute_type="int8")
    held.append(model)

    # 2. the job in flight: audio, transcribe, forced alignment. In a burst every
    #    slot is mid-job, so the align model is resident too.
    audio = whisperx.load_audio(AUDIO_WAV)
    held.append(audio)
    result = model.transcribe(audio, language=ALIGN_LANG)
    align_model, align_meta = whisperx.load_align_model(
        language_code=ALIGN_LANG, device="cpu")
    held.append(align_model)
    aligned = whisperx.align(result["segments"], align_model, align_meta,
                             audio, device="cpu", return_char_alignments=False)
    held.append(aligned)

    # 3. prefetch: a busy worker has already pulled the next job(s) off the broker,
    #    so hold that many extra decoded audio buffers
    for _ in range(prefetch):
        held.append(whisperx.load_audio(AUDIO_WAV))

    ready_q.put(os.getpid())
    release_evt.wait(timeout=180)     # hold the peak while the parent samples
    del held


def main():
    ap = argparse.ArgumentParser(description="In-container memory probe with the real whisper models.")
    ap.add_argument("--concurrency", type=int, required=True, help="prefork workers (-c).")
    ap.add_argument("--prefetch", type=int, default=1,
                    help="Buffered next-task audio payloads per worker (Celery prefetch).")
    args = ap.parse_args()

    c = args.concurrency
    cap = _cg_read("memory.max")
    cap_gb = cap / GiB if cap else float("nan")

    print(f"[probe] -c {c} | real whisperx '{MODEL_SIZE}' int8 + {ALIGN_LANG} align "
          f"+ (1+{args.prefetch}) audio working-sets | cgroup cap {cap_gb:.2f}G",
          flush=True)

    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    release_evt = ctx.Event()
    procs = [ctx.Process(target=_worker, args=(args.prefetch, ready_q, release_evt))
             for _ in range(c)]
    for p in procs:
        p.start()

    # wait for every worker to reach full footprint (slow on CPU, so a long deadline).
    # If a worker gets OOM-killed first the container dies with 137 before we
    # sample and the host script catches OOMKilled.
    ready = 0
    deadline = time.time() + 600
    while ready < c and time.time() < deadline:
        try:
            ready_q.get(timeout=10)
            ready += 1
        except Exception:
            if not any(p.is_alive() for p in procs):
                break  # all workers gone without signalling -> failure (likely OOM)

    peak = _cg_read("memory.peak")
    cur = _cg_read("memory.current")
    fit = ready == c
    release_evt.set()
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    peak_gb = peak / GiB if peak else float("nan")
    cur_gb = cur / GiB if cur else float("nan")
    per_worker_gb = peak_gb / c if (c and peak) else float("nan")
    print(f"RESULT concurrency={c} fit={fit} peak_gb={peak_gb:.4f} "
          f"current_gb={cur_gb:.4f} cap_gb={cap_gb:.4f} "
          f"per_worker_gb={per_worker_gb:.4f}", flush=True)
    sys.exit(0 if fit else 1)


if __name__ == "__main__":
    main()
