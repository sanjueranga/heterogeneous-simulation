"""Append-only CSV logging. Raw rows land in RESULTS_DIR and the scripts in probes/
roll them up into crossover.csv etc."""
import csv
import os
import time
import pathlib

try:
    import fcntl  # lock so concurrent workers don't interleave rows
except ImportError:
    fcntl = None

RESULTS_DIR = pathlib.Path(os.environ.get("RESULTS_DIR", "/data/results"))


def _append(name, header, row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    new = not path.exists()
    with open(path, "a", newline="") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_UN)


_STAGE_HEADER = ["ts", "run_id", "mode", "heavy_c", "cut_threads", "classify_threads",
                 "video_id", "stage", "wall_s"]


def log_stage(video_id, stage, wall_s):
    _append("stage_timings.csv", _STAGE_HEADER, {
        "ts": time.time(),
        "run_id": os.environ.get("RUN_ID", ""),
        "mode": os.environ.get("PIPELINE_MODE", ""),
        "heavy_c": os.environ.get("HEAVY_C", ""),
        "cut_threads": os.environ.get("CUT_THREADS", ""),
        "classify_threads": os.environ.get("CLASSIFY_THREADS", ""),
        "video_id": video_id, "stage": stage, "wall_s": round(wall_s, 4),
    })


_CLASSIFY_HEADER = ["ts", "run_id", "classify_mode", "model", "scope_id", "cache_scopes",
                    "classify_threads", "video_id", "scene_id", "input_tokens",
                    "cache_creation_input_tokens", "cache_read_input_tokens",
                    "output_tokens", "cost_usd", "latency_ms", "cache_hit", "cycles"]


def log_classify(usage, video_id="", scene_id=""):
    _append("classify_calls.csv", _CLASSIFY_HEADER, {
        "ts": time.time(),
        "run_id": os.environ.get("RUN_ID", ""),
        "classify_mode": os.environ.get("CLASSIFY_MODE", "injection"),
        "model": usage.get("model", ""),
        "scope_id": usage.get("scope_id", ""),
        "cache_scopes": os.environ.get("CACHE_SCOPES", ""),
        "classify_threads": os.environ.get("CLASSIFY_THREADS", ""),
        "video_id": video_id, "scene_id": scene_id,
        "input_tokens": usage.get("input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": round(usage.get("cost_usd", 0.0), 8),
        "latency_ms": round(usage.get("latency_ms", 0.0), 2),
        "cache_hit": usage.get("cache_hit", False),
        "cycles": usage.get("cycles", 1),
    })


class timed:
    """Times a block and logs it as a stage row."""
    def __init__(self, video_id, stage):
        self.video_id, self.stage = video_id, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        log_stage(self.video_id, self.stage, time.perf_counter() - self.t0)
