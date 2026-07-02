#!/usr/bin/env python3
"""Experiment B: memory wall. Runs on the host, needs the docker CLI.

For each (HEAVY_C x CUT_THREADS x MEM_CAP_GB) it recreates worker_heavy with that
config (which resets the cgroup peak), pushes K videos through the v1 pipeline so
every heavy slot is busy, then reads memory.peak from the cgroup and checks docker
for an OOM kill. Rows go to _data/results/memory_blowup_detail.csv.

  python3 probes/mem_probe.py --heavy-c 1 2 3 4 5 6 --cut-threads 1 2 4 \
      --caps 5 3 --k 6 --repeats 3

Run from harness/ with the stack already built (docker compose build).
A standalone version without the full stack is in ../container/real_whisper.
"""
import argparse
import csv
import os
import pathlib
import subprocess
import time

HARNESS = pathlib.Path(__file__).resolve().parent.parent
RESULTS = HARNESS / "_data" / "results"
# Set DOCKER_COMPOSE=docker-compose if the v2 `docker compose` plugin isn't present.
DC = os.environ.get("DOCKER_COMPOSE", "docker compose").split()
HEADER = ["concurrency", "cut_threads", "total_rss_gb", "per_worker_gb", "cap_gb",
          "exceeds_cap", "oom_killed", "backend", "rep"]


def dc(*args, env=None, check=True, capture=False):
    return subprocess.run([*DC, *args], cwd=HARNESS,
                          env={**os.environ, **(env or {})}, check=check,
                          text=True, capture_output=capture)


def container_id(service):
    return dc("ps", "-q", service, capture=True).stdout.strip()


def read_peak_bytes(service):
    cid = container_id(service)
    for path in ("/sys/fs/cgroup/memory.peak",       # cgroup v2
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):  # v1
        r = subprocess.run(["docker", "exec", cid, "cat", path],
                           text=True, capture_output=True)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            return int(r.stdout.strip())
    # fallback to current usage
    r = subprocess.run(["docker", "exec", cid, "cat", "/sys/fs/cgroup/memory.current"],
                       text=True, capture_output=True)
    return int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0


def oom_killed(service):
    cid = container_id(service)
    r = subprocess.run(["docker", "inspect", "--format", "{{.State.OOMKilled}}", cid],
                       text=True, capture_output=True)
    return r.stdout.strip() == "true"


def append_row(row):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "memory_blowup_detail.csv"   # full sweep; aggregate -> clean CSV
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy-c", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--cut-threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--caps", type=int, nargs="+", default=[5, 3])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    o = ap.parse_args()

    for cap in o.caps:
        for c in o.heavy_c:
            for th in o.cut_threads:
                for rep in range(1, o.repeats + 1):
                    env = {"HEAVY_C": str(c), "CUT_THREADS": str(th),
                           "MEM_CAP_GB": str(cap), "PIPELINE_MODE": "v1"}
                    # recreating the worker resets memory.peak
                    dc("up", "-d", "--force-recreate", "worker_heavy", env=env, check=False)
                    time.sleep(8)  # let it register with the broker
                    # push K videos through (the driver runs in web)
                    dc("exec", "-T", "web", "python", "manage.py", "submit_batch",
                       "--k", str(o.k), "--mode", "v1", "--timeout", "150",
                       "--run-id", f"B_c{c}_t{th}_cap{cap}_rep{rep}",
                       env=env, check=False)
                    peak = read_peak_bytes("worker_heavy")
                    oom = oom_killed("worker_heavy")
                    total_gb = round(peak / 1e9, 4)
                    # a killed container can't report its peak (reads 0), so use the cap
                    if oom and total_gb < cap:
                        total_gb = float(cap)
                    row = {
                        "concurrency": c, "cut_threads": th,
                        "total_rss_gb": total_gb,
                        "per_worker_gb": round(total_gb / max(1, c), 4),
                        "cap_gb": float(cap),
                        "exceeds_cap": (total_gb > cap) or oom,
                        "oom_killed": "true" if oom else "false",
                        "backend": "real-whisper", "rep": rep,
                    }
                    append_row(row)
                    print(row)


if __name__ == "__main__":
    main()
