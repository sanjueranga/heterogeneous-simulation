#!/usr/bin/env python3
"""
mem_wall.py -- runs inside a memory-capped container (--memory=Ng --memory-swap=Ng).

The host-side sims/measure_memory_blowup.py only captures the static floor
(weights x concurrency), and on a laptop macOS evicts pages so RSS understates
the real peak. Inside a Linux container the cgroup enforces the cap, memory.peak
is what docker accounts for, and going over the cap is a real OOM kill.

Workload: `-c` prefork workers. Each holds the model weights for its whole life,
plus (1 + prefetch) working-set buffers: the job it's on now and the next
job(s) celery already pulled off the broker. Everyone touches their full
footprint and waits at a barrier, so the parent reads the true simultaneous peak.

    per-worker peak = weights_gb + (1 + prefetch) * working_set_gb
    total peak     ~= base + concurrency * per-worker

The sizes are made-up defaults, set them from a docker stats peak of your own
worker.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

GiB = 1024 ** 3
CG = "/sys/fs/cgroup"  # cgroup v2 unified hierarchy


def _cg_read(name):
    """Read an integer cgroup-v2 memory file; None if unavailable ('max')."""
    try:
        with open(os.path.join(CG, name)) as fh:
            val = fh.read().strip()
        return None if val == "max" else int(val)
    except (FileNotFoundError, ValueError):
        return None


def _touch(n_bytes):
    """Allocate n_bytes of NON-compressible memory and force it resident.

    Random doubles (not zeros): the kernel/zswap compresses uniform pages, so a
    zero-filled buffer would never stay resident and would understate footprint.
    """
    import numpy as np
    arr = np.random.default_rng().random(int(n_bytes) // 8)
    arr += 1.0  # write every page so it is actually faulted in (RSS, not just VSZ)
    return arr


def _worker(weights_gb, working_set_gb, prefetch, ready_q, release_evt):
    """Allocate a worker's full peak footprint, signal ready, hold until released."""
    held = [_touch(weights_gb * GiB)]                      # model weights (steady)
    for _ in range(1 + prefetch):                          # in-flight + prefetched
        held.append(_touch(working_set_gb * GiB))
    ready_q.put(os.getpid())                               # "I am at peak footprint"
    release_evt.wait(timeout=120)                          # hold the peak co-resident
    del held


def main():
    ap = argparse.ArgumentParser(description="In-container memory-wall probe.")
    ap.add_argument("--concurrency", type=int, required=True, help="prefork workers (-c).")
    # defaults are made up, override from real docker stats peaks
    ap.add_argument("--weights-gb", type=float, default=0.8,
                    help="Per-worker model weights held for the worker's life.")
    ap.add_argument("--working-set-gb", type=float, default=0.35,
                    help="Per in-flight job decode/transcode buffer.")
    ap.add_argument("--prefetch", type=int, default=1,
                    help="Buffered next-task payloads per worker (Celery prefetch).")
    args = ap.parse_args()

    c = args.concurrency
    per_worker = args.weights_gb + (1 + args.prefetch) * args.working_set_gb
    cap = _cg_read("memory.max")
    cap_gb = cap / GiB if cap else float("nan")

    print(f"[probe] -c {c} | weights {args.weights_gb}G + "
          f"(1+{args.prefetch}) x working-set {args.working_set_gb}G "
          f"= {per_worker:.2f}G/worker | cgroup cap {cap_gb:.2f}G", flush=True)

    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    release_evt = ctx.Event()
    procs = [ctx.Process(target=_worker,
                         args=(args.weights_gb, args.working_set_gb, args.prefetch,
                               ready_q, release_evt))
             for _ in range(c)]
    for p in procs:
        p.start()

    # Wait for every worker to reach its full footprint. If the cgroup OOM-kills a
    # worker first, the container exits 137 before we get here -- the host detects
    # that via docker's OOMKilled flag, so a clean "fit" only prints on success.
    ready = 0
    deadline = time.time() + 90
    while ready < c and time.time() < deadline:
        try:
            ready_q.get(timeout=5)
            ready += 1
        except Exception:
            if not any(p.is_alive() for p in procs):
                break  # all workers gone without signalling -> treat as failure

    peak = _cg_read("memory.peak")
    cur = _cg_read("memory.current")
    fit = ready == c
    release_evt.set()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    peak_gb = peak / GiB if peak else float("nan")
    cur_gb = cur / GiB if cur else float("nan")
    # Single machine-parseable line the host orchestrator greps for.
    print(f"RESULT concurrency={c} fit={fit} peak_gb={peak_gb:.4f} "
          f"current_gb={cur_gb:.4f} cap_gb={cap_gb:.4f} "
          f"per_worker_gb={per_worker:.4f}", flush=True)
    sys.exit(0 if fit else 1)


if __name__ == "__main__":
    main()
