"""Experiment A: submit K videos as one batch and time it end to end.

Picks K files from videos_in/ (repeats them if there are fewer), queues them for
the chosen mode, waits until all are completed, and appends a row to
batch_runs.csv.

Usage:
  python manage.py submit_batch --k 5 --mode v1 --run-id rA_v1_k5_rep1
"""
import time
import uuid
import itertools
from django.core.management.base import BaseCommand

from pipeline import stages, instrument
from pipeline.models import Job, Video
from pipeline.tasks_celery import process_video_pipeline
from pipeline.tasks_serial import process_video_serial

_BATCH_HEADER = ["ts", "run_id", "mode", "k", "heavy_c", "cut_threads",
                 "classify_threads", "classify_mode", "wall_s", "completed", "failed"]


class Command(BaseCommand):
    help = "Submit a batch of K videos and time end-to-end."

    def add_arguments(self, p):
        p.add_argument("--k", type=int, default=1)
        p.add_argument("--mode", choices=["v0", "v1"], default="v1")
        p.add_argument("--run-id", default="")
        p.add_argument("--timeout", type=int, default=3600)

    def handle(self, *a, **o):
        import os
        os.environ["PIPELINE_MODE"] = o["mode"]
        run_id = o["run_id"] or f"{o['mode']}_k{o['k']}_{uuid.uuid4().hex[:6]}"
        os.environ["RUN_ID"] = run_id

        vids = stages.list_videos()
        if not vids:
            self.stderr.write("No .mp4 in videos_in/. Add files and retry.")
            return
        chosen = list(itertools.islice(itertools.cycle(vids), o["k"]))

        job = Job.objects.create(mode=o["mode"], k=o["k"], run_id=run_id, status="running")
        video_ids = []
        for path in chosen:
            v = Video.objects.create(job=job, filename=os.path.basename(path), path=path)
            video_ids.append(v.id)

        task = process_video_serial if o["mode"] == "v0" else process_video_pipeline
        t0 = time.perf_counter()
        for vid in video_ids:
            task.delay(str(job.id), vid)

        deadline = t0 + o["timeout"]
        while time.perf_counter() < deadline:
            done = Video.objects.filter(job=job, status="completed").count()
            if done >= o["k"]:
                break
            time.sleep(1.0)
        wall = time.perf_counter() - t0

        completed = Video.objects.filter(job=job, status="completed").count()
        job.status = "completed" if completed >= o["k"] else "timeout"
        job.save(update_fields=["status"])

        instrument._append("batch_runs.csv", _BATCH_HEADER, {
            "ts": time.time(), "run_id": run_id, "mode": o["mode"], "k": o["k"],
            "heavy_c": os.environ.get("HEAVY_C", ""),
            "cut_threads": os.environ.get("CUT_THREADS", ""),
            "classify_threads": os.environ.get("CLASSIFY_THREADS", ""),
            "classify_mode": os.environ.get("CLASSIFY_MODE", "injection"),
            "wall_s": round(wall, 4), "completed": completed,
            "failed": o["k"] - completed,
        })
        self.stdout.write(f"[{run_id}] {o['mode']} K={o['k']} wall={wall:.2f}s "
                          f"completed={completed}/{o['k']}")
