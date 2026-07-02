import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
app = Celery("repro", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,   # one task per slot -> honest concurrency for Fig 2
    task_acks_late=True,
    task_default_queue="general",
)
app.autodiscover_tasks(["pipeline"])  # finds pipeline/tasks.py
