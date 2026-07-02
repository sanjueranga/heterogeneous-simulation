# Celery autodiscover entry point — re-export tasks from both orchestration modes.
from .tasks_celery import process_video_pipeline, classify_scenes_task  # noqa: F401
from .tasks_serial import process_video_serial  # noqa: F401
