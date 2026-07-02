"""v0, the serial monolith. One task does transcribe -> segment -> cut -> classify
for a video, one thing at a time. The whisper model is reloaded every task and
there's no light queue."""
from celery import shared_task
from . import stages, instrument, runner
from .models import Video


@shared_task(queue="video_processing")
def process_video_serial(job_id, video_id):
    import os
    os.environ["PIPELINE_MODE"] = "v0"
    v = Video.objects.get(id=video_id)
    v.status = "processing"
    v.save(update_fields=["status"])
    v.duration = stages.probe_duration(v.path)

    with instrument.timed(video_id, "transcribe"):
        segs = stages.transcribe(v.path, reuse=False)
    with instrument.timed(video_id, "segment"):
        atoms = stages.segment(segs, v.duration)
    scenes = runner.make_scenes(v, atoms)

    # fully serial: one cut at a time, one classify call at a time
    with instrument.timed(video_id, "cut"):
        runner.cut_all(v, scenes, threads=1)
    Video.objects.filter(id=video_id).update(cut_done=True)

    with instrument.timed(video_id, "classify"):
        runner.classify_scenes(v, threads=1)
    Video.objects.filter(id=video_id).update(classify_done=True)
    runner.check_complete(video_id)
