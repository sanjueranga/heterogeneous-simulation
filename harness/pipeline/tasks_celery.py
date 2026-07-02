"""v1, the fork-join fleet. The heavy worker transcribes and makes the scenes,
then forks classify onto the light queue and cuts clips itself while that runs.
The whisper model is loaded once per worker child."""
from celery import shared_task
from . import stages, instrument, runner
from .models import Video


@shared_task(queue="video_processing")
def process_video_pipeline(job_id, video_id):
    import os
    os.environ["PIPELINE_MODE"] = "v1"   # so stage timings get tagged right
    v = Video.objects.get(id=video_id)
    v.status = "transcribing"
    v.save(update_fields=["status"])
    v.duration = stages.probe_duration(v.path)

    with instrument.timed(video_id, "transcribe"):
        segs = stages.transcribe(v.path, reuse=True)
    with instrument.timed(video_id, "segment"):
        atoms = stages.segment(segs, v.duration)
    scenes = runner.make_scenes(v, atoms)

    # fork: classify on the light queue, returns right away
    classify_scenes_task.delay(job_id, video_id)

    # meanwhile cut on this worker
    with instrument.timed(video_id, "cut"):
        runner.cut_all(v, scenes)
    Video.objects.filter(id=video_id).update(cut_done=True)
    runner.check_complete(video_id)


@shared_task(queue="api_calls")
def classify_scenes_task(job_id, video_id):
    v = Video.objects.get(id=video_id)
    with instrument.timed(video_id, "classify"):
        runner.classify_scenes(v)
    Video.objects.filter(id=video_id).update(classify_done=True)
    runner.check_complete(video_id)
