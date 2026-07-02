"""Shared stage helpers used by both v0 (serial) and v1 (fork-join) tasks."""
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.db import transaction

from . import stages, classify, instrument
from .models import Video, Scene, ClassifyCall

CUT_THREADS = int(os.environ.get("CUT_THREADS", "2"))
CLASSIFY_THREADS = int(os.environ.get("CLASSIFY_THREADS", "4"))
CLIP_DIR = pathlib.Path(os.environ.get("CLIP_DIR", "/data/clips"))


def make_scenes(video, atoms):
    scenes = []
    with transaction.atomic():
        for i, a in enumerate(atoms):
            scenes.append(Scene.objects.create(
                video=video, idx=i, start=a["start"], end=a["end"], text=a["text"]))
    video.n_scenes = len(scenes)
    video.save(update_fields=["n_scenes"])
    return scenes


def classify_one(scene, scope_id=0):
    label, usage = classify.classify_scene(scene.text, scope_id=scope_id)
    instrument.log_classify(usage, video_id=scene.video_id, scene_id=scene.id)
    scene.category_id = label.get("category_id")
    scene.subcategory_id = label.get("subcategory_id")
    scene.tone = label.get("tone") or ""
    scene.angle = label.get("angle") or ""
    scene.persona = label.get("persona") or ""
    scene.categorized = True
    scene.save()
    ClassifyCall.objects.create(
        scene=scene, classify_mode=os.environ.get("CLASSIFY_MODE", "injection"),
        model=usage.get("model", ""), scope_id=scope_id,
        input_tokens=usage.get("input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cost_usd=usage.get("cost_usd", 0.0), latency_ms=usage.get("latency_ms", 0.0),
        cache_hit=usage.get("cache_hit", False), cycles=usage.get("cycles", 1))
    return usage


def classify_scenes(video, threads=None):
    threads = threads or CLASSIFY_THREADS
    scenes = list(video.scenes.all())
    if threads <= 1:
        for s in scenes:
            classify_one(s)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(classify_one, s) for s in scenes]
            for f in as_completed(futs):
                f.result()


def cut_all(video, scenes, threads=None):
    threads = threads or CUT_THREADS
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    def _cut(scene):
        out = CLIP_DIR / f"clip_{video.id}_{scene.id}.mp4"
        stages.cut(video.path, {"start": scene.start, "end": scene.end}, out)
        scene.clip_path = str(out)
        scene.save(update_fields=["clip_path"])

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(_cut, s) for s in scenes]
        for f in as_completed(futs):
            f.result()


def check_complete(video_id):
    with transaction.atomic():
        v = Video.objects.select_for_update().get(id=video_id)
        if v.cut_done and v.classify_done and v.status != "completed":
            v.status = "completed"
            v.save(update_fields=["status"])
