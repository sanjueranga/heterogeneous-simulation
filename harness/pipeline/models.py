import uuid
from django.db import models


class Job(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    mode = models.CharField(max_length=8, default="v1")   # v0 | v1
    k = models.IntegerField(default=1)                     # videos in this batch
    run_id = models.CharField(max_length=64, blank=True, default="")
    status = models.CharField(max_length=16, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)


class Video(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="videos")
    filename = models.CharField(max_length=512)
    path = models.CharField(max_length=1024)
    status = models.CharField(max_length=24, default="pending")
    duration = models.FloatField(default=0.0)
    n_scenes = models.IntegerField(default=0)
    cut_done = models.BooleanField(default=False)
    classify_done = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")


class Scene(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="scenes")
    idx = models.IntegerField(default=0)
    start = models.FloatField(default=0.0)
    end = models.FloatField(default=0.0)
    text = models.TextField(blank=True, default="")
    category_id = models.CharField(max_length=64, blank=True, null=True)
    subcategory_id = models.CharField(max_length=64, blank=True, null=True)
    tone = models.CharField(max_length=32, blank=True, default="")
    angle = models.CharField(max_length=32, blank=True, default="")
    persona = models.CharField(max_length=64, blank=True, default="")
    categorized = models.BooleanField(default=False)
    clip_path = models.CharField(max_length=1024, blank=True, default="")


class ClassifyCall(models.Model):
    """One Claude call (same rows also go to classify_calls.csv)."""
    scene = models.ForeignKey(Scene, on_delete=models.CASCADE, null=True,
                              related_name="calls")
    classify_mode = models.CharField(max_length=16, default="injection")
    model = models.CharField(max_length=48, default="")
    scope_id = models.IntegerField(default=0)
    input_tokens = models.IntegerField(default=0)
    cache_creation_input_tokens = models.IntegerField(default=0)
    cache_read_input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    cost_usd = models.FloatField(default=0.0)
    latency_ms = models.FloatField(default=0.0)
    cache_hit = models.BooleanField(default=False)
    cycles = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
