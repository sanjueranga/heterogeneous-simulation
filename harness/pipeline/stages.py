"""The three real stages: transcribe (WhisperX base int8 on CPU), segment (group
whisper segments into scenes) and cut (ffmpeg)."""
import os
import subprocess
import pathlib

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
ATOMS_TARGET = int(os.environ.get("ATOMS_TARGET", "12"))
# cap threads per transcribe so several can run side by side instead of one
# taking every core. 0 = all cores, e.g. 2 to fit ~4 transcribes on 8 cores.
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "0"))

_model_cache = {}


def _load_model():
    import whisperx
    kw = {"device": "cpu", "compute_type": WHISPER_COMPUTE}
    if WHISPER_THREADS > 0:
        kw["threads"] = WHISPER_THREADS
    return whisperx.load_model(WHISPER_MODEL, **kw)


def _model(reuse=True):
    # reuse=True (v1): load once per worker child. reuse=False (v0): load every task.
    if not reuse:
        return _load_model()
    key = (WHISPER_MODEL, WHISPER_COMPUTE)
    if key not in _model_cache:
        _model_cache[key] = _load_model()
    return _model_cache[key]


def transcribe(video_path, reuse=True):
    """Returns [{start, end, text}]."""
    import whisperx
    audio = whisperx.load_audio(str(video_path))  # 16k mono
    result = _model(reuse=reuse).transcribe(audio, batch_size=8)
    return [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text", "").strip()}
        for s in result.get("segments", [])
    ]


def segment(whisper_segments, video_duration=None):
    """Group segments into roughly ATOMS_TARGET scenes. No speech -> one scene for the whole video."""
    segs = [s for s in whisper_segments if s.get("text")]
    if not segs:
        end = video_duration or 0.0
        return [{"start": 0.0, "end": end, "text": ""}]
    n = len(segs)
    target = max(1, min(ATOMS_TARGET, n))
    chunk = max(1, (n + target - 1) // target)
    atoms = []
    for i in range(0, n, chunk):
        grp = segs[i:i + chunk]
        atoms.append({
            "start": grp[0]["start"],
            "end": grp[-1]["end"],
            "text": " ".join(g["text"] for g in grp).strip(),
        })
    return atoms


def cut(video_path, atom, out_path):
    """Cut one scene out with ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(atom["start"]), "-to", str(atom["end"]),
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=120)
    return out_path


def probe_duration(video_path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def list_videos():
    d = pathlib.Path(__file__).resolve().parent.parent / "videos_in"
    return sorted(str(p) for p in d.glob("*.mp4"))
