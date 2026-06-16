import os
import subprocess
import uuid
from pathlib import Path
from faster_whisper import WhisperModel
from .config import settings

_model: WhisperModel | None = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(settings.model_size, device=settings.device, compute_type=settings.compute_type)
    return _model

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
ALL_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

def extract_audio(input_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")

def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seconds_to_srt_time(seg["start"])
        end = seconds_to_srt_time(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(lines)

def transcribe(file_path: str, language: str | None = None) -> dict:
    ext = Path(file_path).suffix.lower()
    audio_path = file_path

    tmp_audio = None
    if ext in VIDEO_EXTENSIONS:
        tmp_audio = f"/tmp/transcribe/{uuid.uuid4()}.wav"
        os.makedirs("/tmp/transcribe", exist_ok=True)
        extract_audio(file_path, tmp_audio)
        audio_path = tmp_audio

    try:
        model = get_model()
        kwargs = {"beam_size": 5}
        if language:
            kwargs["language"] = language

        raw_segments, info = model.transcribe(audio_path, **kwargs)
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segments]

        return {
            "language": info.language,
            "duration": round(info.duration, 2),
            "text": " ".join(s["text"].strip() for s in segments),
            "srt_content": segments_to_srt(segments),
            "segments": segments,
        }
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)
