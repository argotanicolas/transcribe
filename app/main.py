import os
import uuid
import threading
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from .config import settings
from .transcriber import transcribe, ALL_EXTENSIONS
from .database import init_db, create_job, set_processing, set_done, set_error, get_job

app = FastAPI(title="Transcribe API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(settings.temp_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

init_db()


def require_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _run_transcription(job_id: str, file_path: str, filename_stem: str, language: str | None):
    path = Path(file_path)
    try:
        set_processing(job_id)
        result = transcribe(file_path, language)

        txt_path = UPLOAD_DIR / f"{job_id}.txt"
        srt_path = UPLOAD_DIR / f"{job_id}.srt"
        txt_path.write_text(result["text"], encoding="utf-8")
        srt_path.write_text(result["srt_content"], encoding="utf-8")

        set_done(job_id, result["language"], result["duration"], result["text"], result["srt_content"])
    except Exception as e:
        set_error(job_id, str(e))
    finally:
        if path.exists():
            path.unlink()


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_size, "device": settings.device}


@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str | None = None,
    _: None = Depends(require_api_key),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALL_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Allowed: {', '.join(ALL_EXTENSIONS)}")

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}{ext}"
    filename_stem = Path(file.filename or "audio").stem

    try:
        async with aiofiles.open(file_path, "wb") as out:
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_file_size_mb}MB limit")
                await out.write(chunk)
    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise

    create_job(job_id, file.filename or "audio")

    thread = threading.Thread(
        target=_run_transcription,
        args=(job_id, str(file_path), filename_stem, language),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "pending"}, status_code=202)


@app.get("/status/{job_id}")
def get_status(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response: dict = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
    }

    if job["status"] == "done":
        response.update({
            "language": job["language"],
            "duration": job["duration"],
            "text": job["text_content"],
            "srt_content": job["srt_content"],
        })
    elif job["status"] == "error":
        response["error"] = job["error"]

    return JSONResponse(response)


@app.get("/download/{job_id}/txt")
def download_txt(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    txt_path = UPLOAD_DIR / f"{job_id}.txt"
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    stem = Path(job["filename"]).stem
    return FileResponse(str(txt_path), media_type="text/plain", filename=f"{stem}.txt")


@app.get("/download/{job_id}/srt")
def download_srt(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    srt_path = UPLOAD_DIR / f"{job_id}.srt"
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    stem = Path(job["filename"]).stem
    return FileResponse(str(srt_path), media_type="text/plain", filename=f"{stem}.srt")
