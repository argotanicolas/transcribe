import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from .config import settings
from .transcriber import transcribe, ALL_EXTENSIONS

app = FastAPI(title="Transcribe API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}
UPLOAD_DIR = Path(settings.temp_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def require_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

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

    try:
        async with aiofiles.open(file_path, "wb") as out:
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_file_size_mb}MB limit")
                await out.write(chunk)

        result = transcribe(str(file_path), language)

        txt_path = UPLOAD_DIR / f"{job_id}.txt"
        srt_path = UPLOAD_DIR / f"{job_id}.srt"
        txt_path.write_text(result["text"], encoding="utf-8")
        srt_path.write_text(result["srt_content"], encoding="utf-8")

        JOBS[job_id] = {"txt": str(txt_path), "srt": str(srt_path), "filename": Path(file.filename or "audio").stem}

        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "language": result["language"],
            "duration": result["duration"],
            "text": result["text"],
            "srt_content": result["srt_content"],
            "segments": result["segments"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()

@app.get("/download/{job_id}/txt")
def download_txt(job_id: str, _: None = Depends(require_api_key)):
    job = JOBS.get(job_id)
    if not job or not Path(job["txt"]).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(job["txt"], media_type="text/plain", filename=f"{job['filename']}.txt")

@app.get("/download/{job_id}/srt")
def download_srt(job_id: str, _: None = Depends(require_api_key)):
    job = JOBS.get(job_id)
    if not job or not Path(job["srt"]).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(job["srt"], media_type="text/plain", filename=f"{job['filename']}.srt")
