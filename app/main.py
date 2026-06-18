import logging
import uuid
import threading
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from .config import settings
from .transcriber import transcribe_stream, segments_to_srt, ALL_EXTENSIONS, get_model_size, set_model_size
from .database import (
    init_db, reset_stuck_jobs, create_job,
    set_processing, update_partial, set_done, set_error,
    get_job, list_jobs, delete_job,
)

logger = logging.getLogger("uvicorn.error")

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
reset_stuck_jobs()
logger.info(
    f"=== Transcribe API v2 lista — modelo={settings.model_size} device={settings.device} "
    f"compute={settings.compute_type} threads={settings.cpu_threads} workers={settings.num_workers} "
    f"lang_default={settings.default_language} temp={settings.temp_dir} ==="
)

# cancel flags: job_id → threading.Event
_cancel_flags: dict[str, threading.Event] = {}


def require_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _run_transcription(job_id: str, file_path: str, filename_stem: str, language: str | None, cancel_event: threading.Event):
    path = Path(file_path)
    size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
    logger.info(f"[{job_id}] ▶ Iniciando transcripción — archivo={filename_stem!r} size={size_mb:.1f}MB language={language or f'default({settings.default_language})'}")
    try:
        set_processing(job_id)
        logger.info(f"[{job_id}] ⚙ Cargando modelo Whisper ({settings.model_size})...")

        info, segment_iter = transcribe_stream(file_path, language)
        logger.info(f"[{job_id}] 🎙 Modelo listo — idioma detectado={info.language} duración={info.duration:.1f}s — transcribiendo...")

        segments: list[dict] = []

        for seg in segment_iter:
            if cancel_event.is_set():
                logger.info(f"[{job_id}] 🛑 Cancelación — deteniendo en segmento {len(segments) + 1}")
                break
            segments.append(seg)
            partial = " ".join(s["text"].strip() for s in segments)
            update_partial(job_id, partial)
            logger.info(f"[{job_id}] 📝 [{seg['start']:.1f}s→{seg['end']:.1f}s] {seg['text'].strip()[:80]!r}")

        final_text = " ".join(s["text"].strip() for s in segments)
        srt_content = segments_to_srt(segments)

        if cancel_event.is_set():
            logger.info(f"[{job_id}] ✂️ Parcial guardado — {len(segments)} segmentos, palabras≈{len(final_text.split())}")
        else:
            logger.info(f"[{job_id}] ✔ Completo — {len(segments)} segmentos, palabras≈{len(final_text.split())}")

        txt_path = UPLOAD_DIR / f"{job_id}.txt"
        srt_path = UPLOAD_DIR / f"{job_id}.srt"
        txt_path.write_text(final_text, encoding="utf-8")
        srt_path.write_text(srt_content, encoding="utf-8")

        set_done(job_id, info.language, info.duration, final_text, srt_content)
        logger.info(f"[{job_id}] ✅ Job guardado en SQLite")
    except Exception as e:
        logger.error(f"[{job_id}] ❌ Error en transcripción: {e}", exc_info=True)
        set_error(job_id, str(e))
    finally:
        _cancel_flags.pop(job_id, None)
        if path.exists():
            path.unlink()
            logger.info(f"[{job_id}] 🗑 Archivo temporal eliminado")


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
        logger.warning(f"Formato rechazado: {ext} — archivo={file.filename!r}")
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Allowed: {', '.join(ALL_EXTENSIONS)}")

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}{ext}"
    filename_stem = Path(file.filename or "audio").stem

    logger.info(f"[{job_id}] 📥 Recibido archivo={file.filename!r} ext={ext}")

    try:
        async with aiofiles.open(file_path, "wb") as out:
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    logger.warning(f"[{job_id}] Archivo demasiado grande: {total / 1024 / 1024:.1f}MB > {settings.max_file_size_mb}MB")
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_file_size_mb}MB limit")
                await out.write(chunk)
    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise

    size_mb = file_path.stat().st_size / 1024 / 1024
    logger.info(f"[{job_id}] 💾 Guardado en disco: {size_mb:.1f}MB — lanzando thread...")

    create_job(job_id, file.filename or "audio")

    cancel_event = threading.Event()
    _cancel_flags[job_id] = cancel_event

    thread = threading.Thread(
        target=_run_transcription,
        args=(job_id, str(file_path), filename_stem, language, cancel_event),
        daemon=True,
    )
    thread.start()

    logger.info(f"[{job_id}] 🚀 Thread iniciado — respondiendo 202 al cliente")
    return JSONResponse({"job_id": job_id, "status": "pending"}, status_code=202)


@app.get("/status/{job_id}")
def get_status(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        logger.warning(f"[{job_id}] GET /status — job no encontrado")
        raise HTTPException(status_code=404, detail="Job not found")
    logger.info(f"[{job_id}] GET /status → {job['status']}")

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
    elif job["status"] == "processing":
        if job.get("partial_text"):
            response["partial_text"] = job["partial_text"]
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


# ── Admin: job management ─────────────────────────────────────────────────────

@app.get("/jobs")
def list_all_jobs(_: None = Depends(require_api_key)):
    jobs = list_jobs()
    logger.info(f"GET /jobs → {len(jobs)} jobs")
    return JSONResponse(jobs)

@app.delete("/jobs/{job_id}")
def delete_job_endpoint(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    for ext in ["txt", "srt"]:
        p = UPLOAD_DIR / f"{job_id}.{ext}"
        if p.exists():
            p.unlink()
    delete_job(job_id)
    logger.info(f"[{job_id}] 🗑 Job eliminado por admin")
    return JSONResponse({"ok": True})

@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("pending", "processing"):
        raise HTTPException(status_code=400, detail=f"No se puede cancelar un job en estado '{job['status']}'")

    if job_id in _cancel_flags:
        _cancel_flags[job_id].set()
        logger.info(f"[{job_id}] 🛑 Cancelación solicitada via API")
        return JSONResponse({"ok": True, "message": "Cancelando — el resultado parcial se guardará al terminar el segmento actual"})
    else:
        set_error(job_id, "Cancelado por el usuario")
        return JSONResponse({"ok": True, "message": "Job cancelado"})

@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, _: None = Depends(require_api_key)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "error":
        raise HTTPException(status_code=400, detail="Only error jobs can be retried")
    raise HTTPException(status_code=400, detail="El archivo original fue eliminado. Subí el archivo de nuevo.")

# ── Admin: config ─────────────────────────────────────────────────────────────

@app.get("/config")
def get_config(_: None = Depends(require_api_key)):
    return JSONResponse({
        "model_size": get_model_size(),
        "compute_type": settings.compute_type,
        "device": settings.device,
        "available_models": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    })

@app.patch("/config")
def update_config(body: dict, _: None = Depends(require_api_key)):
    new_size = body.get("model_size")
    allowed = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
    if not new_size or new_size not in allowed:
        raise HTTPException(status_code=400, detail=f"model_size debe ser uno de: {', '.join(sorted(allowed))}")
    old = get_model_size()
    set_model_size(new_size)
    logger.info(f"⚙ Modelo cambiado: {old} → {new_size} (se recargará en próxima transcripción)")
    return JSONResponse({"ok": True, "model_size": new_size})
