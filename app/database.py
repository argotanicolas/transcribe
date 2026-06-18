import sqlite3
from pathlib import Path

DB_PATH = Path("/tmp/transcribe/jobs.db")

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'pending',
                filename     TEXT,
                language     TEXT,
                duration     REAL,
                text_content TEXT,
                srt_content  TEXT,
                partial_text TEXT,
                error        TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN partial_text TEXT")
        except Exception:
            pass
        conn.commit()

def reset_stuck_jobs():
    """On startup: jobs stuck in processing/pending have no live thread — mark as error."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status='error', error='Servicio reiniciado durante la transcripción' "
            "WHERE status IN ('processing', 'pending')"
        )
        conn.commit()

def create_job(job_id: str, filename: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status, filename) VALUES (?, 'pending', ?)",
            (job_id, filename)
        )
        conn.commit()

def set_processing(job_id: str):
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET status='processing' WHERE id=?", (job_id,))
        conn.commit()

def update_partial(job_id: str, partial_text: str):
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET partial_text=? WHERE id=?", (partial_text, job_id))
        conn.commit()

def set_done(job_id: str, language: str, duration: float, text: str, srt: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status='done', language=?, duration=?, text_content=?, srt_content=?, partial_text=NULL WHERE id=?",
            (language, duration, text, srt, job_id)
        )
        conn.commit()

def set_error(job_id: str, error: str):
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET status='error', error=?, partial_text=NULL WHERE id=?", (error, job_id))
        conn.commit()

def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

def list_jobs() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def delete_job(job_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.commit()
