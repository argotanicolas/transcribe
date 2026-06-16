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
                error        TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
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

def set_done(job_id: str, language: str, duration: float, text: str, srt: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status='done', language=?, duration=?, text_content=?, srt_content=? WHERE id=?",
            (language, duration, text, srt, job_id)
        )
        conn.commit()

def set_error(job_id: str, error: str):
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET status='error', error=? WHERE id=?", (error, job_id))
        conn.commit()

def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None
