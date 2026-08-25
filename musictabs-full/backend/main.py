"""
main.py — REST API для анализа треков.

Эндпоинты:
  POST /analyze          — загрузить аудиофайл, получить job_id
  GET  /analyze/{job_id}  — статус/результат анализа

Для MVP обработка идёт в фоновом потоке (BackgroundTasks) и хранится
в памяти. Для продакшена: заменить на очередь (Celery/RQ) + Redis/Postgres,
и хранить файлы в S3/аналоге вместо локального диска.
"""

import os
import uuid
import shutil
import tempfile
from enum import Enum

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from chord_recognition import analyze_chords

app = FastAPI(title="MusicTabs API", version="0.1")

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "musictabs_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory job store: {job_id: {"status": ..., "result": ..., "error": ...}}
JOBS: dict[str, dict] = {}


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class AnalyzeResponse(BaseModel):
    job_id: str
    status: JobStatus


def _run_analysis(job_id: str, file_path: str):
    JOBS[job_id]["status"] = JobStatus.PROCESSING
    try:
        result = analyze_chords(file_path)
        JOBS[job_id]["status"] = JobStatus.DONE
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = JobStatus.FAILED
        JOBS[job_id]["error"] = str(e)
    finally:
        # чистим временный файл
        try:
            os.remove(file_path)
        except OSError:
            pass


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Неподдерживаемый формат файла")

    job_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": JobStatus.QUEUED, "result": None, "error": None}
    background_tasks.add_task(_run_analysis, job_id, dest_path)

    return AnalyzeResponse(job_id=job_id, status=JobStatus.QUEUED)


@app.get("/analyze/{job_id}")
async def get_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job не найден")

    if job["status"] == JobStatus.FAILED:
        return JSONResponse(status_code=500, content={"status": job["status"], "error": job["error"]})

    return {"status": job["status"], "result": job["result"]}


@app.get("/health")
async def health():
    return {"ok": True}
