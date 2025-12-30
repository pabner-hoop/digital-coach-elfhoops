import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends

from app.db import get_db, init_db
from app.models import Shot
from app.workers.jobs import enqueue_process_shot

UPLOAD_DIR = "/data/uploads"

app = FastAPI(title="Digital Coach ElfHoops")

@app.on_event("startup")
def on_startup():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    init_db()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/shots")
async def create_shot(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Envie um arquivo de vídeo.")

    shot_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{shot_id}.mp4")

    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    shot = Shot(id=shot_id, status="queued", video_path=path)
    db.add(shot)
    db.commit()

    job_id = enqueue_process_shot(shot_id)

    return {"shot_id": shot_id, "status": "queued", "job_id": job_id}

@app.get("/shots/{shot_id}")
def get_shot(shot_id: str, db: Session = Depends(get_db)):
    shot = db.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot não encontrado.")

    return {
        "shot_id": shot.id,
        "status": shot.status,
        "result": shot.result_json,
        "created_at": shot.created_at.isoformat(),
    }
