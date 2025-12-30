import os
import json
from redis import Redis
from rq import Queue
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Shot
from app.settings import settings

def _redis_conn() -> Redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(redis_url)

def enqueue_process_shot(shot_id: str) -> str:
    q = Queue("default", connection=_redis_conn())
    job = q.enqueue("app.workers.jobs.process_shot", shot_id)
    return job.id

def _db_session() -> Session:
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    return Session(engine)

def process_shot(shot_id: str) -> None:
    db = _db_session()
    try:
        shot = db.get(Shot, shot_id)
        if not shot:
            print(f"[JOB] shot_id {shot_id} não encontrado")
            return

        shot.status = "processing"
        db.commit()

        # MVP: por enquanto "mock" do resultado
        # depois aqui entra MediaPipe + métricas
        result = {
            "overall_score": 78,
            "alerts": [
                {"code": "LOW_RELEASE", "severity": "medium", "message": "Release baixo"},
                {"code": "TRUNK_FORWARD", "severity": "low", "message": "Tronco levemente inclinado"},
                {"code": "ELBOW_OPEN", "severity": "low", "message": "Cotovelo um pouco aberto"},
            ],
        }

        shot.result_json = json.dumps(result, ensure_ascii=False)
        shot.status = "done"
        db.commit()

        print(f"[JOB] processado shot {shot_id} OK")

    except Exception as e:
        print(f"[JOB] erro processando {shot_id}: {e}")
        shot = db.get(Shot, shot_id)
        if shot:
            shot.status = "failed"
            db.commit()
        raise
    finally:
        db.close()
