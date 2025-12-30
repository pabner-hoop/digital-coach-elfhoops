import os
import json

import cv2
from redis import Redis
from rq import Queue
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Shot
from app.settings import settings
from app.pose.video_io import read_frame_by_index
from app.pose.draw import draw_pose


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
    """
    Job executado pelo worker.
    """
    # ⬇️ IMPORTS PESADOS DEVEM FICAR AQUI DENTRO
    from app.pose.pipeline import (
        extract_pose_timeseries,
        choose_shooting_side,
        keyframes_simple,
        compute_metrics,
        score_and_alerts,
    )

    db = _db_session()

    try:
        shot = db.get(Shot, shot_id)
        if not shot:
            print(f"[JOB] shot {shot_id} não encontrado")
            return

        shot.status = "processing"
        db.commit()

        # ---------- PIPELINE REAL ----------
        frames = extract_pose_timeseries(
            shot.video_path,
            sample_fps=10,
            max_side=720,
        )

        ok_ratio = sum(1 for f in frames if f.get("ok")) / max(1, len(frames))
        if ok_ratio < 0.5:
            shot.status = "failed"
            shot.result_json = json.dumps(
                {"error": "Pose não detectada com qualidade suficiente"},
                ensure_ascii=False,
            )
            db.commit()
            return

        side = choose_shooting_side(frames)
        kf = keyframes_simple(frames, side)
        if not kf:
            shot.status = "failed"
            shot.result_json = json.dumps(
                {"error": "Não foi possível identificar set/release"},
                ensure_ascii=False,
            )
            db.commit()
            return

        metrics = compute_metrics(frames, side, kf)
        overall, alerts = score_and_alerts(metrics)

        # ---------- REPLAYS (opcional, não pode quebrar o job) ----------
        replays = []
        try:
            media_dir = f"/data/shots/{shot_id}/replays"
            os.makedirs(media_dir, exist_ok=True)

            # Tenta chaves comuns que podem existir no seu pipeline
            possible_kp_keys = [
                "keypoints_px",
                "keypoints",
                "landmarks_px",
                "landmarks",
                "pose_landmarks",
            ]

            for name in ["set", "release", "follow"]:
                frame_idx = kf.get(name)
                if frame_idx is None:
                    continue

                frame = read_frame_by_index(shot.video_path, frame_idx)

                frame_data = frames[frame_idx]
                kp = None
                for key in possible_kp_keys:
                    if key in frame_data:
                        kp = frame_data[key]
                        break

                if kp is None:
                    print(
                        f"[JOB] replay '{name}' ignorado: nenhum keypoints encontrado "
                        f"(keys disponíveis: {list(frame_data.keys())})"
                    )
                    continue

                frame = draw_pose(frame, kp)

                out_path = os.path.join(media_dir, f"{name}.png")
                cv2.imwrite(out_path, frame)

                replays.append(
                    {
                        "name": name,
                        "url": f"/media/shots/{shot_id}/replays/{name}.png",
                        "frame_idx": frame_idx,
                    }
                )

        except Exception as e:
            # nunca falha o job inteiro por causa de replay
            print(f"[JOB] falha gerando replays para {shot_id}: {e}")

        result = {
            "overall_score": overall,
            "side": side,
            "metrics": metrics,
            "alerts": alerts,
            "replays": replays,
            "keyframes": kf,
            "quality": {
                "pose_ok_ratio": ok_ratio,
                "frames_used": len(frames),
            },
        }

        shot.result_json = json.dumps(result, ensure_ascii=False)
        shot.status = "done"
        db.commit()

        print(f"[JOB] processado shot {shot_id} OK (score={overall})")

    except Exception as e:
        print(f"[JOB] erro processando {shot_id}: {e}")
        shot = db.get(Shot, shot_id)
        if shot:
            shot.status = "failed"
            db.commit()
        raise

    finally:
        db.close()

