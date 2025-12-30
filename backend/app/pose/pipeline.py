import cv2
import numpy as np
import mediapipe as mp

from app.pose.metrics import angle_3pts, vertical_angle

mp_pose = mp.solutions.pose

# índices do MediaPipe Pose
LM = mp_pose.PoseLandmark

def extract_pose_timeseries(video_path: str, sample_fps: int = 10, max_side: int = 720):
    """
    Retorna lista de dict por frame:
      - t (seg)
      - landmarks: dict com pontos normalizados (x,y,vis)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / sample_fps)), 1)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step != 0:
            idx += 1
            continue

        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        t = idx / fps
        if res.pose_landmarks is None:
            frames.append({"t": t, "ok": False, "lm": None})
        else:
            lm = {}
            for p in LM:
                pt = res.pose_landmarks.landmark[p.value]
                lm[p.name] = {"x": float(pt.x), "y": float(pt.y), "v": float(pt.visibility)}
            frames.append({"t": t, "ok": True, "lm": lm})

        idx += 1

    cap.release()
    pose.close()
    return frames

def choose_shooting_side(frames) -> str:
    """Escolhe 'right' ou 'left' baseado em visibilidade média do punho."""
    r = []
    l = []
    for f in frames:
        if not f["ok"]:
            continue
        lm = f["lm"]
        r.append(lm["RIGHT_WRIST"]["v"])
        l.append(lm["LEFT_WRIST"]["v"])
    if not r and not l:
        return "right"
    return "right" if (np.mean(r) if r else 0) >= (np.mean(l) if l else 0) else "left"

def keyframes_simple(frames, side: str):
    """
    Heurística simples:
    - set: frame onde punho está mais alto (menor y) antes do release
    - release: frame de pico de velocidade vertical do punho (descendo->subindo)
    - follow: alguns frames após release (clamp)
    """
    wrist = "RIGHT_WRIST" if side == "right" else "LEFT_WRIST"

    valid = [(i, f) for i, f in enumerate(frames) if f["ok"]]
    if len(valid) < 10:
        return None

    ys = np.array([f["lm"][wrist]["y"] for _, f in valid], dtype=float)
    ts = np.array([f["t"] for _, f in valid], dtype=float)

    # derivada aproximada dy/dt
    dy = np.gradient(ys, ts + 1e-6)

    # release: ponto de mínimo dy (punho subindo mais rápido => dy bem negativo em coords imagem)
    rel_k = int(np.argmin(dy))
    rel_idx = valid[rel_k][0]

    # set: mínimo y nos frames antes do release (punho mais alto)
    before = ys[: rel_k + 1]
    set_k = int(np.argmin(before))
    set_idx = valid[set_k][0]

    follow_idx = min(rel_idx + 3, len(frames) - 1)

    return {"set": set_idx, "release": rel_idx, "follow": follow_idx}

def compute_metrics(frames, side: str, kf: dict):
    """Calcula métricas base do MVP."""
    sh = "RIGHT_SHOULDER" if side == "right" else "LEFT_SHOULDER"
    el = "RIGHT_ELBOW" if side == "right" else "LEFT_ELBOW"
    wr = "RIGHT_WRIST" if side == "right" else "LEFT_WRIST"

    lh = "LEFT_HIP"
    rh = "RIGHT_HIP"

    def pt(frame, name):
        return (frame["lm"][name]["x"], frame["lm"][name]["y"])

    setf = frames[kf["set"]]
    relf = frames[kf["release"]]
    folf = frames[kf["follow"]]

    # elbow angles
    elbow_set = angle_3pts(pt(setf, sh), pt(setf, el), pt(setf, wr))
    elbow_release = angle_3pts(pt(relf, sh), pt(relf, el), pt(relf, wr))

    # release height: wrist vs shoulder (quanto acima, melhor). Em coords normalizadas, menor y = mais alto.
    release_height = pt(relf, sh)[1] - pt(relf, wr)[1]  # positivo => punho acima do ombro

    # trunk angle: usa centro do quadril -> centro do ombro
    hip = ((pt(relf, lh)[0] + pt(relf, rh)[0]) / 2, (pt(relf, lh)[1] + pt(relf, rh)[1]) / 2)
    shoulder_center = ((pt(relf, "LEFT_SHOULDER")[0] + pt(relf, "RIGHT_SHOULDER")[0]) / 2,
                       (pt(relf, "LEFT_SHOULDER")[1] + pt(relf, "RIGHT_SHOULDER")[1]) / 2)
    trunk_ang = vertical_angle(hip, shoulder_center)

    # follow-through: punho acima do cotovelo e cotovelo acima do ombro
    w_y = pt(folf, wr)[1]
    e_y = pt(folf, el)[1]
    s_y = pt(folf, sh)[1]
    follow_ok = (w_y < e_y) and (e_y < s_y)  # menor y = mais alto
    follow_score = 1.0 if follow_ok else 0.0

    return {
        "elbow_angle_set": elbow_set,
        "elbow_angle_release": elbow_release,
        "release_height": release_height,
        "trunk_angle": trunk_ang,
        "followthrough_score": follow_score,
    }

def score_and_alerts(m):
    """
    Transforma métricas em score + alertas simples (MVP).
    Ajuste thresholds depois com dados reais.
    """
    alerts = []

    # cotovelo
    def elbow_subscore(a):
        if 70 <= a <= 110:
            return 1.0
        # cai linear fora da faixa
        dist = min(abs(a - 70), abs(a - 110))
        return max(0.0, 1.0 - dist / 40.0)

    elbow_s = elbow_subscore(m["elbow_angle_release"])
    if m["elbow_angle_release"] < 70:
        alerts.append(("ELBOW_CLOSED", "medium", "Cotovelo muito fechado no release"))
    elif m["elbow_angle_release"] > 110:
        alerts.append(("ELBOW_OPEN", "medium", "Cotovelo muito aberto no release"))

    # release height
    # >= 0.03 acima do ombro (normalizado) é ok (depende do enquadramento, mas serve pro MVP)
    rh = m["release_height"]
    if rh <= 0.0:
        rel_s = 0.0
        alerts.append(("LOW_RELEASE", "high", "Release baixo (punho abaixo do ombro)"))
    else:
        rel_s = min(1.0, rh / 0.06)

    # tronco (0–25° aceitável)
    ta = m["trunk_angle"]
    if ta > 25:
        trunk_s = max(0.0, 1.0 - (ta - 25) / 25.0)
        alerts.append(("TRUNK_FORWARD", "medium", "Tronco inclinado demais para frente"))
    else:
        trunk_s = 1.0

    # follow-through
    ft = m["followthrough_score"]
    if ft < 1.0:
        alerts.append(("NO_EXTENSION", "low", "Faltou extensão no follow-through"))

    # score ponderado
    score = (
        30 * elbow_s +
        25 * rel_s +
        25 * trunk_s +
        20 * ft
    )

    # ordena por severidade
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    alerts = sorted(alerts, key=lambda x: sev_rank.get(x[1], 9))[:5]
    alerts_out = [{"code": c, "severity": s, "message": msg} for c, s, msg in alerts]

    return int(round(score)), alerts_out
