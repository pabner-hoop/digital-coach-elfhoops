import math
import numpy as np

def angle_3pts(a, b, c) -> float:
    """Ângulo em graus no ponto b formado por a-b-c."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def vertical_angle(hip, shoulder) -> float:
    """Ângulo do vetor hip->shoulder vs vertical (0 = reto)."""
    hx, hy = hip
    sx, sy = shoulder
    vx, vy = (sx - hx), (sy - hy)
    # vertical "para cima" = (0, -1) em coordenadas de imagem
    denom = (math.hypot(vx, vy) * 1.0) + 1e-9
    cosang = (vx * 0 + vy * (-1)) / denom
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))
