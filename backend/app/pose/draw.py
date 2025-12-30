import cv2

# pares simples (MediaPipe Pose landmarks em 2D)
POSE_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

def draw_pose(frame_bgr, keypoints: dict, thickness: int = 2):
    """
    keypoints: dict {name: {"x":..., "y":..., "score":...}} em pixels
    """
    h, w = frame_bgr.shape[:2]

    def pt(name):
        p = keypoints.get(name)
        if not p:
            return None
        x, y = int(p["x"]), int(p["y"])
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return (x, y)

    # desenha joints
    for name, p in keypoints.items():
        x, y = int(p["x"]), int(p["y"])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)

    # desenha “ossos”
    for a, b in POSE_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame_bgr, pa, pb, (0, 255, 255), thickness)

    return frame_bgr
