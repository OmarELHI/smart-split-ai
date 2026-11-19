"""Etape 3 : suivi du visage en full cam.

On detecte le visage quelques fois par seconde (modele YuNet d'OpenCV, telecharge une
fois dans data\\models), on lisse sa position, puis on la transforme en expression FFmpeg
qui deplace le recadrage vertical au fil du temps, sans a-coups.
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import DATA

# OpenCV 5 affiche un avertissement sans consequence au chargement du modele : on le masque
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # noqa: BLE001
    pass

MODEL = DATA / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
DW, DH = 640, 360          # resolution de detection
TRACK_FPS = 5.0            # detections par seconde
MIN_FACE = 0.05            # visage plus petit que 5 % de la largeur : ignore (poster, ecran...)
DEAD_ZONE = 0.03           # le cadre ne bouge pas tant que le visage reste a moins de 3 % du centre
SMOOTH_S = 0.7            # duree du lissage du mouvement (plus petit = cadre plus reactif)
KEY_STEP = 0.5             # une position cle toutes les 0,5 s dans l'expression FFmpeg


def ensure_model() -> Path:
    """Telecharge le modele de detection de visage (environ 230 Ko) s'il manque."""
    if MODEL.exists() and MODEL.stat().st_size > 100_000:
        return MODEL
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, MODEL)
            if MODEL.stat().st_size > 100_000:
                return MODEL
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"telechargement du modele de visage impossible : {last}")


class FaceDetector:
    def __init__(self, score: float = 0.6):
        self.det = cv2.FaceDetectorYN.create(str(ensure_model()), "", (DW, DH), score, 0.3, 5)

    def largest(self, img: np.ndarray) -> tuple[float, float, float, float, float] | None:
        """Plus grand visage : (x, y, w, h, score) en coordonnees relatives, ou None."""
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
        if fw / w < MIN_FACE:
            return None
        return x / w, y / h, fw / w, fh / h, score


def _frames(src: Path, start: float, end: float, fps: float):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(src), "-vf", f"fps={fps},scale={DW}:{DH}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = DW * DH * 3
    i = 0
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(DH, DW, 3)
        i += 1
    p.wait()


def track(src: Path, start: float, end: float, detector: FaceDetector | None = None) -> list[tuple[float, float | None]]:
    """Centre horizontal du visage (0 a 1) a chaque instant, en secondes depuis start."""
    det = detector or FaceDetector()
    out = []
    for t, img in _frames(src, start, end, TRACK_FPS):
        f = det.largest(img)
        out.append((t, (f[0] + f[2] / 2) if f else None))
    return out


def smooth(points: list[tuple[float, float | None]], duration: float) -> list[tuple[float, float]]:
    """Trajectoire douce du cadre : trous combles, zone morte, lissage sans retard."""
    if not points or all(c is None for _, c in points):
        return [(0.0, 0.5), (duration, 0.5)]
    ts = np.array([t for t, _ in points])
    cs = np.array([np.nan if c is None else c for _, c in points], dtype=float)
    ok = ~np.isnan(cs)
    cs = np.interp(ts, ts[ok], cs[ok])                       # trous : interpolation
    k = 3                                                    # mediane sur 0,6 s : supprime les detections aberrantes
    pad = np.pad(cs, k // 2, mode="edge")
    cs = np.array([np.median(pad[i:i + k]) for i in range(len(cs))])
    # zone morte : le cadre ne suit que les vrais deplacements
    held = np.empty_like(cs)
    cur = cs[0]
    for i, c in enumerate(cs):
        if abs(c - cur) > DEAD_ZONE:
            cur = c - np.sign(c - cur) * DEAD_ZONE
        held[i] = cur
    # lissage gaussien (aller-retour, donc sans retard) sur environ SMOOTH_S secondes
    sigma = SMOOTH_S * TRACK_FPS / 2
    r = int(3 * sigma)
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    ker /= ker.sum()
    sm = np.convolve(np.pad(held, r, mode="edge"), ker, mode="valid")
    keys = np.arange(0, duration + 1e-6, KEY_STEP)
    vals = np.interp(keys, ts, sm)
    if keys[-1] < duration:
        keys, vals = np.append(keys, duration), np.append(vals, vals[-1])
    return list(zip(keys.tolist(), vals.tolist()))


def crop_x_expr(keys: list[tuple[float, float]], src_w: int, crop_w: int) -> str:
    """Expression FFmpeg de la position x du recadrage, interpolee entre les positions cles."""
    xs = [min(max(c * src_w - crop_w / 2, 0), src_w - crop_w) for _, c in keys]
    ts = [t for t, _ in keys]
    if len(keys) == 1 or max(xs) - min(xs) < 2:
        return f"{xs[0]:.1f}"
    expr = f"{xs[-1]:.1f}"
    for i in range(len(ts) - 2, -1, -1):
        t0, t1, x0, x1 = ts[i], ts[i + 1], xs[i], xs[i + 1]
        seg = f"{x0:.1f}+({x1 - x0:.1f})*(t-{t0:.2f})/{max(t1 - t0, 1e-3):.2f}"
        expr = f"if(lt(t,{t1:.2f}),{seg},{expr})"
    return expr


def face_crop_expr(src: Path, start: float, end: float, src_w: int, crop_w: int,
                   detector: FaceDetector | None = None) -> str:
    """Raccourci : suivi + lissage + expression pour le passage [start, end]."""
    return crop_x_expr(smooth(track(src, start, end, detector), end - start), src_w, crop_w)
