"""Etape 1 : mise en page 9:16 selon la scene OBS et rendu d'un clip avec FFmpeg."""
import json
import subprocess
from enum import Enum
from pathlib import Path

from .config import BOTTOM_MARGIN, CAM_SHARE, CAM_TOP_RIGHT, GAME_RATIO, OUT_H, OUT_W, Rect


class Scene(str, Enum):
    CAM_JEU = "cam_jeu"    # cam incrustee + jeu
    JEU = "jeu"            # jeu seul
    FULL_CAM = "full_cam"  # webcam plein cadre


def probe(src: Path) -> dict:
    """Largeur, hauteur et duree de la video source."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(src)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    st = info["streams"][0]
    return {"width": st["width"], "height": st["height"], "duration": float(info["format"]["duration"])}


def _even(v: float) -> int:
    return int(round(v / 2)) * 2


def _center_crop_width(src_w: int, src_h: int, ratio: float, center_x: float = 0.5) -> tuple[int, int]:
    """Largeur et position x d'un recadrage pleine hauteur au ratio voulu (largeur / hauteur)."""
    w = min(_even(src_h * ratio), src_w)
    x = _even(center_x * src_w - w / 2)
    return w, max(0, min(x, src_w - w))


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def game_crop(src_w: int, src_h: int, ratio: float, cam_px: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Plus grande zone de jeu au ratio voulu qui ne contient pas l'incrustation de la cam.

    1. pleine hauteur, centree ; 2. decalee a l'oppose de la cam ;
    3. sinon on retire le haut de l'image (au-dessus du bas de la cam).
    """
    cx, cy, cw, ch = cam_px
    for y0 in (0, _even(cy + ch)):
        h = src_h - y0
        w = _even(h * ratio)
        if w > src_w:
            w = src_w
            h = _even(w / ratio)
        x = max(0, min(_even((src_w - w) / 2), src_w - w))
        rect = (x, y0, w, h)
        if not _overlaps(rect, cam_px):
            return rect
        # decale a l'oppose de la cam
        x = cx - w if cx + cw / 2 > src_w / 2 else cx + cw
        if 0 <= x <= src_w - w:
            rect = (_even(x), y0, w, h)
            if not _overlaps(rect, cam_px):
                return rect
    return (0, src_h - h, w, h)  # dernier recours


def build_filter(scene: Scene, src_w: int, src_h: int, cam: Rect = CAM_TOP_RIGHT,
                 face_x: float = 0.5, game_ratio: float = GAME_RATIO,
                 bottom_margin: float = BOTTOM_MARGIN) -> str:
    """Construit le filtre FFmpeg qui produit l'image 1080x1920 dans [v]."""
    if scene is Scene.CAM_JEU:
        cam_h = _even(OUT_H * CAM_SHARE)                   # 596 px
        game_h = _even(OUT_W / game_ratio)                 # 810 px en 4:3
        game_y = OUT_H - _even(OUT_H * bottom_margin) - game_h
        cam_y = game_y - cam_h
        if cam_y < 0:
            raise ValueError("cam + jeu plus hauts que l'ecran : reduire le ratio ou la marge")
        cam_px = cam.to_px(src_w, src_h)
        cx, cy, cw, ch = cam_px
        # zone a eviter : cam + cadre rose + pseudo sous la cam
        avoid = (cx - 12, cy - 12, cw + 24, ch + _even(src_h * 0.035) + 12)
        gx, gy, gw, gh = game_crop(src_w, src_h, game_ratio, avoid)
        bw, bx = _center_crop_width(src_w, src_h, OUT_W / OUT_H)   # fond : centre de l'image en 9:16
        return (
            f"[0:v]split=3[s0][s1][s2];"
            f"[s0]crop={bw}:{src_h}:{bx}:0,scale=270:480,boxblur=16:3,"
            f"scale={OUT_W}:{OUT_H},eq=brightness=-0.08[bg];"
            f"[s1]crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{cam_h}:flags=lanczos[cam];"
            f"[s2]crop={gw}:{gh}:{gx}:{gy},scale={OUT_W}:{game_h}:flags=lanczos[game];"
            f"[bg][cam]overlay=0:{cam_y}[t];[t][game]overlay=0:{game_y},setsar=1[v]"
        )
    if scene is Scene.JEU:
        gw, gx = _center_crop_width(src_w, src_h, OUT_W / OUT_H)
        return f"[0:v]crop={gw}:{src_h}:{gx}:0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v]"
    if scene is Scene.FULL_CAM:
        # face_x : position horizontale du visage (0 a 1). Le suivi automatique arrive a l'etape 3.
        fw, fx = _center_crop_width(src_w, src_h, OUT_W / OUT_H, face_x)
        return f"[0:v]crop={fw}:{src_h}:{fx}:0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v]"
    raise ValueError(scene)


def render_clip(src: Path, start: float, end: float, scene: Scene, out: Path,
                cam: Rect = CAM_TOP_RIGHT, face_x: float = 0.5, encoder: str = "h264_nvenc",
                game_ratio: float = GAME_RATIO, bottom_margin: float = BOTTOM_MARGIN) -> Path:
    """Rend le passage [start, end] de la source en clip vertical MP4."""
    info = probe(src)
    if not 0 <= start < end <= info["duration"] + 0.01:
        raise ValueError(f"intervalle {start}-{end} hors de la video ({info['duration']:.1f} s)")
    vf = build_filter(scene, info["width"], info["height"], cam, face_x, game_ratio, bottom_margin)
    # Le decodage GPU sort du NV12, que certains filtres (flou) lisent mal : on repasse en yuv420p
    vf = vf.replace("[0:v]", "[0:v]format=yuv420p,", 1)
    out.parent.mkdir(parents=True, exist_ok=True)

    if encoder == "h264_nvenc":
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
        hw = ["-hwaccel", "cuda"]
    else:  # repli sans GPU NVIDIA
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        hw = []

    if out.suffix.lower() == ".png":  # apercu : une seule image
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
               "-filter_complex", vf, "-map", "[v]", "-frames:v", "1", str(out)]
        subprocess.run(cmd, check=True)
        return out

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *hw, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
        *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
