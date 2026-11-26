"""Etape 1 : mise en page 9:16 selon la scene OBS et rendu d'un clip avec FFmpeg."""
import json
import subprocess
from enum import Enum
from pathlib import Path

from . import watermark
from .audio import audio_tracks, mix_track
from .config import BOTTOM_MARGIN, CAM_SHARE, CAM_TOP_RIGHT, GAME_RATIO, OUT_H, OUT_W, TITLE_Y, WATERMARK_SIZE, Rect


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


_INTERNAL = ("s0", "s1", "s2", "bg", "cam", "game", "t", "a", "b")


def build_filter(scene: Scene, src_w: int, src_h: int, cam: Rect | None = CAM_TOP_RIGHT,
                 face_x: float | str = 0.5, game_ratio: float = GAME_RATIO,
                 bottom_margin: float = BOTTOM_MARGIN, inp: str = "0:v", out: str = "v", tag: str = "") -> str:
    """Filtre FFmpeg qui lit [inp] et produit l'image 1080x1920 dans [out]."""
    f = _build_filter(scene, src_w, src_h, cam or CAM_TOP_RIGHT, face_x, game_ratio, bottom_margin)
    for lab in _INTERNAL:
        f = f.replace(f"[{lab}]", f"[{tag}{lab}]")
    return f.replace("[0:v]", f"[{inp}]").replace("[v]", f"[{out}]")


def _build_filter(scene: Scene, src_w: int, src_h: int, cam: Rect, face_x: float | str,
                  game_ratio: float, bottom_margin: float) -> str:
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
        # zone a eviter : cam + cadre rose (large marge) + pseudo sous la cam
        mx, my = _even(src_w * 0.02), _even(src_h * 0.02)
        avoid = (cx - mx, cy - my, cw + 2 * mx, ch + my + _even(src_h * 0.06))
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
        # face_x : position fixe du visage (0 a 1) ou expression FFmpeg du suivi (etape 3)
        if isinstance(face_x, str):
            fw, _ = _center_crop_width(src_w, src_h, OUT_W / OUT_H)
            x = f"'{face_x}'"
        else:
            fw, x = _center_crop_width(src_w, src_h, OUT_W / OUT_H, face_x)
        return f"[0:v]crop=w={fw}:h={src_h}:x={x}:y=0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v]"
    raise ValueError(scene)


def _face_x(src: Path, start: float, end: float, src_w: int, src_h: int, face_x):
    """face_x = "auto" : suivi du visage sur [start, end] ; repli au centre si impossible."""
    if face_x != "auto":
        return face_x
    try:
        from .face import face_crop_expr
        fw, _ = _center_crop_width(src_w, src_h, OUT_W / OUT_H)
        return face_crop_expr(src, start, end, src_w, fw)
    except Exception as e:  # noqa: BLE001
        print(f"  [suivi du visage indisponible, cadrage centre] {e}")
        return 0.5


def _watermark(vf: str, title: str | None = None, zoom: float = 1.0) -> tuple[list[str], str, str]:
    """Entrees supplementaires, graphe et etiquette de sortie : zoom eventuel, puis titre d'accroche
    et texte publicitaire par-dessus (ils ne zooment pas)."""
    label, inputs, idx = "v", [], 1
    if zoom > 1.001:
        vf += f";[v]crop=iw/{zoom:.3f}:ih/{zoom:.3f},scale={OUT_W}:{OUT_H},setsar=1[vz]"
        label = "vz"
    wm_top = None
    if title:
        png = watermark.title_png(title)
        if png is not None:
            import cv2
            th = cv2.imread(str(png), cv2.IMREAD_UNCHANGED).shape[0]
            top = round(OUT_H * TITLE_Y)
            inputs += ["-i", str(png)]
            vf += f";[{label}][{idx}:v]overlay=(W-w)/2:{top}:format=auto[vt]"
            label, idx = "vt", idx + 1
            wm_top = top + th - 18 + 6             # bas de la carte du titre (ombre exclue) + petit ecart
    wm = watermark.png()
    if wm is not None:
        if wm_top is not None:                     # le PNG du texte a des marges : on aligne le haut des lettres
            import cv2
            hw = cv2.imread(str(wm), cv2.IMREAD_UNCHANGED).shape[0]
            wm_top = round(wm_top - (hw / 2 - 0.45 * WATERMARK_SIZE))
        inputs += ["-i", str(wm)]
        vf = watermark.apply(vf, idx, label_in=label, top_px=wm_top)
        label = "vw"
    return inputs, vf, f"[{label}]"


def _codec(encoder: str) -> tuple[list[str], list[str]]:
    if encoder == "h264_nvenc":
        return ["-hwaccel", "cuda"], ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    return [], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]  # repli sans GPU NVIDIA


def _audio(src: Path) -> str:
    """Son du clip, a ajouter au graphe (sortie [aout]).
    - Plusieurs pistes (OBS multipiste) : le melange est baisse quand Omar parle (compression pilotee par le
      micro), puis la voix est remise par-dessus : le jeu passe sous la voix.
    - Une seule piste (voix et jeu melanges) : on ne peut pas separer le jeu, on ecrase seulement ses pics
      (explosions, musique forte) pour que la voix ressorte mieux."""
    n = audio_tracks(src)
    if n >= 2:
        return (f";[0:a:0]asplit[mic1][mic2];[0:a:{n - 1}][mic1]sidechaincompress=threshold=0.02:ratio=8:"
                f"attack=15:release=350[duck];[duck][mic2]amix=inputs=2:weights=1 0.6:normalize=0,"
                f"alimiter=limit=0.95[aout]")
    if n == 1:
        return ";[0:a:0]acompressor=threshold=0.08:ratio=4:attack=5:release=250:makeup=1.5,alimiter=limit=0.95[aout]"
    return ""


def render_clip(src: Path, start: float, end: float, scene: Scene, out: Path,
                cam: Rect = CAM_TOP_RIGHT, face_x: float | str = 0.5, encoder: str = "h264_nvenc",
                game_ratio: float = GAME_RATIO, bottom_margin: float = BOTTOM_MARGIN,
                title: str | None = None, zoom: float = 1.0) -> Path:
    """Rend le passage [start, end] de la source en clip vertical MP4."""
    info = probe(src)
    if not 0 <= start < end <= info["duration"] + 0.01:
        raise ValueError(f"intervalle {start}-{end} hors de la video ({info['duration']:.1f} s)")
    if scene is Scene.FULL_CAM:
        face_x = _face_x(src, start, end, info["width"], info["height"], face_x)
    vf = build_filter(scene, info["width"], info["height"], cam, face_x, game_ratio, bottom_margin)
    # Le decodage GPU sort du NV12, que certains filtres (flou) lisent mal : on repasse en yuv420p
    vf = vf.replace("[0:v]", "[0:v]format=yuv420p,", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    hw, venc = _codec(encoder)

    wm_in, vf, vout = _watermark(vf, title, zoom)

    if out.suffix.lower() == ".png":  # apercu : une seule image
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
               *wm_in, "-filter_complex", vf, "-map", vout, "-frames:v", "1", str(out)]
        subprocess.run(cmd, check=True)
        return out

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *hw, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src), *wm_in,
        "-filter_complex", vf + _audio(src), "-map", vout, *(["-map", "[aout]"] if audio_tracks(src) else []),
        *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def render_segments(src: Path, start: float, end: float, pieces: list, out: Path,
                    face_x: float | str = 0.5, encoder: str = "h264_nvenc",
                    game_ratio: float = GAME_RATIO, bottom_margin: float = BOTTOM_MARGIN,
                    title: str | None = None, zoom: float = 1.0) -> Path:
    """Rend [start, end] en changeant de mise en page a chaque changement de scene.

    pieces : liste de (debut, fin, scene, cam) en secondes absolues, issue de scenes.segments_for().
    """
    if not pieces:
        raise ValueError("aucune scene sur cet intervalle")
    if len(pieces) == 1:
        a, b, scene, cam = pieces[0]
        return render_clip(src, start, end, Scene(scene), out, cam or CAM_TOP_RIGHT, face_x, encoder,
                           game_ratio, bottom_margin, title, zoom)
    info = probe(src)
    w, h, k = info["width"], info["height"], len(pieces)
    parts = [f"[0:v]format=yuv420p,split={k}" + "".join(f"[i{n}]" for n in range(k)) + ";"]
    for n, (a, b, scene, cam) in enumerate(pieces):
        parts.append(f"[i{n}]trim=start={a - start:.3f}:end={b - start:.3f},setpts=PTS-STARTPTS[t{n}];")
        fx = _face_x(src, a, b, w, h, face_x) if Scene(scene) is Scene.FULL_CAM else 0.5
        parts.append(build_filter(Scene(scene), w, h, cam, fx, game_ratio, bottom_margin,
                                  inp=f"t{n}", out=f"o{n}", tag=f"p{n}") + ";")
    parts.append("".join(f"[o{n}]" for n in range(k)) + f"concat=n={k}:v=1:a=0[v]")
    wm_in, graph, vout = _watermark("".join(parts), title, zoom)
    out.parent.mkdir(parents=True, exist_ok=True)
    hw, venc = _codec(encoder)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *hw, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src), *wm_in,
        "-filter_complex", graph + _audio(src), "-map", vout, *(["-map", "[aout]"] if audio_tracks(src) else []),
        *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
