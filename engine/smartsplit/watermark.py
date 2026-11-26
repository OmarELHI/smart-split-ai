"""Texte publicitaire ("Twitch : OmarELHI") incruste sur tous les clips.

Le texte est dessine une seule fois dans un PNG transparent (mis en cache dans data/watermark),
puis superpose a chaque rendu. On peut aussi fournir son propre PNG (WATERMARK_IMAGE).
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import (DATA, OUT_W, TITLE_MAX_CHARS, TITLE_SIZE, WATERMARK, WATERMARK_COLOR, WATERMARK_IMAGE,
                     WATERMARK_OPACITY, WATERMARK_SIZE, WATERMARK_Y)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
FONT = "Poppins-Bold.ttf"          # police libre (SIL OFL), fournie dans engine/assets


def png() -> Path | None:
    """PNG transparent du texte, ou None si le texte est desactive."""
    if WATERMARK_IMAGE:
        p = Path(WATERMARK_IMAGE)
        if not p.exists():
            raise FileNotFoundError(f"WATERMARK_IMAGE introuvable : {p}")
        return p
    if not WATERMARK.strip():
        return None
    key = hashlib.md5(f"{WATERMARK}|{WATERMARK_SIZE}|{WATERMARK_COLOR}|{OUT_W}".encode()).hexdigest()[:10]
    out = DATA / "watermark" / f"texte_{key}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "texte.txt").write_text(WATERMARK, encoding="utf-8", newline="\n")
    h = WATERMARK_SIZE * 2 + 40
    border = max(2, round(WATERMARK_SIZE / 16))
    shadow = max(2, round(WATERMARK_SIZE / 13))
    # chemins relatifs (police copiee a cote, texte dans un fichier) : evite l'echappement de "D:\" et ":"
    font = out.parent / FONT
    if not font.exists():
        font.write_bytes((ASSETS / FONT).read_bytes())
    vf = (f"drawtext=fontfile={FONT}:textfile=texte.txt:fontsize={WATERMARK_SIZE}:fontcolor=white:"
          f"borderw={border}:bordercolor=0x{WATERMARK_COLOR}:shadowx={shadow}:shadowy={shadow + 1}:"
          f"shadowcolor=0x{WATERMARK_COLOR}:x=(w-text_w)/2:y=(h-text_h)/2")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c=black@0.0:s={OUT_W}x{h},format=rgba", "-vf", vf,
                    "-frames:v", "1", "-update", "1", out.name], cwd=out.parent, check=True)
    return out


def apply(filter_graph: str, input_index: int, label_in: str = "v", label_out: str = "vw",
          top_px: int | None = None) -> str:
    """Ajoute au graphe la superposition du PNG (entree n° input_index) sur [label_in]."""
    op = max(0.0, min(1.0, WATERMARK_OPACITY))
    return (f"{filter_graph};[{input_index}:v]format=rgba,colorchannelmixer=aa={op:.2f}[wmk];"
            f"[{label_in}][wmk]overlay=(W-w)/2:{f'{top_px}' if top_px is not None else f'H*{WATERMARK_Y}-h/2'}"
            f":format=auto[{label_out}]")


def _clean_title(text: str) -> str:
    """Retire les emojis et symboles que la police ne sait pas dessiner."""
    keep = "’'«»…!?.,:;-()%$€&"
    t = "".join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in keep)
    return " ".join(t.split())


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def _two_lines(text: str) -> list[str]:
    """2 lignes maximum : on elargit un peu la ligne, puis on coupe avec des points de suspension."""
    for width in (TITLE_MAX_CHARS, TITLE_MAX_CHARS + 4):
        lines = _wrap(text, width)
        if len(lines) <= 2:
            return lines
    lines = _wrap(text, TITLE_MAX_CHARS + 4)[:2]
    return [lines[0], lines[1].rstrip(" ,.!?") + "…"]


def _rounded_box(w: int, h: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    return m


def title_png(text: str) -> Path | None:
    """Bandeau du titre d'accroche : carte blanche aux coins arrondis avec une ombre douce, texte noir gras
    centre, 2 lignes maximum."""
    text = _clean_title(text)
    if not text:
        return None
    lines = _two_lines(text)
    key = hashlib.md5(f"v3|{text}|{TITLE_SIZE}|{TITLE_MAX_CHARS}|{OUT_W}".encode()).hexdigest()[:12]
    out = DATA / "watermark" / "titres" / f"titre_{key}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    font = out.parent / FONT
    if not font.exists():
        font.write_bytes((ASSETS / FONT).read_bytes())
    txt = out.with_suffix(".txt")
    txt.write_text("\n".join(lines), encoding="utf-8", newline="\n")   # \n seul : sinon ligne vide sous Windows
    raw = out.with_name(out.stem + "_texte.png")
    canvas_h = TITLE_SIZE * 4
    vf = (f"drawtext=fontfile={FONT}:textfile={txt.name}:fontsize={TITLE_SIZE}:fontcolor=0x111111:"
          f"line_spacing={round(TITLE_SIZE * 0.12)}:text_align=C:x=(w-text_w)/2:y=(h-text_h)/2")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c=black@0.0:s={OUT_W}x{canvas_h},format=rgba", "-vf", vf,
                    "-frames:v", "1", "-update", "1", raw.name], cwd=out.parent, check=True)
    t = cv2.imread(str(raw), cv2.IMREAD_UNCHANGED)
    ys, xs = np.where(t[:, :, 3] > 0)
    t = t[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    th, tw = t.shape[:2]
    padx, pady, r, sh = round(TITLE_SIZE * 0.7), round(TITLE_SIZE * 0.45), round(TITLE_SIZE * 0.45), 18
    bw, bh = min(OUT_W - 60, tw + 2 * padx), th + 2 * pady
    W, H = bw + 2 * sh, bh + 2 * sh
    img = np.zeros((H, W, 4), np.float32)
    box = _rounded_box(bw, bh, r).astype(np.float32) / 255
    shadow = np.zeros((H, W), np.float32)                         # ombre douce, decalee vers le bas
    shadow[sh + 6:sh + 6 + bh, sh:sh + bw] = box
    shadow = cv2.GaussianBlur(shadow, (0, 0), 7) * 0.35
    img[:, :, 3] = shadow                                        # ombre noire
    a = np.zeros((H, W), np.float32)
    a[sh:sh + bh, sh:sh + bw] = box
    img[:, :, :3] = img[:, :, :3] * (1 - a[..., None]) + 255 * a[..., None]
    img[:, :, 3] = a * 0.97 + img[:, :, 3] * (1 - a)
    ox, oy = sh + (bw - tw) // 2, sh + (bh - th) // 2
    ta = t[:, :, 3:4].astype(np.float32) / 255
    reg = img[oy:oy + th, ox:ox + tw]
    reg[:, :, :3] = reg[:, :, :3] * (1 - ta) + t[:, :, :3].astype(np.float32) * ta
    reg[:, :, 3:4] = np.maximum(reg[:, :, 3:4], ta)
    img[:, :, 3] *= 255                                           # opacite de 0..1 vers 0..255
    cv2.imwrite(str(out), np.clip(img, 0, 255).astype(np.uint8))
    raw.unlink(missing_ok=True)
    return out
