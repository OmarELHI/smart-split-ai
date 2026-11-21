"""Texte publicitaire ("Twitch : OmarELHI") incruste sur tous les clips.

Le texte est dessine une seule fois dans un PNG transparent (mis en cache dans data/watermark),
puis superpose a chaque rendu. On peut aussi fournir son propre PNG (WATERMARK_IMAGE).
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .config import (DATA, OUT_W, TITLE_MAX_CHARS, TITLE_SIZE, WATERMARK, WATERMARK_COLOR, WATERMARK_IMAGE,
                     WATERMARK_SIZE, WATERMARK_Y)

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
    (out.parent / "texte.txt").write_text(WATERMARK, encoding="utf-8")
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


def apply(filter_graph: str, input_index: int, label_in: str = "v", label_out: str = "vw") -> str:
    """Ajoute au graphe la superposition du PNG (entree n° input_index) sur [label_in]."""
    return (f"{filter_graph};[{label_in}][{input_index}:v]"
            f"overlay=(W-w)/2:H*{WATERMARK_Y}-h/2:format=auto[{label_out}]")


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
    return lines[:3]


def title_png(text: str) -> Path | None:
    """Bandeau du titre d'accroche : fond blanc, texte noir gras, centre, 3 lignes maximum."""
    text = _clean_title(text)
    if not text:
        return None
    lines = _wrap(text, TITLE_MAX_CHARS)
    key = hashlib.md5(f"{text}|{TITLE_SIZE}|{TITLE_MAX_CHARS}|{OUT_W}".encode()).hexdigest()[:12]
    out = DATA / "watermark" / "titres" / f"titre_{key}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    font = out.parent / FONT
    if not font.exists():
        font.write_bytes((ASSETS / FONT).read_bytes())
    txt = out.with_suffix(".txt")
    txt.write_text("\n".join(lines), encoding="utf-8")
    pad = round(TITLE_SIZE * 0.45)
    h = round(len(lines) * TITLE_SIZE * 1.35 + 2 * pad + 20)
    vf = (f"drawtext=fontfile={FONT}:textfile={txt.name}:fontsize={TITLE_SIZE}:fontcolor=black:"
          f"line_spacing={round(TITLE_SIZE * 0.2)}:text_align=C:box=1:boxcolor=white:boxborderw={pad}:"
          f"x=(w-text_w)/2:y=(h-text_h)/2")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c=black@0.0:s={OUT_W}x{h},format=rgba", "-vf", vf,
                    "-frames:v", "1", "-update", "1", out.name], cwd=out.parent, check=True)
    return out
