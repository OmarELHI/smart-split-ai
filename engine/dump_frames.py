"""Outil de mise au point : exporte les images cles d'une video en JPG (960x540).

Exemple (depuis D:\\Logiciels\\Smart Split) :
  python engine\\dump_frames.py "2026-11-19 16-11-36.mp4"
Resultat : data\\analysis\\<nom de la video>\\frames\\<seconde>.jpg
"""
import subprocess
import sys
from pathlib import Path

from smartsplit.config import DATA

src = Path(sys.argv[1])
out = DATA / "analysis" / src.stem / "frames"
out.mkdir(parents=True, exist_ok=True)
# une image par image cle, nommee par son instant (en dixiemes de seconde)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey", "-i", str(src),
                "-vf", "scale=960:540", "-fps_mode", "passthrough", "-frame_pts", "1", "-q:v", "3",
                str(out / "%d.jpg")], check=True)
files = sorted(out.glob("*.jpg"), key=lambda p: int(p.stem))
print(f"{len(files)} images dans {out}")
