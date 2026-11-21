"""Outil de mise au point : exporte les images cles d'une video en JPG (960x540).

Exemples (depuis D:\\Logiciels\\Smart Split) :
  python engine\\dump_frames.py "2026-11-19 16-11-36.mp4"
  python engine\\dump_frames.py "F:\\Videos stream\\long.mp4" --every 15     (1 image cle sur 15, environ 1 par minute)
Resultat : data\\analysis\\<nom de la video>\\frames\\<instant en dixiemes de seconde>.jpg
"""
import argparse
import subprocess
from pathlib import Path

from smartsplit.config import DATA

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("--every", type=int, default=1, help="garde 1 image cle sur N")
args = ap.parse_args()

out = DATA / "analysis" / args.src.stem / "frames"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.jpg"):
    f.unlink()

# instants des images cles (lecture des paquets, sans decodage)
pk = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                     "-of", "csv=p=0", str(args.src)], capture_output=True, text=True).stdout
times = sorted(float(l.split(",")[0]) for l in pk.splitlines() if ",K" in l and l.split(",")[0] not in ("", "N/A"))
times = times[:: max(1, args.every)]

vf = "scale=960:540" if args.every <= 1 else f"select='not(mod(n\\,{args.every}))',scale=960:540"
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey", "-i", str(args.src),
                "-vf", vf, "-fps_mode", "passthrough", "-q:v", "3", "-start_number", "0",
                str(out / "raw_%d.jpg")], check=True)
raws = sorted(out.glob("raw_*.jpg"), key=lambda p: int(p.stem[4:]))
for f, t in zip(raws, times):
    f.rename(out / f"{round(t * 10)}.jpg")
for f in out.glob("raw_*.jpg"):
    f.unlink()
print(f"{min(len(raws), len(times))} images dans {out}")
