"""Test de la vision sur un passage : ce que l'IA voit a l'ecran, et en combien de temps.

  python engine\\vision_test.py "F:\\Videos stream\\2026-11-19 21-26-39.mp4" --start 25:00 --end 29:00

Affiche une ligne par fenetre de 30 s et enregistre les mosaiques envoyees a l'IA dans
data\\analysis\\<video>\\vision_test\\ (pour verifier ce qu'elle a recu).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2

from smartsplit import llm, scenes, vision


def secs(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    v = 0.0
    for p in parts:
        v = v * 60 + p
    return v


ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("--start", default="0")
ap.add_argument("--end", required=True)
args = ap.parse_args()
a, b = secs(args.start), secs(args.end)

out = scenes.analysis_dir(args.src) / "vision_test"
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
print("Extraction des images...")
frames = []
for ts, img in scenes.frames_between(args.src, a, b, fps=0.25):   # une image toutes les 4 s, comme les images cles
    if ts is None:
        continue
    p = out / f"{round(ts * 10)}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frames.append((ts, p))
an = scenes.load(args.src)
segments = an.segments if an else []
print(f"{len(frames)} images ; modele {llm.check()}")

t0 = time.time()
wins = vision.analyze(args.src, segments, b, start=a, end=b, frames=frames)
for w in wins:
    m, s = divmod(int(w["start"]), 60)
    h, m = divmod(m, 60)
    print(f"  {h}:{m:02d}:{s:02d}  {w['action']:12s} {w['intensite']:2d}/10  {w['description']}"
          + (f"   [{w['top3']}]" if w.get("top3") else ""))
    inside = [p for t, p in frames if w["start"] <= t < w["end"]]
    if inside:
        (out / f"mosaique_{int(w['start'])}.jpg").write_bytes(vision._mosaic(inside[:: max(1, len(inside) // 4)][:4]))
(out / "resultat.json").write_text(json.dumps(wins, ensure_ascii=False, indent=1), encoding="utf-8")
el = time.time() - t0
n = max(1, len(wins) - 1)
t_one = vision.LAST_TIMES[-1] if getattr(vision, "LAST_TIMES", None) else el / n
print(f"\n{len(wins)} fenetres en {el:.0f} s (chargement du modele compris)")
print(f"Hors chargement : {t_one:.2f} s par fenetre, soit environ {t_one * 600 / 60:.0f} min pour 5 h de stream")
print(f"Mosaiques : {out}")
